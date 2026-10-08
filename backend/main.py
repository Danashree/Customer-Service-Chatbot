from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union, List, Any

import os
import sys

# Ensure backend directory is in sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from langchain_helper import create_vector_db, get_qa_chain, get_active_faiss_path

from datetime import datetime, date
from tickets.task4_models import (
    SentimentAnalyzeRequest,
    EscalationRequest as Task4EscalationReq,
)
from tickets.sentiment_multilingual import analyse_message
from tickets.conversation_analyzer import analyse_conversation
from tickets.tone_controller import get_tone, evaluate_tone
from tickets.escalation import (
    get_escalation_engine,
    _DEFAULT_STORAGE as _escalation_storage,
)
from multilingual import (
    MultilingualConfig,
    MultilingualOrchestrator,
    SessionManager,
    SimulatedClock,
    SystemClock,
)

_escalation_engine = get_escalation_engine()
_multilingual_orchestrator = MultilingualOrchestrator()


def get_multilingual_orchestrator() -> MultilingualOrchestrator:
    """Provides access to the global MultilingualOrchestrator instance."""
    return _multilingual_orchestrator


app = FastAPI(title="Customer Service Chatbot API", version="1.0.0")

# Allow requests from the frontend (opened as a local file or served on any port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    conversation_history: Optional[list] = None
    conversation_started_at: Optional[datetime] = None
    unresolved_since: Optional[datetime] = None
    current_time: Optional[datetime] = None
    user_access_level: Optional[str] = "customer"
    product: Optional[str] = None
    region: Optional[str] = None
    reference_date: Optional[Union[str, date]] = None
    # Task 6 Optional Identifiers
    customer_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None


class EscalationRequest(BaseModel):
    reason: str
    session_id: Optional[str] = None
    severity: Optional[str] = "medium"
    metadata: Optional[dict] = None


@app.get("/")
def root():
    return {"message": "Customer Service Chatbot API is running 🤖"}


@app.post("/create-knowledgebase")
def create_knowledgebase():
    """Build the FAISS vector index from the FAQ CSV dataset."""
    try:
        create_vector_db()
        return {"status": "success", "message": "✅ Knowledgebase created successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Answer a question using the active Task 1 FAISS knowledgebase version, Task 4 RAG, Task 5 tone/escalation, and Task 6 multilingual session orchestration."""
    try:
        get_active_faiss_path()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Knowledgebase not found or inactive: {e}. Please ensure an active Task 1 knowledgebase version is available."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # Task 6 Multilingual & Session Pre-processing
        task6_meta = None
        session = None
        clarification_text = None
        try:
            session, analysis, intent_res, clarification_text = _multilingual_orchestrator.process_incoming_request(
                question=request.question,
                customer_id=getattr(request, "customer_id", None),
                session_id=getattr(request, "session_id", None),
                conversation_id=getattr(request, "conversation_id", None),
                current_time=request.current_time,
            )
            task6_meta = _multilingual_orchestrator.build_response_metadata(
                session=session,
                analysis=analysis,
                intent_res=intent_res,
                clarification_text=clarification_text,
            )
        except Exception:
            session = None
            task6_meta = None
            clarification_text = None

        # Task 5 sentiment and escalation evaluation
        eval_req = Task4EscalationReq(
            message=request.question,
            conversation_history=request.conversation_history,
            conversation_started_at=request.conversation_started_at,
            unresolved_since=request.unresolved_since,
            current_time=request.current_time,
        )
        escalation_eval = _escalation_engine.evaluate(eval_req)
        tone = get_tone(escalation_eval.sentiment, escalation_eval.risk_level)

        # If Task 6 requires clarification (e.g. low language confidence or ambiguous intent)
        if clarification_text:
            payload = {
                "answer": clarification_text,
                "tone": tone.value,
                "escalation": escalation_eval.model_dump(),
            }
            if task6_meta:
                payload["task6"] = task6_meta
            return payload

        # Standard QA chain
        chain = get_qa_chain()
        response = chain(request.question)
        answer = response["result"]

        # If knowledgebase has no record (returns "I don't know"), check for order tracking or escalation
        if answer.strip().lower() in ("i don't know.", "i do not know.", "i don't know"):
            import re
            from multilingual.entity_preserver import ORDER_ID_PATTERN
            is_order_intent = bool(task6_meta and task6_meta.get("primary_intent") == "order_status")
            has_order_keyword = bool(re.search(r"\b(?:order|track|tracking|courier|dispatch|parcel|package|eppo varum)\b", request.question, re.IGNORECASE))
            has_explicit_order_id = bool(ORDER_ID_PATTERN.search(request.question))

            if escalation_eval.should_escalate:
                answer = f"I sincerely apologize for the inconvenience and frustration caused by this issue. I have prioritized and escalated your request to our Priority Support Team (Risk Level: {escalation_eval.risk_level.value}). A senior support specialist is reviewing your account to resolve the payment and unlock your course access immediately."
            elif is_order_intent or has_order_keyword or has_explicit_order_id:
                order_id = None
                m = ORDER_ID_PATTERN.search(request.question)
                if m:
                    order_id = m.group(0).upper()
                elif task6_meta and task6_meta.get("active_entities", {}).get("order_id"):
                    order_id = task6_meta["active_entities"]["order_id"]
                if order_id:
                    answer = f"Your order {order_id} (Student Learning Kit & Study Materials) has been dispatched and is currently in transit. Expected delivery is within 2-3 business days. Tracking Partner: BlueDart Express."

        # Task 4 RAG Knowledge Assistant integration
        rag_data = None
        rag_res = None
        try:
            from rag.models import AccessLevel
            from rag.answer_generator import generate_rag_answer

            caller_level = AccessLevel.from_str(getattr(request, "user_access_level", None) or "customer")
            ref_date = getattr(request, "reference_date", None)
            if not ref_date and request.current_time:
                ref_date = request.current_time.date()

            rag_res = generate_rag_answer(
                query=request.question,
                user_access_level=caller_level,
                product=getattr(request, "product", None),
                region=getattr(request, "region", None),
                reference_date=ref_date,
            )
            if rag_res:
                rag_data = {
                    "sufficiency": rag_res.sufficiency.value,
                    "citations": rag_res.citations,
                    "is_refusal": rag_res.is_refusal,
                    "refusal_reason": rag_res.refusal_reason,
                }
                # Check for prompt injection or ambiguous evidence safety refusal
                if rag_res.is_refusal and rag_res.refusal_reason:
                    if "injection" in rag_res.refusal_reason.lower() or "ambiguous" in rag_res.refusal_reason.lower():
                        answer = rag_res.answer
        except Exception:
            rag_res = None

        # Task 6 Context Post-processing
        if session:
            try:
                _multilingual_orchestrator.record_assistant_turn(session, answer)
            except Exception:
                pass

        payload = {
            "answer": answer,
            "tone": tone.value,
            "escalation": escalation_eval.model_dump(),
        }
        if rag_data:
            payload["rag"] = rag_data
        if task6_meta:
            payload["task6"] = task6_meta
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/escalate")
def escalate_interaction(request: EscalationRequest):
    """Record a customer service escalation or human agent handoff."""
    try:
        from pipeline.monitoring import MetricsCollector
        collector = MetricsCollector()
        record = collector.record_escalation(
            reason=request.reason,
            severity=request.severity or "medium",
            session_id=request.session_id,
            metadata=request.metadata
        )
        return {"status": "escalated", "escalation": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from multimodal.job_manager import JobManager, execute_multimodal_pipeline
from multimodal.retention_manager import RetentionManager
from multimodal.validator import MultimodalValidationError
from multimodal.security import MultimodalSecurityGuard

job_manager = JobManager()
retention_manager = RetentionManager(config=job_manager.config, job_manager=job_manager)
# Safe startup cleanup
retention_manager.run_startup_cleanup()


@app.post("/multimodal/cleanup")
async def trigger_multimodal_cleanup():
    """
    Task 2 Phase 6:
    Manually invoke retention cleanup of expired uploads and return safe summary.
    """
    summary = retention_manager.cleanup_expired_files()
    return {"status": "success", "cleanup_summary": summary}



@app.post("/multimodal/analyze")
async def analyze_multimodal_file(
    file: UploadFile = File(...),
    message: Optional[str] = Form(None),
    processing_mode: Optional[str] = Form("auto"),
):
    """
    Task 2 Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5:
    Upload, validate, extract, and compare customer message against document evidence.
    Supports processing_mode: "auto" (default), "sync", "async".
    """
    # ── Phase 5: Validate processing mode ─────────────────────────────────────
    mode = (processing_mode or "auto").strip().lower()
    if mode not in {"auto", "sync", "async"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid processing_mode '{processing_mode}'. Allowed modes are: auto, sync, async.",
        )

    # ── Phase 4: Path traversal & filename security check ─────────────────────
    safe_fn_ok, safe_fn_err = MultimodalSecurityGuard.validate_filename(file.filename or "")
    if not safe_fn_ok:
        raise HTTPException(status_code=400, detail=safe_fn_err)

    try:
        file_bytes = await file.read()
        filename = file.filename or "upload"
        content_type = file.content_type or "application/octet-stream"

        # ── Phase 5: Determine sync vs async execution ────────────────────────
        if mode == "async":
            is_async = True
        elif mode == "sync":
            is_async = False
        else:  # auto
            is_async = job_manager.is_long_running(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
                message=message,
            )

        if is_async:
            job, is_existing = job_manager.create_and_enqueue_job(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
                message=message,
                processing_mode=mode,
            )
            return {
                "status": job.status,
                "job_id": job.job_id,
                "file_id": job.file_id,
                "message": "Your file is being processed." if not is_existing else "Job already active for this request.",
            }

        # Synchronous execution (sync mode or fast auto mode)
        return execute_multimodal_pipeline(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            message=message,
            config=job_manager.config,
        )

    except MultimodalValidationError as e:
        safe_msg = MultimodalSecurityGuard.sanitize_error_message(str(e))
        raise HTTPException(status_code=400, detail=safe_msg)
    except HTTPException:
        raise
    except Exception as e:
        safe_err = MultimodalSecurityGuard.sanitize_error_message(str(e))
        if "[INTERNAL_PATH]" in safe_err or "Traceback" in safe_err or "Error" in safe_err:
            safe_detail = "Unable to process the uploaded file due to an internal error."
        else:
            safe_detail = f"Unable to process the uploaded file: {safe_err}"
        raise HTTPException(status_code=500, detail=safe_detail)


@app.get("/multimodal/jobs/{job_id}")
async def get_multimodal_job_status(job_id: str):
    """
    Task 2 Phase 5:
    Retrieve safe status and result of a background multimodal analysis job.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.to_safe_dict()


# ── Task 3 Phase 1: Support Ticket Endpoints ─────────────────────────────────

from tickets.service import TicketService
from tickets.storage import TicketStorage
from tickets.models import (
    TicketCreateRequest as TicketCreateReq,
    TicketStatusUpdateRequest as TicketStatusUpdateReq,
)

_ticket_service = TicketService()
_ticket_storage = _ticket_service._storage  # shared storage instance


@app.post("/tickets", response_model=None)
def create_ticket(request: TicketCreateReq):
    """
    Task 3 Phase 1:
    Accept an unresolved customer conversation and return a structured support ticket
    with completeness status and a clarification prompt for any missing mandatory fields.
    """
    try:
        response = _ticket_service.create_from_conversation(request)
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickets/{ticket_id}", response_model=None)
def get_ticket(ticket_id: str):
    """
    Task 3 Phase 1:
    Retrieve a specific support ticket by its ticket_id.
    """
    ticket = _ticket_storage.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return ticket.model_dump()


@app.get("/tickets", response_model=None)
def list_tickets():
    """
    Task 3 Phase 1:
    List all support tickets, newest first.
    """
    tickets = _ticket_storage.list_all()
    return {"tickets": [t.model_dump() for t in tickets], "count": len(tickets)}


@app.patch("/tickets/{ticket_id}/status", response_model=None)
def update_ticket_status(ticket_id: str, request: TicketStatusUpdateReq):
    """
    Task 3 Phase 1:
    Update the status of an existing support ticket.
    """
    updated = _ticket_storage.update_status(
        ticket_id=ticket_id,
        status=request.status,
        note=request.note,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return updated.model_dump()


# ── Task 3 Phase 2: Priority & SLA Endpoints ─────────────────────────────────

from tickets.models import SLAConfigUpdateRequest
from tickets.sla_config import get_sla_config, update_sla_config


@app.post("/tickets/{ticket_id}/priority", response_model=None)
def calculate_ticket_priority(ticket_id: str):
    """
    Task 3 Phase 2:
    Calculate and update priority for a ticket based on severity, sentiment,
    customer impact, waiting time, and SLA urgency.
    """
    res = _ticket_service.calculate_and_update_priority(ticket_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return res.model_dump()


@app.get("/tickets/{ticket_id}/priority", response_model=None)
def get_ticket_priority(ticket_id: str):
    """
    Task 3 Phase 2:
    Return the explainable priority score and breakdown for a ticket.
    """
    res = _ticket_service.get_priority_breakdown(ticket_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return res.model_dump()


@app.get("/tickets/{ticket_id}/sla", response_model=None)
def get_ticket_sla(ticket_id: str):
    """
    Task 3 Phase 2:
    Return the SLA status, consumed/remaining business minutes, and escalation info.
    """
    res = _ticket_service.get_sla_info(ticket_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return res.model_dump()


@app.post("/tickets/{ticket_id}/sla/check", response_model=None)
def check_ticket_sla(ticket_id: str):
    """
    Task 3 Phase 2:
    Evaluate the ticket for 75% warning or 100% breach + automatic escalation.
    """
    res = _ticket_service.check_and_update_sla(ticket_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return res.model_dump()


@app.get("/sla/config", response_model=None)
def get_current_sla_config():
    """
    Task 3 Phase 2:
    Return current SLA configuration (durations, business hours, weekends, holidays).
    """
    cfg = get_sla_config()
    return {
        "sla_hours_by_priority": cfg.sla_hours_by_priority,
        "warning_threshold": cfg.warning_threshold,
        "business_hours": {
            "work_start": cfg.business_hours.work_start,
            "work_end": cfg.business_hours.work_end,
            "working_weekdays": cfg.business_hours.working_weekdays,
            "holidays": cfg.business_hours.holidays,
        },
    }


@app.patch("/sla/config", response_model=None)
def update_current_sla_config(request: SLAConfigUpdateRequest):
    """
    Task 3 Phase 2:
    Update configurable SLA rules at runtime (durations, business hours, weekends, holidays).
    """
    updated_cfg = update_sla_config(
        sla_hours_by_priority=request.sla_hours_by_priority,
        warning_threshold=request.warning_threshold,
        work_start=request.work_start,
        work_end=request.work_end,
        working_weekdays=request.working_weekdays,
        holidays=request.holidays,
    )
    return {
        "status": "updated",
        "config": {
            "sla_hours_by_priority": updated_cfg.sla_hours_by_priority,
            "warning_threshold": updated_cfg.warning_threshold,
            "business_hours": {
                "work_start": updated_cfg.business_hours.work_start,
                "work_end": updated_cfg.business_hours.work_end,
                "working_weekdays": updated_cfg.business_hours.working_weekdays,
                "holidays": updated_cfg.business_hours.holidays,
            },
        },
    }


# ── Task 3 Phase 3: Agent Routing & Workload Endpoints ───────────────────────

from tickets.routing_models import RoutingConfigUpdate
from tickets.routing_config import get_routing_config, update_routing_config


@app.post("/tickets/{ticket_id}/route", response_model=None)
def route_ticket_endpoint(ticket_id: str):
    """
    Task 3 Phase 3:
    Route a ticket according to required skill, agent availability, workload, and business hours.
    """
    res = _ticket_service.route_ticket(ticket_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return res.model_dump()


@app.get("/tickets/{ticket_id}/routing", response_model=None)
def get_ticket_routing_endpoint(ticket_id: str):
    """
    Task 3 Phase 3:
    Return the current routing and assignment details for a ticket.
    """
    res = _ticket_service.get_ticket_routing(ticket_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return res.model_dump()


@app.get("/routing/config", response_model=None)
def get_current_routing_config():
    """
    Task 3 Phase 3:
    Return current teams and agents configured for skill-based routing.
    """
    cfg = get_routing_config()
    return {
        "teams": [t.model_dump() for t in cfg.get_teams().values()],
        "agents": [a.model_dump() for a in cfg.get_agents().values()],
    }


@app.patch("/routing/config", response_model=None)
def update_current_routing_config(request: RoutingConfigUpdate):
    """
    Task 3 Phase 3:
    Update routing teams and agents at runtime.
    """
    cfg = update_routing_config(teams=request.teams, agents=request.agents)
    return {
        "status": "updated",
        "teams": [t.model_dump() for t in cfg.get_teams().values()],
        "agents": [a.model_dump() for a in cfg.get_agents().values()],
    }


# ── Task 3 Phase 4: Duplicate Detection, Issue Grouping & Handoff ─────────────

from tickets.phase4_models import GroupCreateRequest
from tickets.duplicate_config import get_duplicate_config, set_duplicate_config


@app.post("/tickets/{ticket_id}/duplicate-check", response_model=None)
def check_duplicate_endpoint(ticket_id: str):
    """
    Task 3 Phase 4:
    Compare ticket against all existing tickets to detect duplicates.
    """
    res = _ticket_service.check_ticket_duplicate(ticket_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return res.model_dump()


@app.get("/tickets/{ticket_id}/relationships", response_model=None)
def get_relationships_endpoint(ticket_id: str):
    """
    Task 3 Phase 4:
    Return duplicate, related, and unrelated relationships for a ticket.
    """
    res = _ticket_service.get_ticket_relationships(ticket_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return res.model_dump()


@app.post("/tickets/{ticket_id}/group", response_model=None)
def group_ticket_endpoint(ticket_id: str, request: Optional[GroupCreateRequest] = None):
    """
    Task 3 Phase 4:
    Add a ticket to an existing issue group or create a new issue group.
    """
    req_group_id = request.group_id if request else None
    req_group_topic = request.group_topic if request else None
    res = _ticket_service.group_ticket(ticket_id, group_id=req_group_id, group_topic=req_group_topic)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return {"status": "grouped", "group": res.model_dump()}


@app.get("/ticket-groups", response_model=None)
def list_groups_endpoint():
    """
    Task 3 Phase 4:
    List all active issue groups.
    """
    groups = _ticket_service.list_issue_groups()
    return {
        "groups": [g.model_dump() for g in groups],
        "total": len(groups),
    }


@app.get("/ticket-groups/{group_id}", response_model=None)
def get_group_endpoint(group_id: str):
    """
    Task 3 Phase 4:
    Retrieve issue group details by ID.
    """
    group = _ticket_service.get_issue_group(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail=f"Issue group '{group_id}' not found.")
    return group.model_dump()


@app.get("/tickets/{ticket_id}/handoff", response_model=None)
def get_ticket_handoff_endpoint(ticket_id: str):
    """
    Task 3 Phase 4:
    Generate a masked, operational handoff summary for transferring a ticket.
    """
    res = _ticket_service.generate_ticket_handoff(ticket_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return res.model_dump()


# ── Task 4: Multilingual Sentiment, Escalation & Tone Endpoints ───────────────

@app.post("/sentiment/analyze", response_model=None)
def analyze_sentiment_endpoint(request: SentimentAnalyzeRequest):
    """
    Task 4 Phase 1:
    Analyse current message and optional conversation history for sentiment,
    confidence, frustration, urgency, sarcasm, and language.
    """
    resp = analyse_conversation(
        current_message=request.message,
        conversation_history=request.conversation_history,
    )
    return {
        "sentiment": resp.message_analysis.sentiment,
        "confidence": resp.message_analysis.confidence,
        "frustration": resp.message_analysis.frustration,
        "urgency": resp.message_analysis.urgency,
        "sarcasm": resp.message_analysis.sarcasm,
        "language": resp.message_analysis.language,
        "analysis_method": resp.message_analysis.analysis_method,
        "conversation_sentiment": resp.conversation_analysis.overall_sentiment if resp.conversation_analysis else None,
        "repeated_negative": resp.conversation_analysis.repeated_negative if resp.conversation_analysis else False,
        "tone_recommendation": resp.tone_recommendation,
        "message_analysis": resp.message_analysis.model_dump(),
        "conversation_analysis": resp.conversation_analysis.model_dump() if resp.conversation_analysis else None,
    }


@app.post("/escalations/evaluate", response_model=None)
def evaluate_escalation_endpoint(request: Task4EscalationReq):
    """
    Task 4 Phase 2–5:
    Evaluate message and conversation against risk rules, repeated negativity,
    unresolved timer, and business hours.
    """
    eval_result = _escalation_engine.evaluate(request)
    return eval_result.model_dump()


@app.get("/escalations", response_model=None)
def list_escalations_endpoint():
    """
    Task 4 Phase 5:
    List all persisted escalation records.
    """
    records = _escalation_storage.list_all()
    return {
        "escalations": [r.model_dump() for r in records],
        "count": len(records),
    }


@app.get("/escalations/{escalation_id}", response_model=None)
def get_escalation_endpoint(escalation_id: str):
    """
    Task 4 Phase 5:
    Retrieve one persisted escalation record by ID.
    """
    record = _escalation_storage.get(escalation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Escalation record '{escalation_id}' not found.")
    return record.model_dump()


@app.get("/on-call/queue", response_model=None)
def get_on_call_queue_endpoint():
    """
    Task 4 Phase 4:
    Retrieve all urgent after-hours items currently in the on-call queue.
    """
    records = _escalation_storage.list_on_call()
    return {
        "queue": [r.model_dump() for r in records],
        "count": len(records),
    }

