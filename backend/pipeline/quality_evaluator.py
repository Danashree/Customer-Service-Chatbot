import os
import json
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from langchain_community.vectorstores import FAISS

from .config import PipelineConfig
from .vector_store_manager import VectorStoreManager

logger = logging.getLogger("pipeline.quality_evaluator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [QualityEvaluator] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default baseline evaluation questions grounded in the knowledge base
DEFAULT_BENCHMARK_CASES = [
    {
        "id": "case_1",
        "question": "What is the refund policy?",
        "expected_keywords": ["refund", "policy"]
    },
    {
        "id": "case_2",
        "question": "Can I use Power BI on a Mac computer?",
        "expected_keywords": ["power bi", "mac"]
    },
    {
        "id": "case_3",
        "question": "Do you offer EMI payment plans?",
        "expected_keywords": ["emi"]
    },
    {
        "id": "case_4",
        "question": "How do I change my email address?",
        "expected_keywords": ["email", "account"]
    },
    {
        "id": "case_5",
        "question": "How to cancel an order that has already shipped?",
        "expected_keywords": ["cancel", "order"]
    },
    {
        "id": "case_6",
        "question": "What are the features of the wireless noise-canceling headphones?",
        "expected_keywords": ["headphones", "battery"]
    }
]

# Known prompt injection pattern signatures to detect in documents
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(all\s+)?(rules|instructions|prompts)",
    r"reveal\s+(the\s+)?(system\s+prompt|developer\s+instructions)",
    r"system\s+override",
    r"act\s+as\s+(an?\s+)?unrestricted",
    r"bypass\s+(all\s+)?security",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"exfiltrate|leak\s+confidential",
]


class QualityEvaluator:
    """
    Task 1: Phase 3 Quality Evaluation Gate.
    Evaluates candidate knowledge-base versions against quality thresholds and
    regression bounds before any version can be activated.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        vector_store_manager: Optional[VectorStoreManager] = None,
        embeddings=None,
        benchmark_cases: Optional[List[Dict]] = None
    ):
        self.config = config or PipelineConfig()
        self.vsm = vector_store_manager or VectorStoreManager(config=self.config, embeddings=embeddings)
        self.embeddings = embeddings
        self.benchmark_cases = benchmark_cases or DEFAULT_BENCHMARK_CASES

    def get_embeddings(self):
        """Returns the configured embeddings from self or VectorStoreManager."""
        if self.embeddings is not None:
            return self.embeddings
        return self.vsm.get_embeddings()

    # ------------------------------------------------------------------
    # Prompt-Injection Pattern Inspection (Document Content Guard)
    # ------------------------------------------------------------------
    @staticmethod
    def scan_for_malicious_instructions(text: str) -> List[str]:
        """
        Scans document text for malicious prompt-injection patterns.
        Treats them strictly as untrusted text content, ensuring instructions
        are never executed or allowed to alter evaluation outcomes.
        """
        detected = []
        for pattern in INJECTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                detected.append(match.group(0))
        return detected

    # ------------------------------------------------------------------
    # Single-Version Retrieval & Grounding Evaluation
    # ------------------------------------------------------------------
    def evaluate_version_raw(
        self,
        version: str,
        test_cases: Optional[List[Dict]] = None
    ) -> Tuple[bool, float, float, List[Dict], List[str]]:
        """
        Evaluates retrieval and grounding scores for a specific version.
        Returns: (success, retrieval_score, grounding_score, case_results, errors)
        """
        cases = test_cases or self.benchmark_cases
        if not cases:
            return False, 0.0, 0.0, [], ["No benchmark evaluation cases provided."]

        version_dir = Path(self.config.versions_dir) / version
        if not (version_dir / "index.faiss").exists() or not (version_dir / "index.pkl").exists():
            return False, 0.0, 0.0, [], [f"Version directory '{version}' is missing valid FAISS index files."]

        try:
            emb = self.get_embeddings()
            vectordb = FAISS.load_local(str(version_dir), emb, allow_dangerous_deserialization=True)
        except Exception as e:
            return False, 0.0, 0.0, [], [f"Failed to load FAISS index for version '{version}': {e}"]

        case_results = []
        total_retrieval_points = 0.0
        total_grounding_points = 0.0

        for case in cases:
            q = case.get("question", "")
            expected_kws = [kw.lower() for kw in case.get("expected_keywords", [])]

            try:
                retrieved_docs = vectordb.similarity_search(q, k=4)
            except Exception as e:
                logger.error(f"Error querying FAISS for case '{case.get('id', q)}': {e}")
                retrieved_docs = []

            combined_text = " ".join([d.page_content for d in retrieved_docs])
            combined_lower = combined_text.lower()

            # Check for prompt injection instructions inside retrieved document content
            malicious_patterns = self.scan_for_malicious_instructions(combined_text)
            untrusted_flag = len(malicious_patterns) > 0

            # Calculate keyword match
            if expected_kws:
                found_kws = [kw for kw in expected_kws if kw in combined_lower]
                kw_ratio = len(found_kws) / len(expected_kws)
            else:
                found_kws = []
                kw_ratio = 1.0 if retrieved_docs else 0.0

            # Retrieval score: Was relevant knowledge retrieved?
            case_retrieval_score = round(kw_ratio, 4)
            # Grounding score: Evidence support for factual answer
            case_grounding_score = round(kw_ratio, 4)

            total_retrieval_points += case_retrieval_score
            total_grounding_points += case_grounding_score

            case_results.append({
                "case_id": case.get("id"),
                "question": q,
                "expected_keywords": expected_kws,
                "matched_keywords": found_kws,
                "retrieved_chunk_count": len(retrieved_docs),
                "retrieval_score": case_retrieval_score,
                "grounding_score": case_grounding_score,
                "untrusted_instruction_detected": untrusted_flag,
                "flagged_patterns": malicious_patterns
            })

        avg_retrieval = round(total_retrieval_points / len(cases), 4)
        avg_grounding = round(total_grounding_points / len(cases), 4)

        return True, avg_retrieval, avg_grounding, case_results, []

    # ------------------------------------------------------------------
    # Full Quality Gate Decision Pipeline
    # ------------------------------------------------------------------
    def evaluate_candidate(
        self,
        candidate_version: str,
        parent_version: Optional[str] = None,
        test_cases: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Runs the full quality gate evaluation for candidate_version against
        configured thresholds and parent_version regression limits.
        Persists report and updates versions.json metadata.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        failure_reasons = []

        # Determine parent version
        metadata = self.vsm.load_versions_metadata()
        versions_data = metadata.get("versions", {})

        if candidate_version not in versions_data:
            return {
                "version": candidate_version,
                "overall_status": "FAIL",
                "decision": "REJECTED",
                "failure_reasons": [f"Candidate version '{candidate_version}' does not exist in versions registry."]
            }

        candidate_entry = versions_data[candidate_version]
        parent_v = parent_version or candidate_entry.get("parent_version") or self.vsm.get_active_version()

        # If evaluating same version as active, treat parent as None
        if parent_v == candidate_version:
            parent_v = None

        logger.info(f"Evaluating candidate '{candidate_version}' (comparison baseline: '{parent_v}')...")

        # 1. Evaluate candidate version
        c_ok, c_retrieval, c_grounding, c_cases, c_errors = self.evaluate_version_raw(
            candidate_version, test_cases=test_cases
        )
        if not c_ok:
            failure_reasons.extend(c_errors)

        # 2. Evaluate previous/parent version for regression check
        prev_retrieval = None
        prev_grounding = None
        regression_status = "PASS"
        retrieval_diff = 0.0
        grounding_diff = 0.0

        if parent_v and parent_v in versions_data:
            # Check if parent already has stored scores
            parent_eval = versions_data[parent_v].get("evaluation")
            if parent_eval and "retrieval_score" in parent_eval and "grounding_score" in parent_eval:
                prev_retrieval = parent_eval["retrieval_score"]
                prev_grounding = parent_eval["grounding_score"]
            else:
                p_ok, p_retrieval, p_grounding, _, _ = self.evaluate_version_raw(parent_v, test_cases=test_cases)
                if p_ok:
                    prev_retrieval = p_retrieval
                    prev_grounding = p_grounding

            if prev_retrieval is not None:
                retrieval_diff = round(c_retrieval - prev_retrieval, 4)
                # If quality decreased more than allowed regression
                if retrieval_diff < -self.config.max_allowed_regression:
                    regression_status = "FAIL"
                    failure_reasons.append(
                        f"Retrieval regression of {abs(retrieval_diff):.4f} exceeds allowed threshold of {self.config.max_allowed_regression:.4f} "
                        f"(previous: {prev_retrieval}, candidate: {c_retrieval})"
                    )

            if prev_grounding is not None:
                grounding_diff = round(c_grounding - prev_grounding, 4)
                if grounding_diff < -self.config.max_allowed_regression:
                    regression_status = "FAIL"
                    failure_reasons.append(
                        f"Grounding regression of {abs(grounding_diff):.4f} exceeds allowed threshold of {self.config.max_allowed_regression:.4f} "
                        f"(previous: {prev_grounding}, candidate: {c_grounding})"
                    )

        # 3. Check absolute quality thresholds
        if c_retrieval < self.config.min_retrieval_score:
            failure_reasons.append(
                f"Retrieval score {c_retrieval:.4f} is below minimum required threshold of {self.config.min_retrieval_score:.4f}"
            )

        if c_grounding < self.config.min_grounding_score:
            failure_reasons.append(
                f"Grounding score {c_grounding:.4f} is below minimum required threshold of {self.config.min_grounding_score:.4f}"
            )

        # 4. Determine final decision
        c_confidence = round(0.5 * c_retrieval + 0.5 * c_grounding, 4)

        if not failure_reasons:
            overall_status = "PASS"
            decision = "APPROVED"
        else:
            overall_status = "FAIL"
            decision = "REJECTED"

        # 5. Build detailed evaluation report
        report_data = {
            "version": candidate_version,
            "parent_version": parent_v,
            "timestamp": now_iso,
            "overall_status": overall_status,
            "decision": decision,
            "retrieval_score": c_retrieval,
            "grounding_score": c_grounding,
            "response_confidence": c_confidence,
            "previous_retrieval_score": prev_retrieval,
            "previous_grounding_score": prev_grounding,
            "retrieval_difference": retrieval_diff,
            "grounding_difference": grounding_diff,
            "regression_status": regression_status,
            "thresholds": {
                "min_retrieval_score": self.config.min_retrieval_score,
                "min_grounding_score": self.config.min_grounding_score,
                "max_allowed_regression": self.config.max_allowed_regression
            },
            "failure_reasons": failure_reasons,
            "evaluation_cases": c_cases
        }

        # 6. Persist evaluation report file
        reports_dir = Path(self.config.evaluation_reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / f"{candidate_version}_evaluation.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 7. Update versions.json metadata
        eval_summary = {
            "status": "approved" if decision == "APPROVED" else "rejected",
            "decision": decision,
            "overall_status": overall_status,
            "retrieval_score": c_retrieval,
            "grounding_score": c_grounding,
            "response_confidence": c_confidence,
            "previous_retrieval_score": prev_retrieval,
            "previous_grounding_score": prev_grounding,
            "regression_status": regression_status,
            "evaluated_at": now_iso,
            "report_file": str(report_file),
            "failure_reasons": failure_reasons
        }


        # Update candidate entry in versions metadata without activating it
        metadata["versions"][candidate_version]["evaluation"] = eval_summary
        self.vsm.save_versions_metadata(metadata)

        logger.info(
            f"Quality evaluation finished for '{candidate_version}': Decision={decision}, "
            f"Retrieval={c_retrieval}, Grounding={c_grounding}, Regression={regression_status}"
        )

        return report_data
