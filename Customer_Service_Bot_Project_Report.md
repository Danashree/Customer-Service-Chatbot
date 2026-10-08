# 🤖 Enterprise Customer Service Chatbot — Project Report & Architecture Guide

> **Enterprise Multilingual Generative AI Customer Service Bot with RAG, Multimodal Verification, Support Ticketing & Escalation**  
> **Company / Use Case:** Nullclass E-Learning & Services  
> **Tech Stack:** FastAPI, HTML5/CSS3/JavaScript, LangChain LCEL, FAISS, Google Gemini (Flash), OCR, Python 3.13  

---

## 📌 1. Executive Summary & Project Overview

The **Enterprise Customer Service Chatbot** is an AI-powered conversational automation system designed for Nullclass, an e-learning and virtual internship platform. Learners and prospective students frequently submit hundreds of inquiries across discord, web portals, and email regarding courses, course materials, tablets, order shipments, billing issues, and technical difficulties.

Addressing these repetitive inquiries manually creates heavy overhead for human support agents. This system automates resolution across 6 interconnected modules:
1. **Production Knowledge-Base Pipeline**: Ingestion, hash deduplication, automated health checks, and 5-minute rollback.
2. **Multimodal Evidence Verification**: OCR extraction from invoice receipts and error screenshots with sensitive payment masking.
3. **Support Ticket Automation & SLA Engine**: Automatic ticket extraction, multi-factor priority scoring, and calendar-aware SLA calculation.
4. **Temporal Grounded RAG Assistant**: Strict source citations, version conflict resolution, and anti-hallucination refusals.
5. **Multilingual Sentiment & Escalation Engine**: Tone adaptation (empathetic, urgent, reassuring) and high-risk complaint escalation.
6. **Multilingual Foundation & Session Isolation**: Support for English, Tamil, Hindi, and Malayalam (including code-mixed and transliterated queries) with session memory and entity preservation.

---

## 🏗️ 2. System Architecture & Components

The application is structured into a modern two-tier architecture:

```text
customer_service_bot/
├── backend/
│   ├── main.py                     # FastAPI REST API endpoints (/ask, /multimodal, /tickets)
│   ├── langchain_helper.py         # FAISS vector store & Gemini QA chain
│   ├── pipeline/                   # Task 1: Ingestion, Version Control, Rollback, RBAC
│   ├── multimodal/                 # Task 2: OCR Extraction, Evidence Verification, PII Masking
│   ├── tickets/                    # Task 3 & 5: Ticketing Engine, SLA Engine, Sentiment
│   ├── rag/                        # Task 4: Grounded RAG, Citations & Temporal Filter
│   ├── multilingual/               # Task 6: Multilingual NLP, Session Context & Entity Manager
│   └── tests/                      # 891 automated unit and integration tests
├── dataset/
│   └── dataset.csv                 # Core FAQ dataset (SHA256 verified)
├── frontend/
│   ├── index.html                  # Responsive chat interface with bottom docked input
│   ├── script.js                   # Client fetch client, evidence cards, polling logic
│   └── style.css                   # Dark theme, status pills, and animated bubbles
└── run.py                          # Unified CLI entrypoint controller
```

### Component Breakdown
| Layer | Technology | Role |
|---|---|---|
| **Backend API** | FastAPI + Uvicorn | Asynchronous, high-performance REST API with CORS support. |
| **Frontend UI** | HTML5, CSS3, Vanilla JS | Modern WhatsApp/ChatGPT-style bubble interface with multimodal attachment cards. |
| **Embedding Engine** | HuggingFace Instructor (`hkunlp/instructor-large`) | Transforms FAQ texts and policies into dense semantic vectors. |
| **Vector Database** | FAISS | Ultra-fast nearest-neighbor similarity search. |
| **Generative LLM** | Google Gemini | Generates grounded answers strictly from retrieved context and policies. |
| **OCR & Vision** | Tesseract / PDF Parser | Extracts Order IDs, amounts, and error codes from images and invoices. |

---

## 📜 3. Detailed Task Implementation Breakdown

### Task 1: Production Knowledge Base Pipeline
- **Hash-based Ingestion**: Uses SHA-256 document hashing to process only new or modified documents.
- **Automated Version Control**: Maintains versioned checkpoints (`v1`, `v2`, etc.).
- **Automatic Rollback**: Runs automated health checks on deployment; if any check fails within 5 minutes, automatically rolls back to the previous stable version.
- **Enterprise Security**: Role-Based Access Control (Admin, Operator, Viewer), prompt injection detection, and PII masking.

### Task 2: Multimodal Analysis & OCR Verification
- **Evidence Extraction**: Extracts Order IDs (`ORD-xxxxx`), dates, total amounts, and product codes from PNG, JPG, WEBP, and PDF documents.
- **Cross-Verification**: Compares customer message statements against the extracted document data to detect conflicts.
- **Security & Redaction**: Automatically redacts sensitive payment card numbers (`[PAYMENT_REDACTED]`).
- **Asynchronous Queueing**: Requests requiring >30s move to a background queue with customer progress notifications.

### Task 3: Support Ticket Automation & SLA Engine
- **Conversation-to-Ticket Conversion**: Converts unresolved interactions into structured tickets with customer, order, product, and issue fields.
- **Multi-Factor Priority Scoring**: Calculates priority using severity, sentiment, waiting time, customer impact, and SLA rules.
- **Calendar-Aware SLA Calculations**: Automatically excludes weekends and regional holidays; triggers warning notifications at 75% elapsed time and escalates upon breach.
- **Smart Routing**: Routes tickets to human agents based on skill requirements, current workload capacity, and business hours.

### Task 4: Grounded RAG Knowledge Assistant
- **Strict Evidence Grounding**: Every answer is grounded in factual document chunks with precise source citations.
- **Zero Hallucination Guardrails**: Explicitly states *"I don't know"* or requests clarification when evidence is insufficient or ambiguous.
- **Temporal & Version Conflict Resolution**: Resolves policy conflicts by checking policy effective dates, expiry dates, and version recency.

### Task 5: Multilingual Sentiment & Escalation Engine
- **Multilingual Sentiment Detection**: Identifies emotions (frustrated, urgent, negative, neutral, positive) across English, Tamil, Hindi, and Malayalam.
- **Dynamic Tone Adjustment**: Adapts assistant tone (empathetic, reassuring, professional, or formal).
- **High-Risk Escalation**: High-risk triggers (double billing, unauthorized account access, legal threats) automatically escalate to priority support queues.

### Task 6: Multilingual Foundation & Session Management
- **Languages Supported**: English, Tamil, Hindi, Malayalam, and transliterated code-mixed queries (Tanglish/Hinglish).
- **Session Continuity**: 10-message sliding window context memory; isolates simultaneous customer sessions.
- **Lifecycle & Restoration**: 30-minute inactivity session expiry with a 24-hour summary restoration window.
- **Smart Clarification**: Accurately recognizes genuine questions while politely requesting clarification for ambiguous single-word inputs.

---

## 🔄 4. End-to-End Data Flow

```text
User Input (Text / Image / Mixed Language)
    │
    ▼
[Task 6] Multilingual Orchestrator ──▶ Language Detection & Session Context
    │
    ├── (Ambiguous) ───────────────▶ Return Smart Clarification Prompt
    │
    ▼
[Task 5] Sentiment & Risk Engine ──▶ Evaluate Emotion & Escalation Triggers
    │
    ▼
[Task 1 & 4] FAISS / Temporal RAG ──▶ Semantic Search & Grounded Retrieval
    │
    ▼
Google Gemini LLM ─────────────────▶ Synthesize Grounded, Tone-Adapted Answer
    │
    ▼
[Task 2 & 3] Evidence / Ticketing ──▶ Cross-Check OCR Evidence or Generate Ticket
    │
    ▼
Unified JSON Payload ──────────────▶ Rendered in Web Frontend UI
```

---

## 🧪 5. Testing & Verification Summary

- **Total Automated Tests**: **891 Passing Tests** across all 6 tasks.
- **Live Interactive Demo**: 16 pipeline health and quality gate checks passing (`python run.py demo`).
- **Dataset Verification**: Core `dataset/dataset.csv` maintained with SHA-256 hash `930649d927881235ecbd3b53b29de92ddf8dbeca084657774335630eb9361b9a`.

---

## 🛠️ 6. How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Start Backend Server**:
   ```bash
   python run.py server
   ```
3. **Open Frontend**:
   Open `frontend/index.html` in your web browser.
4. **Interactive Swagger API Documentation**:
   Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
