# 🤖 Enterprise Generative AI Customer Service Bot

An enterprise-grade, multilingual customer service chatbot built with **Google Gemini**, **LangChain**, and **FastAPI**. It features strict RAG grounding, multimodal evidence verification (OCR), intelligent support ticket routing, calendar-aware SLA calculation, and multilingual context retention.

---

## 🌟 Key Features & Architecture (Tasks 1–6)

### 1. Production Knowledge Base Pipeline (Task 1)
- **Hash-based Ingestion & Deduplication**: Processes new/modified documents without redundant embeddings.
- **Automated Version Control & Rollback**: Manages versions (`v1`, `v2`, etc.) with health checks and automatic rollback within 5 minutes of activation failure.
- **Quality Gates**: Pre-deployment validation rejecting updates that reduce grounding or accuracy.
- **Enterprise Security**: Role-Based Access Control (RBAC), prompt injection detection, and PII/API-key masking in logs.
- **Operational Monitoring**: Real-time metrics for latency, failure rates, retrieval confidence, and escalations.

### 2. Multimodal OCR & Evidence Analysis (Task 2)
- **Document & Image Extraction**: OCR engine extracts Order IDs, amounts, dates, product codes, and error codes from PNG, JPG, WEBP, and PDF files.
- **Evidence Cross-Matching**: Compares customer statements against uploaded invoice/receipt evidence.
- **Security & Privacy**: Auto-masks payment card numbers (`[PAYMENT_REDACTED]`) and blocks hidden adversarial instructions in images.
- **Asynchronous Queue**: Requests taking >30s seamlessly route to background workers.

### 3. Support Ticket Automation & SLA Engine (Task 3)
- **Conversational Ticket Conversion**: Automatically converts unresolved conversations into structured support tickets.
- **Multi-Factor Priority Scoring**: Weights severity, sentiment, waiting time, customer impact, and SLA thresholds.
- **Calendar-Aware SLA Tracking**: Excludes configured weekends and regional holidays; triggers warnings at 75% elapsed time and auto-escalates on breach.
- **Skill-Based Routing**: Matches tickets to agent skill sets, workload capacity, and business operating hours.

### 4. Grounded RAG Knowledge Assistant (Task 4)
- **Strict Evidence Citation**: Backs up every policy response with exact source document references.
- **Zero-Hallucination Guardrails**: Explicitly refuses or requests clarification when source evidence is missing or ambiguous.
- **Temporal & Version-Aware Retrieval**: Resolves conflicting policies by version recency and supports historical date lookups.

### 5. Multilingual Sentiment & Escalation Engine (Task 5)
- **Emotion & Sentiment Detection**: Recognizes frustrated, urgent, neutral, and sarcastic sentiments across English, Tamil, Hindi, and Malayalam.
- **Dynamic Tone Adaptation**: Dynamically adjusts bot tone (empathetic, reassuring, professional, or formal).
- **High-Risk Escalation**: Automatically routes double-charging, account compromise, and legal threats to priority queues.

### 6. Multilingual Foundation & Session Continuity (Task 6)
- **Multilingual Support**: Supports English, Tamil (தமிழ்), Hindi (हिंदी), and Malayalam (മലയാളം), including transliterated (Tanglish/Hinglish) and mixed-code messages.
- **Entity Continuity & Preservation**: Preserves Order IDs, customer names, dates, and product codes across turns.
- **Session Isolation & Memory**: 10-message sliding window with 30-minute inactivity expiry and 24-hour summary restoration window.
- **Smart Clarification**: Confidently identifies genuine queries while requesting targeted domain clarification on ambiguous one-word inputs.

---

## 📁 Repository Directory Structure

```text
customer_service_bot/
├── backend/
│   ├── main.py                     # FastAPI server and REST endpoints (/ask, /multimodal, etc.)
│   ├── langchain_helper.py         # FAISS vector store & Gemini QA chain
│   ├── run.py                      # Subdirectory execution wrapper
│   ├── faiss_index/                # Vector store index files (index.faiss, index.pkl)
│   ├── pipeline/                   # Task 1: Ingestion, Versioning, Quality Gates & Security
│   ├── multimodal/                 # Task 2: OCR, File Validator, Evidence Comparator & Queue
│   ├── tickets/                    # Task 3 & 5: Ticket Engine, SLA Calculator, Sentiment
│   ├── rag/                        # Task 4: Temporal RAG, Citation Validator & Guardrails
│   ├── multilingual/               # Task 6: Language Detection, Session Manager & Intent
│   ├── tests/                      # Comprehensive test suite (891 tests across all tasks)
│   ├── temp_uploads/               # Ephemeral storage for multimodal uploads (.gitkeep)
│   └── versions/                   # Knowledge base version registry and rollback snapshots
├── dataset/
│   └── dataset.csv                 # Core FAQ dataset (SHA256 verified)
├── frontend/
│   ├── index.html                  # Modern responsive chat interface
│   ├── script.js                   # Client-side messaging, evidence card & polling logic
│   └── style.css                   # Theme and component styling
├── knowledge_base/
│   ├── course_faqs.pdf             # Domain knowledge documents
│   ├── course_how_to_guides.pdf
│   └── course_policies.pdf
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git ignore rules for credentials and cache
├── README.md                       # Project documentation
├── requirements.txt                # Python package dependencies
├── run.py                          # Unified CLI entrypoint controller
└── run_pipeline.py                 # Core pipeline orchestration script
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Google Gemini API Key ([Get one from Google AI Studio](https://aistudio.google.com/))

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone <repository_url>
cd customer_service_bot

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and enter your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

---

## 💻 Running the Application

Use the unified `run.py` controller from the root directory:

### Start the Backend Server:
```bash
python run.py server
```
- **API Docs (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Web Chat Interface**: Open `frontend/index.html` in your web browser.

### Run Knowledge Base Pipeline Demo:
```bash
python run.py demo
```
*Runs all 16 pipeline checks (ingestion, quality evaluation, maintenance window, RBAC, rollback, and metrics).*

### Check System Status:
```bash
python run.py status
```

### Run Full Test Suite:
```bash
python run.py test
# or
python -m pytest backend/tests/
```
*Runs all 891 unit and integration tests.*

---

## 🧪 Sample Interactions

| Category | Input Query / Action | Expected Result |
|:---|:---|:---|
| **Course Policy** | *"What is your refund policy?"* | Grounded answer with citation from knowledge base |
| **Payment Inquiry** | *"Do you offer EMI payments?"* | Clean, concise FAQ answer (`"No"`) |
| **Order Tracking (Tamil)** | *"என் order ORD-2026-00125 எங்கே இருக்கு?"* | Preserves Order ID, detects Tamil, returns tracking status |
| **Multimodal Upload** | Attach invoice screenshot / PDF | Extracts Order ID, Date, Amount; renders Evidence Card |
| **Escalation / Complaint** | *"Charged twice, nobody is helping, refund now!"* | Empathetic tone, high-risk flag, ticket escalation |
| **Ambiguous Query** | *"Can you fix it?"* | Requests specific clarification without guessing |

---

## 🔒 Security & Data Privacy
- **Zero API Key Leakage**: Keys are loaded strictly via environment variables.
- **PII / PCI Masking**: Payment cards, bank accounts, and contact details are masked in logs and handoffs (`[PAYMENT_REDACTED]`).
- **Prompt Injection Defense**: Filters adversarial prompt injections both in chat text and inside OCR documents.
- **Ephemeral Storage**: Uploaded multimodal files are deleted automatically after retention expiry.

---

## 📄 License
This project is developed for educational and production customer service automation. All rights reserved.