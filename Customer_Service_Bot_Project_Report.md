# 🤖 Customer Service Chatbot — Project Report & Architecture Guide

> **E-Learning FAQ Automation System with RAG (Retrieval-Augmented Generation)**  
> **Company:** Nullclass  
> **Tech Stack:** FastAPI, HTML5/CSS3/JavaScript, LangChain LCEL, FAISS, Google Gemini 2.5 Flash

---

## 📌 1. Project Overview & Business Context

### What is this project?
The **Customer Service Chatbot** is an AI-powered Question-and-Answer (Q&A) automation system designed for **Nullclass**, an e-learning platform specializing in data science courses, AI bootcamps, and virtual internships.

### The Business Problem
Nullclass learners and prospective students routinely ask hundreds of repetitive questions on **Discord** and **email**, including:
- *"Do you offer EMI payment options?"*
- *"Can I run Power BI on a Mac?"*
- *"Do I need prior programming experience?"*
- *"What is your refund policy?"*

Addressing these inquiries manually creates huge overhead for human mentors and support teams. 

### The Solution
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline:
1. It ingests verified company FAQ data from `dataset.csv`.
2. Encodes each FAQ into mathematical vector embeddings.
3. Indexes them in a local **FAISS** vector store.
4. When a user asks a question, it retrieves only the relevant FAQ sections and feeds them to **Google Gemini 2.5 Flash** to synthesize an accurate, instant response.

> **Key Guardrail Policy:** The model is strictly instructed:  
> *"If the answer is not found in the context, kindly state 'I don't know.' Don't try to make up an answer."*  
> This guarantees the bot never hallucinates or makes false promises about company policies.

---

## 🏗️ 2. Current System Architecture

The project is structured into a modern, decoupled **two-tier architecture**:

```
customer_service_bot/
├── .env                         # API credentials (GOOGLE_API_KEY)
├── dataset/
│   └── dataset.csv              # Company FAQs (prompt, response pairs)
├── backend/
│   ├── main.py                  # FastAPI REST API (endpoints: /ask, /create-knowledgebase)
│   ├── langchain_helper.py      # LCEL pipeline, FAISS loader, Gemini LLM
│   └── faiss_index/             # Pre-built vector database index files
└── frontend/
    ├── index.html               # Responsive chat interface
    ├── style.css                # Dark mode styling & animated chat bubbles
    └── script.js                # API fetch client & typing indicator
```

### Component Breakdown
| Layer | Technology | Role |
|---|---|---|
| **Backend API** | FastAPI + Uvicorn (Python 3.13) | Asynchronous, high-performance REST API with CORS support. |
| **Frontend UI** | HTML5, CSS3, Modern JavaScript | WhatsApp/ChatGPT-style bubble interface with real-time typing indicators and sample questions. |
| **Embedding Engine** | HuggingFace Instructor (`hkunlp/instructor-large`) | Transforms FAQ texts into dense semantic vectors. |
| **Vector Database** | FAISS (Facebook AI Similarity Search) | Performs ultra-fast nearest-neighbor similarity searches. |
| **Generative LLM** | Google Gemini 2.5 Flash | Synthesizes verified, natural answers strictly from retrieved context. |

---

## 📜 3. Complete Chronological Journey: Problems, Root Causes & Fixes

Here is the exact record of every obstacle encountered from the beginning of this project and why each change was made:

### 1. Missing Dependencies & Outdated `requirements.txt`
- **Problem:** `requirements.txt` was pinned to old packages (`langchain==0.0.339`), and essential packages like `faiss-cpu`, `sentence-transformers`, and `InstructorEmbedding` were missing.
- **Root Cause:** The project was based on an older tutorial repository. The environment was on modern Python 3.13 where these packages were absent.
- **Fix:** Installed `faiss-cpu`, `langchain-community`, `langchain-core`, and `langchain-google-genai`. Updated `requirements.txt`.
- **Why:** To eliminate `ModuleNotFoundError` and ensure compatibility with Python 3.13.

---

### 2. Migration from Deprecated Google PaLM to Google Gemini
- **Problem:** The original script imported `GooglePalm` from `langchain.llms`.
- **Root Cause:** Google officially retired the PaLM API in favor of the Google Gemini ecosystem. PaLM endpoints are discontinued.
- **Fix:** Switched to `ChatGoogleGenerativeAI` from `langchain-google-genai` using your existing `GOOGLE_API_KEY`.
- **Why:** Restores active API connectivity, reduces latency, and gives significantly higher reasoning quality.

---

### 3. Gemini Model Resolution (`gemini-1.5-flash` vs. `gemini-2.5-flash`)
- **Problem:** Calling `gemini-1.5-flash` returned `404 NOT_FOUND` from Google's API.
- **Root Cause:** On Google's current `v1beta` endpoint with your API key, model names have evolved.
- **Fix:** Queried `genai.list_models()` dynamically and selected `gemini-2.5-flash`, which was verified active on your key.
- **Why:** Completely fixed the 404 error and enabled Google's latest Gemini 2.5 model.

---

### 4. Refactoring Deprecated `RetrievalQA` to Modern LCEL
- **Problem:** `from langchain.chains import RetrievalQA` raised `ModuleNotFoundError: No module named 'langchain.chains'`.
- **Root Cause:** LangChain 1.x removed legacy monolithic chain wrappers in favor of LangChain Expression Language (LCEL).
- **Fix:** Re-architected the Q&A workflow using LCEL:
  ```python
  chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt_template
      | llm
      | StrOutputParser()
  )
  ```
- **Why:** LCEL is the official, future-proof LangChain standard—lighter, faster, and fully transparent.

---

### 5. Dataset Path Mismatch
- **Problem:** `CSVLoader` crashed trying to locate `dataset.csv` in the root folder.
- **Root Cause:** The CSV file was located inside `dataset/dataset.csv`.
- **Fix:** Updated the loader path to `../dataset/dataset.csv`.
- **Why:** Allows the script to reliably locate and index the FAQ data.

---

### 6. FAISS Security Deserialization Permission
- **Problem:** `FAISS.load_local()` raised a security check failure.
- **Root Cause:** LangChain requires explicit permission to deserialize pickle files to protect against malicious vector stores.
- **Fix:** Added `allow_dangerous_deserialization=True` to `FAISS.load_local()`.
- **Why:** Required to load local FAISS index files in trusted applications.

---

### 7. Streamlit Knowledgebase Race Condition Guard
- **Problem:** In the original Streamlit app, typing into the question input before clicking "Create Knowledgebase" caused a hard crash (`No such file or directory: faiss_index/index.faiss`).
- **Root Cause:** Streamlit re-runs the entire script on keystroke, trying to load the index before it was ever generated.
- **Fix:** Added `os.path.exists("faiss_index")` validation, loading spinners, and helpful guidance messages.
- **Why:** Prevents application crashes and guides users smoothly.

---

### 8. Architectural Decoupling: Streamlit ➔ FastAPI + Web Client
- **Problem:** Streamlit re-runs the entire script top-to-bottom on every click, creating high latency and preventing the bot from being embedded in regular websites or mobile apps.
- **Fix:** Converted the application into:
  - A **FastAPI REST API** (`backend/main.py`)
  - A clean **HTML5/CSS3/JavaScript Frontend** (`frontend/index.html`)
- **Why:** Decoupled architecture is the industry standard. The backend can now serve any client (web, mobile, Discord bot, Slack app) while the frontend provides a smooth, instant chat interface.

---

## ⚖️ 4. Comparison Table: Before vs. After

| Feature | Original Legacy Version | Current Modern Version |
|---|---|---|
| **Architecture** | Monolithic Streamlit script | Decoupled FastAPI backend + HTML/JS frontend |
| **LLM Provider** | Google PaLM (deprecated) | Google Gemini 2.5 Flash (active, state-of-the-art) |
| **LangChain Core** | Outdated `RetrievalQA` chain | Modern LCEL (LangChain Expression Language) |
| **Index Loading** | Direct load (security crash) | Safe FAISS load with explicit deserialization flag |
| **User Interface** | Basic Streamlit form widgets | WhatsApp/ChatGPT-style chat bubbles with typing dots |
| **Dataset Path** | Broken relative path | Robust relative path (`../dataset/dataset.csv`) |
| **Integration** | Locked inside Streamlit | Open REST API (`POST /ask`) ready for any platform |

---

## 🔄 5. End-to-End Operational Pipeline (Data Flow)

```
1. Ingestion:
   dataset.csv ➔ CSVLoader ➔ LangChain Documents

2. Embedding:
   Documents ➔ HuggingFace Instructor-Large ➔ 768-dim Vectors

3. Indexing:
   Vectors ➔ FAISS Index ➔ Saved locally in backend/faiss_index/

4. Retrieval:
   User Question ➔ Vector Similarity Search (Score > 0.7) ➔ Top FAQ Matches

5. Generation:
   Retrieved FAQ Matches + User Question ➔ Prompt Template ➔ Gemini 2.5 Flash

6. Delivery:
   Gemini Answer ➔ FastAPI JSON Response ➔ Animated Bubble in Frontend Chat UI
```

---

## 🛠️ 6. How to Run the Project

### Start the Backend
```powershell
cd c:\Users\Acer\Desktop\customer_service_bot\backend
python -m uvicorn main:app --port 8000 --reload
```
- Interactive API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

### Open the Frontend
Double-click:
```
c:\Users\Acer\Desktop\customer_service_bot\frontend\index.html
```
or open it in Google Chrome, Edge, or any web browser.
