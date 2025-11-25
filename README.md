# 🌈 SpectraDoc AI

SpectraDoc AI is a **multi-modal, multi-agent Retrieval-Augmented Generation (RAG)** system that lets you **upload any PDF report and chat with it** using **text, tables, and OCR-extracted images**.  

It uses **LangGraph** for multi-agent orchestration, **FAISS** for retrieval, and **Gemini 2.5 Flash** as the LVLM to keep answers **grounded, faithful, and low-hallucination**.  

---

## ✨ Key Features

- **📤 Upload-based PDF QA** – Users can upload their own PDFs from the UI, and all answers are grounded only on that document.  
- **🧠 Multi-Agent Reasoning** – General, critical, text-only, and image-aware agents collaborate and a final summarizing agent fuses their answers.  
- **🖼️ Multi-Modal Understanding** – The system reads **text**, **tables**, and **images via OCR**, so charts and scanned pages also contribute.  
- **🔍 FAISS Vector Retrieval** – Each document is chunked, embedded, and stored in a FAISS index for efficient similarity search.  
- **🧩 LangGraph Orchestration** – The full pipeline is implemented as a LangGraph state graph with explicit nodes for each agent.  
- **💬 Conversation Memory** – The system keeps a short memory of previous Q&A turns and uses it as context for follow-up questions.  
- **🌐 Gemini 2.5 Flash LVLM** – Uses the official `google-generativeai` SDK for robust, low-latency, vision-text generation.  
- **🖥️ Streamlit Chat UI** – Clean, dark-themed chat interface with typing-style streaming and per-document sessions.  

---

## 🧱 High-Level Architecture

**1. Document Ingestion**  
- User uploads a PDF via Streamlit.  
- The file is saved to `data/raw/document.pdf`.  
- `ingestion.py` runs a multi-modal extraction pipeline:  
  - Extracts **text blocks** per page via PyMuPDF.  
  - Extracts **page images**, runs **Tesseract OCR**, and stores OCR text plus image paths.  
  - Produces a unified list of chunks with metadata: `{type, content, page, source, image_path, id}`.  

**2. Vector Store Construction**  
- `vector_store.py` embeds all chunks using **Sentence-Transformers**.  
- Builds a **FAISS** index as a unified vector space over text and image-OCR content.  
- Saves the index and chunk metadata under `data/vector_store/`.  

**3. Multi-Agent RAG Pipeline (LangGraph)**  
- `graph_pipeline.py` defines a **DocQAState** and a LangGraph **StateGraph**.  
- Nodes:  
  - `retrieve` → retrieves top-K text and image chunks from FAISS.  
  - `general` → a general multi-modal agent, given question + context.  
  - `critical` → extracts critical text and image hints in structured JSON.  
  - `text_agent` → text-focused reasoning agent using only textual evidence.  
  - `image_agent` → image-focused agent using OCR content and image hints.  
  - `summarize` → fuses all agent answers into one final, grounded response and updates chat history.  

**4. LVLM Layer (Gemini Flash)**  
- `llm_clients.py` uses `google-generativeai` to call **Gemini 2.5 Flash**.  
- Supports prompts with and without images (via PIL).  
- All agents in `agents.py` call this client with specialized prompts from `prompts.py`.  

**5. UI Layer (Streamlit)**  
- `app.py` provides a dark-themed chat interface.  
- Sidebar: PDF upload, session info, clear chat and memory.  
- Main area: chat messages, typing-style streaming of answers.  
- Each new upload triggers a fresh ingestion → vector build → graph build, so responses are **document-specific**.  

---

## 🗂️ Project Structure

```
SpectraDoc_AI/
├─ app.py                  # Streamlit UI (upload + chat + streaming)
├─ cli_app.py              # Optional CLI interface for quick testing
├─ config.py               # Paths, model names, directory helpers
├─ ingestion.py            # PDF → text/image chunks with OCR
├─ vector_store.py         # FAISS index creation and search
├─ graph_pipeline.py       # LangGraph multi-agent state graph
├─ agents.py               # General, critical, text, image, summarizing agents
├─ prompts.py              # Prompt templates for all agents
├─ llm_clients.py          # Gemini 2.5 Flash client using google-generativeai
├─ requirements.txt        # Python dependencies
├─ data/
│  ├─ raw/                 # Uploaded PDFs (ignored in git)
│  ├─ processed/           # Extracted chunks JSON (ignored)
│  ├─ vector_store/        # FAISS index + metadata (ignored)
│  └─ images/              # Extracted page images (ignored)
└─ README.md               # You are here

```

## ⚙️ Setup & Installation
1. Clone the repository
```
git clone https://github.com/Dev7g/SpectraDoc_AI.git
cd SpectraDoc_AI
```

2. Create and activate a virtual environment
```
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS / Linux:
source env/bin/activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Configure environment variables
```
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

---
## ▶️ Running SpectraDoc AI
```
python -m streamlit run app.py
```

## 🧠 Design Highlights

- Multi-Modal Coverage: Text + tables + image OCR chunks are all embedded into a single vector space.

- Multi-Agent Pipeline: Different specialized agents reason over the same retrieved context for robustness.

- Evidence-Driven Answers: Agents are prompted to stay grounded in retrieved chunks and admit uncertainty when evidence is lacking.

- Memory-Aware Conversations: Previous Q&A turns are summarized and passed as compact history, improving follow-up questions.
