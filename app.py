import os
import time
import streamlit as st

import config
from ingestion import run_ingestion
from vector_store import build_vector_store
from graph_pipeline import build_graph


# ----------------------------#
#  Page Config & CSS          #
# ----------------------------#
st.set_page_config(
    page_title="Multi-Modal MDocAgent RAG",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #e5e7eb;
    }
    .user-msg {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
        word-wrap: break-word;
    }
    .assistant-msg {
        background: #111827;
        color: #e5e7eb;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #1f2937;
        font-size: 0.95rem;
        word-wrap: break-word;
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 1rem;
    }
    .meta-pill {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    .footer-note {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------#
#  Session State              #
# ----------------------------#
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"question": str, "answer": str}]

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str}]

if "graph_app" not in st.session_state:
    st.session_state.graph_app = None

if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

if "current_doc_name" not in st.session_state:
    st.session_state.current_doc_name = None

# ----------------------------#
#  Sidebar (Upload + Controls)#
# ----------------------------#
with st.sidebar:
    st.markdown("### 📚 Multi-Modal IMF RAG")
    st.markdown(
        "Upload a **PDF report** and chat with a multi-agent, multi-modal RAG "
        "system based on that document."
    )

    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])

    if uploaded_file is not None:
        # Only re-process if it's a new file
        if uploaded_file.name != st.session_state.current_doc_name:
            st.session_state.pipeline_ready = False
            st.session_state.graph_app = None
            st.session_state.chat_history = []
            st.session_state.messages = []

            with st.spinner("Processing document (extracting text, OCR, building index)..."):
                # Ensure directories exist
                config.ensure_dirs()

                # Save uploaded file to the standard path used by ingestion/vector_store
                with open(config.PDF_PATH, "wb") as f:
                    f.write(uploaded_file.read())

                # Run ingestion and vector store build for THIS document
                run_ingestion()
                build_vector_store()

                # Build a new LangGraph app bound to this vector store
                st.session_state.graph_app = build_graph()
                st.session_state.pipeline_ready = True
                st.session_state.current_doc_name = uploaded_file.name

            st.success(f"✅ Document loaded: {uploaded_file.name}")

    if st.session_state.pipeline_ready:
        st.markdown(
            f"**Current document:** `{st.session_state.current_doc_name}`"
        )
    else:
        st.markdown("**Current document:** _none loaded_")

    st.markdown("---")
    st.markdown("**Session Info**")
    st.write(f"🧠 Memory turns: `{len(st.session_state.chat_history)}`")

    if st.button("🧹 Clear Chat & Memory"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
        
    st.markdown("---")
    st.markdown(
        "💡 **Tips**\n"
        "- Upload any IMF-like PDF.\n"
        "- Ask follow-up questions; the system uses **memory**.\n"
        "- Mention tables/figures/charts to trigger OCR context."
    )

    st.markdown(
        "<span class='meta-pill'>Backend: FAISS • LangGraph • Gemini 1.5 Flash</span>",
        unsafe_allow_html=True,
    )

# ----------------------------#
#  Main Header                #
# ----------------------------#
st.markdown(
    """
    <div class="header-title">📄 Multi-Modal RAG Assistant</div>
    <div class="header-subtitle">
        Upload a policy or financial report PDF and ask questions about it. 
        The system combines text, tables, and OCR from images with multi-agent reasoning and memory.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------#
#  Show Chat History          #
# ----------------------------#
for msg in st.session_state.messages:
    role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
    with st.chat_message(msg["role"]):
        st.markdown(
            f"<div class='{role_class}'>{msg['content']}</div>",
            unsafe_allow_html=True,
        )

# ----------------------------#
#  Chat Input                 #
# ----------------------------#
if not st.session_state.pipeline_ready:
    st.info("👆 Please upload a PDF in the sidebar to start chatting.")
    user_input = None
else:
    user_input = st.chat_input("Type your question here...")

if user_input and st.session_state.pipeline_ready:
    app = st.session_state.graph_app

    # 1) Add user message to UI history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(
            f"<div class='user-msg'>{user_input}</div>",
            unsafe_allow_html=True,
        )

    # 2) Assistant container + placeholder for streaming
    assistant_container = st.chat_message("assistant")
    answer_placeholder = assistant_container.empty()

    # 3) Build initial state for this query (with memory)
    state = {
        "question": user_input,
        "chat_history": st.session_state.chat_history,
        "text_docs": None,
        "image_docs": None,
        "general_answer": None,
        "critical_text": None,
        "critical_image": None,
        "text_answer": None,
        "image_answer": None,
        "final_answer": None,
    }

    # 4) Run the LangGraph pipeline once
    with st.spinner("🔍 Retrieving, reasoning, and fusing multi-modal evidence..."):
        result = app.invoke(state)

    # 5) Get final answer + updated memory
    answer = result.get("final_answer", "") or "No answer produced."
    st.session_state.chat_history = result.get(
        "chat_history", st.session_state.chat_history
    )

    # 6) Typing-style streaming of the final answer
    streamed_text = ""
    for token in answer.split():
        streamed_text += token + " "
        answer_placeholder.markdown(
            f"<div class='assistant-msg'>{streamed_text}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(0.03)  # adjust streaming speed if needed

    # 7) Save full assistant answer to UI history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# ----------------------------#
#  Footer                     #
# ----------------------------#
st.markdown(
    "<div class='footer-note'>Demo prototype: Multi-modal, multi-agent RAG over user-uploaded documents.</div>",
    unsafe_allow_html=True,
)
