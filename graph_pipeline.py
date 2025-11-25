# graph_pipeline.py

from typing import TypedDict, List, Any, Optional, Dict
from langgraph.graph import StateGraph, END

import config
from vector_store import MultiModalVectorStore
from agents import (
    general_agent,
    critical_agent,
    text_agent,
    image_agent,
    summarizing_agent,
)


class DocQAState(TypedDict):
    # User question (given once at start, then only read)
    question: str

    # Conversation memory: list of {"question": str, "answer": str}
    chat_history: Optional[List[Dict[str, str]]]

    # Retrieval results
    text_docs: Optional[List[Any]]
    image_docs: Optional[List[Any]]

    # Agent outputs
    general_answer: Optional[str]
    critical_text: Optional[str]
    critical_image: Optional[str]
    text_answer: Optional[str]
    image_answer: Optional[str]
    final_answer: Optional[str]


def build_graph():
    """
    Build a LangGraph pipeline that uses a fresh MultiModalVectorStore
    loaded from the current FAISS index on disk.
    This is called again every time you upload a new document in Streamlit.
    """

    # ---- IMPORTANT CHANGE: create & load vector store INSIDE build_graph ----
    store = MultiModalVectorStore()
    store.load(config.VECTOR_FAISS_PATH)

    # Node functions capture `store` via closure

    def node_retrieve(state: DocQAState) -> DocQAState:
        q = state["question"]
        state["text_docs"] = store.search_text(q, config.TOP_K_TEXT)
        state["image_docs"] = store.search_image(q, config.TOP_K_IMAGE)
        return state

    def node_general(state: DocQAState) -> DocQAState:
        state["general_answer"] = general_agent(
            state["question"],
            state["text_docs"],
            state["image_docs"],
            state.get("chat_history"),
        )
        return state

    def node_critical(state: DocQAState) -> DocQAState:
        info = critical_agent(
            state["question"],
            state["text_docs"],
            state["image_docs"],
            state["general_answer"],
            state.get("chat_history"),
        )
        state["critical_text"] = info.get("text", "")
        state["critical_image"] = info.get("image", "")
        return state

    def node_text_agent(state: DocQAState) -> DocQAState:
        state["text_answer"] = text_agent(
            state["question"],
            state["text_docs"],
            state["critical_text"],
            state.get("chat_history"),
        )
        return state

    def node_image_agent(state: DocQAState) -> DocQAState:
        state["image_answer"] = image_agent(
            state["question"],
            state["image_docs"],
            state["critical_image"],
            state.get("chat_history"),
        )
        return state

    def node_summarize(state: DocQAState) -> DocQAState:
        state["final_answer"] = summarizing_agent(
            state["question"],
            state["general_answer"],
            state["text_answer"],
            state["image_answer"],
            state.get("chat_history"),
        )

        # Update memory with this turn
        history = state.get("chat_history") or []
        history.append(
            {
                "question": state["question"],
                "answer": state["final_answer"] or "",
            }
        )
        state["chat_history"] = history
        return state

    # ---- Build the LangGraph graph ----
    graph = StateGraph(DocQAState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("general", node_general)
    graph.add_node("critical", node_critical)
    graph.add_node("text_agent", node_text_agent)
    graph.add_node("image_agent", node_image_agent)
    graph.add_node("summarize", node_summarize)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "general")
    graph.add_edge("general", "critical")
    graph.add_edge("critical", "text_agent")
    graph.add_edge("text_agent", "image_agent")
    graph.add_edge("image_agent", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
