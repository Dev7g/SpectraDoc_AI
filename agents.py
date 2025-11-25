import json
from typing import List, Dict, Any, Optional

from llm_clients import call_gemini
import prompts


def format_text_context(text_docs: Optional[List[Any]]) -> str:
    if not text_docs:
        return "No text context."
    lines = []
    for d in text_docs:
        meta = d.metadata
        lines.append(
            f"[page {meta.get('page')}, source={meta.get('source')}] "
            f"{d.page_content[:500].strip()}"
        )
    return "\n\n".join(lines)


def format_image_context(img_docs: Optional[List[Any]]) -> str:
    if not img_docs:
        return "No image context."
    lines = []
    for d in img_docs:
        meta = d.metadata
        lines.append(
            f"[image page {meta.get('page')}, source={meta.get('source')}] "
            f"OCR: {d.page_content[:400].strip()}"
        )
    return "\n\n".join(lines)


def format_history(chat_history: Optional[List[Dict[str, str]]]) -> str:
    if not chat_history:
        return "No previous turns."
    parts = []
    # use last 5 turns
    for turn in chat_history[-5:]:
        q = turn.get("question", "").strip()
        a = turn.get("answer", "").strip()
        if q or a:
            parts.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(parts) if parts else "No previous turns."


def general_agent(
    question: str,
    text_docs: Optional[List[Any]],
    img_docs: Optional[List[Any]],
    chat_history: Optional[List[Dict[str, str]]],
) -> str:
    text_ctx = format_text_context(text_docs)
    img_ctx = format_image_context(img_docs)
    hist = format_history(chat_history)

    prompt = prompts.GENERAL_AGENT_PROMPT.format(
        question=question,
        text_context=text_ctx,
        image_context=img_ctx,
        history=hist,
    )
    return call_gemini(prompt)


def critical_agent(
    question: str,
    text_docs: Optional[List[Any]],
    img_docs: Optional[List[Any]],
    general_answer: str,
    chat_history: Optional[List[Dict[str, str]]],
) -> Dict[str, str]:
    text_ctx = format_text_context(text_docs)
    img_ctx = format_image_context(img_docs)
    hist = format_history(chat_history)

    prompt = prompts.CRITICAL_AGENT_PROMPT.format(
        question=question,
        text_context=text_ctx,
        image_context=img_ctx,
        general_answer=general_answer,
        history=hist,
    )
    raw = call_gemini(prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        json_str = raw[start:end]
        data = json.loads(json_str)
        return {
            "text": data.get("text", ""),
            "image": data.get("image", ""),
        }
    except Exception:
        return {"text": "", "image": ""}


def text_agent(
    question: str,
    text_docs: Optional[List[Any]],
    critical_text: str,
    chat_history: Optional[List[Dict[str, str]]],
) -> str:
    text_ctx = format_text_context(text_docs)
    hist = format_history(chat_history)

    prompt = prompts.TEXT_AGENT_PROMPT.format(
        question=question,
        text_context=text_ctx,
        critical_text=critical_text,
        history=hist,
    )
    return call_gemini(prompt)


def image_agent(
    question: str,
    img_docs: Optional[List[Any]],
    critical_image: str,
    chat_history: Optional[List[Dict[str, str]]],
) -> str:
    img_ctx = format_image_context(img_docs)
    hist = format_history(chat_history)

    prompt = prompts.IMAGE_AGENT_PROMPT.format(
        question=question,
        image_context=img_ctx,
        critical_image=critical_image,
        history=hist,
    )

    image_paths = [
        d.metadata.get("image_path")
        for d in (img_docs or [])
        if d.metadata.get("image_path")
    ]
    return call_gemini(prompt, image_paths=image_paths)


def summarizing_agent(
    question: str,
    general_answer: str,
    text_answer: str,
    image_answer: str,
    chat_history: Optional[List[Dict[str, str]]],
) -> str:
    hist = format_history(chat_history)

    prompt = prompts.SUMMARIZING_AGENT_PROMPT.format(
        question=question,
        general_answer=general_answer,
        text_answer=text_answer,
        image_answer=image_answer,
        history=hist,
    )
    return call_gemini(prompt)
