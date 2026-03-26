"""Note-scoped chat: Gemini uses a fixed template (generated notes + user question / history)."""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from notes_chain import _gemini_api_key


def build_note_reference_context(
    *,
    title: str,
    summary: str,
    key_takeaways: list[str],
    questions: list[str],
    answers: dict[str, str],
    transcript: str,
    youtube_url: str,
    transcript_max_chars: int | None = None,
) -> str:
    """Pack Firestore-shaped note + transcript into one reference block (trimmed)."""
    max_tr = (
        transcript_max_chars
        if transcript_max_chars is not None
        else int(os.getenv("NOTE_CHAT_TRANSCRIPT_MAX_CHARS", "28000"))
    )
    tr = (transcript or "").strip()
    if len(tr) > max_tr:
        tr = tr[:max_tr] + "\n\n[Transcript truncated for chat context length.]"

    qa_lines: list[str] = []
    for q in questions:
        a = (answers or {}).get(q, "")
        qa_lines.append(f"Q: {q}\nA: {a}")

    takeaways = "\n".join(f"- {t}" for t in key_takeaways if t and str(t).strip())

    parts = [
        f"## Episode / note title\n{title.strip() or '(untitled)'}",
        f"## Source\n{youtube_url.strip() or '(no URL)'}",
        f"## Summary\n{summary.strip() or '(none)'}",
        f"## Key takeaways\n{takeaways or '(none)'}",
        f"## Prior Q&A from note generation\n{chr(10).join(qa_lines) if qa_lines else '(none)'}",
        f"## Full transcript (primary evidence)\n{tr or '(no transcript stored for this note)'}",
    ]
    return "\n\n".join(parts)


def _format_prior_chat(messages: list[tuple[Literal["user", "assistant"], str]]) -> str:
    lines: list[str] = []
    for role, content in messages:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content.strip()}")
    return "\n".join(lines) if lines else ""


def build_chat_prompt_document(
    *,
    generated_notes: str,
    messages: list[tuple[Literal["user", "assistant"], str]],
) -> str:
    """
    Single structured document: saved/generated notes + optional prior turns + current user question.
    `messages` must be non-empty; last item must be role=user.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    if messages[-1][0] != "user":
        raise ValueError("last message must be from the user")

    prior = messages[:-1]
    current_question = messages[-1][1].strip()
    if not current_question:
        raise ValueError("current user question must not be empty")

    prior_text = _format_prior_chat(prior)
    if not prior_text:
        prior_section = "(This is the first message in this chat; there is no prior conversation.)"
    else:
        prior_section = prior_text

    return (
        "=== GENERATED NOTES (saved note + transcript; your only factual sources) ===\n"
        f"{generated_notes.strip()}\n"
        "=== END GENERATED NOTES ===\n\n"
        "=== PRIOR CHAT (for context only; do not contradict the generated notes) ===\n"
        f"{prior_section}\n\n"
        "=== CURRENT USER QUESTION (answer this now) ===\n"
        f"{current_question}"
    )


def run_note_chat(
    *,
    reference_context: str,
    messages: list[tuple[Literal["user", "assistant"], str]],
    expand_for_more: bool = False,
    max_output_tokens: int = 700,
) -> str:
    """
    Stateless turn: one system prompt + one human blob built from template
    (generated notes + prior chat + current user question).
    """
    api_key = _gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in backend/.env for note chat."
        )

    user_document = build_chat_prompt_document(
        generated_notes=reference_context,
        messages=messages,
    )

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.35,
        google_api_key=api_key,
        generation_config={"max_output_tokens": max_output_tokens},
    )

    system_text = (
        "You help the user discuss ONE podcast episode using the materials in their message.\n\n"
        "Rules:\n"
        "- Use only facts supported by the section GENERATED NOTES (summary, takeaways, stored Q&A, transcript). "
        "If the notes do not contain the answer, say so clearly—do not invent.\n"
        "- Answer the CURRENT USER QUESTION. Use PRIOR CHAT only for continuity.\n"
        "- Be concise by default (short bullets or 2-4 short paragraphs).\n"
        "- If the user explicitly asks for more detail (e.g., 'more', 'in detail', 'full transcript'), you may expand.\n"
        f"- User asked for more detail: {expand_for_more}.\n"
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_text),
            HumanMessage(content=user_document),
        ]
    )
    out = getattr(response, "content", None)
    if isinstance(out, str):
        return out.strip() or "(No response.)"
    if isinstance(out, list):
        texts: list[str] = []
        for block in out:
            if isinstance(block, str):
                texts.append(block)
            else:
                t = getattr(block, "text", None)
                if t:
                    texts.append(str(t))
        return "\n".join(texts).strip() or "(No response.)"
    return str(out or "").strip() or "(No response.)"
