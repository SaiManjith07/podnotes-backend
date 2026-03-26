"""LangChain LCEL pipeline: user context + transcript -> structured notes (Google Gemini)."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


class GeneratedNote(BaseModel):
    """LLM output shape (parsed from JSON; avoids Gemini tool-schema array bugs)."""

    answers: dict[str, str] = Field(
        description="Map each provided question string (exact key) to a grounded answer from the transcript only.",
    )
    summary: str = Field(description="2–4 sentence summary; weigh user interests when relevant.")
    keyTakeaways: list[str] = Field(description="4–8 short takeaway strings.")


def _gemini_api_key() -> str | None:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip() or None


def _build_chain():
    api_key = _gemini_api_key()
    if not api_key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    parser = PydanticOutputParser(pydantic_object=GeneratedNote)
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.25,
        google_api_key=api_key,
        generation_config={"response_mime_type": "application/json"},
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You extract podcast notes from transcripts only. If the transcript does not support "
                "a claim, say it is not discussed — do not invent facts. Prefer clarity and actionable phrasing. "
                "When user interests are provided, emphasize angles relevant to those topics without ignoring "
                "the rest of the episode.\n\n"
                "Return a single JSON object only (no markdown code fences). It must match this schema:\n"
                "{format_instructions}",
            ),
            (
                "human",
                "User interest topics: {interest_labels}\n"
                "User interest ids: {interest_ids}\n\n"
                "Episode title: {video_title}\n"
                "Source URL: {source_url}\n\n"
                "Transcript:\n{transcript}\n\n"
                "Answer each question below. In `answers`, use EXACTLY these keys (same spelling and punctuation):\n"
                "{questions_json}\n\n"
                "Questions as a list:\n{questions_bulleted}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def run_generate_notes(
    transcript: str,
    video_title: str,
    source_url: str,
    user_interests: list[str],
    user_interest_labels: list[str],
    questions: list[str],
) -> dict[str, Any]:
    chain = _build_chain()
    if chain is None:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in backend/.env for LangChain + Gemini note generation."
        )

    max_chars = int(os.getenv("NOTES_TRANSCRIPT_MAX_CHARS", "24000"))
    text = transcript if len(transcript) <= max_chars else transcript[:max_chars]

    questions_bulleted = "\n".join(f"- {q}" for q in questions)
    questions_json = json.dumps(questions, ensure_ascii=False)

    try:
        result: GeneratedNote = chain.invoke(
            {
                "interest_labels": ", ".join(user_interest_labels)
                if user_interest_labels
                else "(none specified)",
                "interest_ids": ", ".join(user_interests) if user_interests else "(none)",
                "video_title": video_title,
                "source_url": source_url,
                "transcript": text,
                "questions_json": questions_json,
                "questions_bulleted": questions_bulleted,
            }
        )
    except OutputParserException as e:
        raise RuntimeError(
            "The model returned text that could not be parsed as JSON notes. "
            "Try again, shorten the transcript (NOTES_TRANSCRIPT_MAX_CHARS), or switch GEMINI_MODEL."
        ) from e

    answers: dict[str, str] = {}
    for q in questions:
        raw = result.answers.get(q) if result.answers else None
        if raw and raw.strip():
            answers[q] = raw.strip()
        else:
            nested = None
            if result.answers:
                for k, v in result.answers.items():
                    if k.strip().lower() == q.strip().lower():
                        nested = v
                        break
            answers[q] = (nested or "").strip() or "Not clearly addressed in the transcript."
    return {
        "answers": answers,
        "summary": result.summary.strip(),
        "keyTakeaways": [k.strip() for k in result.keyTakeaways if k and str(k).strip()],
    }
