from __future__ import annotations

import logging
import os
import threading
import tempfile
import uuid
from typing import Any, Literal, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, field_validator, model_validator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("podnotes")

# Vite default is :8000; if port is busy it may use :8001 — include both so the SPA can call the API.
_default_cors = (
    "http://localhost:8000,http://127.0.0.1:8000,"
    "http://localhost:8001,http://127.0.0.1:8001"
)
FRONTEND_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", os.getenv("FRONTEND_URL", _default_cors)).split(",")
    if o.strip()
]
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

app = FastAPI(
    title="PodNotes API",
    description="YouTube → Whisper → Gemini notes for PodNotes.",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return JSON for unexpected errors; never leak raw tracebacks to clients."""
    if isinstance(exc, HTTPException):
        detail: Union[str, list[Any]] = exc.detail
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. See terminal logs for the traceback.",
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    youtubeUrl: HttpUrl
    userId: Optional[str] = None


class GenerateNotesRequest(BaseModel):
    transcript: str
    videoTitle: str
    sourceUrl: str
    userInterests: list[str] = []
    userInterestLabels: list[str] = []
    questions: list[str]


DEFAULT_NOTE_QUESTIONS = [
    "What are the main topics discussed?",
    "What are the key takeaways?",
    "Who are the speakers/guests?",
    "What actionable advice is given?",
    "Summary in 3 bullet points",
]


class NoteChatContextPayload(BaseModel):
    """Firestore-shaped note + transcript text; used as retrieval context (no vector DB)."""

    noteId: str
    title: str = ""
    summary: str = ""
    keyTakeaways: list[str] = []
    questions: list[str] = []
    answers: dict[str, str] = {}
    transcript: str = ""
    youtubeUrl: str = ""


class NoteChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        s = (v or "").strip()
        if len(s) > 12000:
            raise ValueError("message too long (max 12000 characters)")
        return s


class NoteChatRequest(BaseModel):
    userId: str = ""
    context: NoteChatContextPayload
    messages: list[NoteChatTurn]

    @model_validator(mode="after")
    def validate_conversation(self) -> NoteChatRequest:
        msgs = self.messages
        if len(msgs) < 1:
            raise ValueError("at least one message is required")
        if len(msgs) > 48:
            raise ValueError("too many messages in one request (max 48)")
        if msgs[-1].role != "user":
            raise ValueError("last message must be from the user")
        for i, m in enumerate(msgs):
            if not m.content.strip():
                raise ValueError(f"message {i} is empty")
        return self


class FromYoutubeRequest(BaseModel):
    """Single-call pipeline: YouTube URL -> transcript -> LangChain notes."""

    youtubeUrl: str
    userId: Optional[str] = None
    userInterests: list[str] = []
    userInterestLabels: list[str] = []
    questions: list[str] = []
    thumbnail: Optional[str] = None
    duration: Optional[str] = None

    @field_validator("youtubeUrl")
    @classmethod
    def normalize_youtube_url(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("youtubeUrl is required")
        if not s.startswith(("http://", "https://")):
            s = "https://" + s
        low = s.lower()
        if "youtube.com" not in low and "youtu.be" not in low:
            raise ValueError("URL must be a YouTube link (youtube.com or youtu.be)")
        return s


def _transcribe_youtube_to_text(youtube_url: str) -> tuple[str, str]:
    """
    Downloads YouTube audio -> transcribes with Whisper (faster-whisper) -> returns (title, transcript).
    """
    # Lazy-load yt-dlp + faster-whisper so server startup is fast.
    import yt_dlp
    from faster_whisper import WhisperModel

    # Thread-safe lazy model load (server may handle multiple requests).
    global _MODEL, _MODEL_LOCK
    model: WhisperModel
    with _MODEL_LOCK:
        model = _MODEL
        if model is None:
            model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
            _MODEL = model

    with tempfile.TemporaryDirectory() as tmpdir:
        cookies_path = (os.getenv("YOUTUBE_COOKIES_FILE") or "").strip()
        ydl_opts: dict = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "retries": 10,
            "fragment_retries": 10,
            "socket_timeout": 30,
            # Download to a temp file; ffmpeg postprocessor converts to wav for transcription.
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
        }
        if cookies_path and os.path.isfile(cookies_path):
            ydl_opts["cookiefile"] = cookies_path

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(youtube_url, download=True)
            except Exception as e:
                err = str(e)
                if "429" in err or "Too Many Requests" in err:
                    raise HTTPException(
                        status_code=429,
                        detail=(
                            "YouTube rate-limited downloads (HTTP 429). Wait several minutes, run "
                            "`pip install -U yt-dlp` in backend, and optionally set YOUTUBE_COOKIES_FILE "
                            "in backend/.env to a cookies.txt exported while logged into YouTube in a browser. "
                            f"Original error: {err[:500]}"
                        ),
                    ) from e
                raise HTTPException(status_code=400, detail=f"Failed to download YouTube audio: {e}") from e

        title = info.get("title") if isinstance(info, dict) else None
        title = title or "YouTube video"

        # Find produced audio (FFmpeg postprocessor usually yields .wav; if ffmpeg is missing, may stay .m4a/.webm).
        audio_path: Optional[str] = None
        prefer = (".wav", ".m4a", ".mp3", ".webm", ".opus", ".ogg", ".flac")
        names = os.listdir(tmpdir)
        for ext in prefer:
            for name in names:
                if name.lower().endswith(ext):
                    audio_path = os.path.join(tmpdir, name)
                    break
            if audio_path:
                break
        if not audio_path:
            raise HTTPException(
                status_code=500,
                detail="Audio download succeeded but no audio file found in temp dir "
                f"(have ffmpeg installed? files={names}).",
            )

        try:
            size = os.path.getsize(audio_path)
        except OSError:
            size = 0
        if size < 2048:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Downloaded audio file is unexpectedly small; YouTube may be blocking or rate-limiting "
                    "this IP. Update yt-dlp (`pip install -U yt-dlp`), wait, try another video, or set "
                    "YOUTUBE_COOKIES_FILE in backend/.env."
                ),
            )

        try:
            segments, _info = model.transcribe(
                audio_path,
                language=None,
                vad_filter=True,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Whisper transcription failed (corrupt or unsupported audio?): {e}",
            ) from e

        transcript = " ".join([seg.text.strip() for seg in segments if seg.text and seg.text.strip()]).strip()
        if not transcript:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Transcription produced no text. The download may be incomplete (see server log for "
                    "YouTube 429 / signature warnings). Try again later, another video, `pip install -U yt-dlp`, "
                    "or YOUTUBE_COOKIES_FILE."
                ),
            )
        return title, transcript


_MODEL: Optional["object"] = None
_MODEL_LOCK = threading.Lock()


def _gemini_configured() -> bool:
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    return bool(key)


@app.get("/api/podcast/health")
def health() -> dict:
    """Lightweight readiness: no Whisper load, no network."""
    return {
        "status": "ok",
        "service": "podnotes-fastapi",
        "pipeline": "POST /api/podcast/from-youtube",
        "noteChat": "POST /api/podcast/note-chat",
        "geminiConfigured": _gemini_configured(),
        "whisperModel": WHISPER_MODEL_SIZE,
        "corsOrigins": len(FRONTEND_ORIGINS),
    }


@app.post("/api/podcast/extract-audio")
def extract_audio(req: ExtractRequest) -> dict:
    title, transcript = _transcribe_youtube_to_text(str(req.youtubeUrl))
    return {
        "videoTitle": title,
        "sourceType": "youtube",
        "sourceUrl": str(req.youtubeUrl),
        "transcript": transcript,
    }


@app.post("/api/podcast/process-audio")
def process_audio(req: ExtractRequest) -> dict:
    # For test purposes we treat this the same way as extract-audio.
    title, transcript = _transcribe_youtube_to_text(str(req.youtubeUrl))
    return {
        "videoTitle": title,
        "sourceType": "youtube",
        "sourceUrl": str(req.youtubeUrl),
        "transcript": transcript,
    }


@app.post("/api/podcast/from-youtube")
def from_youtube(req: FromYoutubeRequest) -> dict:
    """
    One request: download audio -> Whisper transcript -> LangChain (interests + transcript) -> notes.
    """
    if not _gemini_configured():
        raise HTTPException(
            status_code=503,
            detail="Note generation is unavailable: set GEMINI_API_KEY or GOOGLE_API_KEY in backend/.env.",
        )

    url = str(req.youtubeUrl)
    title, transcript = _transcribe_youtube_to_text(url)
    questions = req.questions if req.questions else list(DEFAULT_NOTE_QUESTIONS)
    transcript_id = str(uuid.uuid4())
    note_id = str(uuid.uuid4())
    try:
        from notes_chain import run_generate_notes

        notes = run_generate_notes(
            transcript=transcript,
            video_title=title,
            source_url=url,
            user_interests=req.userInterests,
            user_interest_labels=req.userInterestLabels,
            questions=questions,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Note generation failed: {e}") from e

    firestore_saved = False
    uid = (req.userId or "").strip()
    if uid:
        from firestore_sync import try_save_pipeline

        firestore_saved = try_save_pipeline(
            uid,
            transcript_id,
            note_id,
            youtube_url=url,
            video_title=title,
            thumbnail=(req.thumbnail or "").strip(),
            transcript=transcript,
            duration=(req.duration or "").strip(),
            questions=questions,
            answers=notes["answers"],
            summary=notes["summary"],
            key_takeaways=notes["keyTakeaways"],
        )

    return {
        "transcriptId": transcript_id,
        "noteId": note_id,
        "firestoreSaved": firestore_saved,
        "videoTitle": title,
        "sourceType": "youtube",
        "sourceUrl": url,
        "transcript": transcript,
        **notes,
    }


@app.post("/api/podcast/generate-notes")
def generate_notes(req: GenerateNotesRequest) -> dict:
    if not _gemini_configured():
        raise HTTPException(
            status_code=503,
            detail="Note generation is unavailable: set GEMINI_API_KEY or GOOGLE_API_KEY in backend/.env.",
        )
    if not req.transcript or not str(req.transcript).strip():
        raise HTTPException(status_code=422, detail="transcript must be non-empty.")
    if not req.questions:
        raise HTTPException(status_code=422, detail="questions must contain at least one question.")

    try:
        from notes_chain import run_generate_notes

        return run_generate_notes(
            transcript=req.transcript,
            video_title=req.videoTitle,
            source_url=req.sourceUrl,
            user_interests=req.userInterests,
            user_interest_labels=req.userInterestLabels,
            questions=req.questions,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Note generation failed: {e}") from e


@app.post("/api/podcast/note-chat")
def note_chat(req: NoteChatRequest) -> dict:
    """
    Chat about a single note using only that note's saved fields + transcript as context
    (prompt-based \"retrieval\", no vector database).
    """
    if not _gemini_configured():
        raise HTTPException(
            status_code=503,
            detail="Chat is unavailable: set GEMINI_API_KEY or GOOGLE_API_KEY in backend/.env.",
        )

    cid = (req.context.noteId or "").strip()
    if not cid:
        raise HTTPException(status_code=422, detail="context.noteId is required")

    try:
        from note_chat import build_note_reference_context, run_note_chat

        last_question = (req.messages[-1].content or "").strip() if req.messages else ""
        q_low = last_question.lower()
        expand_for_more = any(
            k in q_low
            for k in [
                "more detail",
                "in detail",
                "full transcript",
                "full",
                "transcript",
                "elaborate",
                "explain",
                "continue",
                "more",
                "detailed",
            ]
        )
        brief_max_chars = int(os.getenv("NOTE_CHAT_TRANSCRIPT_MAX_CHARS_BRIEF", "6000"))
        expanded_max_chars = int(os.getenv("NOTE_CHAT_TRANSCRIPT_MAX_CHARS", "28000"))
        transcript_max_chars = expanded_max_chars if expand_for_more else brief_max_chars

        brief_output_tokens = int(os.getenv("NOTE_CHAT_MAX_OUTPUT_TOKENS_BRIEF", "700"))
        expanded_output_tokens = int(os.getenv("NOTE_CHAT_MAX_OUTPUT_TOKENS_EXPANDED", "1400"))
        max_output_tokens = expanded_output_tokens if expand_for_more else brief_output_tokens

        reference = build_note_reference_context(
            title=req.context.title,
            summary=req.context.summary,
            key_takeaways=req.context.keyTakeaways,
            questions=req.context.questions,
            answers=req.context.answers,
            transcript=req.context.transcript,
            youtube_url=req.context.youtubeUrl,
            transcript_max_chars=transcript_max_chars,
        )
        turns: list[tuple[Literal["user", "assistant"], str]] = [
            (m.role, m.content) for m in req.messages
        ]
        reply = run_note_chat(
            reference_context=reference,
            messages=turns,
            expand_for_more=expand_for_more,
            max_output_tokens=max_output_tokens,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Note chat failed: {e}") from e

    return {"role": "assistant", "content": reply}

