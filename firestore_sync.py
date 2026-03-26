"""Optional server-side Firestore writes (Firebase Admin). Matches client `firestoreSerial` document shape."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_db: Any = None


def _admin_db():
    """Lazy Firestore client; returns None if Admin SDK is not configured or init fails."""
    global _db
    if _db is not None:
        return _db

    json_blob = (os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or "").strip()
    path = (
        os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or ""
    ).strip()

    if not json_blob and not (path and os.path.isfile(path)):
        return None

    try:
        import firebase_admin
        from firebase_admin import credentials

        if json_blob:
            cred = credentials.Certificate(json.loads(json_blob))
        else:
            cred = credentials.Certificate(path)

        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(cred)

        _db = firebase_admin.firestore.client()
        return _db
    except Exception as e:
        logger.warning("Firebase Admin could not initialize: %s", e)
        return None


def try_save_pipeline(
    user_id: str,
    transcript_id: str,
    note_id: str,
    *,
    youtube_url: str,
    video_title: str,
    thumbnail: str,
    transcript: str,
    duration: str,
    questions: list[str],
    answers: dict[str, str],
    summary: str,
    key_takeaways: list[str],
) -> bool:
    db = _admin_db()
    if db is None:
        return False

    try:
        from google.cloud.firestore import SERVER_TIMESTAMP

        batch = db.batch()
        tr_ref = db.collection("users").document(user_id).collection("transcripts").document(transcript_id)
        note_ref = db.collection("users").document(user_id).collection("notes").document(note_id)

        batch.set(
            tr_ref,
            {
                "youtubeUrl": youtube_url,
                "title": video_title,
                "thumbnail": thumbnail,
                "transcript": transcript,
                "duration": duration,
                "createdAt": SERVER_TIMESTAMP,
            },
        )
        batch.set(
            note_ref,
            {
                "transcriptId": transcript_id,
                "youtubeUrl": youtube_url,
                "title": video_title,
                "thumbnail": thumbnail,
                "questions": questions,
                "answers": answers,
                "summary": summary,
                "keyTakeaways": key_takeaways,
                "createdAt": SERVER_TIMESTAMP,
            },
        )
        batch.commit()
        return True
    except Exception as e:
        logger.warning("Firestore pipeline save failed: %s", e)
        return False
