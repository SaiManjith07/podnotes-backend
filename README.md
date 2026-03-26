# PodNotes Backend (FastAPI)

Backend for YouTube -> transcription -> AI notes -> note chat.

## Local setup (Windows / PowerShell)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\pip.exe install -r .\requirements.txt

# Optional: copy env template
copy .env.example .env

# Run API
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 5000 --reload
```

Health check:

- `GET http://127.0.0.1:5000/api/podcast/health`

## Core endpoints

- `GET /api/podcast/health`
- `POST /api/podcast/extract-audio`
- `POST /api/podcast/from-youtube`
- `POST /api/podcast/generate-notes`
- `POST /api/podcast/note-chat`

## Environment variables

See `backend/.env.example` for full comments.

Required for notes/chat:

- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)

Required for browser access from frontend:

- `CORS_ORIGINS` (comma-separated origins)

Common optional values:

- `GEMINI_MODEL=gemini-2.5-flash`
- `WHISPER_MODEL_SIZE=base`
- `NOTES_TRANSCRIPT_MAX_CHARS=24000`
- `NOTE_CHAT_TRANSCRIPT_MAX_CHARS_BRIEF=6000`
- `NOTE_CHAT_TRANSCRIPT_MAX_CHARS=28000`
- `NOTE_CHAT_MAX_OUTPUT_TOKENS_BRIEF=700`
- `NOTE_CHAT_MAX_OUTPUT_TOKENS_EXPANDED=1400`

## Runtime requirements

- `ffmpeg` must be installed and available in PATH.
- Whisper model files are downloaded on first use.

## Deploy backend on Render (Docker)

This repo includes:

- `backend/Dockerfile` (installs `ffmpeg`)
- `render.yaml` (Render Blueprint)

### Deploy steps

1. Push repo to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select the repo (Render reads `render.yaml`).
4. Set env vars in Render service:
   - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
   - `CORS_ORIGINS=https://<your-vercel-app>.vercel.app`
   - optional Firebase Admin: `FIREBASE_SERVICE_ACCOUNT_JSON`
5. Deploy.

### Verify deployed backend

- `GET https://<render-service>.onrender.com/api/podcast/health`

### Connect frontend (Vercel)

In Vercel env vars:

- `VITE_API_BASE_URL=https://<render-service>.onrender.com`
