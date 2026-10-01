# VoiceNG

VoiceNG is a static browser chat frontend with a FastAPI backend, MongoDB Atlas vector search, Hugging Face embeddings and speech recognition, and an optional Gemini chat model.

## Run the backend

```sh
cd voiceng-bot/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn index:app --reload
```

Set `MONGO_URI` and `HF_TOKEN` in the backend environment. `GEMINI_API_KEY` is optional; if present, chat uses `GEMINI_MODEL` (default `gemini-3.1-flash-lite`). Without it, chat uses `CHAT_MODEL` (default `Qwen/Qwen2.5-3B-Instruct`). Both model settings can be changed without changing any secret names. The Gemini Developer API offers a free tier, and Hugging Face currently includes a small monthly inference credit for free accounts. To keep provider usage at $0, use a Gemini key from a project without billing enabled and do not purchase extra Hugging Face credits; requests can be throttled or rejected when free quotas run out.

Repeated embeddings are cached in process memory, and the backend reuses worker threads and its Gemini HTTP connection. Retrieval uses the four strongest matching chunks to limit model input size. Conversation indexes are created automatically at backend startup.

Set `CORS_ORIGINS` to the deployed frontend origin (comma separated if there are multiple). Keep the existing secret variable names configured in Vercel. The speech recording feature requires HTTPS and browser microphone permission.

## Run the frontend

Open `voiceng-bot/frontend/index.html` through a local static server or deploy that directory to Vercel. Set `API_URL` near the top of its script to the backend's HTTPS origin, without a trailing slash. The browser prefers an available English female system voice; the exact voice depends on voices installed in the user's browser/OS.

## API endpoints

The backend exposes these routes (interactive schema at `/docs`):

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` or `/health` | Health check |
| `POST` | `/chat` | Chat; JSON body: `message`, `user_id`, optional `cid`, and `persona` (`voiceng` or `fia`) |
| `POST` | `/stt` | Transcribe multipart audio in the `audio` field |
| `GET` | `/conversations?user_id=...` | List a user's conversations |
| `GET` | `/history/{cid}?user_id=...` | Read one conversation |
| `DELETE` | `/conversations/{cid}?user_id=...` | Delete one conversation |
| `GET` | `/debug` | Backend dependency diagnostics; restrict or remove before public production use |

## Knowledge base

`voiceng-bot/backend/ingest.py` ingests site content into MongoDB Atlas. Configure the persona site URLs and ensure the Atlas vector index is named `vector_index` with a `persona` filter and 384-dimensional `embedding` field before using retrieval.
