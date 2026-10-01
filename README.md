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

Set `MONGO_URI` in the backend environment. Add `GROQ_API_KEY` for the recommended fast free chat provider; it uses `GROQ_MODEL` (default `llama-3.3-70b-versatile`). `GEMINI_API_KEY` is optional and remains a fallback; it uses `GEMINI_MODEL` (default `gemini-3.1-flash-lite`). `HF_TOKEN` is still used when available for embeddings, speech transcription, and final chat fallback, but chat and retrieval can continue without it. The Groq, Gemini, and Hugging Face free tiers can be throttled or rejected when quotas run out.

Repeated embeddings and searches are cached in process memory, and the backend reuses worker threads plus its provider HTTP connections. Retrieval uses the four strongest matching chunks to limit model input size. Conversation indexes are created automatically at backend startup. Groq is primary when `GROQ_API_KEY` is set, followed by Gemini and Hugging Face. If Atlas or embedding retrieval fails, retrieval automatically uses MongoDB keyword matching, so the knowledge base remains usable after HF credits expire.

The raw “I couldn't generate a full response...” text means no chat provider produced an answer; it is not a MongoDB failure. Check `/debug` or the backend logs for the `GROQ_API_KEY`, `GEMINI_API_KEY`, and `HF_TOKEN` status plus `GROQ CHAT ERROR`, `GEMINI CHAT ERROR`, or `HF CHAT ERROR`. Common causes are a missing Vercel variable, exhausted quota, an invalid key, a temporarily unavailable provider, or an unavailable model. Retrieved knowledge is returned as a grounded excerpt when generation is unavailable. Keep `MONGO_URI`, `HF_TOKEN`, and the existing optional provider key names unchanged.

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

`voiceng-bot/backend/ingest.py` ingests site content into MongoDB Atlas. With `HF_TOKEN`, it stores embeddings and uses Atlas vector search. Without `HF_TOKEN`, ingestion still stores text chunks and the backend uses MongoDB keyword retrieval; Groq cannot create embeddings, so no Groq key is needed for ingestion. Configure the persona site URLs and ensure the Atlas vector index is named `vector_index` with a `persona` filter and 384-dimensional `embedding` field when using vector search.
