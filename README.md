# VOICING# VoiceNG Bot

Docker-free VoiceNG RAG voice assistant.

## Architecture

Browser
  |
  v
Vercel
  |
  v
FastAPI on Render
  |
  +-- Groq Whisper STT
  |
  +-- Sentence Transformers embeddings
  |
  +-- MongoDB Atlas Vector Search
  |
  +-- LangGraph
  |
  +-- LiteLLM on Render
          |
          v
      Hugging Face
      Granite 4.2 3B

Langfuse is used for observability.

## Backend

cd backend

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload

## LiteLLM

cd litellm

pip install -r requirements.txt

export HF_TOKEN="your_token"

export LITELLM_MASTER_KEY="your_key"

litellm --config litellm_config.yaml --port 4000

## Ingestion

From the repository root:

python -m backend.app.ingest --persona both

## API

GET /health

POST /stt

POST /chat

GET /conversations

GET /history/{cid}

DELETE /conversations/{cid}

POST /admin/ingest

GET /admin/ingest/{run_id}