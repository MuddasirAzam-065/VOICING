import os
import re
import time
import requests

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pymongo import MongoClient

from personas import PERSONAS, PROMPT

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
STT_MODEL = os.getenv("STT_MODEL", "openai/whisper-large-v3-turbo")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))  # must match EMBED_MODEL
TOP_K = 4
MIN_SCORE = 0.60  # Atlas cosine score = (1 + cos) / 2. Tune this.
_EMBED_POOL = ThreadPoolExecutor(max_workers=4)
_GEMINI_SESSION = requests.Session()
_GROQ_SESSION = requests.Session()

ERROR_MESSAGE = "Sorry, I'm having trouble answering right now. Please try again."
PROVIDER_RETRIES = 2

db = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=10000)["voiceng"]
hf = InferenceClient(api_key=os.environ["HF_TOKEN"])


def _embed_one(text: str) -> list[float]:
    vec = np.array(hf.feature_extraction(text, model=EMBED_MODEL), dtype="float32")
    if vec.ndim > 1:                      # token-level output -> mean pool
        vec = vec.reshape(-1, vec.shape[-1]).mean(axis=0)
    vec = vec / (np.linalg.norm(vec) or 1.0)
    return vec.tolist()


@lru_cache(maxsize=256)
def _cached_embedding(text: str) -> tuple[float, ...]:
    return tuple(_embed_one(text))


def embed(texts: list[str], task: str = "") -> list[list[float]]:
    # `task` is kept so ingest.py needs no change; MiniLM doesn't use it.
    if not texts:
        return []
    vectors = map(_cached_embedding, texts) if len(texts) == 1 else _EMBED_POOL.map(_cached_embedding, texts)
    return [list(vector) for vector in vectors]


@lru_cache(maxsize=128)
def _cached_search(query: str, persona: str) -> tuple[dict, ...]:
    vector = embed([query], "query")[0]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": 100,
                "limit": TOP_K,
                "filter": {"persona": persona},
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "url": 1,
                "title": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    docs = [d for d in db.chunks.aggregate(pipeline) if d["score"] >= MIN_SCORE]
    return tuple(docs)


def search(query: str, persona: str) -> list[dict]:
    try:
        return list(_cached_search(query, persona))
    except Exception as exc:
        print("VECTOR SEARCH ERROR:", type(exc).__name__)
        return _keyword_search(query, persona)


@lru_cache(maxsize=128)
def _keyword_search(query: str, persona: str) -> tuple[dict, ...]:
    terms = {
        term.lower()
        for term in re.findall(r"[a-zA-Z0-9]{3,}", query)
        if term.lower() not in {"what", "where", "when", "which", "does", "about", "voice"}
    }
    if not terms:
        return ()

    docs = list(db.chunks.find(
        {"persona": persona},
        {"_id": 0, "text": 1, "url": 1, "title": 1},
    ))
    ranked = []
    for doc in docs:
        words = set(re.findall(r"[a-zA-Z0-9]{3,}", doc.get("text", "").lower()))
        score = len(terms & words)
        if score:
            ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return tuple(doc for _, doc in ranked[:TOP_K])


def build_context(docs: list[dict]) -> str:
    if not docs:
        return "No relevant knowledge-base content was found."
    return "\n\n".join(
        f"SOURCE {i}\nTitle: {d['title']}\nURL: {d['url']}\n\n{d['text'][:2800]}"
        for i, d in enumerate(docs, start=1)
    )


def answer_question(persona: str, question: str, history: list[dict]):
    cfg = PERSONAS[persona]
    # A temporary embedding or Atlas outage should not turn the whole /chat
    # request into a 500. The prompt tells the model to say when facts are
    # unavailable, so it can still answer gracefully without retrieved facts.
    try:
        docs = search(f"{cfg['org']} {question}", persona)
    except Exception as exc:
        print("SEARCH ERROR:", type(exc).__name__)
        docs = []

    system_prompt = PROMPT.format(
        label=cfg["label"],
        org=cfg["org"],
        context=build_context(docs),
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": question})

    answer = ""
    if os.getenv("GROQ_API_KEY"):
        try:
            answer = _groq_answer(messages)
        except Exception as exc:
            print("GROQ CHAT ERROR:", type(exc).__name__)

    if os.getenv("GEMINI_API_KEY"):
        try:
            answer = _gemini_answer(system_prompt, history, question)
        except Exception as exc:
            # Gemini free-tier quotas and transient service failures can reject
            # requests. HF_TOKEN is already required for embeddings/STT, so use
            # its configured chat model as an automatic backup.
            print("GEMINI CHAT ERROR:", type(exc).__name__)

    if not answer:
        try:
            answer = _hf_answer(messages)
        except Exception as exc:
            print("HF CHAT ERROR:", type(exc).__name__)

    answer = answer or _retrieval_fallback(docs, cfg)
    answer = answer or ERROR_MESSAGE

    sources = list(dict.fromkeys(d["url"] for d in docs if d.get("url")))
    return answer, sources


def _hf_answer(messages: list[dict]) -> str:
    last_error = None
    for attempt in range(PROVIDER_RETRIES):
        try:
            response = hf.chat_completion(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=500,
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                continue
            message = getattr(choices[0], "message", None)
            answer = getattr(message, "content", "") if message else ""
            if isinstance(answer, list):
                answer = "".join(
                    part.get("text", "") for part in answer if isinstance(part, dict)
                )
            answer = answer.strip() if isinstance(answer, str) else ""
            if answer:
                return answer
        except Exception as exc:
            last_error = exc
            if attempt + 1 == PROVIDER_RETRIES:
                break
            time.sleep(0.35)

    try:
        prompt = "\n\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in messages
        ) + "\n\nAssistant:"
        answer = hf.text_generation(
            prompt,
            model=CHAT_MODEL,
            temperature=0.2,
            max_new_tokens=500,
            return_full_text=False,
        )
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    except Exception as exc:
        last_error = exc

    if last_error:
        raise last_error
    return ""


def _groq_answer(messages: list[dict]) -> str:
    response = _GROQ_SESSION.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 500,
        },
        timeout=25,
    )
    response.raise_for_status()
    choices = response.json().get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def _retrieval_fallback(docs: list[dict], cfg: dict) -> str:
    if not docs:
        return ""
    excerpts = []
    for doc in docs[:2]:
        text = " ".join(doc.get("text", "").split())
        if text:
            excerpts.append(text[:700].rstrip())
    if not excerpts:
        return ""
    return (
        f"I couldn't generate a full response, but I found this information in the "
        f"{cfg['org']} knowledge base:\n\n"
        + "\n\n".join(f"- {excerpt}" for excerpt in excerpts)
    )


def _gemini_answer(system_prompt: str, history: list[dict], question: str) -> str:
    contents = [
        {
            "role": "model" if item["role"] == "assistant" else "user",
            "parts": [{"text": item["content"]}],
        }
        for item in history
        if item.get("role") in {"user", "assistant"}
    ]
    contents.append({"role": "user", "parts": [{"text": question}]})
    response = _GEMINI_SESSION.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        params={"key": os.environ["GEMINI_API_KEY"]},
        json={
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500},
        },
        timeout=25,
    )
    response.raise_for_status()
    result = response.json()
    candidates = result.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip()


def transcribe(audio_bytes: bytes, mime_type: str) -> str:
    # The provider detects the codec from the uploaded bytes.
    result = hf.automatic_speech_recognition(audio_bytes, model=STT_MODEL)
    return (result.text or "").strip()
