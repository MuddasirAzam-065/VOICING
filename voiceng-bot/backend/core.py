import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pymongo import MongoClient

from personas import PERSONAS, PROMPT

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
STT_MODEL = os.getenv("STT_MODEL", "openai/whisper-large-v3-turbo")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))  # must match EMBED_MODEL
TOP_K = 6
MIN_SCORE = 0.60  # Atlas cosine score = (1 + cos) / 2. Tune this.

ERROR_MESSAGE = "Sorry, I'm having trouble answering right now. Please try again."

db = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=10000)["voiceng"]
hf = InferenceClient(api_key=os.environ["HF_TOKEN"])


def _embed_one(text: str) -> list[float]:
    vec = np.array(hf.feature_extraction(text, model=EMBED_MODEL), dtype="float32")
    if vec.ndim > 1:                      # token-level output -> mean pool
        vec = vec.reshape(-1, vec.shape[-1]).mean(axis=0)
    vec = vec / (np.linalg.norm(vec) or 1.0)
    return vec.tolist()


def embed(texts: list[str], task: str = "") -> list[list[float]]:
    # `task` is kept so ingest.py needs no change; MiniLM doesn't use it.
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(_embed_one, texts))


def search(query: str, persona: str) -> list[dict]:
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

    docs = list(db.chunks.aggregate(pipeline))
    return [d for d in docs if d["score"] >= MIN_SCORE]


def build_context(docs: list[dict]) -> str:
    if not docs:
        return "No relevant knowledge-base content was found."
    return "\n\n".join(
        f"SOURCE {i}\nTitle: {d['title']}\nURL: {d['url']}\n\n{d['text']}"
        for i, d in enumerate(docs, start=1)
    )


def answer_question(persona: str, question: str, history: list[dict]):
    cfg = PERSONAS[persona]
    docs = search(f"{cfg['org']} {question}", persona)

    system_prompt = PROMPT.format(
        label=cfg["label"],
        org=cfg["org"],
        context=build_context(docs),
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": question})

    try:
        response = hf.chat_completion(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
        answer = (response.choices[0].message.content or "").strip() or ERROR_MESSAGE
    except Exception as exc:
        print("CHAT ERROR:", repr(exc))
        answer = ERROR_MESSAGE

    sources = list(dict.fromkeys(d["url"] for d in docs if d.get("url")))
    return answer, sources


def transcribe(audio_bytes: bytes, mime_type: str) -> str:
    result = hf.automatic_speech_recognition(audio_bytes, model=STT_MODEL)
    return (result.text or "").strip()
