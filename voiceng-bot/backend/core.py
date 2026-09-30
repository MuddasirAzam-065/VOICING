import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pymongo import MongoClient

from personas import PERSONAS, PROMPT

load_dotenv()

CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
TOP_K = 6
MIN_SCORE = 0.70  # Atlas cosine score = (1 + cos) / 2, so 0.70 is about cos 0.40. Tune this.

ERROR_MESSAGE = "Sorry, I'm having trouble answering right now. Please try again."

db = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=10000)["voiceng"]
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def embed(texts: list[str], task: str) -> list[list[float]]:
    result = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task,
            output_dimensionality=EMBED_DIM,
        ),
    )
    return [e.values for e in result.embeddings]


def search(query: str, persona: str) -> list[dict]:
    vector = embed([query], "RETRIEVAL_QUERY")[0]

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

    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    try:
        response = gemini.models.generate_content(
            model=CHAT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        answer = (response.text or "").strip() or ERROR_MESSAGE
    except Exception as exc:
        print("GEMINI ERROR:", repr(exc))
        answer = ERROR_MESSAGE

    sources = list(dict.fromkeys(d["url"] for d in docs if d.get("url")))
    return answer, sources


def transcribe(audio_bytes: bytes, mime_type: str) -> str:
    response = gemini.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            "Transcribe this audio exactly. Return only the transcript, "
            "or an empty string if there is no speech.",
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()
