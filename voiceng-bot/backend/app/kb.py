import os

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


load_dotenv()

MONGO_URI = os.environ["MONGO_URI"]

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=10000,
)

db = client["voiceng"]

# Small CPU-friendly embedding model.
embedder = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu",
)


def embed_text(text: str) -> list[float]:
    vector = embedder.encode(
        text,
        normalize_embeddings=True,
    )

    return vector.tolist()


def search(
    query: str,
    persona: str,
    k: int = 6,
) -> list[dict]:

    vector = embed_text(query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": max(50, k * 15),
                "limit": k,
                "filter": {
                    "persona": persona,
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "url": 1,
                "title": 1,
                "section": 1,
                "score": {
                    "$meta": "vectorSearchScore",
                },
            }
        },
    ]

    return list(
        db.chunks.aggregate(pipeline)
    )