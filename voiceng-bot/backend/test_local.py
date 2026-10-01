import traceback
from core import db, embed

print("chunks in db:", db.chunks.count_documents({}))
print("by persona:", list(db.chunks.aggregate([
    {"$group": {"_id": "$persona", "n": {"$sum": 1}}}
])))

sample = db.chunks.find_one({}, {"_id": 0, "embedding": 1, "title": 1})
print("stored embedding dim:", len(sample["embedding"]) if sample else None)
print("sample title:", sample["title"] if sample else None)

for q in ["What is VoiceNG?", "VoiceNG What is VoiceNG?", "services"]:
    try:
        vec = embed([q], "query")[0]
        hits = list(db.chunks.aggregate([
            {"$vectorSearch": {"index": "vector_index", "path": "embedding",
                               "queryVector": vec, "numCandidates": 100,
                               "limit": 5, "filter": {"persona": "voiceng"}}},
            {"$project": {"_id": 0, "url": 1,
                          "score": {"$meta": "vectorSearchScore"}}},
        ]))
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  {h['score']:.3f}  {h['url']}")
        if not hits:
            print("  NO HITS")
    except Exception:
        traceback.print_exc()
