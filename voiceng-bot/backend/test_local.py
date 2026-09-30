import traceback
from core import db, hf, embed, search, CHAT_MODEL

print("chunks in db:", db.chunks.count_documents({}))

try:
    print("embed dim:", len(embed(["test"], "query")[0]))
except Exception:
    traceback.print_exc()

try:
    hits = db.chunks.aggregate([
        {"$vectorSearch": {"index": "vector_index", "path": "embedding",
                           "queryVector": embed(["What is VoiceNG?"], "query")[0],
                           "numCandidates": 100, "limit": 5,
                           "filter": {"persona": "voiceng"}}},
        {"$project": {"_id": 0, "url": 1, "score": {"$meta": "vectorSearchScore"}}},
    ])
    print("raw search hits:", list(hits))
except Exception:
    traceback.print_exc()

for model in [CHAT_MODEL, "meta-llama/Llama-3.1-8B-Instruct",
              "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]:
    try:
        r = hf.chat_completion(model=model,
                               messages=[{"role": "user", "content": "Say hi"}],
                               max_tokens=20)
        print("OK  ", model, "->", r.choices[0].message.content)
    except Exception as e:
        print("FAIL", model, "->", repr(e)[:200])
