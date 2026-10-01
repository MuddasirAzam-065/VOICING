import datetime
import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import answer_question, db, transcribe
from personas import PERSONAS

app = FastAPI(title="VoiceNG API")

cors = os.getenv("CORS_ORIGINS", "*")
origins = ["*"] if cors == "*" else [o.strip() for o in cors.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def ensure_message_indexes():
    # These cover chat/history lookups and the conversation-list aggregation.
    db.messages.create_index([("cid", 1), ("user_id", 1), ("created_at", 1)])
    db.messages.create_index([("user_id", 1), ("created_at", -1)])


class ChatRequest(BaseModel):
    cid: str | None = None
    persona: str = "voiceng"
    message: str
    user_id: str


class ChatResponse(BaseModel):
    cid: str
    answer: str
    sources: list[str]


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/stt")
def speech_to_text(audio: UploadFile = File(...)):
    data = audio.file.read()
    if not data:
        raise HTTPException(400, "Empty audio file.")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Audio file is too large (15 MB maximum).")

    mime = (audio.content_type or "audio/webm").split(";")[0]
    try:
        return {"text": transcribe(data, mime)}
    except Exception as exc:
        print("STT ERROR:", repr(exc))
        raise HTTPException(500, "Transcription failed.")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.persona not in PERSONAS:
        raise HTTPException(400, "Invalid persona.")

    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Message cannot be empty.")
    if len(message) > 4000:
        raise HTTPException(413, "Message is too long (4,000 characters maximum).")
    if not req.user_id.strip():
        raise HTTPException(400, "user_id cannot be empty.")

    cid = req.cid or str(uuid.uuid4())

    rows = list(
        db.messages.find(
            {"cid": cid, "user_id": req.user_id},
            {"_id": 0, "role": 1, "content": 1},
        )
        .sort("created_at", -1)
        .limit(6)
    )
    history = rows[::-1]

    answer, sources = answer_question(req.persona, message, history)

    now = datetime.datetime.now(datetime.timezone.utc)
    base = {"cid": cid, "user_id": req.user_id, "persona": req.persona}
    db.messages.insert_many(
        [
            {**base, "role": "user", "content": message, "created_at": now},
            {
                **base,
                "role": "assistant",
                "content": answer,
                "created_at": now + datetime.timedelta(milliseconds=1),
            },
        ]
    )

    return ChatResponse(cid=cid, answer=answer, sources=sources)


@app.get("/conversations")
def conversations(user_id: str):
    rows = db.messages.aggregate(
        [
            {"$match": {"user_id": user_id}},
            {"$sort": {"created_at": -1}},
            {
                "$group": {
                    "_id": "$cid",
                    "last_message": {"$first": "$content"},
                    "created_at": {"$first": "$created_at"},
                    "persona": {"$first": "$persona"},
                }
            },
            {"$sort": {"created_at": -1}},
            {"$limit": 50},
        ]
    )
    return [
        {
            "cid": r["_id"],
            "last_message": r["last_message"],
            "created_at": r["created_at"],
            "persona": r["persona"],
        }
        for r in rows
    ]


@app.get("/history/{cid}")
def history(cid: str, user_id: str):
    return list(
        db.messages.find(
            {"cid": cid, "user_id": user_id},
            {"_id": 0, "role": 1, "content": 1, "created_at": 1},
        ).sort("created_at", 1)
    )


@app.delete("/conversations/{cid}")
def delete_conversation(cid: str, user_id: str):
    db.messages.delete_many({"cid": cid, "user_id": user_id})
    return {"deleted": True, "cid": cid}


@app.get("/debug")
def debug():
    import traceback
    from core import embed, groq_probe, search

    out = {}
    for name, fn in [
        ("env", lambda: {k: bool(os.getenv(k)) for k in [
            "MONGO_URI", "HF_TOKEN", "GROQ_API_KEY", "GEMINI_API_KEY",
        ]}),
        ("mongo", lambda: db.command("ping")["ok"]),
        ("chunks_count", lambda: db.chunks.count_documents({})),
        ("embed", lambda: len(embed(["test"], "query")[0])),
        ("search", lambda: len(search("VoiceNG", "voiceng"))),
        ("groq", groq_probe),
    ]:
        try:
            out[name] = fn()
        except Exception:
            out[name] = "FAILED: " + traceback.format_exc()[-600:]
    return out
