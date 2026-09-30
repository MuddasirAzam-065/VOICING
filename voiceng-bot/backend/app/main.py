import datetime
import os
import tempfile
import uuid

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langfuse import observe, propagate_attributes
from pydantic import BaseModel
from google import genai
from google.genai import types

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

from .graph import graph
from .kb import db
from .llm import chat
from .personas import PERSONAS
from . import ingest


load_dotenv()


app = FastAPI(
    title="VoiceNG API",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

cors_value = os.getenv(
    "CORS_ORIGINS",
    "*",
)

if cors_value == "*":
    origins = ["*"]
else:
    origins = [
        item.strip()
        for item in cors_value.split(",")
        if item.strip()
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Clients
# --------------------------------------------------

gemini_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


# --------------------------------------------------
# Models
# --------------------------------------------------

class ChatRequest(BaseModel):
    cid: str | None = None
    persona: str = "voiceng"
    message: str
    user_id: str | None = None


class ChatResponse(BaseModel):
    cid: str
    answer: str
    sources: list[str]


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "voiceng-api",
    }


# --------------------------------------------------
# STT
# --------------------------------------------------

@app.post("/stt")
def speech_to_text(audio: UploadFile = File(...)):

    audio_bytes = audio.file.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = "." + audio.filename.lower().rsplit(".", 1)[1]

    mime_types = {
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".m4a": "audio/aac",
    }
    mime = mime_types.get(suffix, "audio/webm")

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Transcribe this audio exactly. Return only the transcript "
                "and nothing else. If there is no speech, return an empty string.",
                types.Part.from_bytes(data=audio_bytes, mime_type=mime),
            ],
        )
    except Exception as exc:
        print("STT ERROR:", repr(exc))
        raise HTTPException(status_code=502, detail="Transcription failed.")

    return {"text": (response.text or "").strip()}

# --------------------------------------------------
# Chat
# --------------------------------------------------

@observe(name="chat-request")
def run_chat(
    cid: str,
    persona: str,
    message: str,
    user_id: str | None,
):

    with propagate_attributes(
        session_id=cid,
        user_id=user_id,
        tags=[persona],
    ):

        history_docs = list(
            db.messages.find(
                {
                    "cid": cid,
                },
                {
                    "_id": 0,
                    "role": 1,
                    "content": 1,
                },
            )
            .sort(
                "created_at",
                1,
            )
            .limit(20)
        )

        history = [
            {
                "role": item["role"],
                "content": item["content"],
            }
            for item in history_docs
        ]

        result = graph.invoke(
            {
                "cid": cid,
                "persona": persona,
                "question": message,
                "history": history,
            },
            config={
                "configurable": {
                    "thread_id": cid,
                }
            },
        )

        return result


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat_endpoint(
    request: ChatRequest,
):

    if request.persona not in PERSONAS:
        raise HTTPException(
            status_code=400,
            detail="Invalid persona.",
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    cid = (
        request.cid
        or str(uuid.uuid4())
    )

    result = run_chat(
        cid=cid,
        persona=request.persona,
        message=message,
        user_id=request.user_id,
    )

    return ChatResponse(
        cid=cid,
        answer=result.get(
            "answer",
            "Sorry, I could not answer that.",
        ),
        sources=result.get(
            "sources",
            [],
        ),
    )


# --------------------------------------------------
# Conversations
# --------------------------------------------------

@app.get("/conversations")
def conversations():

    pipeline = [
        {
            "$sort": {
                "created_at": -1
            }
        },
        {
            "$group": {
                "_id": "$cid",
                "last_message": {
                    "$first": "$content"
                },
                "created_at": {
                    "$first": "$created_at"
                },
                "persona": {
                    "$first": "$persona"
                },
            }
        },
        {
            "$sort": {
                "created_at": -1
            }
        },
        {
            "$limit": 50
        },
    ]

    rows = list(
        db.messages.aggregate(
            pipeline
        )
    )

    return [
        {
            "cid": row["_id"],
            "last_message": row[
                "last_message"
            ],
            "created_at": row[
                "created_at"
            ],
            "persona": row[
                "persona"
            ],
        }
        for row in rows
    ]


@app.get(
    "/history/{cid}"
)
def history(cid: str):

    rows = list(
        db.messages.find(
            {
                "cid": cid,
            },
            {
                "_id": 0,
                "role": 1,
                "content": 1,
                "created_at": 1,
            },
        ).sort(
            "created_at",
            1,
        )
    )

    return rows


@app.delete(
    "/conversations/{cid}"
)
def delete_conversation(
    cid: str,
):

    db.messages.delete_many(
        {
            "cid": cid,
        }
    )

    return {
        "deleted": True,
        "cid": cid,
    }


# --------------------------------------------------
# Admin ingestion
# --------------------------------------------------

ingestion_running = False


def perform_ingestion(
    persona: str,
    run_id: str,
):

    global ingestion_running

    try:

        db.ingest_runs.update_one(
            {
                "run_id": run_id,
            },
            {
                "$set": {
                    "status": "running",
                    "started_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ),
                }
            },
        )

        ingest.run(persona)

        db.ingest_runs.update_one(
            {
                "run_id": run_id,
            },
            {
                "$set": {
                    "status": "completed",
                    "finished_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ),
                }
            },
        )

    except Exception as exc:

        print(
            "INGEST ERROR:",
            repr(exc),
        )

        db.ingest_runs.update_one(
            {
                "run_id": run_id,
            },
            {
                "$set": {
                    "status": "failed",
                    "error": str(exc),
                    "finished_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ),
                }
            },
        )

    finally:

        ingestion_running = False


@app.post("/admin/ingest")
def start_ingestion(
    background_tasks: BackgroundTasks,
    persona: str = "both",
    x_api_key: str | None = Header(
        default=None
    ),
):

    global ingestion_running

    admin_key = os.getenv(
        "ADMIN_API_KEY"
    )

    if not admin_key:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_API_KEY is not configured.",
        )

    if x_api_key != admin_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    if persona not in {
        "voiceng",
        "fia",
        "both",
    }:
        raise HTTPException(
            status_code=400,
            detail="Invalid persona.",
        )

    if ingestion_running:
        raise HTTPException(
            status_code=409,
            detail="Ingestion already running.",
        )

    ingestion_running = True

    run_id = str(uuid.uuid4())

    db.ingest_runs.insert_one(
        {
            "run_id": run_id,
            "persona": persona,
            "status": "queued",
            "created_at": datetime.datetime.now(
                datetime.timezone.utc
            ),
        }
    )

    background_tasks.add_task(
        perform_ingestion,
        persona,
        run_id,
    )

    return {
        "run_id": run_id,
        "status": "queued",
    }


@app.get(
    "/admin/ingest/{run_id}"
)
def ingestion_status(
    run_id: str,
    x_api_key: str | None = Header(
        default=None
    ),
):

    admin_key = os.getenv(
        "ADMIN_API_KEY"
    )

    if x_api_key != admin_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    result = db.ingest_runs.find_one(
        {
            "run_id": run_id,
        },
        {
            "_id": 0,
        },
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="Ingestion run not found.",
        )

    return result
