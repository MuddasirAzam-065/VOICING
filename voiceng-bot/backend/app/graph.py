import datetime
import operator

from typing import Annotated, TypedDict

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from langgraph.checkpoint.mongodb import MongoDBSaver

from .kb import client, db, search
from .llm import chat, get_system_prompt
from .personas import PERSONAS


TOP_K = 6
MIN_SCORE = 0.30

ERROR_MESSAGE = (
    "Sorry, I'm having trouble answering right now. "
    "Please try again."
)


class State(TypedDict, total=False):
    cid: str
    persona: str
    question: str
    query: str
    docs: list
    answer: str
    sources: list
    history: Annotated[list, operator.add]


def rewrite_query(state: State):

    question = state["question"]
    persona = state["persona"]

    org = PERSONAS[persona]["org"]

    query = f"{org} {question}"

    return {
        "query": query,
    }


def retrieve(state: State):

    query = state["query"]
    persona = state["persona"]

    docs = search(
        query,
        persona,
        TOP_K,
    )

    docs = [
        doc
        for doc in docs
        if float(doc.get("score", 0)) >= MIN_SCORE
    ]

    return {
        "docs": docs,
    }


def build_context(docs: list[dict]) -> str:

    if not docs:
        return "No relevant knowledge-base content was found."

    parts = []

    for index, doc in enumerate(docs, start=1):

        title = doc.get("title", "")
        section = doc.get("section", "")
        text = doc.get("text", "")
        url = doc.get("url", "")

        parts.append(
            f"""
SOURCE {index}

Title: {title}
Section: {section}
URL: {url}

Content:
{text}
""".strip()
        )

    return "\n\n".join(parts)


def generate_answer(state: State):

    persona = state["persona"]
    question = state["question"]

    docs = state.get("docs", [])
    history = state.get("history", [])

    context = build_context(docs)

    system_prompt = get_system_prompt(
        persona,
        context,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # Only send a small amount of history.
    for item in history[-6:]:
        messages.append(item)

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:
        answer = chat(
            messages,
            max_tokens=500,
        )

    except Exception:
        answer = ERROR_MESSAGE

    sources = []

    seen = set()

    for doc in docs:

        url = doc.get("url")

        if url and url not in seen:
            seen.add(url)
            sources.append(url)

    return {
        "answer": answer,
        "sources": sources,
    }


def save_conversation(state: State):

    cid = state["cid"]

    persona = state["persona"]

    question = state["question"]

    answer = state["answer"]

    now = datetime.datetime.now(
        datetime.timezone.utc
    )

    db.messages.insert_many(
        [
            {
                "cid": cid,
                "persona": persona,
                "role": "user",
                "content": question,
                "created_at": now,
            },
            {
                "cid": cid,
                "persona": persona,
                "role": "assistant",
                "content": answer,
                "created_at": now,
            },
        ]
    )

    return {}


builder = StateGraph(State)

builder.add_node(
    "rewrite_query",
    rewrite_query,
)

builder.add_node(
    "retrieve",
    retrieve,
)

builder.add_node(
    "generate_answer",
    generate_answer,
)

builder.add_node(
    "save_conversation",
    save_conversation,
)

builder.add_edge(
    START,
    "rewrite_query",
)

builder.add_edge(
    "rewrite_query",
    "retrieve",
)

builder.add_edge(
    "retrieve",
    "generate_answer",
)

builder.add_edge(
    "generate_answer",
    "save_conversation",
)

builder.add_edge(
    "save_conversation",
    END,
)


checkpointer = MongoDBSaver(
    client,
    db_name="voiceng",
    checkpoint_collection_name="checkpoints",
    writes_collection_name="checkpoint_writes",
)


graph = builder.compile(
    checkpointer=checkpointer
)