import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


LITELLM_URL = os.environ["LITELLM_URL"].rstrip("/")
LITELLM_KEY = os.environ["LITELLM_KEY"]

MODEL = os.getenv(
    "LLM_MODEL",
    "chat-model",
)


client = OpenAI(
    base_url=f"{LITELLM_URL}/v1",
    api_key=LITELLM_KEY,
)


def get_system_prompt(
    persona: str,
    context: str,
    note: str = "",
) -> str:

    from .personas import PERSONAS

    cfg = PERSONAS[persona]

    prompt = cfg["fallback"]

    prompt += "\n\nKNOWLEDGE BASE CONTEXT:\n"
    prompt += context

    if note:
        prompt += "\n\nADDITIONAL NOTE:\n"
        prompt += note

    return prompt


def chat(
    messages: list[dict],
    max_tokens: int = 500,
) -> str:

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
        extra_body={
            "include_reasoning": False,
        },
    )

    elapsed = time.perf_counter() - start

    print(
        f"LLM TIME: {elapsed:.2f}s"
    )

    print(
        "LLM TOKENS:",
        getattr(response, "usage", None),
    )

    message = response.choices[0].message

    content = (
        message.content or ""
    ).strip()

    if not content:
        return (
            "Sorry, I could not generate an answer right now."
        )

    return content