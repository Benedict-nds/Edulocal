# OpenAI chat completions wrapper: loads API key from environment / .env via dotenv.

import os
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# Quick check for Streamlit validation without constructing a client or raising.
def api_key_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


# Build a client; raises RuntimeError if the key is missing (caller shows UI error).
def get_client() -> OpenAI:
    if not api_key_configured():
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in .env or your environment."
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())


# Single chat completion; low temperature for steadier grounding-style answers.
def chat_completion(
    messages: Iterable[dict[str, str]],
    model: str | None = None,
) -> str:
    client = get_client()
    model_name = (model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip()
    response = client.chat.completions.create(
        model=model_name,
        messages=list(messages),
        temperature=0.2,
    )
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        return ""
    return content.strip()
