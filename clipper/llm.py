"""Small wrapper for structured Claude calls (text and images)."""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def image_block(path: Path) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def parse(model: str, system: str, content: list[dict] | str, schema: type[T],
          effort: str = "high", max_tokens: int = 32000) -> T:
    import anthropic

    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=model,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        output_format=schema,
        system=system,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined: {msg.stop_details}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("Response hit max_tokens before finishing")
    return msg.parsed_output
