"""Helpers for Hugging Face chat-template behavior used in the project."""

from __future__ import annotations

from typing import Any


def chat_template_kwargs(model_name: str | None) -> dict[str, Any]:
    """Return per-model chat-template options for deterministic prompting."""
    normalized = model_name.strip() if model_name else None
    if not normalized:
        return {}
    if normalized.lower().startswith("qwen/qwen3.5-") or normalized.lower().startswith("qwen/qwen3-"):
        return {"enable_thinking": False}
    return {}
