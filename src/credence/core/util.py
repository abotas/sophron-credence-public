"""Minimal stub — only provider() and short_model() are needed by the viz app."""

import re


def provider(model_id: str) -> str:
    """Extract provider from model ID.

    'openai/gpt-5-mini-2025-08-07' -> 'openai'
    'openai-api/deepseek/deepseek-chat' -> 'deepseek'
    """
    parts = model_id.split("/")
    if parts[0] == "openai-api" and len(parts) >= 3:
        return parts[1]
    return parts[0]


def short_model(model_id: str) -> str:
    """Strip provider prefix and date suffix from model ID."""
    name = model_id.split("/")[-1]
    return re.sub(r"[-_]\d{4}[-_]?\d{2}[-_]?\d{2}$", "", name)
