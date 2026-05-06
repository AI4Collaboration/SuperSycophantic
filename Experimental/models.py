"""Shared model panel definitions for SuperSycophantic runs."""

from __future__ import annotations

from collections.abc import Iterable


MAIN_MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
    "mistralai/mistral-medium-3.1",
    "cohere/command-r-08-2024",
]

SMOKE_TEST_MODELS = ["google/gemini-3.1-flash-lite-preview"]

MODEL_ALIASES = {
    "main": MAIN_MODELS,
    "smoke": SMOKE_TEST_MODELS,
    "gpt-5.4": "openai/gpt-5.4",
    "mini": "openai/gpt-5.4-mini",
    "nano": "openai/gpt-5.4-nano",
    "claude-opus": "anthropic/claude-opus-4.5",
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
    "claude-sonnet": "anthropic/claude-sonnet-4.5",
    "claude-sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "claude-haiku": "anthropic/claude-haiku-4.5",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "gemini-flash-lite": "google/gemini-3.1-flash-lite-preview",
    "gemini-flash-lite-preview": "google/gemini-3.1-flash-lite-preview",
    "mistral-medium": "mistralai/mistral-medium-3.1",
    "mistral-medium-3.1": "mistralai/mistral-medium-3.1",
    "command-r": "cohere/command-r-08-2024",
    "cohere-command-r": "cohere/command-r-08-2024",
}


def resolve_model(name: str) -> str:
    value = MODEL_ALIASES.get(name, name)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{name!r} expands to multiple models; use resolve_models instead")
        return value[0]
    return value


def resolve_models(names: Iterable[str] | None) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for name in names or []:
        value = MODEL_ALIASES.get(name, name)
        expanded = value if isinstance(value, list) else [value]
        for model in expanded:
            if model not in seen:
                resolved.append(model)
                seen.add(model)
    return resolved
