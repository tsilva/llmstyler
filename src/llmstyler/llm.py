"""Shared inference settings for previews, chats, and dataset rewrites."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from llmstyler.agent import AgentModeError, agent_mode, validate_base_url

DEFAULT_PROVIDER = "agentbridge"
DEFAULT_MODEL = "codex/gpt-6.1-sol"
DEFAULT_BASE_URL = "http://127.0.0.1:8082/api/v1"
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def resolve_llm_settings(style: dict[str, Any]) -> dict[str, Any]:
    settings = dict(style)
    provider = settings.get("provider", DEFAULT_PROVIDER)
    if provider != "agentbridge":
        raise ValueError("LLMStyler uses only AgentBridge; select the model, not a provider")
    settings["provider"] = "agentbridge"
    settings.setdefault("model", DEFAULT_MODEL)
    model = settings["model"]
    if not (
        model.startswith(("codex/", "claudecode/"))
        or (model.startswith("openrouter/") and model.count("/") >= 2)
        or model == "openrouter/default"
    ):
        settings["model"] = f"openrouter/{model}"
    settings.setdefault("base_url", os.getenv("AGENTBRIDGE_BASE_URL") or DEFAULT_BASE_URL)
    parsed = urlparse(settings["base_url"])
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.hostname == "openrouter.ai"
    ):
        raise ValueError("Configure an AgentBridge HTTP endpoint")
    if settings["model"].startswith("codex/"):
        settings.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
    settings.setdefault("request_timeout", 630.0)
    settings.setdefault("workers", 1)
    settings["base_url"] = settings["base_url"].rstrip("/")
    effort = settings.get("reasoning_effort")
    if effort is not None and effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
    if agent_mode():
        validate_base_url(settings["base_url"])
    return settings


def provider_api_key(style: dict[str, Any]) -> str | None:
    resolve_llm_settings(style)
    if agent_mode():
        if os.environ.get("AGENTBRIDGE_API_KEY"):
            raise AgentModeError("Agent mode does not accept application API keys.")
        return None
    return os.getenv("AGENTBRIDGE_API_KEY") or None
