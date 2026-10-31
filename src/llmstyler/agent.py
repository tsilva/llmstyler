"""Explicit keyless AgentBridge boundary for agent-operated CLI processes."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODE_VARIABLE = "LLMSTYLER_AGENT_MODE"
MODE = "keyless-loopback-v1"
BASE_URL = "http://127.0.0.1:8082/api/v1"
ENVIRONMENT_KEYS = frozenset(
    {"HOME", "CODEX_HOME", "PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "WINDIR"}
)


class AgentModeError(ValueError):
    """Fixed diagnostics which never include credentials or rejected URLs."""


def agent_mode() -> bool:
    mode = os.environ.get(MODE_VARIABLE)
    if mode not in (None, "", MODE):
        raise AgentModeError("Invalid LLMStyler agent mode.")
    return mode == MODE


def agent_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key in ENVIRONMENT_KEYS}
    environment[MODE_VARIABLE] = MODE
    environment["AGENTBRIDGE_BASE_URL"] = BASE_URL
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return environment


def validate_base_url(value: str) -> None:
    if value not in (BASE_URL, BASE_URL + "/"):
        raise AgentModeError("Agent mode requires the reviewed keyless loopback endpoint.")


def reject_agent_operation(operation: str) -> None:
    if agent_mode():
        raise AgentModeError(f"{operation} has no reviewed agent credential route.")


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise urllib.error.HTTPError(
            request.full_url, code, "Redirects are disabled in agent mode.", headers, response
        )


def open_local(request: str | urllib.request.Request, *, timeout: float):
    request = urllib.request.Request(request) if isinstance(request, str) else request
    allowed = {
        (BASE_URL + "/models", "GET"),
        (BASE_URL + "/chat/completions", "POST"),
    }
    if (request.full_url, request.get_method()) not in allowed:
        raise AgentModeError("Agent HTTP requests must use the reviewed loopback routes.")
    if any(name.lower() not in {"content-type", "accept"} for name, _ in request.header_items()):
        raise AgentModeError("Agent HTTP requests cannot carry authentication headers.")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirects())
    return opener.open(request, timeout=timeout)


def http_child() -> int:
    try:
        request = json.loads(sys.stdin.read())
        http_request = urllib.request.Request(
            request["url"],
            data=json.dumps(request["payload"]).encode("utf-8"),
            headers=request["headers"],
            method="POST",
        )
        with open_local(http_request, timeout=request["timeout"]) as response:
            sys.stdout.write(response.read().decode("utf-8"))
        return 0
    except Exception:
        sys.stderr.write("AgentBridge request failed in keyless agent mode.")
        return 1


def launch(arguments: list[str], *, root: Path) -> None:
    if arguments == ["check"]:
        print(f"Agent mode: keyless {BASE_URL}; no service or credential request performed.")
        return
    if not arguments or arguments[0] != "run" or len(arguments) < 2:
        raise AgentModeError("Usage: python scripts/agent.py check|run <llmstyler arguments>")
    environment = agent_environment()
    os.chdir(root)
    os.execve(
        sys.executable,
        [sys.executable, "-m", "llmstyler.cli", *arguments[1:]],
        environment,
    )


if __name__ == "__main__":
    raise SystemExit(http_child())
