"""Explicit keyless AgentBridge boundary for agent-operated CLI processes."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

MODE_VARIABLE = "LLMSTYLER_AGENT_MODE"
MODE = "keyless-loopback-v1"
VAULT_VARIABLE = "LLMSTYLER_AGENT_VAULT"
VAULT_MARKER = "llmstyler_native_placeholder"
CA_SHA256 = "d18b13df74a7fcaf752cc2ee2c1eaf18d3e03abd89630ef611b2b0a7a765005a"
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


def native_proxy() -> str | None:
    """Validate an explicit native session without fetching any service key."""
    marker = os.environ.get(VAULT_VARIABLE)
    if marker in (None, ""):
        if os.environ.get("INFISICAL_AGENT_VAULT_URL"):
            raise AgentModeError("Native Vault sessions require the reviewed placeholder.")
        return None
    try:
        if (marker != VAULT_MARKER
                or os.environ.get("INFISICAL_AGENT_VAULT_URL") != "https://app.infisical.com"):
            raise ValueError
        proxy = os.environ["HTTPS_PROXY"]
        parsed = urlsplit(proxy)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
                or parsed.port != 8283 or not parsed.password
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment
                or os.environ.get("HTTP_PROXY") != proxy
                or os.environ.get("NO_PROXY") or os.environ.get("no_proxy")):
            raise ValueError
        if any(os.environ.get(key, proxy) != proxy for key in ("http_proxy", "https_proxy")):
            raise ValueError
        ca = Path(os.environ["LLMSTYLER_VAULT_CA_FILE"])
        if (ca != Path.home() / ".config/infisical/agent-vault-ca.pem" or ca.resolve() != ca
                or hashlib.sha256(ca.read_bytes()).hexdigest() != CA_SHA256):
            raise ValueError
        return proxy
    except (OSError, ValueError, KeyError):
        raise AgentModeError("Expected the reviewed native Vault session and placeholder.") from None


def agent_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key in ENVIRONMENT_KEYS}
    environment[MODE_VARIABLE] = MODE
    environment["AGENTBRIDGE_BASE_URL"] = BASE_URL
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    proxy = native_proxy()
    if proxy:
        environment[VAULT_VARIABLE] = VAULT_MARKER
        environment["LLMSTYLER_VAULT_CA_FILE"] = os.environ["LLMSTYLER_VAULT_CA_FILE"]
        environment["INFISICAL_AGENT_VAULT_URL"] = "https://app.infisical.com"
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            environment[key] = proxy
        environment["NO_PROXY"] = environment["no_proxy"] = ""
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
    proxy = native_proxy()
    proxies = {"http": proxy, "https": proxy} if proxy else {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies), _NoRedirects())
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
    if arguments == ["vault-check"]:
        try:
            if native_proxy() is None:
                raise ValueError
            with open_local(BASE_URL + "/models", timeout=5) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError
            value = json.loads(raw)
            if not isinstance(value, dict) or not isinstance(value.get("data"), list):
                raise ValueError
        except Exception:
            raise AgentModeError("Native Vault catalog check failed; provider details withheld.") from None
        print("Native Agent Vault gateway catalog verified; no inference performed.")
        return
    if arguments == ["check"]:
        print(f"Agent mode: keyless {BASE_URL}; no service or credential request performed.")
        return
    if not arguments or arguments[0] != "run" or len(arguments) < 2:
        raise AgentModeError("Usage: python scripts/agent.py check|vault-check|run <llmstyler arguments>")
    environment = agent_environment()
    os.chdir(root)
    os.execve(
        sys.executable,
        [sys.executable, "-m", "llmstyler.cli", *arguments[1:]],
        environment,
    )


if __name__ == "__main__":
    raise SystemExit(http_child())
