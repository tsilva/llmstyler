"""Fresh native-session propagation without source keys or network traffic."""
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

from llmstyler import agent


@pytest.fixture
def native(monkeypatch, tmp_path):
    ca = tmp_path / ".config/infisical/agent-vault-ca.pem"
    ca.parent.mkdir(parents=True)
    ca.write_bytes(b"synthetic-public-ca")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(agent, "CA_SHA256", hashlib.sha256(ca.read_bytes()).hexdigest())
    proxy = "http://fixture:synthetic-session@127.0.0.1:8283"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, proxy)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv(agent.VAULT_VARIABLE, agent.VAULT_MARKER)
    monkeypatch.setenv("INFISICAL_AGENT_VAULT_URL", "https://app.infisical.com")
    monkeypatch.setenv("LLMSTYLER_VAULT_CA_FILE", str(ca))
    return proxy


def test_nested_environment_retains_only_reviewed_session_and_no_service_keys(native, monkeypatch):
    for name in ("OPENROUTER_API_KEY", "AGENTBRIDGE_API_KEY", "HF_TOKEN", "WANDB_API_KEY",
                 "INFISICAL_TOKEN", "MODAL_TOKEN_SECRET", "ALL_PROXY", "PYTHONPATH"):
        monkeypatch.setenv(name, "fake-private-value")
    env = agent.agent_environment()
    assert "fake-private-value" not in env.values()
    assert env[agent.VAULT_VARIABLE] == agent.VAULT_MARKER
    assert env[agent.MODE_VARIABLE] == agent.MODE
    assert env["INFISICAL_AGENT_VAULT_URL"] == "https://app.infisical.com"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert env[key] == native
    assert env["NO_PROXY"] == env["no_proxy"] == ""
    calls = []
    monkeypatch.setattr(agent.os, "chdir", lambda _path: None)
    monkeypatch.setattr(agent.os, "execve", lambda *args: calls.append(args))
    agent.launch(["run", "preview", "--compile-only"], root=Path("/synthetic"))
    assert calls[0][1] == [sys.executable, "-m", "llmstyler.cli", "preview", "--compile-only"]
    assert calls[0][2]["HTTPS_PROXY"] == native


@pytest.mark.parametrize("key,value", [
    (agent.VAULT_VARIABLE, "unknown"), (agent.VAULT_VARIABLE, ""),
    ("INFISICAL_AGENT_VAULT_URL", "https://unreviewed.invalid"),
    ("HTTPS_PROXY", "http://fixture@127.0.0.1:8283"),
    ("HTTP_PROXY", "http://other:other@127.0.0.1:8283"),
    ("HTTPS_PROXY", "http://fixture:session@unreviewed.invalid:8283"),
    ("http_proxy", "http://other:other@127.0.0.1:8283"),
    ("NO_PROXY", "*"), ("no_proxy", "127.0.0.1"),
    ("LLMSTYLER_VAULT_CA_FILE", "/unreviewed.pem"),
])
def test_invalid_native_session_never_becomes_direct_transport(native, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(agent.AgentModeError):
        agent.agent_environment()
    with pytest.raises(agent.AgentModeError):
        agent.open_local(agent.BASE_URL + "/models", timeout=1)


def test_ca_drift_fails_before_transport(native, monkeypatch):
    monkeypatch.setattr(agent, "CA_SHA256", "0" * 64)
    with pytest.raises(agent.AgentModeError):
        agent.native_proxy()


def test_native_proxy_is_used_by_the_existing_opener(native, monkeypatch):
    handlers = []

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == agent.BASE_URL + "/models" and timeout == 1
            return io.BytesIO(b'{"data":[]}')

    monkeypatch.setattr(agent.urllib.request, "build_opener", lambda *items: (handlers.extend(items), Opener())[1])
    with agent.open_local(agent.BASE_URL + "/models", timeout=1) as response:
        assert json.load(response) == {"data": []}
    assert handlers[0].proxies == {"http": native, "https": native}
    assert isinstance(handlers[1], agent._NoRedirects)


@pytest.mark.parametrize("body,ok", [(b'{"data":[]}', True), (b'{"private":"fake-private-value"}', False)])
def test_live_check_returns_only_sanitized_catalog_result(native, monkeypatch, capsys, body, ok):
    monkeypatch.setattr(agent, "open_local", lambda *_a, **_k: io.BytesIO(body))
    if ok:
        agent.launch(["vault-check"], root=Path("/unused"))
    else:
        with pytest.raises(agent.AgentModeError, match="details withheld"):
            agent.launch(["vault-check"], root=Path("/unused"))
    assert "fake-private-value" not in capsys.readouterr().out
