from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.response import addinfourl

import pytest

from llmstyler import agent
from llmstyler.ease import _check_hugging_face_auth, _check_llm
from llmstyler.hub import _api
from llmstyler.io import load_env_file
from llmstyler.llm import provider_api_key, resolve_llm_settings
from llmstyler.mix import build_mix, datamixxer_artifact_dir
from llmstyler.restyle import chat_completion, load_source_rows, rewrite_style_hash
from llmstyler.runbook import run_generated_runbook

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def operator_mode(monkeypatch):
    monkeypatch.delenv(agent.MODE_VARIABLE, raising=False)
    monkeypatch.delenv("AGENTBRIDGE_API_KEY", raising=False)
    monkeypatch.delenv("AGENTBRIDGE_BASE_URL", raising=False)


@pytest.fixture
def agent_mode(monkeypatch):
    monkeypatch.setenv(agent.MODE_VARIABLE, agent.MODE)


@pytest.mark.parametrize("script", ["agent.py", "infisical.py"])
def test_launch_is_keyless_and_never_calls_a_secret_manager(script, monkeypatch):
    spec = importlib.util.spec_from_file_location("agent_launcher", ROOT / "scripts" / script)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    for name in (
        "AGENTBRIDGE_API_KEY",
        "OPENROUTER_API_KEY",
        "HF_TOKEN",
        "WANDB_API_KEY",
        "MODAL_TOKEN_SECRET",
        "INFISICAL_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "PYTHONPATH",
        "PYTHONSTARTUP",
    ):
        monkeypatch.setenv(name, "synthetic-private-value")
    monkeypatch.setenv("AGENTBRIDGE_BASE_URL", "https://unreviewed.invalid/private")
    monkeypatch.setenv("HOME", "/synthetic/home")
    monkeypatch.setenv("CODEX_HOME", "/synthetic/codex")
    calls = []
    monkeypatch.setattr(agent.os, "chdir", lambda path: None)
    monkeypatch.setattr(agent.os, "execve", lambda *args: calls.append(args))
    launcher.main(["run", "preview", "--style", "voice.yaml"])
    executable, arguments, environment = calls[0]
    assert executable == sys.executable
    assert arguments == [sys.executable, "-m", "llmstyler.cli", "preview", "--style", "voice.yaml"]
    assert environment["HOME"] == "/synthetic/home"
    assert environment["CODEX_HOME"] == "/synthetic/codex"
    assert environment["AGENTBRIDGE_BASE_URL"] == agent.BASE_URL
    assert environment[agent.MODE_VARIABLE] == agent.MODE
    assert set(environment) <= agent.ENVIRONMENT_KEYS | {
        agent.MODE_VARIABLE,
        "AGENTBRIDGE_BASE_URL",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    }
    assert "synthetic-private-value" not in environment.values()
    with pytest.raises(agent.AgentModeError):
        launcher.main(["add"])
    assert not hasattr(launcher, "read_key")


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8082/api/v1",
        "http://localhost:8082/api/v1",
        "http://127.0.0.1:8083/api/v1",
        "http://127.0.0.1.evil.invalid:8082/api/v1",
        "http://synthetic-secret@127.0.0.1:8082/api/v1",
        agent.BASE_URL + "?token=synthetic-secret",
        agent.BASE_URL + "#synthetic-secret",
        agent.BASE_URL + "/../other",
    ],
)
def test_agent_endpoint_rejections_do_not_echo_configuration(agent_mode, url):
    with pytest.raises(ValueError) as error:
        resolve_llm_settings({"base_url": url})
    assert "synthetic-secret" not in str(error.value)


def test_marker_is_not_serialized_or_added_to_cache_fingerprints(monkeypatch):
    style = {"id": "voice", "rewrite_prompt": "Keep facts.", "reasoning_effort": "high"}
    ordinary = resolve_llm_settings(style)
    fingerprint = rewrite_style_hash(style)
    monkeypatch.setenv(agent.MODE_VARIABLE, agent.MODE)
    assert resolve_llm_settings(style) == ordinary
    assert rewrite_style_hash(style) == fingerprint
    assert agent.MODE_VARIABLE not in json.dumps(resolve_llm_settings(style))


def test_agent_mode_rejects_explicit_and_restored_keys(agent_mode, monkeypatch):
    with pytest.raises(agent.AgentModeError, match="explicit"):
        chat_completion({}, [], api_key="synthetic-private-value")
    monkeypatch.setenv("AGENTBRIDGE_API_KEY", "synthetic-private-value")
    with pytest.raises(agent.AgentModeError, match="API keys"):
        provider_api_key({})
    monkeypatch.setenv(agent.MODE_VARIABLE, "unknown")
    with pytest.raises(agent.AgentModeError, match="Invalid"):
        provider_api_key({})


def test_agent_http_child_has_no_keys_or_proxy_environment(agent_mode, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-private-value")
    monkeypatch.setenv("HTTPS_PROXY", "http://synthetic-private-value@proxy.invalid")
    captured = []

    def complete(command, **options):
        captured.append((command, options))
        return SimpleNamespace(
            stdout=json.dumps(
                {"choices": [{"message": {"content": "ordinary answer"}}], "usage": {}}
            )
        )

    monkeypatch.setattr("llmstyler.restyle.subprocess.run", complete)
    assert (
        chat_completion({}, [{"role": "user", "content": "ordinary"}], json_mode=False)[0]
        == "ordinary answer"
    )
    command, options = captured[0]
    assert command == [sys.executable, "-m", "llmstyler.agent"]
    assert "synthetic-private-value" not in options["input"]
    assert "synthetic-private-value" not in options["env"].values()
    request = json.loads(options["input"])
    assert request["url"] == agent.BASE_URL + "/chat/completions"
    assert request["headers"] == {"Content-Type": "application/json"}


def test_operator_authentication_and_custom_endpoints_are_unchanged(monkeypatch):
    monkeypatch.setenv("AGENTBRIDGE_API_KEY", "operator-key")
    assert provider_api_key({"base_url": "https://operator.invalid/api"}) == "operator-key"
    captured = []

    def complete(command, **options):
        captured.append((command, options))
        return SimpleNamespace(
            stdout=json.dumps({"choices": [{"message": {"content": "operator answer"}}]})
        )

    monkeypatch.setattr("llmstyler.restyle.subprocess.run", complete)
    chat_completion({"base_url": "https://operator.invalid/api"}, [], api_key="explicit-operator")
    assert captured[0][0][1] == "-c"
    assert "env" not in captured[0][1]
    assert (
        json.loads(captured[0][1]["input"])["headers"]["Authorization"]
        == "Bearer explicit-operator"
    )


def test_doctor_uses_the_same_keyless_models_transport(agent_mode, monkeypatch):
    calls = []

    def open_models(url, **options):
        calls.append((url, options))
        return io.StringIO('{"data":[]}')

    monkeypatch.setattr("llmstyler.ease.open_local", open_models)
    assert _check_llm({}).ok
    assert calls == [(agent.BASE_URL + "/models", {"timeout": 5})]


def test_transport_disables_proxies_redirects_and_auth_headers(monkeypatch):
    original_handler = urllib.request.HTTPHandler
    seen = []

    class FakeHttp(original_handler):
        def http_open(self, request):
            seen.append(request.full_url)
            headers = Message()
            headers["Location"] = "https://unreviewed.invalid/synthetic-secret"
            response = addinfourl(io.BytesIO(b""), headers, request.full_url, code=302)
            response.msg = "Found"
            return response

    monkeypatch.setattr(urllib.request, "HTTPHandler", FakeHttp)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid")
    with pytest.raises(urllib.error.HTTPError, match="Redirects are disabled"):
        agent.open_local(agent.BASE_URL + "/models", timeout=1)
    assert seen == [agent.BASE_URL + "/models"]
    request = urllib.request.Request(
        agent.BASE_URL + "/chat/completions",
        data=b"{}",
        headers={"aUtHoRiZaTiOn": "synthetic-private-value"},
    )
    with pytest.raises(agent.AgentModeError, match="authentication"):
        agent.open_local(request, timeout=1)


def test_http_child_rechecks_payload_boundary_without_network(monkeypatch, capsys):
    monkeypatch.setattr(
        agent.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "url": "https://unreviewed.invalid/synthetic-secret",
                    "payload": {},
                    "headers": {},
                    "timeout": 1,
                }
            )
        ),
    )
    monkeypatch.setattr(
        agent, "open_local", lambda *a, **kw: (_ for _ in ()).throw(agent.AgentModeError("fixed"))
    )
    assert agent.http_child() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "synthetic-secret" not in captured.err
    assert captured.err == "AgentBridge request failed in keyless agent mode."


def test_agent_mode_does_not_load_dotenv_or_touch_unreviewed_services(agent_mode, tmp_path):
    path = tmp_path / ".env"
    path.write_text("AGENTBRIDGE_API_KEY=synthetic-private-value\n")
    load_env_file(path)
    assert provider_api_key({}) is None
    for operation in (
        _api,
        lambda: build_mix("unused.yaml"),
        lambda: datamixxer_artifact_dir("unused.yaml"),
        lambda: run_generated_runbook({}, Path("unused.ipynb"), Path("unused")),
        lambda: load_source_rows({"hub_repo_id": "unreviewed/private"}),
    ):
        with pytest.raises(agent.AgentModeError, match="no reviewed agent credential route"):
            operation()
    assert not _check_hugging_face_auth().ok
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"messages":[]}\n')
    assert load_source_rows({"input_path": str(rows)}) == [{"messages": []}]


@pytest.mark.parametrize("name", ["studio_smoke.py", "review_smoke.py"])
def test_retired_smoke_scripts_fail_before_credentials_or_services(name):
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / name)],
        capture_output=True,
        text=True,
        timeout=5,
        env=agent.agent_environment(),
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "disabled" in result.stderr
