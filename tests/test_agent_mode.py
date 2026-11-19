from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer
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
SYNTHETIC_PROXY = "http://x-agent-vault:agv_synthetic@127.0.0.1:8283"


@pytest.fixture(autouse=True)
def operator_mode(monkeypatch):
    monkeypatch.delenv(agent.MODE_VARIABLE, raising=False)
    monkeypatch.delenv("RUNBOX_POLICY", raising=False)
    monkeypatch.delenv("INFISICAL_AGENT_VAULT_URL", raising=False)
    monkeypatch.delenv("AGENTBRIDGE_API_KEY", raising=False)
    monkeypatch.delenv("AGENTBRIDGE_BASE_URL", raising=False)


@pytest.fixture
def agent_mode(monkeypatch):
    monkeypatch.setenv(agent.MODE_VARIABLE, agent.MODE)


@pytest.fixture
def reviewed_proxy(monkeypatch):
    monkeypatch.setattr(agent, "native_proxy", lambda: SYNTHETIC_PROXY)
    monkeypatch.setenv("LLMSTYLER_VAULT_CA_FILE", "/fixture/ca.pem")


def test_run_cli_uses_the_reviewed_environment_and_preserves_arguments(
    agent_mode, reviewed_proxy, monkeypatch
):
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
    arguments = ["llmstyler", "preview", "--style", "voice.yaml"]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(agent, "os", SimpleNamespace(environ=dict(agent.os.environ)))
    calls = []
    monkeypatch.setattr("llmstyler.cli.main", lambda: calls.append((sys.argv, agent.os.environ)))
    agent.run_cli()
    received_arguments, environment = calls[0]
    assert received_arguments == arguments
    assert environment["HOME"] == "/synthetic/home"
    assert environment["CODEX_HOME"] == "/synthetic/codex"
    assert environment["AGENTBRIDGE_BASE_URL"] == agent.BASE_URL
    assert environment[agent.MODE_VARIABLE] == agent.MODE
    assert environment["HTTPS_PROXY"] == SYNTHETIC_PROXY
    assert environment[agent.VAULT_VARIABLE] == agent.VAULT_MARKER
    assert "synthetic-private-value" not in environment.values()
    assert "ALL_PROXY" not in environment
    assert "INFISICAL_TOKEN" not in environment


def test_agent_requests_refuse_missing_proxy_before_opening_network(agent_mode, monkeypatch):
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: pytest.fail("Network fallback"))
    monkeypatch.delenv(agent.VAULT_VARIABLE, raising=False)
    with pytest.raises(agent.AgentModeError, match="reviewed native Vault"):
        agent.open_local(agent.BASE_URL + "/models", timeout=1)
    with pytest.raises(agent.AgentModeError, match="reviewed native Vault"):
        agent.run_cli()


def test_unrelated_runbox_context_cannot_load_operator_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNBOX_POLICY", "other-profile")
    env_file = tmp_path / ".env"
    env_file.write_text("AGENTBRIDGE_API_KEY=synthetic-private-value\n")
    load_env_file(env_file)
    assert provider_api_key({}) is None
    with pytest.raises(agent.AgentModeError, match="reviewed LLMStyler"):
        agent.run_cli()


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


def test_agent_http_child_only_receives_the_reviewed_proxy(agent_mode, reviewed_proxy, monkeypatch):
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
    assert command == [sys.executable, "-I", "-S", "-B", str(Path(agent.__file__))]
    assert "synthetic-private-value" not in options["input"]
    assert "synthetic-private-value" not in options["env"].values()
    assert options["env"]["HTTPS_PROXY"] == SYNTHETIC_PROXY
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


def test_transport_rejects_redirects_and_auth_headers(reviewed_proxy, monkeypatch):
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


def test_isolated_http_child_sends_request_only_through_proxy(tmp_path):
    received = []

    class Proxy(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, self.headers, json.loads(payload)))
            response = b'{"choices":[]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Proxy) as proxy:
        port = proxy.server_port
        ca = tmp_path / ".config/infisical/agent-vault-ca.pem"
        ca.parent.mkdir(parents=True)
        ca.write_bytes(b"synthetic public CA")
        host = f"127.0.0.1:{port}"
        base_url = f"http://{host}/api/v1"
        source = Path(agent.__file__).read_text()
        child = tmp_path / "agent.py"
        child.write_text(
            source.replace(agent.BASE_URL, base_url)
            .replace(f"PROXY_PORT = {agent.PROXY_PORT}", f"PROXY_PORT = {port}")
            .replace(agent.CA_SHA256, hashlib.sha256(ca.read_bytes()).hexdigest())
        )
        proxy_url = f"http://x-agent-vault:agv_synthetic@{host}"
        environment = {
            "HOME": str(tmp_path),
            agent.VAULT_VARIABLE: agent.VAULT_MARKER,
            "INFISICAL_AGENT_VAULT_URL": "https://app.infisical.com",
            "LLMSTYLER_VAULT_CA_FILE": str(ca),
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
        }
        url = base_url + "/chat/completions"
        payload = {"messages": [{"role": "user", "content": "fixture"}]}
        worker = threading.Thread(target=proxy.serve_forever, kwargs={"poll_interval": 0.01})
        worker.start()
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", str(child)],
                input=json.dumps({"url": url, "payload": payload, "headers": {}, "timeout": 2}),
                text=True,
                capture_output=True,
                check=True,
                timeout=5,
                env=environment,
            )
        finally:
            proxy.shutdown()
            worker.join()

    assert json.loads(result.stdout) == {"choices": []}
    [(request_url, headers, request_payload)] = received
    assert request_url == url
    assert headers.get("Proxy-Authorization", "").startswith("Basic ")
    assert headers.get("Authorization") is None
    assert request_payload == payload


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
        env={agent.MODE_VARIABLE: agent.MODE},
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "disabled" in result.stderr
