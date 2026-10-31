import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml

from llmstyler.cli import main
from llmstyler.ease import _check_llm, create_pipeline_config
from llmstyler.llm import DEFAULT_MODEL, resolve_llm_settings
from llmstyler.pipeline import compose_pipeline_config
from llmstyler.restyle import chat_completion, restyle, rewrite_style_hash


@pytest.fixture
def bridge():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/api/v1/models"
            self.respond({"data": [{"id": DEFAULT_MODEL}]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), payload))
            if payload["messages"] == [{"role": "user", "content": "slow"}]:
                time.sleep(0.1)
            self.respond(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"restyled_content": "The final price is 68 euros."}
                                )
                            }
                        }
                    ],
                    "usage": {"total_tokens": 12},
                }
            )

        def respond(self, body):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                self.wfile.write(json.dumps(body).encode())
            except BrokenPipeError:
                pass

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_default_rewrite_round_trip_without_openrouter_key(tmp_path, monkeypatch, bridge):
    base_url, requests = bridge
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "restyle": True,
                "messages": [
                    {"role": "user", "content": "How much?"},
                    {"role": "assistant", "content": "The final price is 68 euros."},
                ],
            }
        )
        + "\n"
    )
    config = {
        "dataset": {"input_path": str(source)},
        "style": {"id": "voice", "rewrite_prompt": "Keep the price.", "base_url": base_url},
        "output": {"dir": str(tmp_path / "styled"), "push_to_hub": False},
    }
    config_path = tmp_path / "restyle.yaml"
    config_path.write_text(yaml.safe_dump(config))
    restyle(config_path)
    path, headers, payload = requests[0]
    assert path == "/api/v1/chat/completions"
    assert "Authorization" not in headers
    assert payload["model"] == DEFAULT_MODEL
    assert payload["reasoning_effort"] == "high"
    schema = payload["response_format"]
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["strict"] is True
    assert schema["json_schema"]["schema"]["required"] == ["restyled_content"]
    manifest = json.loads((tmp_path / "styled/manifest.json").read_text())
    assert manifest["style"]["provider"] == "agentbridge"
    assert manifest["style"]["model"] == DEFAULT_MODEL
    assert manifest["estimated_cost"]["estimated_total_cost_usd"] is None
    # Compatible checkpoints avoid a duplicate request.
    restyle(config_path)
    assert len(requests) == 1
    config["style"]["reasoning_effort"] = "medium"
    config_path.write_text(yaml.safe_dump(config))
    restyle(config_path)
    assert len(requests) == 2
    assert requests[1][2]["reasoning_effort"] == "medium"


def test_agentbridge_never_sends_openrouter_key(monkeypatch, bridge):
    base_url, requests = bridge
    monkeypatch.setenv("OPENROUTER_API_KEY", "do-not-send-this")
    chat_completion({"base_url": base_url}, [{"role": "user", "content": "Hello"}], json_mode=False)
    assert "Authorization" not in requests[0][1]
    assert "response_format" not in requests[0][2]
    assert _check_llm({"base_url": base_url}).ok


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "codex/gpt-6-astra"),
        ("reasoning_effort", "medium"),
        ("base_url", "http://127.0.0.1:9000/api/v1"),
    ],
)
def test_inference_changes_invalidate_checkpoint(field, value):
    style = {"id": "voice", "rewrite_prompt": "short"}
    edited = {**style, field: value}
    assert rewrite_style_hash(style) != rewrite_style_hash(edited)
    assert rewrite_style_hash(style) == rewrite_style_hash(resolve_llm_settings(style))


def test_new_pipeline_resolves_requested_defaults(tmp_path):
    path = create_pipeline_config(
        pipeline_id="sample", owner="tester", output_path=tmp_path / "sample.yaml"
    )
    style = compose_pipeline_config(path, [])["restyle"]["style"]
    assert style["provider"] == "agentbridge"
    assert style["model"] == DEFAULT_MODEL
    assert style["reasoning_effort"] == "high"
    assert style["workers"] == 1
    assert style["request_timeout"] == 630.0
    assert "pricing_usd_per_1m" not in style


@pytest.mark.parametrize("command", ["preview", "chat"])
def test_voice_cli_defaults_without_model_or_credentials(command, tmp_path, monkeypatch):
    pack = tmp_path / "voice.yaml"
    pack.write_text(yaml.safe_dump({"id": "voice", "voice": ["Be concise."]}))
    captured = []

    def complete(settings, messages, **kwargs):
        captured.append((resolve_llm_settings(settings), kwargs))
        return "Hello", {}

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("llmstyler.styles.chat_completion", complete)
    monkeypatch.setattr("sys.argv", ["llmstyler", command, "--style", str(pack)])
    prompts = iter(["Hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))
    main()
    assert captured[0][0]["provider"] == "agentbridge"
    assert captured[0][0]["model"] == DEFAULT_MODEL
    assert captured[0][0]["reasoning_effort"] == "high"
    assert captured[0][1]["json_mode"] is False


def test_openrouter_requires_infisical_key_and_known_endpoint(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="only AgentBridge"):
        chat_completion({"provider": "openrouter", "model": "x-ai/grok-4.3"}, [])
    with pytest.raises(ValueError, match="AgentBridge HTTP endpoint"):
        resolve_llm_settings({"model": "model", "base_url": "https://openrouter.ai/api/v1"})


def test_bridge_timeout_reports_cause_without_fallback(bridge):
    base_url, requests = bridge
    with pytest.raises(
        RuntimeError, match="agentbridge request failed: Connection failed: timed out"
    ) as error:
        chat_completion(
            {"base_url": base_url, "request_timeout": 0.01, "retries": 0},
            [{"role": "user", "content": "slow"}],
        )
    assert "Traceback" not in str(error.value)
    assert len(requests) == 1
