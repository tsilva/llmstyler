from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from llmstyler.ease import create_pipeline_config
from llmstyler.pipeline import compose_pipeline_config
from llmstyler.restyle import (
    RestyleResult,
    RestyleTarget,
    chat_completion,
    compatible_checkpoint_rewrites,
    rewrite_style_hash,
)
from llmstyler.studio import export_studio
from llmstyler.styles import chat_messages, load_style_pack, rewrite_prompt

ROOT = Path(__file__).resolve().parents[1]


def test_style_pack_is_shared_by_chat_and_pipeline(tmp_path):
    pack_path = tmp_path / "voice.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "id": "custom",
                "voice": ["Use short sentences."],
                "examples": [{"user": "Hello", "assistant": "Hi."}],
            }
        )
    )
    config_path = create_pipeline_config(
        pipeline_id="sample",
        owner="tester",
        style_pack=pack_path,
        output_path=tmp_path / "configs" / "sample.yaml",
    )
    pipeline = compose_pipeline_config(config_path, [])
    pack = load_style_pack(pack_path)
    assert pipeline["restyle"]["style"]["rewrite_prompt"] == rewrite_prompt(pack)
    assert chat_messages(pack)[-1] == {"role": "assistant", "content": "Hi."}
    assert pipeline["runstep"]["dataset"]["restyled_only"] is False
    pack_path.write_text(
        pack_path.read_text().replace("Use short sentences.", "Use detailed examples.")
    )
    changed = compose_pipeline_config(config_path, [])
    assert (
        changed["restyle"]["style"]["rewrite_prompt"]
        != pipeline["restyle"]["style"]["rewrite_prompt"]
    )


def test_style_edit_invalidates_rewrite_checkpoint():
    first = {"id": "voice", "model": "model", "rewrite_prompt": "short"}
    edited = {**first, "rewrite_prompt": "long"}
    target = RestyleTarget(0, 1, "original", 10, 8)
    result = RestyleResult(0, 1, "original", "styled", {}, rewrite_style_hash(first))
    checkpoint = {(0, 1): result}
    assert rewrite_style_hash(first) == rewrite_style_hash({**first, "workers": 12, "retries": 5})
    assert compatible_checkpoint_rewrites(checkpoint, [target], rewrite_style_hash(first)) == (
        checkpoint,
        0,
    )
    assert compatible_checkpoint_rewrites(checkpoint, [target], rewrite_style_hash(edited)) == (
        {},
        1,
    )
    legacy = RestyleResult(0, 1, "original", "old style", {})
    assert compatible_checkpoint_rewrites(
        {(0, 1): legacy}, [target], rewrite_style_hash(first)
    ) == ({}, 1)


def test_chat_transport_does_not_request_json(monkeypatch):
    payloads = []

    def fake_run(*args, **kwargs):
        payloads.append(json.loads(kwargs["input"])["payload"])
        return SimpleNamespace(stdout=json.dumps({"choices": [{"message": {"content": "Hello"}}]}))

    monkeypatch.setattr("llmstyler.restyle.subprocess.run", fake_run)
    messages = [{"role": "user", "content": "Hello"}]
    assert chat_completion({"model": "test"}, messages, json_mode=False)[0] == "Hello"
    assert "response_format" not in payloads[0]
    chat_completion({"model": "test"}, messages)
    assert payloads[1]["response_format"]["type"] == "json_schema"


@pytest.mark.parametrize("name", ["warm-direct", "patient-teacher"])
def test_shipped_style_packs_compile(name):
    pack = load_style_pack(ROOT / "styles" / f"{name}.yaml")
    assert len(chat_messages(pack)) == 7
    assert "restyled_content" in rewrite_prompt(pack)


def test_studio_export_keeps_heldout_separate_and_rejects_leakage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = create_pipeline_config(
        pipeline_id="sample",
        owner="tester",
        style_pack=ROOT / "styles/warm-direct.yaml",
        output_path=tmp_path / "sample.yaml",
    )
    pipeline = compose_pipeline_config(config_path, [])
    base_dir = Path(pipeline["base_mix"]["output"]["dir"])
    styled_dir = Path(pipeline["restyle"]["output"]["dir"])
    base_dir.mkdir(parents=True)
    styled_dir.mkdir(parents=True)
    train = {
        "messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
    }
    test = {
        "messages": [{"role": "user", "content": "3+3?"}, {"role": "assistant", "content": "6"}]
    }
    (base_dir / "train.jsonl").write_text(json.dumps(train) + "\n")
    (base_dir / "test.jsonl").write_text(json.dumps(test) + "\n")
    (styled_dir / "train.jsonl").write_text(
        json.dumps({**train, "restyled_style": "warm-direct"}) + "\n"
    )
    destination = export_studio(config_path)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["training_rows"] == manifest["styled_training_rows"] == 1
    assert json.loads((destination / "heldout.jsonl").read_text()) == test
    assert json.loads((destination / "train.jsonl").read_text()) == train
    (base_dir / "test.jsonl").write_text(json.dumps(train) + "\n")
    with pytest.raises(ValueError, match="overlapping"):
        export_studio(config_path, output_dir=tmp_path / "leaky")
