from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from llmstyler.io import read_jsonl, write_json, write_jsonl
from llmstyler.pipeline import compose_pipeline_config


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Studio export requires non-empty conversational messages")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant", "tool"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("Studio export requires text messages with valid roles")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("Studio export requires an assistant response in every row")
    return messages


def _key(row: dict[str, Any]) -> str:
    value = json.dumps(_messages(row), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def export_studio(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_rows: int | None = None,
) -> Path:
    pipeline = compose_pipeline_config(config_path, [])
    destination = Path(output_dir or f"{pipeline['output_dir']}/studio")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"{destination} is not empty; choose a new output directory")
    base = pipeline["base_mix"]["output"]
    base_train = Path(base["dir"]) / base.get("train_file", "train.jsonl")
    base_test = Path(base["dir"]) / base.get("test_file", "test.jsonl")
    original_train = read_jsonl(base_train)
    test_rows = read_jsonl(base_test)
    if not test_rows:
        raise ValueError("Studio export requires a held-out test split")
    if {_key(row) for row in original_train} & {_key(row) for row in test_rows}:
        raise ValueError("Train and held-out test contain overlapping conversations")

    restyle_output = pipeline["restyle"]["output"]
    train_path = Path(restyle_output["dir"]) / restyle_output.get("train_file", "train.jsonl")
    train_rows = read_jsonl(train_path)
    runstep = pipeline["runstep"]
    style_id = runstep["style"]["id"]
    if runstep["dataset"].get("restyled_only", False):
        train_rows = [row for row in train_rows if row.get("restyled_style") == style_id]
    if max_rows is not None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        train_rows = random.Random(int(pipeline.get("seed", 3407))).sample(
            train_rows, min(max_rows, len(train_rows))
        )
    if not train_rows:
        raise ValueError("Studio export has no training rows")
    train_messages = [{"messages": _messages(row)} for row in train_rows]
    test_messages = [{"messages": _messages(row)} for row in test_rows]
    style_system = runstep["style"].get("system_prompt", "")
    if style_system:
        for row in train_messages:
            messages = row["messages"]
            row["messages"] = [{"role": "system", "content": style_system}] + (
                messages[1:] if messages[0]["role"] == "system" else messages
            )
    write_jsonl(destination / "train.jsonl", train_messages)
    write_jsonl(destination / "heldout.jsonl", test_messages)
    write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "style": pipeline.get("style_contract"),
            "base_model": runstep["model"]["base_model"],
            "training_settings": runstep["training"],
            "training_system_prompt": style_system,
            "training_rows": len(train_rows),
            "styled_training_rows": sum(
                row.get("restyled_style") == style_id for row in train_rows
            ),
            "heldout_rows": len(test_rows),
            "training_source": str(train_path),
            "heldout_source": str(base_test),
            "files": {"training": "train.jsonl", "evaluation": "heldout.jsonl"},
            "evaluation_note": "Keep heldout.jsonl out of training and model selection. Compare voice and answer quality against the base model.",
        },
    )
    print(f"Studio training dataset: {destination / 'train.jsonl'}")
    print(f"Held-out evaluation dataset: {destination / 'heldout.jsonl'}")
    return destination
