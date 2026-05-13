from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmstyler.hub import preflight_upload, upload_folder
from llmstyler.io import read_yaml, write_json, write_jsonl
from llmstyler.standards import (
    artifact_version,
    base_mix_dataset_card,
    now_iso,
    standard_repo_id,
)


def valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    has_user = False
    has_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            return False
        role = message.get("role")
        content = message.get("content")
        if role == "user" and isinstance(content, str) and content.strip():
            has_user = True
        if role == "assistant" and isinstance(content, str) and content.strip():
            has_assistant = True
    return has_user and has_assistant


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": str(message["role"]), "content": str(message["content"]).strip()}
        for message in messages
    ]


def stream_rows(dataset_id: str, config: str | None, split: str, seed: int, buffer_size: int):
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, config, split=split, streaming=True)
    shuffled = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    for row in shuffled:
        messages = row.get("messages")
        if valid_messages(messages):
            yield row


def collect_mix(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = config["source"]
    dataset_id = source["dataset_id"]
    dataset_config = source.get("config")
    seed = int(config.get("seed", 3407))
    buffer_size = int(config.get("buffer_size", 10_000))
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for item_index, item in enumerate(config["plan"]):
        split = item["split"]
        target_count = int(item["count"])
        collected = 0
        iterator = stream_rows(dataset_id, dataset_config, split, seed + item_index, buffer_size)

        for row in iterator:
            messages = clean_messages(row["messages"])
            key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(
                {
                    "messages": messages,
                    "restyle": bool(item.get("restyle", False)),
                    "bucket": item["name"],
                    "source_dataset": dataset_id,
                    "source_config": dataset_config,
                    "source_split": split,
                    "source": row.get("source", split),
                }
            )
            collected += 1
            if collected >= target_count:
                break

        if collected != target_count:
            raise RuntimeError(
                f"Only collected {collected} of {target_count} rows for {item['name']} ({split})"
            )
    return rows


def counts_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_mix_artifacts(
    config: dict[str, Any], rows: list[dict[str, Any]], config_path: str | Path | None = None
) -> None:
    output = config["output"]
    version = artifact_version(config)
    hub_repo_id = standard_repo_id(
        output.get("hub", {}),
        fallback_name=config["id"],
        version=version,
        required=False,
    )
    output_dir = Path(output["dir"])
    train_path = output_dir / output.get("train_file", "train.jsonl")
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_path, rows)

    restyle_true = sum(1 for row in rows if row.get("restyle") is True)
    manifest = {
        "created_at": now_iso(),
        "artifact_type": "base_mix_dataset",
        "artifact_id": config["id"],
        "artifact_version": version,
        "hub_repo_id": hub_repo_id,
        "config_path": str(config_path) if config_path is not None else None,
        "name": config.get("name"),
        "source": config["source"],
        "seed": config.get("seed", 3407),
        "buffer_size": config.get("buffer_size", 10_000),
        "total_rows": len(rows),
        "preserve_rows": len(rows) - restyle_true,
        "restyle_rows": restyle_true,
        "buckets": counts_by_key(rows, "bucket"),
    }
    write_json(manifest_path, manifest)
    readme_path.write_text(
        base_mix_dataset_card(config=config, manifest=manifest, repo_id=hub_repo_id),
        encoding="utf-8",
    )


def build_mix(config_path: str | Path, *, push: bool | None = None) -> None:
    config = read_yaml(config_path)
    output = config["output"]
    should_push = output.get("push_to_hub", False) if push is None else push
    hub_repo_id = None
    if should_push:
        hub = output["hub"]
        hub_repo_id = standard_repo_id(
            hub, fallback_name=config["id"], version=artifact_version(config)
        )
        preflight_upload(
            repo_id=hub_repo_id,
            repo_type="dataset",
            private=bool(hub.get("private", False)),
        )

    rows = collect_mix(config)
    write_mix_artifacts(config, rows, config_path=config_path)
    if should_push:
        hub = output["hub"]
        upload_folder(
            repo_id=hub_repo_id or standard_repo_id(
                hub, fallback_name=config["id"], version=artifact_version(config)
            ),
            repo_type="dataset",
            folder_path=output["dir"],
            private=bool(hub.get("private", False)),
            commit_message=hub.get("commit_message", "Upload llmstyler base mix dataset"),
        )
