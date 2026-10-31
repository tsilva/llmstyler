from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from llmstyler.agent import (
    AgentModeError,
    agent_environment,
    agent_mode,
    reject_agent_operation,
)
from llmstyler.hub import preflight_upload, upload_folder
from llmstyler.io import (
    append_jsonl,
    deep_copy_json,
    load_env_file,
    read_jsonl,
    read_yaml,
    write_json,
    write_jsonl,
)
from llmstyler.llm import provider_api_key, resolve_llm_settings
from llmstyler.standards import (
    artifact_version,
    now_iso,
    standard_repo_id,
    styled_dataset_card,
)

TOKEN_CHARS = 4

AGENTBRIDGE_CHILD_CODE = r"""
import json
import sys
import urllib.error
import urllib.request

request = json.loads(sys.stdin.read())
http_request = urllib.request.Request(
    request["url"],
    data=json.dumps(request["payload"]).encode("utf-8"),
    headers=request["headers"],
    method="POST",
)
try:
    with urllib.request.urlopen(http_request, timeout=request["timeout"]) as response:
        sys.stdout.write(response.read().decode("utf-8"))
except urllib.error.HTTPError as error:
    message = str(error.reason)
    try:
        body = json.loads(error.read().decode("utf-8"))
        message = str(body.get("error", {}).get("message", message))
    except (ValueError, AttributeError, UnicodeDecodeError):
        pass
    authorization = request["headers"].get("Authorization", "")
    secret = authorization.removeprefix("Bearer ")
    if secret:
        message = message.replace(secret, "[redacted]")
    sys.stderr.write(f"HTTP {error.code}: {message[:500]}")
    sys.exit(1)
except (urllib.error.URLError, TimeoutError) as error:
    message = str(getattr(error, "reason", error))
    secret = request["headers"].get("Authorization", "").removeprefix("Bearer ")
    if secret:
        message = message.replace(secret, "[redacted]")
    sys.stderr.write(f"Connection failed: {message[:500]}")
    sys.exit(1)
"""


@dataclasses.dataclass(frozen=True)
class RestyleTarget:
    row_index: int
    assistant_index: int
    original_content: str
    input_chars: int
    original_output_chars: int


@dataclasses.dataclass(frozen=True)
class RestyleResult:
    row_index: int
    assistant_index: int
    original: str
    restyled: str
    usage: dict[str, Any]
    style_hash: str | None = None


def assistant_indexes(row: dict[str, Any]) -> list[int]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("row messages must be a list")
    indexes = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        if message.get("role") == "assistant":
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("assistant message content must be a string")
            indexes.append(index)
    return indexes


def latest_user_message(messages: list[dict[str, Any]], assistant_index: int) -> str | None:
    for message in reversed(messages[:assistant_index]):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return None


def context_before(messages: list[dict[str, Any]], assistant_index: int) -> list[dict[str, str]]:
    context = []
    for message in messages[:assistant_index]:
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user", "assistant"} and isinstance(content, str):
            context.append({"role": role, "content": content})
    return context


def selected_row_indexes(rows: list[dict[str, Any]], sample: int | None, seed: int) -> set[int]:
    candidates = [index for index, row in enumerate(rows) if row.get("restyle") is True]
    if sample is None:
        return set(candidates)
    if sample > len(candidates):
        raise ValueError(
            f"requested {sample} samples but only found {len(candidates)} restyle rows"
        )
    return set(random.Random(seed).sample(candidates, sample))


def build_user_prompt(
    messages: list[dict[str, Any]],
    assistant_index: int,
    original_content: str,
    context_mode: str,
) -> str:
    if context_mode == "full-history":
        payload = {
            "conversation_before_response": context_before(messages, assistant_index),
            "original_assistant_response": original_content,
        }
    elif context_mode == "latest-user":
        payload = {
            "latest_user_message": latest_user_message(messages, assistant_index),
            "original_assistant_response": original_content,
        }
    else:
        raise ValueError(f"unknown context mode: {context_mode}")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_openrouter_messages(
    *,
    style: dict[str, Any],
    messages: list[dict[str, Any]],
    assistant_index: int,
    original_content: str,
    context_mode: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": style["rewrite_prompt"].strip()},
        {
            "role": "user",
            "content": build_user_prompt(messages, assistant_index, original_content, context_mode),
        },
    ]


def iter_targets(
    rows: list[dict[str, Any]], row_indexes: set[int], style: dict[str, Any]
) -> list[RestyleTarget]:
    context_mode = style.get("context_mode", "latest-user")
    targets: list[RestyleTarget] = []
    for row_index in sorted(row_indexes):
        row = rows[row_index]
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"row {row_index}: messages must be a list")
        for assistant_index in assistant_indexes(row):
            original = messages[assistant_index]["content"]
            request_messages = build_openrouter_messages(
                style=style,
                messages=messages,
                assistant_index=assistant_index,
                original_content=original,
                context_mode=context_mode,
            )
            targets.append(
                RestyleTarget(
                    row_index=row_index,
                    assistant_index=assistant_index,
                    original_content=original,
                    input_chars=sum(len(message["content"]) for message in request_messages),
                    original_output_chars=len(original),
                )
            )
    return targets


def estimate_cost(targets: list[RestyleTarget], style: dict[str, Any]) -> dict[str, Any]:
    pricing = style.get("pricing_usd_per_1m", {})
    input_rate = float(pricing["input"]) if "input" in pricing else None
    output_rate = float(pricing["output"]) if "output" in pricing else None
    output_expansion = float(style.get("output_expansion", 1.25))
    input_tokens = math.ceil(sum(target.input_chars for target in targets) / TOKEN_CHARS)
    output_tokens = math.ceil(
        sum(target.original_output_chars for target in targets) * output_expansion / TOKEN_CHARS
    )
    return {
        "assistant_calls": len(targets),
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "input_usd_per_1m": input_rate,
        "output_usd_per_1m": output_rate,
        "estimated_total_cost_usd": (
            input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate
            if input_rate is not None and output_rate is not None
            else None
        ),
        "token_estimate_method": f"ceil(characters / {TOKEN_CHARS})",
    }


def chat_completion(
    style: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    json_mode: bool = True,
):
    style = resolve_llm_settings(style)
    agent_request = agent_mode()
    if agent_request and api_key:
        raise AgentModeError("Agent mode does not accept explicit application API keys.")
    if api_key is None:
        api_key = provider_api_key(style)
    payload = {
        "model": style["model"],
        "messages": messages,
        "temperature": float(style.get("temperature", 0.7)),
        "max_tokens": int(style.get("max_tokens", 4096)),
    }
    if style.get("reasoning_effort") is not None:
        payload["reasoning_effort"] = style["reasoning_effort"]
    if json_mode:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "style_rewrite",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"restyled_content": {"type": "string"}},
                    "required": ["restyled_content"],
                    "additionalProperties": False,
                },
            },
        }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    retries = int(style.get("retries", 3))
    retry_sleep = float(style.get("retry_sleep", 2.0))
    timeout = float(style.get("request_timeout", 90.0))
    url = style["base_url"] + "/chat/completions"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            completed = subprocess.run(
                ([sys.executable, "-m", "llmstyler.agent"]
                 if agent_request else [sys.executable, "-c", AGENTBRIDGE_CHILD_CODE]),
                input=json.dumps(
                    {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
                ),
                text=True,
                capture_output=True,
                timeout=timeout + 5,
                check=True,
                **({"env": agent_environment()} if agent_request else {}),
            )
            parsed = json.loads(completed.stdout)
            return parsed["choices"][0]["message"]["content"], parsed.get("usage", {})
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                detail = stderr or stdout or str(exc)
                last_error = RuntimeError(detail[:1000])
            else:
                last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (2**attempt))
    raise RuntimeError(f"agentbridge request failed: {last_error}") from last_error


def parse_restyle_response(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return valid JSON: {content[:500]}") from exc
    restyled = parsed.get("restyled_content")
    if not isinstance(restyled, str) or not restyled.strip():
        raise ValueError(f"model JSON missing non-empty restyled_content: {content[:500]}")
    return restyled.strip()


def result_to_json(result: RestyleResult) -> dict[str, Any]:
    return {
        "row_index": result.row_index,
        "assistant_index": result.assistant_index,
        "original": result.original,
        "restyled": result.restyled,
        "usage": result.usage,
        "style_hash": result.style_hash,
    }


def result_from_json(row: dict[str, Any]) -> RestyleResult:
    return RestyleResult(
        row_index=int(row["row_index"]),
        assistant_index=int(row["assistant_index"]),
        original=str(row["original"]),
        restyled=str(row["restyled"]),
        usage=row.get("usage", {}),
        style_hash=row.get("style_hash"),
    )


def load_checkpoint(path: Path) -> dict[tuple[int, int], RestyleResult]:
    if not path.exists():
        return {}
    return {
        (result.row_index, result.assistant_index): result
        for result in (result_from_json(row) for row in read_jsonl(path))
    }


def compatible_checkpoint_rewrites(
    checkpoint: dict[tuple[int, int], RestyleResult],
    targets: list[RestyleTarget],
    style_hash: str,
) -> tuple[dict[tuple[int, int], RestyleResult], int]:
    target_by_key = {(target.row_index, target.assistant_index): target for target in targets}
    rewrites: dict[tuple[int, int], RestyleResult] = {}
    ignored = 0
    for key, result in checkpoint.items():
        target = target_by_key.get(key)
        if (
            target is None
            or result.original != target.original_content
            or result.style_hash != style_hash
        ):
            ignored += 1
            continue
        rewrites[key] = result
    return rewrites, ignored


def rewrite_style_hash(style: dict[str, Any]) -> str:
    style = resolve_llm_settings(style)
    generation = {
        "id": style["id"],
        "model": style["model"],
        "provider": style["provider"],
        "base_url": style["base_url"],
        "reasoning_effort": style.get("reasoning_effort"),
        "rewrite_prompt": style["rewrite_prompt"],
        "context_mode": style.get("context_mode", "latest-user"),
        "temperature": float(style.get("temperature", 0.7)),
        "max_tokens": int(style.get("max_tokens", 4096)),
    }
    payload = json.dumps(generation, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def restyle_one(
    api_key: str, style: dict[str, Any], row: dict[str, Any], target: RestyleTarget
) -> RestyleResult:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"row {target.row_index}: messages must be a list")
    response, usage = chat_completion(
        style,
        build_openrouter_messages(
            style=style,
            messages=messages,
            assistant_index=target.assistant_index,
            original_content=target.original_content,
            context_mode=style.get("context_mode", "latest-user"),
        ),
        api_key=api_key,
    )
    return RestyleResult(
        row_index=target.row_index,
        assistant_index=target.assistant_index,
        original=target.original_content,
        restyled=parse_restyle_response(response),
        usage=usage,
        style_hash=rewrite_style_hash(style),
    )


def apply_rewrites(
    rows: list[dict[str, Any]],
    row_indexes: set[int],
    rewrites: dict[tuple[int, int], RestyleResult],
    style: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    output_rows: list[dict[str, Any]] = []
    previews_by_row: dict[int, dict[str, Any]] = {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row_index, row in enumerate(rows):
        if row_index not in row_indexes:
            output_rows.append(row)
            continue
        new_row = deep_copy_json(row)
        messages = new_row["messages"]
        preview = {"row_index": row_index, "bucket": row.get("bucket"), "assistant": []}
        for assistant_index in assistant_indexes(new_row):
            result = rewrites[(row_index, assistant_index)]
            messages[assistant_index]["content"] = result.restyled
            preview["assistant"].append(result_to_json(result))
            for key in usage:
                value = result.usage.get(key)
                if isinstance(value, int):
                    usage[key] += value
        new_row["restyled_by"] = style["model"]
        new_row["restyled_style"] = style["id"]
        output_rows.append(new_row)
        previews_by_row[row_index] = preview
    return output_rows, [previews_by_row[i] for i in sorted(previews_by_row)], usage


def estimate_restyle_cost(config: dict[str, Any]) -> dict[str, Any]:
    rows = load_source_rows(config["dataset"])
    row_indexes = selected_row_indexes(
        rows,
        config["dataset"].get("sample"),
        int(config.get("seed", 3407)),
    )
    targets = iter_targets(rows, row_indexes, config["style"])
    cost = estimate_cost(targets, config["style"])
    print(json.dumps(cost, indent=2, ensure_ascii=False))
    return cost


def restyle(
    config_path: str | Path, *, estimate_only: bool = False, push: bool | None = None
) -> None:
    config = read_yaml(config_path)
    load_env_file(config.get("env_file", ".env"))
    style = config["style"] = resolve_llm_settings(config["style"])
    dataset = config["dataset"]
    output = config["output"]
    version = artifact_version(config)
    hub_repo_id = standard_repo_id(
        output.get("hub", {}),
        fallback_name=f"{style['id']}-dataset",
        version=version,
        required=False,
    )
    should_push = output.get("push_to_hub", True) if push is None else push
    if should_push and not estimate_only:
        hub = output["hub"]
        preflight_upload(
            repo_id=standard_repo_id(hub, fallback_name=f"{style['id']}-dataset", version=version),
            repo_type="dataset",
            private=bool(hub.get("private", False)),
        )

    rows = load_source_rows(dataset)
    row_indexes = selected_row_indexes(
        rows,
        dataset.get("sample"),
        int(config.get("seed", 3407)),
    )
    targets = iter_targets(rows, row_indexes, style)
    cost = estimate_cost(targets, style)
    print(json.dumps(cost, indent=2, ensure_ascii=False))
    if estimate_only:
        return
    api_key = provider_api_key(style)

    output_dir = Path(output["dir"])
    checkpoint_path = output_dir / output.get("checkpoint_file", "checkpoint.jsonl")
    checkpoint = load_checkpoint(checkpoint_path)
    rewrites, ignored_checkpoint = compatible_checkpoint_rewrites(
        checkpoint, targets, rewrite_style_hash(style)
    )
    if ignored_checkpoint:
        print(
            f"Ignoring {ignored_checkpoint} stale checkpoint rewrite(s) for this run",
            file=sys.stderr,
        )
    missing = [
        target for target in targets if (target.row_index, target.assistant_index) not in rewrites
    ]
    workers = int(style.get("workers", 8))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(restyle_one, api_key, style, rows[target.row_index], target): target
            for target in missing
        }
        completed = len(rewrites)
        for future in concurrent.futures.as_completed(futures):
            target = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(
                    f"failed {target.row_index}/{target.assistant_index}: {exc}",
                    file=sys.stderr,
                )
                continue
            rewrites[(result.row_index, result.assistant_index)] = result
            append_jsonl(checkpoint_path, result_to_json(result))
            completed += 1
            print(f"restyled {completed}/{len(targets)} assistant messages", file=sys.stderr)

    missing_after = [
        target for target in targets if (target.row_index, target.assistant_index) not in rewrites
    ]
    if missing_after:
        preview = ", ".join(
            f"{target.row_index}/{target.assistant_index}" for target in missing_after[:10]
        )
        raise RuntimeError(
            f"{len(missing_after)} restyle target(s) are still missing after this pass: {preview}"
        )

    output_rows, previews, usage = apply_rewrites(rows, row_indexes, rewrites, style)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / output.get("train_file", "train.jsonl"), output_rows)
    write_json(output_dir / "preview.json", previews)
    manifest = {
        "created_at": now_iso(),
        "artifact_type": "styled_dataset",
        "artifact_id": style["id"],
        "artifact_version": version,
        "hub_repo_id": hub_repo_id,
        "config_path": str(config_path),
        "source_dataset": dataset.get("hub_repo_id"),
        "source_path": dataset.get("input_path"),
        "style": style,
        "rows": len(output_rows),
        "restyled_rows": len(row_indexes),
        "assistant_calls": len(targets),
        "estimated_cost": cost,
        "actual_usage": usage,
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(
        styled_dataset_card(config=config, manifest=manifest, repo_id=hub_repo_id),
        encoding="utf-8",
    )
    if should_push:
        hub = output["hub"]
        upload_url = upload_folder(
            repo_id=standard_repo_id(hub, fallback_name=f"{style['id']}-dataset", version=version),
            repo_type="dataset",
            folder_path=output_dir,
            private=bool(hub.get("private", False)),
            commit_message=hub.get("commit_message", "Upload llmstyler restyled dataset"),
        )
        print(f"Uploaded dataset to {upload_url}")


def load_source_rows(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    input_path = dataset.get("input_path")
    if input_path and Path(input_path).exists():
        return read_jsonl(input_path)
    hub_repo_id = dataset.get("hub_repo_id")
    if not hub_repo_id:
        raise FileNotFoundError(
            f"{input_path!r} does not exist and dataset.hub_repo_id is not configured"
        )
    reject_agent_operation("Hugging Face dataset access")
    from datasets import load_dataset

    split = dataset.get("split", "train")
    return [dict(row) for row in load_dataset(hub_repo_id, split=split)]
