from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llmstyler.io import load_env_file, read_yaml
from llmstyler.pipeline import STATUS_FILENAME, compose_pipeline_config
from llmstyler.restyle import estimate_restyle_cost
from llmstyler.standards import slugify


DEFAULT_REWRITE_PROMPT = """You rewrite assistant responses for a supervised fine-tuning dataset.

Task:
- Preserve the same factual content, meaning, accuracy, caveats, and user-facing usefulness.
- Rewrite only the assistant response in the requested speaking style.
- Do not claim to be a real person.
- Do not add new factual claims, named entities, accusations, dates, numbers, or policy positions.
- Do not remove important facts, safety caveats, reasoning steps, or conclusions.
- Keep the same language as the original response.
- Keep lists, math, code, citations, and structured outputs intact when they carry meaning.
- Return only valid JSON matching {"restyled_content": "..."}.
"""

BASE_PLAN = [
    ("capability_magpie", "smoltalk_smollm3_smol_magpie_ultra_no_think", 300),
    ("capability_rewrite", "smoltalk_smollm3_smol_rewrite_no_think", 150),
    ("capability_summarize", "smoltalk_smollm3_smol_summarize_no_think", 100),
    ("capability_instruction_following", "tulu_3_sft_personas_instruction_following_no_think", 100),
    ("capability_science", "Mixture_of_Thoughts_science_no_think", 50),
    ("restyle_magpie", "smoltalk_smollm3_smol_magpie_ultra_no_think", 180),
    ("restyle_rewrite", "smoltalk_smollm3_smol_rewrite_no_think", 70),
    ("restyle_summarize", "smoltalk_smollm3_smol_summarize_no_think", 50),
]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def create_pipeline_config(
    *,
    pipeline_id: str,
    owner: str,
    output_path: str | Path | None = None,
    name: str | None = None,
    version: str = "v1",
    base_model: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    style_id: str | None = None,
    style_prompt: str | None = None,
    style_prompt_file: str | Path | None = None,
    rows: int = 1000,
    restyle_rate: float = 0.2,
    force: bool = False,
) -> Path:
    pipeline_slug = slugify(pipeline_id)
    style_slug = slugify(style_id or pipeline_slug)
    resolved_name = name or pipeline_slug.replace("-", " ").replace("_", " ").title()
    prompt = _style_prompt(style_prompt=style_prompt, style_prompt_file=style_prompt_file)
    config = _pipeline_template(
        pipeline_id=pipeline_slug,
        style_id=style_slug,
        name=resolved_name,
        owner=owner,
        version=version,
        base_model=base_model,
        rewrite_prompt=prompt,
        rows=rows,
        restyle_rate=restyle_rate,
    )
    destination = Path(output_path or f"configs/pipelines/{pipeline_slug}.yaml")
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists; pass --force to overwrite it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Wrote {destination}")
    print(f"Estimate cost with: llmstyler estimate {destination}")
    print(f"Run locally with: llmstyler run {destination}")
    print(f"Publish artifacts with: llmstyler run {destination} --publish")
    return destination


def run_doctor(
    config_path: str | Path | None = None,
    *,
    overrides: list[str] | None = None,
    publish: bool = False,
    run: bool = True,
) -> int:
    config = _load_optional_pipeline(config_path, overrides or [])
    env_file = _env_file(config)
    if env_file:
        load_env_file(env_file)

    checks = [
        _check_python(),
        _check_import("yaml", "PyYAML import is available"),
        _check_import("omegaconf", "OmegaConf import is available"),
        _check_import("datasets", "datasets import is available"),
        _check_import("huggingface_hub", "huggingface_hub import is available"),
        _check_executable("datamixxer", "required for base mix generation"),
        _check_executable("runbook", "required for remote training launch") if run else Check(
            "runbook", True, "skipped because --no-run was passed"
        ),
        _check_openrouter(),
    ]
    if publish or _pipeline_wants_publish(config):
        checks.append(_check_hugging_face_auth())
    if config:
        checks.extend(_check_config_paths(config))

    for check in checks:
        marker = "ok" if check.ok else "fail"
        print(f"[{marker}] {check.name}: {check.detail}")
    return 0 if all(check.ok for check in checks) else 1


def estimate_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = _load_pipeline_or_restyle(config_path, overrides or [])
    return estimate_restyle_cost(config)


def inspect_config(config_path: str | Path, overrides: list[str] | None = None) -> None:
    pipeline = compose_pipeline_config(config_path, overrides or [])
    run_dir = Path(pipeline.get("output_dir", f"runs/{pipeline['id']}"))
    status_path = run_dir / "pipeline" / STATUS_FILENAME
    print(f"Pipeline: {pipeline['id']}")
    print(f"Run dir: {run_dir}")
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        steps = status.get("steps", {})
        if isinstance(steps, dict) and steps:
            print("Cached steps:")
            for name, step in steps.items():
                artifacts = step.get("artifacts", []) if isinstance(step, dict) else []
                print(f"- {name}: {len(artifacts)} artifact(s)")
        else:
            print("Cached steps: none")
    else:
        print("Cached steps: none")

    for label, manifest in _manifest_paths(pipeline):
        print(f"{label}: {manifest}")
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            rows = data.get("rows") or data.get("total_rows")
            if rows is not None:
                print(f"  rows: {rows}")
            repo = data.get("hub_repo_id")
            if repo:
                print(f"  hub: {repo}")
        else:
            print("  missing")

    preview = Path(pipeline["restyle"]["output"]["dir"]) / "preview.json"
    if preview.exists():
        previews = json.loads(preview.read_text(encoding="utf-8"))
        print(f"Preview rows: {len(previews)}")

    publishing = pipeline.get("runstep", {}).get("publishing", {})
    if publishing:
        print("Configured model outputs:")
        for label, key in (
            ("adapter", "adapter_repo"),
            ("merged", "merged_repo"),
            ("gguf", "gguf_repo"),
            ("onnx", "onnx_repo"),
        ):
            repo = publishing.get(key)
            if repo:
                print(f"- {label}: https://huggingface.co/{repo}")


def _style_prompt(*, style_prompt: str | None, style_prompt_file: str | Path | None) -> str:
    if style_prompt_file:
        return Path(style_prompt_file).read_text(encoding="utf-8").strip()
    if style_prompt:
        possible_path = Path(style_prompt)
        if possible_path.exists():
            return possible_path.read_text(encoding="utf-8").strip()
        return style_prompt.strip()
    return DEFAULT_REWRITE_PROMPT.strip()


def _scaled_plan(rows: int) -> list[dict[str, Any]]:
    total = sum(count for _, _, count in BASE_PLAN)
    scaled: list[dict[str, Any]] = []
    assigned = 0
    for index, (name, split, count) in enumerate(BASE_PLAN):
        if index == len(BASE_PLAN) - 1:
            item_count = max(1, rows - assigned)
        else:
            item_count = max(1, round(rows * count / total))
            assigned += item_count
        scaled.append({"name": name, "split": split, "count": item_count})
    return scaled


def _pipeline_template(
    *,
    pipeline_id: str,
    style_id: str,
    name: str,
    owner: str,
    version: str,
    base_model: str,
    rewrite_prompt: str,
    rows: int,
    restyle_rate: float,
) -> dict[str, Any]:
    return {
        "id": pipeline_id,
        "name": name,
        "version": version,
        "owner": owner,
        "output_dir": f"runs/{pipeline_id}",
        "style_contract": {
            "id": style_id,
            "system_prompt": (
                "You are a helpful assistant that answers in the configured speaking style."
            ),
        },
        "model": {
            "base_model": base_model,
            "load_in_4bit": True,
            "default_system_prompt": (
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            ),
        },
        "training": {
            "output_dir": f"outputs/{pipeline_id}",
            "run_name": f"{pipeline_id}-qlora",
            "wandb_project": f"{pipeline_id}-qlora",
            "max_seq_length": 2048,
            "num_epochs": 3,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.0003,
            "warmup_ratio": 0.05,
            "lora_r": 32,
            "lora_alpha": 32,
            "dataset_num_proc": 4,
            "eval_fraction": 0.1,
            "eval_steps": 33,
            "save_steps": 33,
            "logging_steps": 5,
            "report_to": ["tensorboard", "wandb"],
        },
        "publishing": {
            "adapter_repo": "${owner}/${id}-qlora-${version}",
            "merged_repo": "${owner}/${id}-merged-${version}",
            "gguf_repo": "${owner}/${id}-gguf-${version}",
            "onnx_repo": "${owner}/${id}-onnx-${version}",
        },
        "exports": {
            "gguf": {"quantization_methods": ["q4_k_m"]},
            "onnx": {"enabled": False, "task": "text-generation-with-past"},
        },
        "runbook": {
            "output_dir": "${output_dir}",
            "runtime": {
                "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
                "gpu": "A10",
                "cpu": 8.0,
                "memory": 49152,
                "timeout": 14400,
            },
            "modal": {"secrets": ["wandb-secret", "huggingface-secret"]},
        },
        "base_mix": {
            "id": f"{pipeline_id}_base_mix",
            "name": f"{name} Base Mix",
            "version": "${version}",
            "seed": 3407,
            "buffer_size": 10000,
            "split": {"test_size": 0.1},
            "tagging": [
                {
                    "rate": restyle_rate,
                    "output_splits": ["train"],
                    "balance_by": ["source"],
                    "tags": {"restyle": True},
                }
            ],
            "source": {"dataset_id": "HuggingFaceTB/smoltalk2", "config": "SFT"},
            "plan": _scaled_plan(max(1, rows)),
            "output": {
                "store_dir": ".datamixxer/mixes",
                "dir": f"datasets/{pipeline_id}_base",
                "train_file": "train.jsonl",
                "test_file": "test.jsonl",
                "push_to_hub": False,
                "hub": {"repo_id": "${owner}/${id}-base-${version}"},
            },
        },
        "restyle": {
            "id": "${style_contract.id}",
            "version": "${version}",
            "seed": 3407,
            "env_file": ".env",
            "dataset": {
                "input_path": "${base_mix.output.dir}/${base_mix.output.train_file}",
                "hub_repo_id": "${base_mix.output.hub.repo_id}",
            },
            "style": {
                "id": "${style_contract.id}",
                "model": "x-ai/grok-4.3",
                "temperature": 0.7,
                "max_tokens": 4096,
                "workers": 8,
                "retries": 3,
                "retry_sleep": 2.0,
                "request_timeout": 90.0,
                "context_mode": "latest-user",
                "output_expansion": 1.25,
                "pricing_usd_per_1m": {"input": 1.25, "output": 2.50},
                "rewrite_prompt": rewrite_prompt,
            },
            "output": {
                "dir": f"datasets/{pipeline_id}_styled",
                "train_file": "train.jsonl",
                "checkpoint_file": "checkpoint.jsonl",
                "push_to_hub": False,
                "hub": {"repo_id": "${owner}/${id}-styled-${version}"},
            },
        },
        "runstep": {
            "id": "${id}",
            "name": "${name}",
            "version": "${version}",
            "style": {
                "id": "${style_contract.id}",
                "system_prompt": "${style_contract.system_prompt}",
            },
            "dataset": {
                "hub_repo_id": "${restyle.output.hub.repo_id}",
                "split": "train",
                "restyled_only": False,
            },
            "model": "${model}",
            "training": "${training}",
            "publishing": "${publishing}",
            "exports": "${exports}",
            "runbook": "${runbook}",
        },
    }


def _load_optional_pipeline(
    config_path: str | Path | None, overrides: list[str]
) -> dict[str, Any] | None:
    if config_path is None:
        return None
    return compose_pipeline_config(config_path, overrides)


def _load_pipeline_or_restyle(config_path: str | Path, overrides: list[str]) -> dict[str, Any]:
    raw = read_yaml(config_path)
    if {"base_mix", "restyle", "runstep"}.issubset(raw):
        return compose_pipeline_config(config_path, overrides)["restyle"]
    if overrides:
        raise ValueError("dotlist overrides are only supported for pipeline configs")
    return raw


def _check_python() -> Check:
    ok = sys.version_info >= (3, 11)
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return Check("python", ok, version)


def _check_import(module_name: str, ok_detail: str) -> Check:
    try:
        __import__(module_name)
    except Exception as exc:
        return Check(module_name, False, str(exc))
    return Check(module_name, True, ok_detail)


def _check_executable(name: str, detail: str) -> Check:
    path = shutil.which(name)
    return Check(name, path is not None, path or detail)


def _check_openrouter() -> Check:
    value = os.environ.get("OPENROUTER_API_KEY")
    return Check(
        "OPENROUTER_API_KEY",
        bool(value),
        "configured" if value else "missing; required for restyle runs",
    )


def _check_hugging_face_auth() -> Check:
    try:
        from huggingface_hub import HfApi

        whoami = HfApi().whoami()
    except Exception as exc:
        return Check("huggingface auth", False, str(exc))
    name = whoami.get("name") or whoami.get("fullname") or "authenticated"
    return Check("huggingface auth", True, str(name))


def _check_config_paths(config: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    restyle = config.get("restyle", {})
    base_mix = config.get("base_mix", {})
    for name, path in (
        ("base mix output", base_mix.get("output", {}).get("dir")),
        ("restyle output", restyle.get("output", {}).get("dir")),
        ("run output", config.get("output_dir")),
    ):
        if not path:
            checks.append(Check(name, False, "not configured"))
            continue
        parent = Path(path).parent
        checks.append(Check(name, parent.exists() or parent == Path("."), str(path)))
    return checks


def _pipeline_wants_publish(config: dict[str, Any] | None) -> bool:
    if not config:
        return False
    for key in ("base_mix", "restyle"):
        output = config.get(key, {}).get("output", {})
        if output.get("push_to_hub") is True:
            return True
    return False


def _env_file(config: dict[str, Any] | None) -> str | Path | None:
    if not config:
        return ".env"
    return config.get("restyle", {}).get("env_file", ".env")


def _manifest_paths(pipeline: dict[str, Any]) -> list[tuple[str, Path]]:
    return [
        ("Base mix manifest", Path(pipeline["base_mix"]["output"]["dir"]) / "manifest.json"),
        ("Restyle manifest", Path(pipeline["restyle"]["output"]["dir"]) / "manifest.json"),
    ]
