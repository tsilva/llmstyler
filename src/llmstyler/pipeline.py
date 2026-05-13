from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from llmstyler.mix import build_mix
from llmstyler.restyle import restyle
from llmstyler.runbook import make_runbook


STEP_CONFIG_FILENAMES = {
    "base_mix": "base_mix.yaml",
    "restyle": "restyle.yaml",
    "runstep": "runstep.yaml",
}
CACHE_SCHEMA_VERSION = 1
STATUS_FILENAME = "status.json"


def compose_pipeline_config(config_path: str | Path, overrides: list[str]) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "OmegaConf is required for pipeline composition. Refresh the llmstyler install with "
            "`uv tool install --reinstall --editable .` from the repo root, then rerun the command."
        ) from exc

    config = OmegaConf.load(config_path)
    override_config = OmegaConf.from_dotlist(overrides)
    resolved = OmegaConf.to_container(
        OmegaConf.merge(config, override_config),
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(resolved, dict):
        raise ValueError(f"{config_path} must resolve to a YAML object")
    return resolved


def train_style(
    config_path: str | Path,
    *,
    overrides: list[str] | None = None,
    run: bool = True,
    push: bool | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    pipeline = compose_pipeline_config(config_path, overrides or [])
    _validate_pipeline(pipeline)
    if run and not dry_run and shutil.which("runbook") is None:
        raise RuntimeError(
            "runbook CLI is required to launch training. Install it first from "
            "https://github.com/tsilva/runbook, or rerun with --no-run to only generate "
            "the training files."
        )

    run_dir = Path(pipeline.get("output_dir", f"runs/{pipeline['id']}"))
    config_dir = run_dir / "pipeline"
    config_dir.mkdir(parents=True, exist_ok=True)
    status_path = config_dir / STATUS_FILENAME
    status = _read_status(status_path)

    base_mix_config = dict(pipeline["base_mix"])
    restyle_config = dict(pipeline["restyle"])
    runstep_config = dict(pipeline["runstep"])
    if push is not None:
        base_mix_config.setdefault("output", {})["push_to_hub"] = push
        restyle_config.setdefault("output", {})["push_to_hub"] = push

    _write_yaml(config_dir / "pipeline.resolved.yaml", pipeline)
    base_mix_path = config_dir / STEP_CONFIG_FILENAMES["base_mix"]
    restyle_path = config_dir / STEP_CONFIG_FILENAMES["restyle"]
    runstep_path = config_dir / STEP_CONFIG_FILENAMES["runstep"]
    _write_yaml(base_mix_path, base_mix_config)
    _write_yaml(restyle_path, restyle_config)
    _write_yaml(runstep_path, runstep_config)

    print(f"Wrote resolved pipeline config to {config_dir / 'pipeline.resolved.yaml'}")
    if dry_run:
        print(f"Wrote step configs to {config_dir}")
        return

    base_mix_hash = _step_hash("base_mix", base_mix_config)
    if _step_complete(status, "base_mix", base_mix_hash, _base_mix_artifacts(base_mix_config), force):
        print("Skipping base mix; inputs unchanged and artifacts exist")
    else:
        print("Building base mix")
        build_mix(base_mix_path, push=push)
        _mark_step_complete(status, "base_mix", base_mix_hash, _base_mix_artifacts(base_mix_config))
        _write_status(status_path, status)

    restyle_hash = _step_hash(
        "restyle",
        {"config": restyle_config, "depends_on": {"base_mix": base_mix_hash}},
    )
    if _step_complete(status, "restyle", restyle_hash, _restyle_artifacts(restyle_config), force):
        print("Skipping restyle; inputs unchanged and artifacts exist")
    else:
        print("Restyling dataset")
        restyle(restyle_path, push=push)
        _mark_step_complete(status, "restyle", restyle_hash, _restyle_artifacts(restyle_config))
        _write_status(status_path, status)

    runstep_hash = _step_hash(
        "runstep",
        {"config": runstep_config, "run": run, "depends_on": {"restyle": restyle_hash}},
    )
    runstep_artifacts = _runstep_artifacts(runstep_config, run=run)
    if _step_complete(status, "runstep", runstep_hash, runstep_artifacts, force):
        print("Skipping training runstep; inputs unchanged and artifacts exist")
    else:
        print("Launching training runstep" if run else "Generating training runstep")
        make_runbook(runstep_path, run=run)
        _mark_step_complete(status, "runstep", runstep_hash, runstep_artifacts)
        _write_status(status_path, status)
    _print_model_outputs(runstep_config)


def _validate_pipeline(pipeline: dict[str, Any]) -> None:
    missing = [key for key in ("id", "base_mix", "restyle", "runstep") if key not in pipeline]
    if missing:
        raise KeyError("pipeline config is missing required key(s): " + ", ".join(missing))
    for key in ("base_mix", "restyle", "runstep"):
        if not isinstance(pipeline[key], dict):
            raise ValueError(f"pipeline.{key} must be a YAML object")


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _read_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": CACHE_SCHEMA_VERSION, "steps": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
    except json.JSONDecodeError:
        return {"schema_version": CACHE_SCHEMA_VERSION, "steps": {}}
    if not isinstance(status, dict) or status.get("schema_version") != CACHE_SCHEMA_VERSION:
        return {"schema_version": CACHE_SCHEMA_VERSION, "steps": {}}
    if not isinstance(status.get("steps"), dict):
        status["steps"] = {}
    return status


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _step_hash(step: str, config: dict[str, Any]) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "step": step,
        "config": config,
    }
    encoded = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _step_complete(
    status: dict[str, Any],
    step: str,
    expected_hash: str,
    artifacts: list[Path],
    force: bool,
) -> bool:
    if force:
        return False
    step_status = status.get("steps", {}).get(step, {})
    if not isinstance(step_status, dict) or step_status.get("hash") != expected_hash:
        return False
    return all(path.exists() for path in artifacts)


def _mark_step_complete(
    status: dict[str, Any],
    step: str,
    step_hash: str,
    artifacts: list[Path],
) -> None:
    missing = [str(path) for path in artifacts if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{step} completed but expected artifact(s) were not found: " + ", ".join(missing)
        )
    status.setdefault("steps", {})[step] = {
        "hash": step_hash,
        "artifacts": [str(path) for path in artifacts],
    }


def _base_mix_artifacts(config: dict[str, Any]) -> list[Path]:
    output = config["output"]
    output_dir = Path(output["dir"])
    return [
        output_dir / output.get("train_file", "train.jsonl"),
        output_dir / "manifest.json",
        output_dir / "README.md",
    ]


def _restyle_artifacts(config: dict[str, Any]) -> list[Path]:
    output = config["output"]
    output_dir = Path(output["dir"])
    return [
        output_dir / output.get("train_file", "train.jsonl"),
        output_dir / "manifest.json",
        output_dir / "preview.json",
        output_dir / "README.md",
    ]


def _runstep_artifacts(config: dict[str, Any], *, run: bool) -> list[Path]:
    run_dir = Path(config.get("runbook", {}).get("output_dir", f"runs/{config['id']}"))
    artifacts = [run_dir / "train.py", run_dir / "train.py.yaml"]
    if run:
        artifacts.append(run_dir / "train.finished.ipynb")
    return artifacts


def _print_model_outputs(runstep_config: dict[str, Any]) -> None:
    publishing = runstep_config.get("publishing", {})
    outputs = {
        "Adapter": publishing.get("adapter_repo"),
        "Merged": publishing.get("merged_repo"),
        "GGUF": publishing.get("gguf_repo"),
        "ONNX": publishing.get("onnx_repo"),
    }
    print("Configured model outputs:")
    for label, repo_id in outputs.items():
        if repo_id:
            print(f"{label}: https://huggingface.co/{repo_id}")
