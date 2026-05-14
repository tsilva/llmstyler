from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from llmstyler.io import read_yaml


def build_mix(config_path: str | Path, *, push: bool | None = None, force: bool = False) -> Path:
    """Build a base dataset mix through datamixxer and mirror it for llmstyler."""
    if shutil.which("datamixxer") is None:
        raise RuntimeError(
            "datamixxer CLI is required for `llmstyler build-mix`. Install it from "
            "https://github.com/tsilva/datamixxer, then rerun the command."
        )

    config = read_yaml(config_path)
    command = ["datamixxer", "build", str(config_path)]
    if push is True:
        command.append("--push-to-hub")
    elif push is False:
        command.append("--no-push-to-hub")
    if force:
        command.append("--force")

    subprocess.run(command, check=True)
    artifact_dir = datamixxer_artifact_dir(config_path)
    mirror_dir = mirror_output_dir(config)
    if mirror_dir is not None:
        mirror_datamixxer_artifact(artifact_dir, mirror_dir)
    return artifact_dir


def datamixxer_artifact_dir(config_path: str | Path) -> Path:
    result = subprocess.run(
        ["datamixxer", "show", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("path: "):
            return Path(line.removeprefix("path: ").strip())
        if line.startswith("Artifact: "):
            return Path(line.removeprefix("Artifact: ").strip())
    raise RuntimeError("datamixxer did not report an artifact path for " + str(config_path))


def mirror_output_dir(config: dict[str, Any]) -> Path | None:
    output = config.get("output") or {}
    if not isinstance(output, dict) or not output.get("dir"):
        return None
    return Path(str(output["dir"]))


def mirror_datamixxer_artifact(artifact_dir: Path, output_dir: Path) -> None:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"datamixxer artifact is missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise RuntimeError(f"datamixxer artifact manifest has no splits: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("manifest.json", "README.md"):
        source = artifact_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)

    for split_info in splits.values():
        if not isinstance(split_info, dict) or not split_info.get("file"):
            continue
        filename = str(split_info["file"])
        shutil.copy2(artifact_dir / filename, output_dir / filename)
