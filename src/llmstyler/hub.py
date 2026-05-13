from __future__ import annotations

from pathlib import Path


def upload_folder(
    *,
    repo_id: str,
    repo_type: str,
    folder_path: str | Path,
    private: bool = False,
    commit_message: str = "Upload llmstyler artifact",
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Hub uploads") from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder_path),
        commit_message=commit_message,
    )

