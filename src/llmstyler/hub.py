from __future__ import annotations

from pathlib import Path


def repo_url(repo_id: str, repo_type: str) -> str:
    path_prefix = "" if repo_type == "model" else f"{repo_type}s/"
    return f"https://huggingface.co/{path_prefix}{repo_id}"


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Hub uploads") from exc
    return HfApi()


def preflight_upload(
    *,
    repo_id: str,
    repo_type: str,
    private: bool = False,
) -> None:
    """Validate auth and repo creation before expensive artifact generation."""

    api = _api()
    try:
        api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face upload preflight failed: authenticate with HF_TOKEN or "
            "`huggingface-cli login` before using --push-to-hub"
        ) from exc

    try:
        # Mirror the first real upload step early so permission failures happen
        # before dataset sampling or model rewrite calls.
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face upload preflight failed: cannot create or access "
            f"{repo_type} repo {repo_id!r}"
        ) from exc


def upload_folder(
    *,
    repo_id: str,
    repo_type: str,
    folder_path: str | Path,
    private: bool = False,
    commit_message: str = "Upload llmstyler artifact",
) -> str:
    api = _api()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder_path),
        commit_message=commit_message,
    )
    return repo_url(repo_id, repo_type)
