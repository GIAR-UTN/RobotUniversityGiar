"""hub.py — keeps policies/<name>/ folders backed up on the Hugging Face Hub.

A policy that only exists on the disk that trained it is one container
restart, one `docker compose down` or one wiped laptop away from being gone.
This module mirrors each policies/<name>/ folder (checkpoint + raw PPO
state + meta.json + train.log) into one Hugging Face model repo, one
sub-folder per policy, so the same name can be pulled back onto any
machine — a teammate's laptop, a Kaggle kernel, a fresh container.

Which repo: `RUGIAR_HF_REPO` (e.g. "GIAR-UTN/rugiar-policies"), or an
explicit `repo_id`. Credentials are huggingface_hub's own: the `HF_TOKEN`
env var or a prior `hf auth login` — never read or stored here.

Automatic upload: TrainingManager.finalize_policy()/fuse_policies() call
push_after_save() once a policy folder is complete. It is a no-op unless
RUGIAR_HF_REPO is set, and it never raises — a failed upload must not turn
a finished training run into a failed one; the policy is still on disk and
`rugiar hub push <name>` retries it by hand.

`huggingface_hub` is an optional extra (`pip install -e .[hub]`), imported
lazily so nothing else in control/ needs it installed.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

REPO_ENV = "RUGIAR_HF_REPO"
POLICIES_DIR = Path(__file__).resolve().parents[2] / "policies"


class HubError(RuntimeError):
    """A hub operation that can't proceed, with a message fit to show a user."""


def _hf():
    try:
        import huggingface_hub
    except ImportError as e:
        raise HubError("huggingface_hub is not installed — run `pip install -e .[hub]`") from e
    return huggingface_hub


def hub_repo(repo_id: Optional[str] = None) -> Optional[str]:
    return repo_id or os.getenv(REPO_ENV) or None


def _require_repo(repo_id: Optional[str]) -> str:
    repo = hub_repo(repo_id)
    if not repo:
        raise HubError(f"no Hugging Face repo given — pass --repo or set {REPO_ENV} "
                       f"(e.g. {REPO_ENV}=GIAR-UTN/rugiar-policies)")
    return repo


def push_policy(name: str, repo_id: Optional[str] = None,
                policies_dir: Path = POLICIES_DIR, private: bool = False) -> str:
    """Upload policies/<name>/ to <repo>/<name>/, creating the repo on first
    use. Returns the policy's URL on the Hub."""
    repo = _require_repo(repo_id)
    folder = Path(policies_dir) / name
    if not (folder / "meta.json").is_file():
        raise HubError(f"'{name}' is not a complete local policy (no {folder / 'meta.json'})")
    hf = _hf()
    api = hf.HfApi()
    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder),
                      path_in_repo=name, commit_message=f"Add/update policy {name}")
    return f"https://huggingface.co/{repo}/tree/main/{name}"


def list_remote_policies(repo_id: Optional[str] = None) -> List[str]:
    """Names of every policy folder in the repo (a top-level folder holding a meta.json)."""
    repo = _require_repo(repo_id)
    files = _hf().HfApi().list_repo_files(repo_id=repo, repo_type="model")
    return sorted({f.split("/", 1)[0] for f in files
                   if f.count("/") == 1 and f.endswith("/meta.json")})


def pull_policy(name: str, repo_id: Optional[str] = None,
                policies_dir: Path = POLICIES_DIR, overwrite: bool = False) -> Path:
    """Download <repo>/<name>/ into policies/<name>/. Refuses to replace an
    existing local folder unless overwrite=True — a local policy with the same
    name may be newer than the Hub copy."""
    repo = _require_repo(repo_id)
    dest = Path(policies_dir) / name
    if dest.exists() and not overwrite:
        raise HubError(f"policies/{name}/ already exists locally — pass --overwrite to replace it")
    with tempfile.TemporaryDirectory() as tmp:
        _hf().snapshot_download(repo_id=repo, repo_type="model",
                                allow_patterns=[f"{name}/*"], local_dir=tmp)
        src = Path(tmp) / name
        if not (src / "meta.json").is_file():
            raise HubError(f"'{name}' not found in {repo} (see `rugiar hub list`)")
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
    return dest


def push_after_save(name: str, policies_dir: Path = POLICIES_DIR) -> Optional[str]:
    """The automatic upload hook — see the module docstring. Returns the URL,
    or None when disabled or failed (the failure is reported on stderr)."""
    if not hub_repo():
        return None
    try:
        url = push_policy(name, policies_dir=policies_dir)
    except Exception as e:  # noqa: BLE001 — must never break the caller, see docstring
        print(f"[hub] could not upload policy '{name}' to the Hugging Face Hub: {e} "
              f"— it is still saved locally; retry with `rugiar hub push {name}`", file=sys.stderr)
        return None
    print(f"[hub] policy '{name}' uploaded: {url}", file=sys.stderr)
    return url
