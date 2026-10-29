"""Optional Hugging Face Hub backup (only when hub.push: true and HF_TOKEN is set). Repos are private."""
from __future__ import annotations

import logging
import os

from .artifacts import art_dir

log = logging.getLogger("covert.hub")


def _repo(name: str) -> tuple[str, str]:
    if name.startswith("speaker"):
        return "covert-speaker", "model"
    if name.startswith("listeners"):
        return "covert-listeners", "model"
    return "covert-data", "dataset"


def push_artifact(cfg, name: str) -> None:
    if not cfg.hub.get("push") or not os.environ.get("HF_TOKEN"):
        return
    from huggingface_hub import HfApi
    org = os.environ.get("HF_ORG") or cfg.hub.org
    rname, rtype = _repo(name)
    api = HfApi(token=os.environ["HF_TOKEN"])
    rid = f"{org}/{rname}"
    api.create_repo(rid, repo_type=rtype, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(art_dir(cfg, name)), repo_id=rid, repo_type=rtype, path_in_repo=name,
                      ignore_patterns=["ckpt.pt", "latest_adapter.pt", "*.tmp"])
    log.info("pushed %s -> %s", name, rid)
