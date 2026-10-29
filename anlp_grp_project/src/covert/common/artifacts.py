"""Artifact storage: artifacts/<name>/ with manifest.json. Stages refuse to start on incomplete inputs."""
from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, cfg_hash
from .device import gpu_info
from .io import read_json, write_json


def art_dir(cfg, name: str) -> Path:
    return Path(cfg["paths"]["artifacts"]) / name


def artifact_status(cfg, name: str) -> str | None:
    m = art_dir(cfg, name) / "manifest.json"
    return read_json(m)["status"] if m.exists() else None


def require(cfg, name: str, stage_hint: str | None = None) -> Path:
    st = artifact_status(cfg, name)
    if st != "complete":
        hint = f" Run `covert status`, then `covert pipeline run --only {stage_hint}`." if stage_hint else " Run `covert status`."
        raise RuntimeError(f"Artifact {name} is not complete (status={st}).{hint}")
    return art_dir(cfg, name)


def _git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


class Artifact:
    """Context manager: marks the artifact running, then complete (or failed) with gpu-hours."""

    def __init__(self, cfg, name: str, stage: str, command: str = "", inputs=(), models=None, extra=None):
        self.cfg, self.name, self.stage = cfg, name, stage
        self.dir = art_dir(cfg, name)
        self.man = {"run_id": name, "stage": stage, "command": command, "inputs": list(inputs), "models": models or {},
                    "outputs": [name], "seeds": {"seed": cfg["seed"]}, "extra": extra or {}}

    def _write(self, status: str) -> None:
        g = gpu_info()
        self.man.update(status=status, created_at=datetime.now(timezone.utc).isoformat(), git_commit=_git(),
                        config_hash=cfg_hash(self.cfg), profile=self.cfg["profile"], gpu_type=g["type"],
                        gpu_hours=round((time.time() - self.t0) / 3600 * (1 if g["mem_gb"] else 0), 4),
                        wall_hours=round((time.time() - self.t0) / 3600, 4),
                        config={k: v for k, v in self.cfg.items() if k not in ("inventory",)})
        write_json(self.dir / "manifest.json", self.man)

    def __enter__(self) -> "Artifact":
        self.t0 = time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write("running")
        return self

    def __exit__(self, et, ev, tb) -> bool:
        self._write("complete" if et is None else "failed")
        return False
