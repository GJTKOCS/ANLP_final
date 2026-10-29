"""Config loading: configs/base.yaml + hardware profile (+ toy overlay) + key=value overrides."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs"


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _read(p: Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _set(cfg: dict, ov: str) -> None:
    k, v = ov.split("=", 1)
    d = cfg
    parts = k.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = yaml.safe_load(v)


def _auto_profile(toy: bool) -> str:
    if toy:
        return "cpu_debug"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. Run on the GPU machine, or use --toy for the CPU smoke test.")
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    return "a100" if mem >= 30 else "t4"


def load_config(overrides: Iterable[str] = (), profile: str | None = None, toy: bool = False) -> Cfg:
    """Load the resolved config. Overrides look like ``speaker.lr=1e-5``."""
    cfg = _read(CONFIGS / "base.yaml")
    if toy:
        cfg = _merge(cfg, _read(CONFIGS / "toy.yaml"))
    hw_over = []
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' must look like section.key=value")
        if ov.startswith("hw."):
            hw_over.append(ov)                                  # applied after the hardware profile is loaded
        else:
            _set(cfg, ov)
    prof = profile or cfg["profile"]
    if prof == "auto":
        prof = _auto_profile(toy)
    hwp = CONFIGS / "hardware" / f"{prof}.yaml"
    if not hwp.exists():
        raise FileNotFoundError(f"Unknown hardware profile '{prof}'. Available: {[p.stem for p in (CONFIGS/'hardware').glob('*.yaml')]}")
    cfg["profile"] = prof
    cfg["hw"] = _read(hwp)
    for ov in hw_over:
        _set(cfg, ov)
    cfg["toy"] = toy
    sub = "toy" if toy else ""
    cfg["paths"] = {k: str((ROOT / v / sub) if sub else ROOT / v) for k, v in cfg["paths"].items()}
    cfg["inventory"] = _read(CONFIGS / "inventory.yaml")
    return Cfg(cfg)


def cfg_hash(cfg: dict) -> str:
    keep = {k: v for k, v in cfg.items() if k not in ("inventory", "paths")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:10]


def load_yaml(name: str) -> Any:
    return _read(CONFIGS / name) if not name.endswith(".yaml") else yaml.safe_load(open(CONFIGS / name))
