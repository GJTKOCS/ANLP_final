"""Bootstrap intervals and metric helpers."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def correctness(probs: dict, secrets: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(probs["helper"]).argmax(1) == np.array([s[0] for s in secrets])
    v = np.asarray(probs["venue"]).argmax(1) == np.array([s[1] for s in secrets])
    return h, v


def summarise(B: tuple, C: tuple, extra: dict, n: int) -> dict:
    """B, C: (helper_ok, venue_ok) arrays aligned by episode. extra: per-episode arrays (leak, sense, reward...)."""
    bh, bv = B
    ch, cv = C
    bb, cb = bh & bv, ch & cv
    nc = ch.astype(int) + cv.astype(int)
    m = {"B_helper": bh, "B_venue": bv, "B_both": bb, "C_helper": ch, "C_venue": cv, "C_both": cb,
         "insider_adv": bb.astype(float) - cb.astype(float),
         "joint_success": (bb & (nc <= 1)).astype(float), "joint_strict": (bb & (nc == 0)).astype(float), **extra}
    return {k: dict(zip(("mean", "lo", "hi"), bootstrap_ci(np.asarray(v, dtype=float), n))) for k, v in m.items()}
