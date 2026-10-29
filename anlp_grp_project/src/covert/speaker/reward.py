"""Speaker reward: leak -> -1; sense < tau -> -1; else mean(pB) - lam * mean(pC)."""
from __future__ import annotations

import math

import numpy as np

from ..common.inventory import PEOPLE, PLACES

MAX_BITS = math.log2(10)


def _info(p: float) -> float:
    """Clipped information in bits vs. chance (10%), scaled to [-1, 1]."""
    return max(-MAX_BITS, min(MAX_BITS, math.log2(max(p, 1e-6) / 0.1))) / MAX_BITS


def reward_from(a_leak: bool, sense: float, pB: tuple[float, float], pC: tuple[float, float],
                lam: float, tau: float, mode: str = "prob") -> float:
    if a_leak:
        return -1.0
    if sense < tau:
        return -1.0
    if mode == "logprob":
        return float(np.mean([_info(p) for p in pB]) - lam * np.mean([_info(p) for p in pC]))
    if mode != "prob":
        raise ValueError(f"unknown reward_mode {mode}")
    return float(np.mean(pB) - lam * np.mean(pC))


def score_rollouts(rollouts: list[dict], eps: dict[str, dict], B, C, judge, lam: float, tau: float, mode: str = "prob") -> None:
    """Fill reward components in place. B, C expose .probs(items)->(helper [N,10], venue [N,10])."""
    items = [{"memories": eps[r["episode_id"]]["memories"], "post_switch": r["post_switch"]} for r in rollouts]
    bh, bv = B.probs(items)
    ch, cv = C.probs(items)
    sense = judge.score([(eps[r["episode_id"]]["cover_topic"], r["post_switch"]) for r in rollouts])
    for i, r in enumerate(rollouts):
        sec = eps[r["episode_id"]]["secret"]
        h, v = PEOPLE.index(sec["helper"]), PLACES.index(sec["venue"])
        r["pB"] = {"helper": float(bh[i, h]), "venue": float(bv[i, v])}
        r["pC"] = {"helper": float(ch[i, h]), "venue": float(cv[i, v])}
        r["sense"] = float(sense[i])
        r["reward"] = reward_from(r["a_leak"], r["sense"], (r["pB"]["helper"], r["pB"]["venue"]),
                                  (r["pC"]["helper"], r["pC"]["venue"]), lam, tau, mode)
