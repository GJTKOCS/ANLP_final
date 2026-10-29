"""Warm-start conversations: big_gen model plays A (with secret) and B (without), K per episode, then scored by
prompted listeners (OptionScorer), leak checker and sense judge. 'good' marks SFT-eligible conversations."""
from __future__ import annotations

import logging
import random

import numpy as np

from ..common import views
from ..common.inventory import PEOPLE, PLACES
from ..common.judge import Judge
from ..common.scorer import OptionScorer
from ..speaker.rollout import rollout

log = logging.getLogger("covert.warmstart")


def pick_examples(convs: list[dict], eps: dict, seed: int, k: int = 3) -> list[tuple[list, dict]]:
    """3 solved training examples for the prompted C (leak-free training transcripts)."""
    ok = [c for c in convs if not c["a_leak"]]
    rng = random.Random(seed)
    pick = rng.sample(ok, min(k, len(ok)))
    return [(c["post_switch"], eps[c["episode_id"]]["secret"]) for c in pick], {c["episode_id"] for c in pick}


def prompted_scores(lm, convs: list[dict], eps: dict, examples, bs: int) -> list[dict]:
    """Prompted B (memories+chat) and prompted C (rules + 3 examples + chat) via option scoring."""
    sc = OptionScorer(lm, bs)
    out = []
    pb = [views.prompted_b_text(eps[c["episode_id"]]["memories"], c["post_switch"]) for c in convs]
    pc = [views.prompted_c_text(c["post_switch"], examples) for c in convs]
    res = {}
    for name, prompts in (("B", pb), ("C", pc)):
        for slot, opts in (("helper", PEOPLE), ("venue", PLACES)):
            res[(name, slot)] = sc.probs([p + f"\n{slot.capitalize()}:" for p in prompts], opts)
    for i, c in enumerate(convs):
        sec = eps[c["episode_id"]]["secret"]
        h, v = PEOPLE.index(sec["helper"]), PLACES.index(sec["venue"])
        d = {}
        for name in ("B", "C"):
            ph, pv = res[(name, "helper")][i], res[(name, "venue")][i]
            d[name] = {"helper_ok": bool(ph.argmax() == h), "venue_ok": bool(pv.argmax() == v),
                       "p_helper": float(ph[h]), "p_venue": float(pv[v])}
        out.append(d)
    return out


def mark_good(c: dict, tau: float, excluded: set[str]) -> bool:
    ps = c["prompted_scores"]
    nC = int(ps["C"]["helper_ok"]) + int(ps["C"]["venue_ok"])
    return (not c["a_leak"]) and c["sense"] >= tau and ps["B"]["helper_ok"] and ps["B"]["venue_ok"] and nC <= 1 \
        and c["episode_id"] not in excluded


def headroom(convs: list[dict]) -> dict:
    ps = [c["prompted_scores"] for c in convs]
    return {"prompted_B_both": float(np.mean([p["B"]["helper_ok"] and p["B"]["venue_ok"] for p in ps])),
            "prompted_C_both": float(np.mean([p["C"]["helper_ok"] and p["C"]["venue_ok"] for p in ps])),
            "a_leak_rate": float(np.mean([c["a_leak"] for c in convs])), "n": len(convs)}


def generate_convs(cfg, lm, episodes: list[dict], K: int, split: str, model_name: str) -> list[dict]:
    ro = rollout(lm, lm, episodes, K, cfg, None, None)
    convs = []
    for r in ro:
        convs.append({"episode_id": r["episode_id"], "condition": "warmstart", "round": 0, "lam": None,
                      "speaker_artifact": model_name, "sample_idx": r["group_idx"], "post_switch": r["post_switch"],
                      "a_leak": r["a_leak"], "b_resamples": r["b_resamples"], "sense": None, "good": None,
                      "prompted_scores": None, "gen_config_hash": model_name})
    return convs


def score_convs(cfg, lm, judge: Judge, convs: list[dict], eps: dict, examples, excluded: set[str], tau: float) -> None:
    sense = judge.score([(eps[c["episode_id"]]["cover_topic"], c["post_switch"]) for c in convs])
    ps = prompted_scores(lm, convs, eps, examples, cfg.hw["score_bs"])
    for c, s, p in zip(convs, sense, ps):
        c["sense"], c["prompted_scores"] = float(s), p
        c["good"] = mark_good(c, tau, excluded)
