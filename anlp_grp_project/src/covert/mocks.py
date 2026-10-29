"""Mock dependencies for fast tests: keyword-overlap listener and a simple judge."""
from __future__ import annotations

import re

import numpy as np

from .common.inventory import PEOPLE, PLACES

_W = re.compile(r"[a-z']+")
_STOP = {"the", "a", "and", "to", "of", "we", "you", "i", "it", "in", "on", "at", "that", "was", "our", "is", "for", "with", "time"}


def _words(t: str) -> set[str]:
    return {w for w in _W.findall(t.lower()) if w not in _STOP and len(w) > 2}


class MockListener:
    """B (use_memories) scores each name/place by overlap of chat words with memories that mention it; C is ~uniform."""

    def __init__(self, view: str):
        self.view = view

    def probs(self, items: list[dict]):
        H, V = [], []
        for it in items:
            cw = _words(" ".join(t for _, t in it["post_switch"]))
            sh, sv = np.full(10, 0.1), np.full(10, 0.1)
            if self.view == "B":
                for m in it["memories"]:
                    ov = len(cw & _words(m)) * 0.5
                    for i, p in enumerate(PEOPLE):
                        if p in m:
                            sh[i] += ov
                    for i, p in enumerate(PLACES):
                        if p in m:
                            sv[i] += ov
            H.append(sh / sh.sum())
            V.append(sv / sv.sum())
        return np.array(H), np.array(V)

    def close(self) -> None:
        pass


class MockJudge:
    def score(self, items):
        return [0.9 if len(p) and all(t.strip() for _, t in p) else 0.1 for _, p in items]
