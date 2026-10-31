"""Prompted listeners (zero-shot, big_gen model, option scoring): B sees memories + chat; C sees only the chat,
the rules of the game and three solved training examples. Same model and scoring for both; no training."""
from __future__ import annotations

import numpy as np

from ..common import views
from ..common.inventory import PEOPLE, PLACES
from ..common.scorer import OptionScorer


class PromptedListener:
    def __init__(self, lm, view: str, examples=None, bs: int = 16):
        self.lm, self.view, self.examples, self.bs = lm, view, examples or [], bs

    def probs(self, items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        if self.view == "B":
            prompts = [views.prompted_b_text(it["memories"], it["post_switch"]) for it in items]
        else:
            prompts = [views.prompted_c_text(it["post_switch"], self.examples) for it in items]
        sc = OptionScorer(self.lm, self.bs)
        h = sc.probs([p + "\nHelper:" for p in prompts], PEOPLE)
        v = sc.probs([p + "\nVenue:" for p in prompts], PLACES)
        return h, v

    def close(self) -> None:
        pass
