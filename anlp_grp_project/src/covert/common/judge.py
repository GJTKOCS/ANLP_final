"""Sense judge: P(Yes)/(P(Yes)+P(No)) from next-token logits, frozen base model (adapters disabled)."""
from __future__ import annotations

import numpy as np
import torch

from . import views
from .llm import LM


class Judge:
    def __init__(self, lm: LM, bs: int = 16):
        self.lm, self.bs = lm, bs
        tok = lm.tok
        ids = lambda words: sorted({tok(w, add_special_tokens=False)["input_ids"][0] for w in words})
        self.yes, self.no = ids(["Yes", " Yes", "yes", " yes"]), ids(["No", " No", "no", " no"])

    def score(self, items: list[tuple[str, list]]) -> list[float]:
        """items: (cover_topic, post_switch). Sees only these two things."""
        prompts = [views.judge_text(c, p) for c, p in items]
        lp = self.lm.next_token_logprobs(prompts, self.yes + self.no, adapter=None, bs=self.bs)
        py = torch.logsumexp(lp[:, : len(self.yes)], -1)
        pn = torch.logsumexp(lp[:, len(self.yes):], -1)
        return torch.sigmoid(py - pn).tolist()
