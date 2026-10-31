"""OptionScorer: distribution over the 10 inventory options via log-probability scoring (no parsing).

Fast path: if the first token of every option is distinct, score the single branching token (one forward
per prompt). Otherwise fall back to full continuation log-probs. ASSUMPTION: first-token scoring is an
approximation of full-sequence scoring; the fallback is exact."""
from __future__ import annotations

import numpy as np
import torch

from .llm import LM


class OptionScorer:
    def __init__(self, lm: LM, bs: int = 16):
        self.lm, self.bs = lm, bs

    def probs(self, prompts: list[str], options: list[str], adapter: str | None = None) -> np.ndarray:
        options = [o[0].upper() + o[1:] for o in options]       # models write 'Bowling alley', not 'bowling alley'
        tok = self.lm.tok
        opt_ids = [tok(" " + o, add_special_tokens=False)["input_ids"] for o in options]
        firsts = [o[0] for o in opt_ids]
        if len(set(firsts)) == len(firsts):
            lp = self.lm.next_token_logprobs(prompts, firsts, adapter, self.bs)
            return torch.softmax(lp, -1).numpy()
        rows = []
        for p in prompts:
            pid = tok(p, add_special_tokens=False)["input_ids"]
            lp, mask = self.lm.logprobs([pid] * len(options), opt_ids, adapter)
            rows.append(lp.sum(1).cpu())
        return torch.softmax(torch.stack(rows), -1).numpy()
