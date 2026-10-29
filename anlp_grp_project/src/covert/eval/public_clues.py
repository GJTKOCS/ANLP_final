"""Public-clue tracker: smoothed log-odds between A-turn words/bigrams and gold venue/helper values."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_W = re.compile(r"[a-z']+")


def _feats(text: str) -> set[str]:
    w = _W.findall(text.lower())
    return set(w) | {a + " " + b for a, b in zip(w, w[1:])}


def clue_table(convs: list[dict], eps: dict[str, dict], top: int = 5, alpha: float = 1.0, min_count: int = 3) -> dict:
    """Return {'venue': {value: [(feature, log_odds, count)]}, 'helper': {...}} from A turns."""
    out = {}
    for slot in ("venue", "helper"):
        by_val = defaultdict(Counter)
        tot = Counter()
        n_val = Counter()
        for c in convs:
            val = eps[c["episode_id"]]["secret"][slot]
            f = set().union(*[_feats(t) for s, t in c["post_switch"] if s == "A"]) if c["post_switch"] else set()
            by_val[val].update(f)
            tot.update(f)
            n_val[val] += 1
        N = sum(n_val.values())
        res = {}
        for val, cnt in by_val.items():
            rows = []
            for f, k in cnt.items():
                if k < min_count:
                    continue
                p_in = (k + alpha) / (n_val[val] + 2 * alpha)
                p_out = (tot[f] - k + alpha) / (N - n_val[val] + 2 * alpha)
                rows.append((f, math.log(p_in / (1 - p_in)) - math.log(p_out / (1 - p_out)), k))
            res[val] = sorted(rows, key=lambda r: -r[1])[:top]
        out[slot] = res
    return out
