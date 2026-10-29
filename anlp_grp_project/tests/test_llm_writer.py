"""LLM memory-writer path (retry + validation + fallback) with a fake generator."""
import re

from covert.common.config import load_config
from covert.data.episodes import build_split
from covert.data.memories import valid_memory


def _fake(bad_first: bool):
    calls = {"n": 0}

    def gen(msgs_list):
        calls["n"] += 1
        outs = []
        for msgs in msgs_list:
            usr = msgs[1]["content"]
            names = re.search(r"verbatim: (.*)\.\n", usr).group(1)
            m = re.match(r"(.*?)(?: and the place '(.*)')?$", names)
            who, place = m.group(1), m.group(2)
            import random
            r = random.Random(hash((names, calls["n"], len(outs))))
            vocab = "umbrella kettle cricket ladder samosa lantern violin pigeon scooter mango blanket guitar puddle ticket whistle".split()
            extra = " ".join(r.sample(vocab, 6))
            at = f" at the {place}" if place else ""
            frames = [f"Remember when {who} showed up{at} wearing a {extra} and nobody could stop laughing for hours?",
                      f"You still tease {who} about the evening{at} when a {extra} caused total chaos in front of everyone we knew.",
                      f"It was {who}{at} who somehow lost a {extra} and then blamed the weather, which made the whole trip unforgettable.",
                      f"We spent an entire afternoon{at} helping {who} hunt for a {extra}, and honestly it was the best day of that year."]
            text = r.choice(frames)
            if bad_first and calls["n"] == 1:
                text = "Too short."
            outs.append(text)
        return outs
    return gen, calls


def test_retry_then_valid():
    cfg = load_config(toy=True)
    gen, calls = _fake(True)
    pub, priv, st = build_split(cfg, "train", 3, gen)
    assert calls["n"] >= 2 and st["retries"] >= 24 and st["fallbacks"] <= 6
    assert all(len(e["memories"]) == 8 for e in pub)


def test_fallback_when_always_bad():
    cfg = load_config(toy=True)
    pub, priv, st = build_split(cfg, "train", 2, lambda msgs: ["bad"] * len(msgs))
    assert st["fallbacks"] == 16 and st["banks_with_fallback"] == 2
