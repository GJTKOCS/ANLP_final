"""Memory banks: plan slots (history first, no secret), write them (LLM or toy templates), validate."""
from __future__ import annotations

import random
import re
from typing import Callable

from ..common.inventory import PEOPLE, PLACES
from ..common.leak import find_inventory


def plan_bank(rng: random.Random, themes: list[str]) -> dict:
    """Return {'birthday_person', 'slots': [{people, place, theme}]*8}. Constraints: every sampled person/place
    appears; >=2 confusable theme pairs (same theme, different people/place); some repeats."""
    for _ in range(200):
        b = rng.choice(PEOPLE)
        people = rng.sample(PEOPLE, rng.randint(5, 6))
        places = rng.sample(PLACES, rng.randint(5, 6))
        th = rng.sample(themes, 6)
        slot_themes = [th[0], th[0], th[1], th[1], th[2], th[3], th[4], th[5]]
        rng.shuffle(slot_themes)
        slots = [{"people": [], "place": None, "theme": t} for t in slot_themes]
        order = list(range(8))
        rng.shuffle(order)
        for k, pl in enumerate(places):                       # each place appears at least once
            slots[order[k]]["place"] = pl
        for i in order[len(places):]:                         # leftovers: reuse a place half the time
            if rng.random() < 0.5:
                slots[i]["place"] = rng.choice(places)
        pord = list(range(8))
        rng.shuffle(pord)
        for k in range(8):                                    # 8 slots >= 6 people: all appear, then repeats
            slots[pord[k]]["people"].append(people[k] if k < len(people) else rng.choice(people))
        for s in slots:
            if rng.random() < 0.3:
                extra = rng.choice([p for p in people if p not in s["people"]])
                s["people"].append(extra)
        if _confusable_pairs(slots) >= 2:
            return {"birthday_person": b, "slots": slots}
    raise RuntimeError("could not plan a valid memory bank (constraints too tight)")


def _confusable_pairs(slots: list[dict]) -> int:
    n = 0
    for i in range(8):
        for j in range(i + 1, 8):
            a, b = slots[i], slots[j]
            if a["theme"] == b["theme"] and set(a["people"]) != set(b["people"]) and a["place"] != b["place"]:
                n += 1
    return n


def _names(slot: dict) -> str:
    p = slot["people"]
    return p[0] if len(p) == 1 else " and ".join(p)


def template_memory(slot: dict) -> str:
    where = f" at the {slot['place']}" if slot["place"] else " on one of our trips"
    return (f"We still laugh about the time {_names(slot)} {slot['theme']}{where}, "
            f"and honestly neither of us will ever let anyone forget it.")


def llm_prompt(slot: dict) -> list[dict]:
    names = _names(slot) + (f" and the place '{slot['place']}'" if slot["place"] else "")
    inv_other = ", ".join(PEOPLE + PLACES)
    sys = ("You write short, vivid memories shared by two close friends, A and B. Output only the memory: one or two "
           "sentences, past tense, from their shared point of view (use 'we' and 'you'), a single concrete, specific "
           "incident with a funny or memorable detail.")
    usr = (f"Theme: someone {slot['theme']}.\nInclude exactly these names verbatim: {names}.\n"
           f"Do not mention any other name or place from this list: {inv_other}.\nMemory:")
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def ngrams(s: str, n: int = 5) -> set[str]:
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i : i + n] for i in range(max(1, len(s) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def valid_memory(text: str, slot: dict, others: list[str], words: tuple[int, int], jmax: float) -> bool:
    ppl, plc = find_inventory(text)
    if ppl != set(slot["people"]) or plc != ({slot["place"]} if slot["place"] else set()):
        return False
    if not (words[0] <= len(text.split()) <= words[1]):
        return False
    g = ngrams(text)
    return all(jaccard(g, ngrams(o)) < jmax for o in others)


def write_banks(plans: list[dict], generate: Callable[[list[list[dict]]], list[str]] | None, cfg) -> tuple[list[list[str]], dict]:
    """Write all memories. generate=None -> toy templates. Returns (texts per bank, stats)."""
    words, jmax, tries = tuple(cfg.data.mem_words), cfg.data.jaccard_mem, cfg.data.writer_retries
    texts = [[None] * 8 for _ in plans]
    stats = {"slots": 8 * len(plans), "fallbacks": 0, "banks_with_fallback": 0, "retries": 0}
    pending = [(i, j) for i in range(len(plans)) for j in range(8)]
    fell = set()
    for attempt in range(tries if generate else 0):
        if not pending:
            break
        outs = generate([llm_prompt(plans[i]["slots"][j]) for i, j in pending])
        nxt = []
        for (i, j), out in zip(pending, outs):
            out = out.strip().split("\n")[0].strip()
            others = [t for k, t in enumerate(texts[i]) if t is not None and k != j]
            if valid_memory(out, plans[i]["slots"][j], others, words, jmax):
                texts[i][j] = out
            else:
                nxt.append((i, j))
        stats["retries"] += len(nxt)
        pending = nxt
    for i, j in pending:                                       # fallback template, logged
        texts[i][j] = template_memory(plans[i]["slots"][j])
        stats["fallbacks"] += generate is not None
        if generate is not None:
            fell.add(i)
    stats["banks_with_fallback"] = len(fell)
    return texts, stats
