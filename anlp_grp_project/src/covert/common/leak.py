"""Leak checker. Banned after the switch: every person name (incl. birthday person), every full place
name, and distinctive place keywords. Generic words (bowling, park, market, garden, cafe, bowler) are
deliberately allowed: they are public clues and catching them is C's job."""
from __future__ import annotations

import re
import unicodedata

from .inventory import PEOPLE, PLACES, PLACE_KEYWORDS


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.lower().replace("’", "'")


def _pat(term: str) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?:'s|s')?(?![a-z0-9])")


_BANNED = [(t, _pat(t)) for t in PEOPLE + PLACES + PLACE_KEYWORDS]
_FULL = [(t, _pat(t)) for t in PEOPLE + PLACES]


def find_leaks(text: str) -> list[str]:
    n = normalise(text)
    return [t for t, p in _BANNED if p.search(n)]


def has_leak(text: str) -> bool:
    return bool(find_leaks(text))


def find_inventory(text: str) -> tuple[set[str], set[str]]:
    """Full inventory strings (people, places) mentioned in text (used for memory validation)."""
    n = normalise(text)
    ppl = {t for t, p in _FULL if t in PEOPLE and p.search(n)}
    plc = {t for t, p in _FULL if t in PLACES and p.search(n)}
    return ppl, plc
