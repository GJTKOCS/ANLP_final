"""Public inventories (fixed order = label order)."""
from __future__ import annotations

from .config import CONFIGS
import yaml

_inv = yaml.safe_load(open(CONFIGS / "inventory.yaml"))
PEOPLE: list[str] = list(_inv["people"])
PLACES: list[str] = list(_inv["places"])
PLACE_KEYWORDS: list[str] = list(_inv["place_keywords"])
ALL_STRINGS: list[str] = PEOPLE + PLACES


def helper_idx(name: str) -> int:
    return PEOPLE.index(name)


def venue_idx(name: str) -> int:
    return PLACES.index(name)
