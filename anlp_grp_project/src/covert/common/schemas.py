"""Minimal record schemas; invalid records fail loudly."""
from __future__ import annotations

REQUIRED = {
    "episode_public": ["episode_id", "split", "birthday_person", "memories", "cover_topic", "pre_switch", "theme_ids"],
    "episode_private": ["episode_id", "secret", "anchors", "seed"],
    "conv": ["episode_id", "condition", "round", "lam", "sample_idx", "post_switch", "a_leak", "b_resamples", "sense"],
    "pred": ["episode_id", "view", "pred", "probs"],
}


def validate(kind: str, rec: dict, where: str = "") -> None:
    missing = [k for k in REQUIRED[kind] if k not in rec]
    if missing:
        raise ValueError(f"{where}: {kind} record missing fields {missing}")
    if kind == "episode_public":
        if len(rec["memories"]) != 8:
            raise ValueError(f"{where}: expected 8 memories, got {len(rec['memories'])}")
    if kind in ("conv",):
        for t in rec["post_switch"]:
            if not (isinstance(t, (list, tuple)) and len(t) == 2 and t[0] in ("A", "B")):
                raise ValueError(f"{where}: bad turn {t!r}")
    if kind == "episode_private":
        if set(rec["secret"]) != {"helper", "venue"}:
            raise ValueError(f"{where}: secret must have helper and venue")
    if kind == "pred":
        for s in ("helper", "venue"):
            if len(rec["probs"][s]) != 10:
                raise ValueError(f"{where}: probs[{s}] must have 10 entries")
