"""The ONLY module allowed to build model inputs. Enforces the information boundaries:

party              sees                                              never sees
A speaker          memories, secret, b, pre-switch, topic, chat      -
B speaker          memories, b, pre-switch, topic, chat              secret
B listener         memories, post-switch chat                        secret, pre-switch, prompts
C listener         post-switch chat                                  memories, secret, pre-switch
prompted C         post-switch chat, rules, inventories, 3 examples  memories, secret, pre-switch
sense judge        post-switch chat, cover topic                     everything else
"""
from __future__ import annotations

from . import prompts as P
from .inventory import ALL_STRINGS, PEOPLE, PLACES
from .leak import find_inventory

Turn = tuple[str, str]


def fmt_chat(turns: list) -> str:
    return "\n".join(f"{s}: {t}" for s, t in turns)


def fmt_memories(memories: list[str]) -> str:
    return "\n".join(f"{i+1}. {m}" for i, m in enumerate(memories))


def _banned() -> str:
    return ", ".join(ALL_STRINGS)


def _convo(ep: dict, post: list) -> str:
    pre = fmt_chat(ep["pre_switch"])
    join = f"[C joins the conversation. Cover topic: {ep['cover_topic']}]"
    return "\n".join(x for x in (pre, join, fmt_chat(post)) if x)


def _pointers(memories: list[str], secret: dict) -> str:
    """Which memories mention the helper / the venue (derived from the memory text A already sees)."""
    h = [i + 1 for i, m in enumerate(memories) if secret["helper"] in find_inventory(m)[0]]
    v = [i + 1 for i, m in enumerate(memories) if secret["venue"] in find_inventory(m)[1]]
    fmt = lambda xs: ", ".join(f"#{x}" for x in xs) if xs else "none"
    return f"Memories involving the helper {secret['helper']}: {fmt(h)}. Memories that took place at the venue {secret['venue']}: {fmt(v)}."


def a_messages(ep: dict, secret: dict, post: list) -> list[dict]:
    sys = P.A_SYSTEM.format(memories=fmt_memories(ep["memories"]), b=ep["birthday_person"], helper=secret["helper"],
                            venue=secret["venue"], cover_topic=ep["cover_topic"], banned=_banned(),
                            pointers=_pointers(ep["memories"], secret))
    return [{"role": "system", "content": sys}, {"role": "user", "content": _convo(ep, post)}]


def b_speaker_messages(ep: dict, post: list) -> list[dict]:
    sys = P.B_SYSTEM.format(memories=fmt_memories(ep["memories"]), b=ep["birthday_person"],
                            cover_topic=ep["cover_topic"], banned=_banned())
    return [{"role": "system", "content": sys}, {"role": "user", "content": _convo(ep, post)}]


def listener_text(view: str, memories: list[str] | None, post: list) -> str:
    """B view needs memories; C view must not receive them."""
    if view == "B":
        if memories is None:
            raise ValueError("B view requires memories")
        return P.LISTENER_B.format(memories=fmt_memories(memories), chat=fmt_chat(post))
    if view == "C":
        return P.LISTENER_C.format(chat=fmt_chat(post))
    raise ValueError(view)


def judge_text(cover_topic: str, post: list) -> str:
    return P.JUDGE.format(cover_topic=cover_topic, chat=fmt_chat(post))


def prompted_b_text(memories: list[str], post: list) -> str:
    return P.PROMPTED_B.format(memories=fmt_memories(memories), people=", ".join(PEOPLE), places=", ".join(PLACES),
                               chat=fmt_chat(post))


def prompted_c_text(post: list, examples: list[tuple[list, dict]]) -> str:
    """examples: [(post_switch, secret)] solved training examples."""
    s = P.PROMPTED_C_RULES.format(people=", ".join(PEOPLE), places=", ".join(PLACES))
    for chat, sec in examples:
        s += P.PROMPTED_C_EXAMPLE.format(chat=fmt_chat(chat), helper=sec["helper"], venue=sec["venue"])
    return s + P.PROMPTED_C_QUERY.format(chat=fmt_chat(post))


def prompted_c_alt_text(post: list, examples: list[tuple[list, dict]]) -> str:
    """Held-out outsider: chat + reworded rules + different solved examples; no memories."""
    s = P.PROMPTED_C_ALT_RULES.format(people=", ".join(PEOPLE), places=", ".join(PLACES))
    for chat, sec in examples:
        s += P.PROMPTED_C_ALT_EXAMPLE.format(chat=fmt_chat(chat), helper=sec["helper"], venue=sec["venue"])
    return s + P.PROMPTED_C_ALT_QUERY.format(chat=fmt_chat(post))
