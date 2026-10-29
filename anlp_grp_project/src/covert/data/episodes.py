"""Episode construction, splits, duplicate checks, and label-aware loading."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml

from ..common.artifacts import Artifact, art_dir, require
from ..common.config import CONFIGS
from ..common.inventory import PEOPLE, PLACES
from ..common.io import read_json, read_jsonl, write_json, write_jsonl
from ..common.leak import find_inventory
from .memories import jaccard, ngrams, plan_bank, write_banks


def _seed(base: int, split: str, idx: int) -> int:
    return int(hashlib.sha1(f"{base}-{split}-{idx}".encode()).hexdigest()[:8], 16)


def build_split(cfg, split: str, n: int, generate=None) -> tuple[list[dict], list[dict], dict]:
    themes = yaml.safe_load(open(CONFIGS / "themes.yaml"))
    topics = yaml.safe_load(open(CONFIGS / "cover_topics.yaml"))
    pre = yaml.safe_load(open(CONFIGS / "pre_switch.yaml"))
    th_pool = themes["test"] if split == "test" else themes["train"]
    tp_pool = topics["test"] if split == "test" else topics["train"]
    plans, rngs = [], []
    for i in range(n):
        sd = _seed(cfg["seed"], split, i)
        rng = random.Random(sd)
        plans.append(plan_bank(rng, th_pool))
        rngs.append((sd, rng))
    texts, stats = write_banks(plans, generate, cfg)
    pub, priv = [], []
    for i, (plan, mem, (sd, rng)) in enumerate(zip(plans, texts, rngs)):
        order = list(range(8))
        rng.shuffle(order)                                      # shuffle memory order last
        mems = [mem[k] for k in order]
        slots = [plan["slots"][k] for k in order]
        b = plan["birthday_person"]
        mentioned = [find_inventory(m) for m in mems]           # draw secret AFTER the history exists
        ppl = sorted({p for pp, _ in mentioned for p in pp if p != b})
        plc = sorted({p for _, pl in mentioned for p in pl})
        if not ppl or not plc:
            raise RuntimeError(f"episode {i}: bank mentions no eligible helper/venue")
        helper, venue = rng.choice(ppl), rng.choice(plc)
        eid = f"{split}-{i:06d}"
        lines = rng.choice(pre)
        pre_switch = [["A", lines[0].format(b=b)], ["B", lines[1].format(b=b)], ["A", lines[2].format(b=b)]]
        pub.append({"episode_id": eid, "split": split, "birthday_person": b, "memories": mems,
                    "cover_topic": rng.choice(tp_pool), "pre_switch": pre_switch, "theme_ids": [s["theme"] for s in slots]})
        priv.append({"episode_id": eid, "secret": {"helper": helper, "venue": venue},
                     "anchors": {"helper": [k for k, (pp, _) in enumerate(mentioned) if helper in pp],
                                 "venue": [k for k, (_, pl) in enumerate(mentioned) if venue in pl]}, "seed": sd})
    return pub, priv, stats


def check_duplicates(splits: dict[str, list[dict]], jmax: float) -> None:
    cat = {s: [" ".join(e["memories"]) for e in eps] for s, eps in splits.items()}
    grams = {s: [ngrams(t) for t in ts] for s, ts in cat.items()}
    exact = {}
    for s, ts in cat.items():
        for t in ts:
            if exact.setdefault(t, s) != s:
                raise RuntimeError(f"Exact duplicate memory bank across splits {exact[t]} and {s}")
    for a, b in (("dev", "train"), ("test", "train"), ("test", "dev")):
        for i, ga in enumerate(grams[a]):
            for j, gb in enumerate(grams[b]):
                if jaccard(ga, gb) > jmax:
                    raise RuntimeError(f"Near-duplicate banks: {a}-{i} vs {b}-{j} (Jaccard>{jmax})")


def build_all(cfg, generate=None, command: str = "data build", model_name: str = "toy-templates") -> None:
    d = cfg.data
    with Artifact(cfg, "episodes", "data", command, models={"memory_writer": model_name}) as art:
        splits, stats_all = {}, {}
        for split, n in (("train", d.n_train), ("dev", d.n_dev), ("test", d.n_test)):
            pub, priv, st = build_split(cfg, split, n, generate)
            splits[split] = pub
            stats_all[split] = st
            write_jsonl(art.dir / f"{split}_public.jsonl", pub, "episode_public")
            (art.dir / "private").mkdir(exist_ok=True)
            write_jsonl(art.dir / "private" / f"{split}_private.jsonl", priv, "episode_private")
        check_duplicates(splits, d.jaccard_split)
        rng = random.Random(cfg["seed"] + 99)
        ids = [e["episode_id"] for e in splits["test"]]
        write_json(art.dir / "showcase_ids.json", sorted(rng.sample(ids, min(d.n_showcase, len(ids)))))
        art.man["extra"]["writer_stats"] = stats_all
        total, fb = sum(s["slots"] for s in stats_all.values()), sum(s["fallbacks"] for s in stats_all.values())
        banks = sum(len(v) for v in splits.values())
        bad = sum(s["banks_with_fallback"] for s in stats_all.values())
        art.man["extra"]["g1"] = {"bank_pass_rate": 1 - bad / banks, "slot_fallback_rate": fb / total}


def load_split(cfg, split: str, with_labels: bool = True) -> list[dict]:
    """Episodes of a split; with_labels merges secret/anchors from the PRIVATE file (label-loading code only)."""
    d = require(cfg, "episodes", "data")
    pub = read_jsonl(d / f"{split}_public.jsonl", "episode_public")
    if not with_labels:
        return pub
    priv = {r["episode_id"]: r for r in read_jsonl(d / "private" / f"{split}_private.jsonl", "episode_private")}
    return [{**e, "secret": priv[e["episode_id"]]["secret"], "anchors": priv[e["episode_id"]]["anchors"]} for e in pub]


def load_showcase_ids(cfg) -> list[str]:
    return read_json(require(cfg, "episodes", "data") / "showcase_ids.json")


def label_idx(secret: dict) -> tuple[int, int]:
    return PEOPLE.index(secret["helper"]), PLACES.index(secret["venue"])
