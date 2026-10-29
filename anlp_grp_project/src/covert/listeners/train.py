"""Train / predict listeners. Same code for reward listeners and evaluation listeners."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from ..common import views
from ..common.device import make_scaler
from ..common.inventory import PEOPLE, PLACES
from ..common.io import read_json, write_json
from .model import build_listener, listener_cfg

log = logging.getLogger("covert.listeners")


def make_items(convs: list[dict], eps: dict[str, dict]) -> list[dict]:
    """Join transcripts with episodes+labels -> training items (labels only here, from label-merged eps)."""
    out = []
    for c in convs:
        e = eps[c["episode_id"]]
        out.append({"memories": e["memories"], "post_switch": c["post_switch"],
                    "helper": PEOPLE.index(e["secret"]["helper"]), "venue": PLACES.index(e["secret"]["venue"])})
    return out


def _texts(view: str, items: list[dict]) -> list[str]:
    return [views.listener_text(view, it["memories"] if view == "B" else None, it["post_switch"]) for it in items]


def _encode(tok, texts, max_len, device):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def predict_probs(model, tok, device, view: str, items: list[dict], cfg, bs: int | None = None):
    model.eval()
    bs = bs or cfg.hw["listener_mb"] * 2
    texts = _texts(view, items)
    H, V = [], []
    for i in range(0, len(texts), bs):
        ids, att = _encode(tok, texts[i : i + bs], cfg.listeners.max_len, device)
        h, v = model(ids, att)
        H.append(torch.softmax(h, -1).cpu().numpy())
        V.append(torch.softmax(v, -1).cpu().numpy())
    if not H:
        return np.zeros((0, 10)), np.zeros((0, 10))
    return np.concatenate(H), np.concatenate(V)


def both_correct(model, tok, device, view, items, cfg) -> float:
    h, v = predict_probs(model, tok, device, view, items, cfg)
    ok = (h.argmax(1) == np.array([i["helper"] for i in items])) & (v.argmax(1) == np.array([i["venue"] for i in items]))
    return float(ok.mean()) if len(ok) else 0.0


def train_listener(cfg, view: str, train_items: list[dict], dev_items: list[dict], out_dir: Path) -> dict:
    """Identical hyperparameters, seed, data order for both views. Early-stops on dev both-correct (per epoch)."""
    lc = listener_cfg(cfg)
    seed = lc["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    model, tok, device = build_listener(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lc["lr"], weight_decay=0.0)
    mb, ga = cfg.hw["listener_mb"], cfg.hw["listener_ga"]
    steps = max(1, lc["epochs"] * ((len(train_items) + mb * ga - 1) // (mb * ga)))
    sched = get_cosine_schedule_with_warmup(opt, int(lc["warmup"] * steps), steps)
    scaler = make_scaler(next(model.parameters()).dtype, device)
    texts = _texts(view, train_items)
    ys_h = torch.tensor([i["helper"] for i in train_items])
    ys_v = torch.tensor([i["venue"] for i in train_items])
    best, best_state, hist = -1.0, None, []
    rng = random.Random(seed)
    for ep in range(lc["epochs"]):
        model.train()
        order = list(range(len(texts)))
        rng.shuffle(order)
        for s in range(0, len(order), mb * ga):
            big = order[s : s + mb * ga]
            opt.zero_grad()
            for m in range(0, len(big), mb):
                idx = big[m : m + mb]
                ids, att = _encode(tok, [texts[i] for i in idx], lc["max_len"], device)
                h, v = model(ids, att)
                loss = (F.cross_entropy(h, ys_h[idx].to(device)) + F.cross_entropy(v, ys_v[idx].to(device))) * (len(idx) / len(big))
                (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt) if scaler else opt.step()
            if scaler:
                scaler.update()
            sched.step()
        acc = both_correct(model, tok, device, view, dev_items, cfg) if dev_items else 0.0
        hist.append({"epoch": ep, "dev_both": acc})
        log.info("listener %s epoch %d dev both-correct %.3f", view, ep, acc)
        if acc > best:
            best, best_state = acc, model.trainable_state()
    model.load_state_dict(best_state, strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "model.pt")
    write_json(out_dir / "info.json", {"view": view, "dev_both": best, "history": hist, "cfg": lc,
                                       "n_train": len(train_items), "n_dev": len(dev_items)})
    del model
    return {"dev_both": best, "history": hist}


class ListenerPredictor:
    """Frozen listener for scoring. .probs(items) -> (helper [N,10], venue [N,10]); C ignores memories."""

    def __init__(self, cfg, dir: Path, view: str):
        self.cfg, self.view = cfg, view
        self.model, self.tok, self.device = build_listener(cfg)
        self.model.load_state_dict(torch.load(dir / "model.pt", map_location="cpu"), strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def probs(self, items: list[dict]):
        return predict_probs(self.model, self.tok, self.device, self.view, items, self.cfg)

    def close(self) -> None:
        from ..common.device import free_memory
        del self.model
        free_memory()
