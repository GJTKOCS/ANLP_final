"""Supervised warm start: LoRA adapter 'sft' trained on good warm-start conversations, loss on A-turn tokens only."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from ..common import views
from ..common.device import make_scaler

log = logging.getLogger("covert.sft")


def build_examples(lm, convs: list[dict], eps: dict[str, dict]) -> list[tuple[list[int], list[int]]]:
    out = []
    for c in convs:
        e = eps[c["episode_id"]]
        post = c["post_switch"]
        for i, (spk, text) in enumerate(post):
            if spk != "A":
                continue
            prompt = lm.chat_prompt(views.a_messages(e, e["secret"], post[:i]))
            p = lm.tok(prompt, add_special_tokens=False)["input_ids"]
            t = lm.tok(text + lm.end_text, add_special_tokens=False)["input_ids"]
            out.append((p, t))
    return out


def train_sft(cfg, lm, examples: list[tuple[list[int], list[int]]], out_path: Path) -> dict:
    sp = cfg.speaker
    lm.new_adapter("sft", cfg.listeners.lora_r, cfg.listeners.lora_alpha, cfg.listeners.lora_dropout)
    params = lm.adapter_params("sft")
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=sp.sft_lr)
    mb = cfg.hw["sft_mb"]
    steps = max(1, sp.sft_epochs * ((len(examples) + mb - 1) // mb))
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    scaler = make_scaler(lm.dtype, lm.device)
    rng = random.Random(cfg["seed"])
    losses = []
    lm.peft.train()
    for ep in range(sp.sft_epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for s in range(0, len(order), mb):
            idx = order[s : s + mb]
            lp, mask = lm.logprobs([examples[i][0] for i in idx], [examples[i][1] for i in idx], "sft", grad=True)
            m = mask.float()
            loss = -((lp * m).sum(1) / m.sum(1).clamp(min=1)).mean()
            opt.zero_grad()
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt) if scaler else opt.step()
            if scaler:
                scaler.update()
            sched.step()
            losses.append(loss.item())
        log.info("sft epoch %d loss %.4f", ep, sum(losses[-10:]) / max(1, len(losses[-10:])))
    lm.peft.eval()
    lm.save_adapter("sft", out_path)
    return {"final_loss": losses[-1] if losses else None, "n_examples": len(examples)}
