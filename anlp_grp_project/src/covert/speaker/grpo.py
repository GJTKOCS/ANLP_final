"""Custom multi-turn GRPO (no TRL). Loss on A-turn tokens only: PPO-clipped surrogate + beta * k3 KL to the
SFT reference adapter. Both A turns of a rollout share the rollout's group-normalised advantage."""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..common.device import make_scaler
from ..common.io import append_jsonl, read_json, write_json
from .reward import score_rollouts
from .rollout import rollout

log = logging.getLogger("covert.grpo")


def group_advantages(rewards: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """rewards [P, G] -> advantages [P, G]; zero-variance groups give exactly 0."""
    r = np.asarray(rewards, dtype=np.float64)
    return (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + eps) * (r.std(1, keepdims=True) > 1e-8)


def grpo_loss(lp_new, lp_old, lp_ref, adv, mask, clip_eps: float, beta: float):
    """lp_* [N, T] token log-probs; adv [N]; mask [N, T] (A tokens). Token-mean within sequence, then mean over sequences."""
    ratio = torch.exp(lp_new - lp_old)
    a = adv.unsqueeze(1)
    surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    d = lp_ref - lp_new
    kl = torch.exp(d) - d - 1                                   # k3 estimator
    per_tok = -(surr) + beta * kl
    m = mask.float()
    per_seq = (per_tok * m).sum(1) / m.sum(1).clamp(min=1)
    return per_seq.mean(), {"kl": ((kl * m).sum(1) / m.sum(1).clamp(min=1)).mean().item()}


def _dev_reward(a_lm, eps_dev, B, C, judge, cfg, lam, tau, n) -> dict:
    sub = eps_dev[:n]
    ro = rollout(a_lm, a_lm, sub, 1, cfg, "policy")
    score_rollouts(ro, {e["episode_id"]: e for e in sub}, B, C, judge, lam, tau, cfg.speaker.reward_mode)
    return {"reward": float(np.mean([r["reward"] for r in ro])), "pB": float(np.mean([np.mean(list(r["pB"].values())) for r in ro])),
            "pC": float(np.mean([np.mean(list(r["pC"].values())) for r in ro]))}


def run_grpo(cfg, a_lm, eps_train, eps_dev, B, C, judge, out_dir: Path, lam: float, tau: float, steps: int,
             sft_path: Path | None, init_path: Path | None = None, wandb_run=None) -> dict:
    """a_lm: LM with PEFT. Adapters: 'policy' (trainable), 'sft' (frozen KL reference; None => base model)."""
    sp, g = cfg.speaker, cfg.hw["grpo"]
    P, G, mb = g["P"], g["G"], cfg.hw["grpo_mb"]
    out_dir.mkdir(parents=True, exist_ok=True)
    a_lm.new_adapter("policy", cfg.listeners.lora_r, cfg.listeners.lora_alpha, 0.0)
    if sp.init == "sft" and sft_path is not None:
        a_lm.new_adapter("sft", cfg.listeners.lora_r, cfg.listeners.lora_alpha, 0.0)
        a_lm.load_adapter_weights("sft", sft_path)
        a_lm.load_adapter_weights("policy", init_path or sft_path)
        ref = "sft"
    else:
        ref = None
    params = a_lm.adapter_params("policy")
    opt = torch.optim.AdamW(params, lr=sp.lr)
    scaler = make_scaler(a_lm.dtype, a_lm.device)
    step0, best = 0, -1e9
    ck = out_dir / "ckpt.pt"
    if ck.exists():                                             # resume
        st = torch.load(ck, map_location="cpu")
        a_lm.load_adapter_weights("policy", out_dir / "latest_adapter.pt")
        opt.load_state_dict(st["opt"])
        step0, best = st["step"], st["best"]
        log.info("resumed GRPO at step %d", step0)
    eps_by = {e["episode_id"]: e for e in eps_train}
    rng_seed = cfg["seed"]
    for step in range(step0, steps):
        t0 = time.time()
        rng = random.Random(rng_seed + step)
        eps = rng.sample(eps_train, min(P, len(eps_train)))
        ro = rollout(a_lm, a_lm, eps, G, cfg, "policy", record_logp=True)
        score_rollouts(ro, eps_by, B, C, judge, lam, tau, sp.reward_mode)
        R = np.array([r["reward"] for r in ro]).reshape(len(eps), G)
        A = group_advantages(R).reshape(-1)
        zero_frac = float((R.std(1) < 1e-8).mean())
        seqs = []                                               # one sequence per A turn
        for r, adv in zip(ro, A):
            r["advantage"] = float(adv)
            for rec in r["a_recs"]:
                if adv != 0 and len(rec["comp_ids"]) > 0:
                    seqs.append((rec, float(adv)))
        a_lm.peft.eval()                                        # reference log-probs BEFORE any policy forward
        for i in range(0, len(seqs), mb):
            chunk = seqs[i : i + mb]
            lp_r, mk = a_lm.logprobs([c[0]["prompt_ids"] for c in chunk], [c[0]["comp_ids"] for c in chunk], ref)
            for j, (rec, _) in enumerate(chunk):
                rec["ref_logp"] = lp_r[j][mk[j]].cpu()
        a_lm.peft.set_adapter("policy")
        for p_ in params:
            p_.requires_grad_(True)
        kls = []
        a_lm.peft.train()
        for _ in range(sp.inner_epochs):
            opt.zero_grad()
            n = max(1, len(seqs))
            for i in range(0, len(seqs), mb):
                chunk = seqs[i : i + mb]
                pids, cids = [c[0]["prompt_ids"] for c in chunk], [c[0]["comp_ids"] for c in chunk]
                lp_new, mask = a_lm.logprobs(pids, cids, "policy", grad=True)
                lp_old = torch.zeros_like(lp_new)
                lp_ref = torch.zeros_like(lp_new)
                for j, (rec, _) in enumerate(chunk):
                    lp_old[j][mask[j]] = rec["old_logp"].to(lp_new.device, lp_new.dtype)
                    lp_ref[j][mask[j]] = rec["ref_logp"].to(lp_new.device, lp_new.dtype)
                adv = torch.tensor([c[1] for c in chunk], device=lp_new.device, dtype=lp_new.dtype)
                loss, st = grpo_loss(lp_new, lp_old, lp_ref, adv, mask, sp.clip_eps, sp.beta)
                loss = loss * (len(chunk) / n)
                (scaler.scale(loss) if scaler else loss).backward()
                kls.append(st["kl"])
            if seqs:
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt) if scaler else opt.step()
                if scaler:
                    scaler.update()
        a_lm.peft.eval()
        toks = sum(len(rec["comp_ids"]) for r in ro for rec in r["a_recs"])
        m = {"step": step + 1, "reward": float(R.mean()), "pB": float(np.mean([np.mean(list(r["pB"].values())) for r in ro])),
             "pC": float(np.mean([np.mean(list(r["pC"].values())) for r in ro])),
             "leak_rate": float(np.mean([r["a_leak"] for r in ro])), "sense_rate": float(np.mean([r["sense"] >= tau for r in ro])),
             "kl": float(np.mean(kls)) if kls else 0.0, "zero_var_frac": zero_frac, "tok_per_s": toks / max(1e-6, time.time() - t0), "lam": lam}
        append_jsonl(out_dir / "metrics.jsonl", m)
        if wandb_run:
            wandb_run.log(m)
        for r in ro:
            rec = {"step": step + 1, "episode_id": r["episode_id"], "group_idx": r["group_idx"], "post_switch": r["post_switch"],
                   "reward": r["reward"], "pB": r["pB"], "pC": r["pC"], "sense": r["sense"], "a_leak": r["a_leak"], "advantage": r["advantage"]}
            append_jsonl(out_dir / "rollouts.jsonl", rec)
            sec = eps_by[r["episode_id"]]["secret"]
            tags = [t for t, c in (("leak", r["a_leak"]), ("sense_fail", r["sense"] < tau),
                                   ("B_wrong", np.mean(list(r["pB"].values())) < 0.3), ("C_right", np.mean(list(r["pC"].values())) > 0.5)) if c]
            if tags:
                append_jsonl(out_dir / "bloopers.jsonl", {**rec, "tags": tags, "secret": sec})
        log.info("step %d reward %.3f pB %.3f pC %.3f leak %.2f sense %.2f kl %.4f", m["step"], m["reward"], m["pB"], m["pC"], m["leak_rate"], m["sense_rate"], m["kl"])
        if (step + 1) % sp.dev_every == 0 or step + 1 == steps:
            d = _dev_reward(a_lm, eps_dev, B, C, judge, cfg, lam, tau, g["dev_n"])
            append_jsonl(out_dir / "dev_metrics.jsonl", {"step": step + 1, **d})
            if d["reward"] > best:
                best = d["reward"]
                a_lm.save_adapter("policy", out_dir / "best_adapter.pt")
        if (step + 1) % sp.save_every == 0 or step + 1 == steps:
            a_lm.save_adapter("policy", out_dir / "latest_adapter.pt")
            torch.save({"opt": opt.state_dict(), "step": step + 1, "best": best}, ck)
    if wandb_run:
        wandb_run.finish()
    if not (out_dir / "best_adapter.pt").exists():
        a_lm.save_adapter("policy", out_dir / "best_adapter.pt")
    return {"best_dev_reward": best, "steps": steps}
