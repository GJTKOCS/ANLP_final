"""Stage implementations (data -> warm start -> judge -> listeners -> SFT -> GRPO -> transcripts)."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

from .common.artifacts import Artifact, art_dir, artifact_status, require
from .common.io import read_json, read_jsonl, write_json, write_jsonl
from .common.judge import Judge
from .common.llm import LM, pick_big_gen
from .common.seed import seed_all
from .data import episodes as E
from .data import warmstart as W
from .listeners.model import listener_cfg
from .listeners.train import ListenerPredictor, make_items, train_listener
from .mocks import MockJudge, MockListener
from .speaker import grpo as G
from .speaker import sft as S
from .speaker.reward import reward_from, score_rollouts
from .speaker.rollout import rollout

log = logging.getLogger("covert.stages")


def policy_lm(cfg, grad_ckpt: bool = False) -> LM:
    return LM(cfg.hw["policy_model"], cfg.hw, grad_ckpt=grad_ckpt and cfg.hw.get("grad_ckpt", False))


def add_adapter(lm: LM, cfg, name: str, path: Path) -> None:
    lm.new_adapter(name, cfg.listeners.lora_r, cfg.listeners.lora_alpha, 0.0)
    lm.load_adapter_weights(name, path)


def eps_map(cfg, split: str) -> dict[str, dict]:
    return {e["episode_id"]: e for e in E.load_split(cfg, split)}


def load_convs(cfg, name: str, split: str) -> list[dict]:
    return read_jsonl(require(cfg, name) / f"{split}.jsonl", "conv")


def get_tau(cfg) -> float:
    if artifact_status(cfg, "judge_calibration") == "complete":
        return float(read_json(art_dir(cfg, "judge_calibration") / "calibration.json")["tau"])
    return float(cfg.speaker.tau)


def big_lm(cfg) -> tuple[LM, str]:
    name = pick_big_gen(cfg.hw)
    log.info("big_gen model: %s", name)
    return LM(name, cfg.hw, max_bs=int(cfg.hw.get("big_gen_bs", 16))), name


def _gen_fn(lm: LM, cfg):
    def gen(msgs: list[list[dict]]) -> list[str]:
        outs = lm.generate([lm.chat_prompt(m) for m in msgs], None, 0.8, 0.95, 90, cfg.hw["gen_bs"])
        return [o["text"] for o in outs]
    return gen


# ---------------------------------------------------------------- data
def stage_data(cfg) -> None:
    if cfg.toy:
        E.build_all(cfg, None, "data build --toy", "toy-templates")
        return
    lm, name = big_lm(cfg)
    E.build_all(cfg, _gen_fn(lm, cfg), "data build", name)
    lm.close()


def stage_warmstart(cfg) -> None:
    seed_all(cfg["seed"])
    tau = float(cfg.speaker.tau)
    train, dev = E.load_split(cfg, "train"), E.load_split(cfg, "dev")
    lm, name = big_lm(cfg)
    with Artifact(cfg, "warmstart_convs", "warmstart", "data warmstart", inputs=["episodes"], models={"big_gen": name}) as art:
        convs = {"train": W.generate_convs(cfg, lm, train, cfg.data.warm_k_train, "train", name),
                 "dev": W.generate_convs(cfg, lm, dev, cfg.data.warm_k_dev, "dev", name)}
        lm.close()
        pl = policy_lm(cfg)                                     # sense judge = policy-profile base model (same as RL)
        judge = Judge(pl, cfg.hw["score_bs"])
        pl_sense = {s: judge.score([(e_by[c["episode_id"]]["cover_topic"], c["post_switch"]) for c in cs])
                    for s, cs in convs.items() for e_by in [{e["episode_id"]: e for e in (train if s == "train" else dev)}]}
        pl.close()
        lm = LM(name, cfg.hw, max_bs=int(cfg.hw.get("big_gen_bs", 16)))
        ex, excluded = W.pick_examples(convs["train"], {e["episode_id"]: e for e in train}, cfg["seed"])
        for s, eps in (("train", train), ("dev", dev)):
            by = {e["episode_id"]: e for e in eps}
            for c, v in zip(convs[s], pl_sense[s]):
                c["sense"] = float(v)
            ps = W.prompted_scores(lm, convs[s], by, ex, cfg.hw["score_bs"])
            for c, p in zip(convs[s], ps):
                c["prompted_scores"] = p
                c["good"] = W.mark_good(c, tau, excluded if s == "train" else set())
            write_jsonl(art.dir / f"{s}.jsonl", convs[s], "conv")
        lm.close()
        art.man["extra"]["headroom"] = W.headroom(convs["train"])
        art.man["extra"]["n_good"] = sum(bool(c["good"]) for c in convs["train"])
        art.man["extra"]["examples_excluded"] = sorted(excluded)


def stage_judge(cfg) -> None:
    """Calibrate tau on 50 good vs 50 corrupted warm-start conversations; re-mark 'good' flags with tau."""
    require(cfg, "warmstart_convs", "warmstart")
    train = E.load_split(cfg, "train", with_labels=False)
    by = {e["episode_id"]: e for e in train}
    convs = load_convs(cfg, "warmstart_convs", "train")
    good = [c for c in convs if c["good"]] or [c for c in convs if not c["a_leak"]] or convs
    rng = random.Random(cfg["seed"])
    n = min(50, len(good))
    good = rng.sample(good, n)
    words = "banana quantum seven purple lorry whisper turnip galaxy marble drift".split()
    bad = []
    for i, c in enumerate(rng.sample(convs, min(n, len(convs)))):
        c2 = [list(t) for t in c["post_switch"]]
        if i % 2 == 0:                                         # turns taken from a different episode
            other = rng.choice(convs)["post_switch"]
            c2 = [[s, other[k % len(other)][1]] for k, (s, _) in enumerate(c2)]
        else:                                                  # one turn replaced with random text
            c2[rng.randrange(len(c2))][1] = " ".join(rng.choices(words, k=8))
        bad.append((c["episode_id"], c2))
    with Artifact(cfg, "judge_calibration", "judge", "judge calibrate", inputs=["warmstart_convs"]) as art:
        lm = policy_lm(cfg)
        judge = Judge(lm, cfg.hw["score_bs"])
        gs = np.array(judge.score([(by[c["episode_id"]]["cover_topic"], c["post_switch"]) for c in good]))
        bs = np.array(judge.score([(by[e]["cover_topic"], p) for e, p in bad]))
        lm.close()
        table = []
        for tau in np.linspace(0.05, 0.95, 19):
            table.append({"tau": round(float(tau), 2), "keep_good": float((gs >= tau).mean()), "reject_bad": float((bs < tau).mean())})
        best = max(table, key=lambda r: (r["keep_good"] + r["reject_bad"], r["tau"]))
        write_json(art.dir / "calibration.json", {"tau": best["tau"], "table": table, "n_good": len(gs), "n_bad": len(bs)})
        write_json(Path(cfg["paths"]["runs"]) / "judge_calibration.json", {"tau": best["tau"], "table": table})
        for s in ("train", "dev"):                              # re-mark good with calibrated tau
            cs = read_jsonl(art_dir(cfg, "warmstart_convs") / f"{s}.jsonl", "conv")
            ex = set(read_json(art_dir(cfg, "warmstart_convs") / "manifest.json")["extra"].get("examples_excluded", [])) if s == "train" else set()
            for c in cs:
                c["good"] = W.mark_good(c, best["tau"], ex)
            write_jsonl(art_dir(cfg, "warmstart_convs") / f"{s}.jsonl", cs, "conv")


# ---------------------------------------------------------------- listeners
def stage_listeners(cfg, rnd: int) -> None:
    seed_all(cfg["seed"])
    train, dev = eps_map(cfg, "train"), eps_map(cfg, "dev")
    convs = load_convs(cfg, "warmstart_convs", "train")
    dev_convs = load_convs(cfg, "warmstart_convs", "dev")
    inputs = ["warmstart_convs"]
    for k in range(1, rnd + 1):                                 # cumulative: all earlier rounds' transcripts
        convs += load_convs(cfg, f"convs_A{k}", "train")
        inputs.append(f"convs_A{k}")
    if rnd >= 1:
        dev_convs = load_convs(cfg, f"convs_A{rnd}", "dev")
    tr, dv = make_items(convs, train), make_items(dev_convs, dev)
    with Artifact(cfg, f"listeners_r{rnd}", "listeners", f"listeners train --round {rnd}", inputs=inputs,
                  models={"listener": cfg.hw["listener_model"]}) as art:
        res = {}
        for view in ("B", "C"):                                 # same recipe, seed, data, steps; only the view differs
            res[view] = train_listener(cfg, view, tr, dv, art.dir / view)
        art.man["extra"] = {"listener_cfg_B": listener_cfg(cfg), "listener_cfg_C": listener_cfg(cfg),
                            "n_train": len(tr), "n_dev": len(dv), "dev_both": {v: res[v]["dev_both"] for v in res}}
        assert art.man["extra"]["listener_cfg_B"] == art.man["extra"]["listener_cfg_C"]


def load_reward_listeners(cfg, rnd: int):
    if cfg.toy and os.environ.get("COVERT_MOCK_LISTENERS") == "1":
        return MockListener("B"), MockListener("C")
    d = require(cfg, f"listeners_r{rnd}", f"listeners_r{rnd}")
    return ListenerPredictor(cfg, d / "B", "B"), ListenerPredictor(cfg, d / "C", "C")


# ---------------------------------------------------------------- speaker
def stage_sft(cfg) -> None:
    seed_all(cfg["seed"])
    eps = {e["episode_id"]: e for e in E.load_split(cfg, "train")}
    convs = [c for c in load_convs(cfg, "warmstart_convs", "train") if c["good"]]
    if not convs and cfg.toy:                                   # random toy model never passes the filters
        convs = load_convs(cfg, "warmstart_convs", "train")
    if not convs:
        raise RuntimeError("No good warm-start conversations: G2 headroom failed. See `covert check G2` for fallbacks.")
    with Artifact(cfg, "speaker_sft", "sft", "speaker sft", inputs=["warmstart_convs"], models={"policy": cfg.hw["policy_model"]}) as art:
        lm = policy_lm(cfg, grad_ckpt=True)
        ex = S.build_examples(lm, convs, eps)
        art.man["extra"] = S.train_sft(cfg, lm, ex, art.dir / "sft_adapter.pt")
        lm.close()


def base_checksum(lm: LM) -> float:
    return float(sum(p.detach().float().sum().item() for n, p in lm.model.named_parameters() if "lora_" not in n))


def stage_grpo(cfg, rnd: int, lam: float, half: bool = False) -> None:
    seed_all(cfg["seed"])
    name = f"speaker_r{rnd}_lam{lam}"
    tr, dv = E.load_split(cfg, "train"), E.load_split(cfg, "dev")
    steps = cfg.hw["grpo"]["steps"] // (2 if half else 1)
    sft_path = require(cfg, "speaker_sft", "sft") / "sft_adapter.pt"
    init = None
    if rnd > 1:
        init = require(cfg, f"speaker_r{rnd-1}_lam{lam}", f"grpo_r{rnd-1}") / "best_adapter.pt"
    with Artifact(cfg, name, "grpo", f"speaker grpo --round {rnd} --lam {lam}", inputs=["speaker_sft", f"listeners_r{rnd-1}"],
                  models={"policy": cfg.hw["policy_model"]}) as art:
        lm = policy_lm(cfg, grad_ckpt=True)
        before = base_checksum(lm)
        B, C = load_reward_listeners(cfg, rnd - 1)
        judge = MockJudge() if (cfg.toy and os.environ.get("COVERT_MOCK_LISTENERS") == "1") else Judge(lm, cfg.hw["score_bs"])
        tau = get_tau(cfg)
        wb = _wandb(cfg, name)
        res = G.run_grpo(cfg, lm, tr, dv, B, C, judge, art.dir, lam, tau, steps, sft_path, init, wb)
        after = base_checksum(lm)
        art.man["extra"] = {**res, "tau": tau, "base_checksum_before": before, "base_checksum_after": after,
                            "base_frozen": abs(before - after) < 1e-6 * max(1.0, abs(before))}
        lm.close()


def _wandb(cfg, name: str):
    if not cfg.wandb.get("enabled") or not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
        return wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=name,
                          config={k: v for k, v in cfg.items() if k not in ("inventory", "paths")})
    except Exception as e:                                      # bad key / offline: train without W&B
        log.warning("W&B disabled: %s", e)
        return None


# ---------------------------------------------------------------- conditions / transcripts
def resolve_cond(cfg, cond: str) -> str:
    """Map S3/S3a/S3b to concrete speaker conditions (A{r}[_lamL])."""
    if cond == "S3":
        r = cfg.speaker.rounds
        while r >= 1 and artifact_status(cfg, f"speaker_r{r}_lam1.0") != "complete":
            r -= 1
        if r < 1:
            raise RuntimeError("No GRPO round complete. Run `covert pipeline run --only grpo_r1`.")
        return f"A{r}"
    return {"S3a": "A1_lam0.5", "S3b": "A1_lam2.0"}.get(cond, cond)


def speaker_artifact(cond: str) -> tuple[str, str]:
    """A{r}[_lam{L}] -> (artifact name, adapter file)."""
    r, _, lam = cond[1:].partition("_lam")
    return f"speaker_r{r}_lam{float(lam) if lam else 1.0}", "best_adapter.pt"


def _records(ro, cond, rnd, lam, tau, sense) -> list[dict]:
    return [{"episode_id": r["episode_id"], "condition": cond, "round": rnd, "lam": lam, "speaker_artifact": cond,
             "sample_idx": r["group_idx"], "post_switch": r["post_switch"], "a_leak": r["a_leak"], "b_resamples": r["b_resamples"],
             "sense": float(s), "good": None, "prompted_scores": None, "gen_config_hash": cond} for r, s in zip(ro, sense)]


def stage_transcripts(cfg, cond: str) -> None:
    cond = resolve_cond(cfg, cond)
    seed_all(cfg["seed"])
    rnd = int(cond[1]) if cond.startswith("A") else 0
    lam = float(cond.partition("_lam")[2] or 1.0) if cond.startswith("A") else None
    sp = cfg.speaker
    inputs = ["episodes"]
    with Artifact(cfg, f"convs_{cond}", "transcripts", f"transcripts --cond {cond}", inputs=inputs) as art:
        b_lm = policy_lm(cfg)
        a_lm, a_adapter = b_lm, None
        if cond == "S1":
            add_adapter(b_lm, cfg, "sft", require(cfg, "speaker_sft", "sft") / "sft_adapter.pt")
            a_adapter = "sft"
        elif cond.startswith("A"):
            n, f = speaker_artifact(cond)
            add_adapter(b_lm, cfg, "policy", require(cfg, n, f"grpo_r{cond[1]}") / f)
            a_adapter = "policy"
        elif cond == "S4":
            a_lm, big = big_lm(cfg)
            art.man["models"] = {"S4_big_gen": big}
        judge = Judge(b_lm, cfg.hw["score_bs"]) if not (cfg.toy and os.environ.get("COVERT_MOCK_LISTENERS") == "1") else MockJudge()
        tau = get_tau(cfg)
        B = C = None
        if cond == "S2":
            B, C = load_reward_listeners(cfg, 0)
        for split in ("train", "dev", "test"):
            eps = E.load_split(cfg, split)
            by = {e["episode_id"]: e for e in eps}
            if cond == "S2":                                    # best-of-N from S0 under the round-0 reward
                N = cfg.eval.best_of_n
                ro = rollout(a_lm, b_lm, eps, N, cfg, None, None, temp_a=sp.transcript_temp)
                score_rollouts(ro, by, B, C, judge, sp.lam, tau, sp.reward_mode)
                best = {}
                for r in ro:
                    if r["episode_id"] not in best or r["reward"] > best[r["episode_id"]]["reward"]:
                        best[r["episode_id"]] = r
                ro = [best[e["episode_id"]] for e in eps]
                for r in ro:
                    r["group_idx"] = 0
                sense = [r["sense"] for r in ro]
            else:
                ro = rollout(a_lm, b_lm, eps, 1, cfg, a_adapter, None, temp_a=sp.transcript_temp)
                sense = judge.score([(by[r["episode_id"]]["cover_topic"], r["post_switch"]) for r in ro])
            write_jsonl(art.dir / f"{split}.jsonl", _records(ro, cond, rnd, lam, tau, sense), "conv")
        if a_lm is not b_lm:
            a_lm.close()
        b_lm.close()
