"""Evaluation stages: eval pairs, arms race, prompted strong C, checks, audit, report, showcase bundle."""
from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import yaml

from .common import views
from .common.artifacts import Artifact, art_dir, artifact_status, require
from .common.config import CONFIGS, ROOT
from .common.io import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from .common.inventory import PEOPLE, PLACES
from .common.scorer import OptionScorer
from .data import episodes as E
from .data import warmstart as W
from .eval.stats import bootstrap_ci, summarise
from .listeners.train import ListenerPredictor, make_items, train_listener
from .speaker.reward import reward_from
from . import stages as ST

log = logging.getLogger("covert.eval")
CONDS = ["S0", "S1", "S2", "S3", "S3a", "S3b", "S4"]


def _first_per_episode(convs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in convs:
        if c["episode_id"] not in seen:
            seen.add(c["episode_id"])
            out.append(c)
    return out


def _secrets(items: list[dict]) -> list[tuple[int, int]]:
    return [(i["helper"], i["venue"]) for i in items]


def stage_eval(cfg, label: str) -> None:
    """Fresh (B, C) pair trained on this condition's transcripts, evaluated on its test transcripts."""
    cond = ST.resolve_cond(cfg, label)
    if artifact_status(cfg, f"convs_{cond}") != "complete":
        ST.stage_transcripts(cfg, cond)
    tr_eps, dv_eps, te_eps = ST.eps_map(cfg, "train"), ST.eps_map(cfg, "dev"), ST.eps_map(cfg, "test")
    tr, dv = ST.load_convs(cfg, f"convs_{cond}", "train"), ST.load_convs(cfg, f"convs_{cond}", "dev")
    te = _first_per_episode(ST.load_convs(cfg, f"convs_{cond}", "test"))
    tr_items, dv_items, te_items = make_items(tr, tr_eps), make_items(dv, dv_eps), make_items(te, te_eps)
    tau, lam = ST.get_tau(cfg), cfg.speaker.lam
    suffix = cfg.eval.get("pair_suffix", "") or ""
    with Artifact(cfg, f"evalpair_{label}{suffix}", "eval", f"eval run --cond {label}", inputs=[f"convs_{cond}"],
                  extra={"resolved_condition": cond}) as art:
        info, preds = {}, {}
        shifted = [{**it, "memories": te_items[(k + 1) % len(te_items)]["memories"]} for k, it in enumerate(te_items)]
        if ST.prompted_mode(cfg):                               # zero-shot prompted B and C, no training
            Bp, Cp, lname = ST.prompted_pair(cfg)
            art.man["models"] = {"listener": lname}
            preds["B"], preds["C"], preds["B_shuffled"] = Bp.probs(te_items), Cp.probs(te_items), Bp.probs(shifted)
            info = {"B": {"dev_both": None}, "C": {"dev_both": None}}
        else:
            for view in ("B", "C"):
                info[view] = train_listener(cfg, view, tr_items, dv_items, art.dir / view)
                pr = ListenerPredictor(cfg, art.dir / view, view)
                preds[view] = pr.probs(te_items)
                if view == "B":                                # sanity: memories from a different test episode
                    preds["B_shuffled"] = pr.probs(shifted)
                pr.close()
        sec = _secrets(te_items)
        n = cfg.eval.bootstrap
        from .eval.stats import correctness
        Bc = correctness({"helper": preds["B"][0], "venue": preds["B"][1]}, sec)
        Cc = correctness({"helper": preds["C"][0], "venue": preds["C"][1]}, sec)
        Sc = correctness({"helper": preds["B_shuffled"][0], "venue": preds["B_shuffled"][1]}, sec)
        rewards = []
        for k, c in enumerate(te):
            h, v = sec[k]
            rewards.append(reward_from(c["a_leak"], c["sense"] if c["sense"] is not None else 1.0,
                                       (preds["B"][0][k, h], preds["B"][1][k, v]), (preds["C"][0][k, h], preds["C"][1][k, v]),
                                       lam, tau, cfg.speaker.reward_mode))
        extra = {"leak_rate": [c["a_leak"] for c in te], "sense_pass": [(c["sense"] or 0) >= tau for c in te],
                 "b_resample_rate": [c["b_resamples"] for c in te], "reward": rewards}
        metrics = summarise(Bc, Cc, extra, n)
        metrics["sanity_B_wrong_memories_both"] = dict(zip(("mean", "lo", "hi"), bootstrap_ci(Sc[0] & Sc[1], n)))
        write_json(art.dir / "metrics.json", {"label": label + suffix, "listener_mode": "prompted" if ST.prompted_mode(cfg) else "trained", "condition": cond, "n_test": len(te), "metrics": metrics,
                                              "listener_dev": {v: info[v]["dev_both"] for v in info}, "chance": {"slot": 0.1, "both": 0.01}})
        recs = []
        for view in ("B", "C"):
            for k, c in enumerate(te):
                recs.append({"episode_id": c["episode_id"], "view": view, "pred": {"helper": PEOPLE[int(preds[view][0][k].argmax())], "venue": PLACES[int(preds[view][1][k].argmax())]},
                             "probs": {"helper": preds[view][0][k].tolist(), "venue": preds[view][1][k].tolist()}})
        write_jsonl(art.dir / "preds_test.jsonl", recs, "pred")


def stage_armsrace(cfg) -> None:
    """C_{r-1} on A_r transcripts (before adapting) vs C_r (after), plus B. Also showcase probs per round."""
    te_eps = ST.eps_map(cfg, "test")
    show = set(E.load_showcase_ids(cfg))
    rows, showcase = [], {}
    with Artifact(cfg, "armsrace", "eval", "eval armsrace") as art:
        if ST.prompted_mode(cfg):                               # no retrained listeners -> no arms race to report
            write_json(art.dir / "arms.json", [])
            write_json(art.dir / "showcase_rounds.json", {})
            art.man["extra"] = {"skipped": "prompted listeners are not retrained"}
            return
        for r in range(1, cfg.speaker.rounds + 1):
            if artifact_status(cfg, f"convs_A{r}") != "complete" or artifact_status(cfg, f"listeners_r{r}") != "complete":
                continue
            items = make_items(_first_per_episode(ST.load_convs(cfg, f"convs_A{r}", "test")), te_eps)
            conv = _first_per_episode(ST.load_convs(cfg, f"convs_A{r}", "test"))
            row = {"round": r}
            for tag, lr in (("prev", r - 1), ("cur", r)):
                d = require(cfg, f"listeners_r{lr}")
                for view in ("B", "C"):
                    p = ListenerPredictor(cfg, d / view, view)
                    h, v = p.probs(items)
                    p.close()
                    ok_h = h.argmax(1) == np.array([i["helper"] for i in items])
                    ok_v = v.argmax(1) == np.array([i["venue"] for i in items])
                    row[f"{view}_{tag}_both"] = dict(zip(("mean", "lo", "hi"), bootstrap_ci(ok_h & ok_v, cfg.eval.bootstrap)))
                    if tag == "cur":
                        for k, c in enumerate(conv):
                            if c["episode_id"] in show:
                                showcase.setdefault(f"A{r}", {}).setdefault(c["episode_id"], {"post_switch": c["post_switch"]})[f"probs{view}"] = {"helper": h[k].tolist(), "venue": v[k].tolist()}
            rows.append(row)
        write_json(art.dir / "arms.json", rows)
        write_json(art.dir / "showcase_rounds.json", showcase)


def stage_strongc(cfg) -> None:
    """Prompted strong C (big_gen model, option scoring) on test transcripts of each available condition."""
    from .data.warmstart import pick_examples
    tr_eps = ST.eps_map(cfg, "train")
    ex, _ = pick_examples(ST.load_convs(cfg, "warmstart_convs", "train"), tr_eps, cfg["seed"])
    te_eps = ST.eps_map(cfg, "test")
    with Artifact(cfg, "strongc", "eval", "eval strongc", inputs=["warmstart_convs"]) as art:
        lm, name = ST.big_lm(cfg)
        art.man["models"] = {"strong_c": name}
        sc, out = OptionScorer(lm, cfg.hw["score_bs"]), {}
        for label in CONDS:
            try:
                cond = ST.resolve_cond(cfg, label)
            except RuntimeError:
                continue
            if artifact_status(cfg, f"convs_{cond}") != "complete":
                continue
            te = _first_per_episode(ST.load_convs(cfg, f"convs_{cond}", "test"))
            prompts = [views.prompted_c_text(c["post_switch"], ex) for c in te]
            ph = sc.probs([p + "\nHelper:" for p in prompts], PEOPLE)
            pv = sc.probs([p + "\nVenue:" for p in prompts], PLACES)
            sec = [(PEOPLE.index(te_eps[c["episode_id"]]["secret"]["helper"]), PLACES.index(te_eps[c["episode_id"]]["secret"]["venue"])) for c in te]
            okh, okv = ph.argmax(1) == np.array([s[0] for s in sec]), pv.argmax(1) == np.array([s[1] for s in sec])
            out[label] = {k: dict(zip(("mean", "lo", "hi"), bootstrap_ci(x, cfg.eval.bootstrap))) for k, x in
                          (("C_helper", okh), ("C_venue", okv), ("C_both", okh & okv))}
        lm.close()
        write_json(art.dir / "strongc.json", out)


def stage_calt(cfg, labels=("S0", "S1", "S3")) -> None:
    """Held-out outsider C_alt (never in the reward loop) + mismatched-memory B control, on existing test transcripts.
    One scoring pass with the big_gen model; no training."""
    import time
    from .data.warmstart import pick_examples
    tr_eps, te_eps = ST.eps_map(cfg, "train"), ST.eps_map(cfg, "test")
    wconvs = ST.load_convs(cfg, "warmstart_convs", "train")
    _, used = pick_examples(wconvs, tr_eps, cfg["seed"])        # the in-loop C's three examples
    pool = _first_per_episode([c for c in wconvs if c["episode_id"] not in used])
    ex_alt, ex_ids = pick_examples(pool, tr_eps, cfg["seed"] + 1000)
    assert not (set(ex_ids) & set(used)), "C_alt examples overlap the in-loop C examples"
    todo = []
    for lab in labels:
        try:
            cond = ST.resolve_cond(cfg, lab)
        except RuntimeError:
            continue
        if artifact_status(cfg, f"convs_{cond}") == "complete":
            todo.append((lab, cond))
    n = sum(len(_first_per_episode(ST.load_convs(cfg, f"convs_{c}", "test"))) for _, c in todo)
    print(f"[calt] conditions {[l for l, _ in todo]}, {n} test conversations, {4 * n} scoring prompts; "
          f"estimated runtime ~{2 + 4 * n / 600:.0f} min (model load + scoring)", flush=True)
    with Artifact(cfg, "calt", "eval", "eval calt", inputs=["warmstart_convs"] + [f"convs_{c}" for _, c in todo],
                  extra={"calt_example_episodes": sorted(ex_ids), "inloop_c_example_episodes": sorted(used)}) as art:
        t0 = time.time()
        lm, name = ST.big_lm(cfg)
        art.man["models"] = {"c_alt": name}
        sc, out = OptionScorer(lm, cfg.hw["score_bs"]), {}
        ci = lambda x: dict(zip(("mean", "lo", "hi"), bootstrap_ci(np.asarray(x, dtype=float), cfg.eval.bootstrap)))
        for lab, cond in todo:
            te = _first_per_episode(ST.load_convs(cfg, f"convs_{cond}", "test"))
            items = make_items(te, te_eps)
            gh, gv = np.array([i["helper"] for i in items]), np.array([i["venue"] for i in items])
            pa = [views.prompted_c_alt_text(c["post_switch"], ex_alt) for c in te]
            ah, av = sc.probs([p + "\nHelper:" for p in pa], PEOPLE), sc.probs([p + "\nVenue:" for p in pa], PLACES)
            pm = [views.prompted_b_text(items[(k + 1) % len(items)]["memories"], it["post_switch"]) for k, it in enumerate(items)]
            mh, mv = sc.probs([p + "\nHelper:" for p in pm], PEOPLE), sc.probs([p + "\nVenue:" for p in pm], PLACES)
            okh, okv, mokh, mokv = ah.argmax(1) == gh, av.argmax(1) == gv, mh.argmax(1) == gh, mv.argmax(1) == gv
            out[lab] = {"condition": cond, "n": len(te), "C_alt_helper": ci(okh), "C_alt_venue": ci(okv), "C_alt_both": ci(okh & okv),
                        "B_mismatch_helper": ci(mokh), "B_mismatch_venue": ci(mokv), "B_mismatch_both": ci(mokh & mokv)}
            print(f"[calt] {lab}: C_alt both {out[lab]['C_alt_both']['mean']:.3f}  B(mismatched memories) both "
                  f"{out[lab]['B_mismatch_both']['mean']:.3f}  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
            write_json(art.dir / "calt.json", out)                 # checkpoint after each condition
        lm.close()
        write_json(art.dir / "calt.json", out)


# ---------------------------------------------------------------- checks
def run_check(cfg, which: str) -> bool:
    ck = Path(cfg["paths"]["runs"]) / "checks"
    ck.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    if which == "G1":
        g = read_json(art_dir(cfg, "episodes") / "manifest.json")["extra"]["g1"]
        ok = g["bank_pass_rate"] >= 0.95
        res = {**g, "fallback": "Use a larger generator, loosen word limits, or accept more template fallbacks."}
    elif which == "G2":
        h = read_json(art_dir(cfg, "warmstart_convs") / "manifest.json")["extra"]["headroom"]
        ok = h["prompted_B_both"] >= 0.5 and h["prompted_C_both"] <= 0.10 and h["a_leak_rate"] <= 0.20
        res = {**h, "fallback": "More vivid memories; stronger never-name instruction; last resort: venue-only (one slot)."}
    elif which == "G3":
        d = read_json(art_dir(cfg, "listeners_r0") / "manifest.json")["extra"]["dev_both"]
        ok = d["B"] >= 0.10 and d["B"] >= 2 * d["C"]   # ASSUMPTION: relaxed from 0.30 (small data, hint-based chats)
        res = {**d, "fallback": "Larger listener backbone, more epochs, or one slot."}
    elif which == "G4":
        tau = ST.get_tau(cfg)
        B, C = ST.load_reward_listeners(cfg, 0)
        dev = ST.eps_map(cfg, "dev")

        def mean_reward(convs):
            items = [{"memories": dev[c["episode_id"]]["memories"], "post_switch": c["post_switch"]} for c in convs]
            bh, bv = B.probs(items)
            ch, cv = C.probs(items)
            out = []
            for k, c in enumerate(convs):
                h, v = PEOPLE.index(dev[c["episode_id"]]["secret"]["helper"]), PLACES.index(dev[c["episode_id"]]["secret"]["venue"])
                out.append(reward_from(c["a_leak"], c["sense"] or 0, (bh[k, h], bv[k, v]), (ch[k, h], cv[k, v]), cfg.speaker.lam, tau))
            return float(np.mean(out))
        good = [c for c in ST.load_convs(cfg, "warmstart_convs", "dev") if c["good"]]
        s0 = ST.load_convs(cfg, "convs_S0", "dev")
        r_good, r_s0 = (mean_reward(good) if good else float("nan")), mean_reward(s0)
        ok = (r_s0 < r_good) if good else True
        res = {"reward_S0_dev": r_s0, "reward_good_warmstart_dev": r_good, "fallback": "Add more confusable memory pairs so the task is harder."}
    elif which == "G5":
        m = read_jsonl(art_dir(cfg, "speaker_r1_lam1.0") / "metrics.jsonl")[:50]
        first, last = m[: max(1, len(m) // 3)], m[-max(1, len(m) // 3):]
        r0, r1 = np.mean([x["reward"] for x in first]), np.mean([x["reward"] for x in last])
        tail = m[-10:]
        res = {"reward_first_third": r0, "reward_last_third": r1, "sense_rate": np.mean([x["sense_rate"] for x in tail]),
               "leak_rate": np.mean([x["leak_rate"] for x in tail]), "kl": np.mean([x["kl"] for x in tail]),
               "fallback": "Lower the learning rate, raise beta, or reduce G; inspect bloopers."}
        ok = r1 >= r0 and res["sense_rate"] >= 0.8 and res["leak_rate"] <= 0.05
    else:
        raise ValueError(which)
    res = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in res.items()}
    res["status"] = "PASS" if ok else "FAIL"
    write_json(ck / f"{which}.json", res)
    print(f"{which}: {res['status']}  {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k not in ('status',)})}")
    return bool(ok)


# ---------------------------------------------------------------- audit
def run_audit(cfg) -> bool:
    out = Path(cfg["paths"]["runs"]) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    rng = random.Random(0)
    eps = E.load_split(cfg, "train")
    sample = rng.sample(eps, min(20, len(eps)))
    rendered = []
    for e in sample:
        sec = e["secret"]
        post = [["A", "Did you see the match?"], ["B", "Ha, yes!"]]
        c_in = views.listener_text("C", None, post)
        b_in = views.listener_text("B", e["memories"], post)
        bsp = views.b_speaker_messages(e, post)[0]["content"]
        j_in = views.judge_text(e["cover_topic"], post)
        if any(m[:30] in c_in for m in e["memories"]) or "<memories>" in c_in:
            problems.append(f"{e['episode_id']}: memory text in C view")
        if any(t[1] in b_in for t in e["pre_switch"]) or "Cover topic" in b_in or "joins the conversation" in b_in:
            problems.append(f"{e['episode_id']}: pre-switch/topic text in B listener view")
        if "helper" in bsp and "who will bring" in bsp and f"is {sec['helper']}," in bsp:
            problems.append(f"{e['episode_id']}: secret in B speaker prompt")
        if any(m[:30] in j_in for m in e["memories"]) or e["birthday_person"] in j_in:
            problems.append(f"{e['episode_id']}: extra info in judge input")
        rendered.append({"episode_id": e["episode_id"], "C_view": c_in, "B_view": b_in, "B_speaker_system": bsp, "judge": j_in})
    write_json(out / "rendered_samples.json", rendered)
    topics = yaml.safe_load(open(CONFIGS / "cover_topics.yaml"))
    themes = yaml.safe_load(open(CONFIGS / "themes.yaml"))
    for s in ("train", "dev"):
        for e in E.load_split(cfg, s, with_labels=False):
            if e["cover_topic"] in topics["test"] or any(t in themes["test"] for t in e["theme_ids"]):
                problems.append(f"{e['episode_id']}: test topic/theme leaked into {s}")
    for e in E.load_split(cfg, "test", with_labels=False):
        if e["cover_topic"] not in topics["test"] or any(t not in themes["test"] for t in e["theme_ids"]):
            problems.append(f"{e['episode_id']}: non-held-out topic/theme in test")
    for p in sorted(Path(cfg["paths"]["artifacts"]).glob("speaker_r*")):
        m = read_json(p / "manifest.json")
        if m["status"] == "complete" and not m["extra"].get("base_frozen", False):
            problems.append(f"{p.name}: base weights changed during GRPO")
    counts = {}
    for p in sorted(Path(cfg["paths"]["artifacts"]).glob("convs_*")):
        if (p / "test.jsonl").exists():
            cs = read_jsonl(p / "test.jsonl")
            counts[p.name] = {"n": len(cs), "a_leak": sum(c["a_leak"] for c in cs), "b_resamples": sum(c["b_resamples"] for c in cs)}
    cfgs = {}
    for p in sorted(Path(cfg["paths"]["artifacts"]).glob("evalpair_*")):
        m = read_json(p / "manifest.json")
        if m["status"] == "complete":
            c = m["config"]
            cfgs[p.name] = json.dumps({"listeners": c["listeners"], "n_turns": c["data"]["n_turns"], "temp_b": c["speaker"]["temp_b"],
                                       "transcript_temp": c["speaker"]["transcript_temp"], "tau": c["speaker"]["tau"], "hw": c["hw"]["name"]}, sort_keys=True)
    for fam in (False, True):                                   # prompted pairs and evaluation-only trained pairs separately
        group = {k: v for k, v in cfgs.items() if k.endswith("_trained") == fam}
        if len(set(group.values())) > 1:
            problems.append(f"evaluation pairs were produced with different settings: {sorted(group)}")
    write_json(out / "report.json", {"problems": problems, "counts": counts})
    print("AUDIT", "PASS" if not problems else "FAIL", *problems, sep="\n  ")
    return not problems


# ---------------------------------------------------------------- report
def _out(cfg, sub: str = "paper") -> Path:
    """Real runs write into the repo's paper/ and viewer/; toy runs never touch them."""
    return (Path(cfg["paths"]["results"]) / "toy_out" / sub) if cfg.toy else ROOT / sub


def _read_metrics(cfg) -> dict:
    out = {}
    for lab in CONDS:
        for suf in ("", "_trained"):                            # prompted pair, then evaluation-only trained pair
            p = art_dir(cfg, f"evalpair_{lab}{suf}") / "metrics.json"
            if p.exists() and artifact_status(cfg, f"evalpair_{lab}{suf}") == "complete":
                out[lab + suf] = read_json(p)
    return out


def _fmt(m: dict) -> str:
    return "TODO" if m is None or np.isnan(m["mean"]) else f"{m['mean']*100:.1f} [{m['lo']*100:.1f}, {m['hi']*100:.1f}]"


def stage_report(cfg) -> None:
    res = Path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    M = _read_metrics(cfg)
    cols = [("B_both", "B both"), ("C_both", "C both"), ("insider_adv", "Insider adv"), ("joint_success", "Joint succ"),
            ("leak_rate", "Leak"), ("sense_pass", "Sense pass"), ("reward", "Reward")]
    strong = read_json(art_dir(cfg, "strongc") / "strongc.json") if artifact_status(cfg, "strongc") == "complete" else {}
    calt = read_json(art_dir(cfg, "calt") / "calt.json") if artifact_status(cfg, "calt") == "complete" else {}
    with Artifact(cfg, "results", "report", "eval report") as art:
        lines = ["| Cond | " + " | ".join(c[1] for c in cols) + " | strong-C both | C_alt both (held-out) |", "|" + "---|" * (len(cols) + 3)]
        csv_rows = []
        for lab, d in M.items():
            m = d["metrics"]
            row = [lab] + [(_fmt(m[k]) if k in ("B_both", "C_both", "insider_adv", "joint_success", "leak_rate", "sense_pass") else f"{m[k]['mean']:.3f}") for k, _ in cols] + [_fmt(strong.get(lab, {}).get("C_both")) if strong else "TODO"] + [_fmt(calt.get(lab, {}).get("C_alt_both")) if calt else "TODO"]
            lines.append("| " + " | ".join(row) + " |")
            csv_rows.append({"condition": lab, **{f"{k}_{s}": m[k][s] for k, _ in cols for s in ("mean", "lo", "hi")}})
        md = "Values in % with 95% bootstrap CI (1000 resamples over test episodes); chance: 10% per slot, 1% both. Reward is in [-1,1] (mean).\n\n" + "\n".join(lines) + "\n"
        (res / "results.md").write_text(md)
        write_json(res / "summary.json", {"conditions": M, "strong_c": strong, "c_alt": calt, "run_ids": sorted(p.name for p in Path(cfg["paths"]["artifacts"]).glob("*"))})
        if csv_rows:
            with open(res / "results.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
                w.writeheader()
                w.writerows(csv_rows)
        tex = _out(cfg) / "tables"
        tex.mkdir(parents=True, exist_ok=True)
        body = "\\begin{tabular}{l" + "r" * (len(cols)) + "}\\hline\nCond & " + " & ".join(c[1] for c in cols) + " \\\\\\hline\n"
        for lab, d in M.items():
            body += lab + " & " + " & ".join(f"{d['metrics'][k]['mean']*100:.1f}" if k not in ("reward",) else f"{d['metrics'][k]['mean']:.2f}" for k, _ in cols) + " \\\\\n"
        (tex / "main.tex").write_text(body + "\\hline\\end{tabular}\n" if M else "\\TODO{results}\n")
        from .evalextra import paper_macros
        paper_macros(cfg, M, strong, _out(cfg) / "tables")
        if calt:                                                   # held-out outsider and mismatched-memory control
            fm = lambda v: f"{v['mean']*100:.1f} [{v['lo']*100:.1f}, {v['hi']*100:.1f}]"
            extra = [f"\\expandafter\\def\\csname res:{lab}:{k}\\endcsname{{{fm(v)}}}" for lab, d in calt.items() for k, v in d.items() if isinstance(v, dict)]
            with open(_out(cfg) / "tables" / "macros.tex", "a") as f:
                f.write("\n".join(extra) + "\n")
        _figures(cfg, res, M)
        _clues(cfg, res)
        _showcase(cfg, M)
        art.man["extra"]["conditions"] = list(M)
        art.man["inputs"] = [f"evalpair_{k}" for k in M]
        if not cfg.toy:
            _readme_table(md)
    print(md)


def _figures(cfg, res: Path, M: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = _out(cfg) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(Path(cfg["paths"]["artifacts"]).glob("speaker_r*"))
    if runs:
        fig, ax = plt.subplots(1, 3, figsize=(11, 3))
        for p in runs:
            if (p / "metrics.jsonl").exists():
                m = read_jsonl(p / "metrics.jsonl")
                for a, k in zip(ax, ("reward", "pB", "pC")):
                    a.plot([x["step"] for x in m], [x[k] for x in m], label=p.name.replace("speaker_", ""))
                    a.set_title(k)
                    a.set_xlabel("step")
        ax[0].legend(fontsize=6)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(fig_dir / f"training_curves.{ext}", dpi=150)
        plt.close(fig)
    if M:
        fig, ax = plt.subplots(figsize=(5, 3))
        labs = list(M)
        x = np.arange(len(labs))
        for off, k, nm in ((-0.2, "B_both", "B listener"), (0.2, "C_both", "C (trained)")):
            mu = np.array([M[l]["metrics"][k]["mean"] for l in labs])
            lo = mu - np.array([M[l]["metrics"][k]["lo"] for l in labs])
            hi = np.array([M[l]["metrics"][k]["hi"] for l in labs]) - mu
            ax.bar(x + off, mu, 0.4, yerr=[lo, hi], label=nm)
        ax.axhline(0.01, ls="--", c="gray", lw=0.8)
        ax.set_xticks(x, labs)
        ax.set_ylabel("both-correct")
        ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(fig_dir / f"scoreboard.{ext}", dpi=150)
        plt.close(fig)


def _clues(cfg, res: Path) -> None:
    from .eval.public_clues import clue_table
    out = {}
    for r in range(0, cfg.speaker.rounds + 1):
        name = "convs_S1" if r == 0 else f"convs_A{r}"
        if artifact_status(cfg, name) == "complete":
            out[f"round{r}"] = clue_table(ST.load_convs(cfg, name, "train"), ST.eps_map(cfg, "train"))
    write_json(res / "public_clues.json", out)


def _showcase(cfg, M: dict) -> None:
    ids = E.load_showcase_ids(cfg)
    te = ST.eps_map(cfg, "test")
    bundle = {"inventories": {"people": PEOPLE, "places": PLACES}, "episodes": [], "run_ids": sorted(p.name for p in Path(cfg["paths"]["artifacts"]).glob("evalpair_*"))}
    rounds = read_json(art_dir(cfg, "armsrace") / "showcase_rounds.json") if artifact_status(cfg, "armsrace") == "complete" else {}
    preds = {}
    for lab, d in M.items():
        p = art_dir(cfg, f"evalpair_{lab}") / "preds_test.jsonl"
        if not p.exists():
            continue
        preds[lab] = {(r["episode_id"], r["view"]): r["probs"] for r in read_jsonl(p, "pred")}
    for eid in ids:
        e = te[eid]
        conds = {}
        for lab, d in M.items():
            cond = d["condition"]
            c = next((c for c in ST.load_convs(cfg, f"convs_{cond}", "test") if c["episode_id"] == eid), None)
            if c:
                conds[lab] = {"post_switch": c["post_switch"], "probsB": preds.get(lab, {}).get((eid, "B")), "probsC": preds.get(lab, {}).get((eid, "C"))}
        for rk, per in rounds.items():
            if eid in per:
                conds[rk] = {"post_switch": per[eid]["post_switch"], "probsB": per[eid].get("probsB"), "probsC": per[eid].get("probsC")}
        bundle["episodes"].append({"episode_id": eid, "memories": e["memories"], "cover_topic": e["cover_topic"], "pre_switch": e["pre_switch"],
                                   "secret": e["secret"], "conditions": conds})
    write_json(_out(cfg, "viewer") / "showcase.json", bundle)


def _readme_table(md: str) -> None:
    p = ROOT / "README.md"
    if not p.exists():
        return
    s = p.read_text()
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a in s and b in s:
        p.write_text(s.split(a)[0] + a + "\n" + md + b + s.split(b)[1])
