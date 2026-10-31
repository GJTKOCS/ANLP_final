"""Orchestrator: ordered stages, skip-if-complete, resume, deadline-aware, one subprocess per stage."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from .common.artifacts import artifact_status
from .common.config import CONFIGS, ROOT
from .common.hub import push_artifact
from .common.io import write_json

log = logging.getLogger("covert.pipeline")
HARD_CHECKS = {"G3"}   # later stages are meaningless if these fail; G1/G4/G5 warn and continue
RANK = {"must": 0, "should": 1, "optional": 2}


def load_stages() -> list[dict]:
    return yaml.safe_load(open(CONFIGS / "pipeline.yaml"))["stages"]


def _cmd(stage: dict, toy: bool, profile: str | None, overrides: list[str]) -> list[str]:
    c = [sys.executable, "-m", "covert.cli"] + stage["cmd"].split()
    if toy:
        c.append("--toy")
    if profile:
        c += ["--profile", profile]
    return c + overrides


def run_pipeline(cfg, until=None, only=None, force=None, skip_optional=False, deadline=None, continue_on_fail=False,
                 overrides: list[str] | None = None, profile: str | None = None) -> int:
    toy = cfg.toy
    runs = Path(cfg["paths"]["runs"])
    runs.mkdir(parents=True, exist_ok=True)
    stages = load_stages()
    dl = datetime.fromisoformat(deadline) if deadline and not toy else None
    state, ratios = {}, []
    t_start = time.time()
    names = [s["name"] for s in stages]
    for name in (until, only, force):
        if name and name not in names:
            raise SystemExit(f"Unknown stage '{name}'. Stages: {names}")
    for st in stages:
        name = st["name"]
        if only and name != only:
            continue
        done = artifact_status(cfg, st["out"]) == "complete"
        if done and name != force:
            state[name] = {"status": "skipped (complete)"}
            continue
        if skip_optional and st["priority"] == "optional":
            state[name] = {"status": "skipped (--skip-optional)"}
            continue
        est = st["est_h"] * (sum(ratios) / len(ratios) if ratios else 1.0)
        if dl and st["priority"] != "must":
            left = (dl - datetime.now(dl.tzinfo)).total_seconds() / 3600
            if est > left:
                state[name] = {"status": f"skipped (needs ~{est:.1f}h, {left:.1f}h left)"}
                log.warning("skip %s: %s", name, state[name]["status"])
                continue
        rest = sum(s["est_h"] for s in stages[names.index(name):])
        print(f"[pipeline] {name} ({st['priority']}) est {est:.1f}h; remaining plan ~{rest:.1f}h", flush=True)
        t0 = time.time()
        rc = subprocess.call(_cmd(st, toy, profile, overrides or []), cwd=ROOT)
        h = (time.time() - t0) / 3600
        ok = rc == 0 and artifact_status(cfg, st["out"]) == "complete"
        state[name] = {"status": "complete" if ok else f"FAILED rc={rc}", "hours": round(h, 3)}
        if ok and not toy and st["est_h"] > 0.5:
            ratios.append(h / st["est_h"])
        if ok:
            try:
                push_artifact(cfg, st["out"])
            except Exception as e:                              # a Hub/network problem must never kill the run
                log.warning("hub push of %s failed (continuing): %s", st["out"], e)
        if not ok and st["priority"] == "must" and not continue_on_fail:
            _log(runs, stages, state, t_start)
            print(f"[pipeline] must-stage '{name}' failed. Fix and re-run `covert pipeline run` (it resumes).", flush=True)
            return 1
        if ok and st.get("check"):
            rc = subprocess.call([sys.executable, "-m", "covert.cli", "check", st["check"]] + (["--toy"] if toy else []) +
                                 (["--profile", profile] if profile else []) + (overrides or []), cwd=ROOT)
            state[name]["check"] = f"{st['check']}: {'PASS' if rc == 0 else 'FAIL'}"
            if rc != 0 and not toy and (st["check"] in HARD_CHECKS and not continue_on_fail):
                _log(runs, stages, state, t_start)
                print(f"[pipeline] go/no-go {st['check']} FAILED; see runs/checks/{st['check']}.json for the fallback. "
                      f"Re-run with --continue-on-fail to proceed anyway.", flush=True)
                return 2
        _log(runs, stages, state, t_start)
        if until and name == until:
            break
    _log(runs, stages, state, t_start)
    print("[pipeline] finished. If on Jarvislabs: STOP THE INSTANCE now to stop billing.", flush=True)
    return 0


def _log(runs: Path, stages, state, t0) -> None:
    write_json(runs / "pipeline_state.json", state)
    lines = ["| stage | status | hours | check |", "|---|---|---|---|"]
    for s in stages:
        v = state.get(s["name"], {})
        lines.append(f"| {s['name']} | {v.get('status', 'pending')} | {v.get('hours', '')} | {v.get('check', '')} |")
    (runs / "pipeline_log.md").write_text("\n".join(lines) + f"\n\nelapsed {(time.time()-t0)/3600:.2f} h\n")


def status(cfg) -> None:
    for s in load_stages():
        print(f"{s['name']:14s} {s['priority']:8s} est {s['est_h']:>5}h  {artifact_status(cfg, s['out']) or 'pending'}")
