"""`covert` command line. Every command: config + key=value overrides, --toy for CPU mock mode."""
from __future__ import annotations

import argparse
import logging
import sys

from .common.config import load_config


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--toy", action="store_true", help="toy data + tiny CPU model")
    p.add_argument("--profile", default=None, help="hardware profile (a100|t4|cpu_debug)")
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    return p


def main(argv: list[str] | None = None) -> int:
    c = _common()
    ap = argparse.ArgumentParser(prog="covert")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data").add_subparsers(dest="sub", required=True)
    d.add_parser("build", parents=[c])
    d.add_parser("warmstart", parents=[c])
    j = sub.add_parser("judge").add_subparsers(dest="sub", required=True)
    j.add_parser("calibrate", parents=[c])
    ls = sub.add_parser("listeners").add_subparsers(dest="sub", required=True)
    lt = ls.add_parser("train", parents=[c])
    lt.add_argument("--round", type=int, required=True)
    sp = sub.add_parser("speaker").add_subparsers(dest="sub", required=True)
    sp.add_parser("sft", parents=[c])
    g = sp.add_parser("grpo", parents=[c])
    g.add_argument("--round", type=int, required=True)
    g.add_argument("--lam", type=float, default=1.0)
    g.add_argument("--half", action="store_true")
    t = sub.add_parser("transcripts", parents=[c])
    t.add_argument("--cond", required=True)
    e = sub.add_parser("eval").add_subparsers(dest="sub", required=True)
    er = e.add_parser("run", parents=[c])
    er.add_argument("--cond", required=True)
    for n in ("armsrace", "strongc", "report", "human-form"):
        e.add_parser(n, parents=[c])
    hs = e.add_parser("human-score", parents=[c])
    hs.add_argument("csv")
    sub.add_parser("preflight", parents=[c])
    ck = sub.add_parser("check", parents=[c])
    ck.add_argument("which")
    sub.add_parser("audit", parents=[c])
    sub.add_parser("status", parents=[c])
    pl = sub.add_parser("pipeline").add_subparsers(dest="sub", required=True)
    pr = pl.add_parser("run", parents=[c])
    pr.add_argument("--until")
    pr.add_argument("--only")
    pr.add_argument("--force")
    pr.add_argument("--skip-optional", action="store_true")
    pr.add_argument("--deadline", default="2026-10-31T06:00+05:30")
    pr.add_argument("--continue-on-fail", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.overrides, a.profile, a.toy)
    logging.getLogger("covert").info("profile=%s toy=%s", cfg["profile"], cfg["toy"])

    from . import evalstages as ES
    from . import stages as ST
    from .pipeline import run_pipeline, status
    key = (a.cmd, getattr(a, "sub", None))
    if key == ("data", "build"):
        ST.stage_data(cfg)
    elif key == ("data", "warmstart"):
        ST.stage_warmstart(cfg)
    elif key == ("judge", "calibrate"):
        ST.stage_judge(cfg)
    elif key == ("listeners", "train"):
        ST.stage_listeners(cfg, a.round)
    elif key == ("speaker", "sft"):
        ST.stage_sft(cfg)
    elif key == ("speaker", "grpo"):
        ST.stage_grpo(cfg, a.round, a.lam, a.half)
    elif a.cmd == "transcripts":
        ST.stage_transcripts(cfg, a.cond)
    elif key == ("eval", "run"):
        ES.stage_eval(cfg, a.cond)
    elif key == ("eval", "armsrace"):
        ES.stage_armsrace(cfg)
    elif key == ("eval", "strongc"):
        ES.stage_strongc(cfg)
    elif key == ("eval", "report"):
        ES.stage_report(cfg)
    elif key == ("eval", "human-form"):
        from .evalextra import human_form
        human_form(cfg)
    elif key == ("eval", "human-score"):
        from .evalextra import human_score
        human_score(cfg, a.csv)
    elif a.cmd == "preflight":
        from .evalextra import preflight
        return 0 if preflight(cfg) else 1
    elif a.cmd == "check":
        return 0 if ES.run_check(cfg, a.which) else 1
    elif a.cmd == "audit":
        return 0 if ES.run_audit(cfg) else 1
    elif a.cmd == "status":
        status(cfg)
    elif key == ("pipeline", "run"):
        return run_pipeline(cfg, a.until, a.only, a.force, a.skip_optional, a.deadline, a.continue_on_fail or cfg.toy, a.overrides, a.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
