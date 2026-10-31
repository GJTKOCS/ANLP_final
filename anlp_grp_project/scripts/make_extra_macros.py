"""Writes paper/tables/extra_macros.tex from run logs (diagnostics that `covert eval report` does not cover).
Usage:  python3 scripts/make_extra_macros.py ~/Documents/ANLP_Final/vm_run
Paper usage: \\x{key}  (renders TODO if missing). Every value is read from a file; nothing is typed by hand."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else ROOT
A, R = V / "artifacts", ROOT / "results"
M: dict[str, str] = {}
pct = lambda x: f"{100 * x:.1f}"
f3 = lambda x: f"{x:.3f}"
rd = lambda p: json.loads(Path(p).read_text())

e = rd(A / "episodes" / "manifest.json")["extra"]
M["slotFallbackPct"] = pct(e["g1"]["slot_fallback_rate"])
M["bankAllLLMPct"] = pct(e["g1"]["bank_pass_rate"])
M["nTrain"] = str(int(e["writer_stats"]["train"]["slots"] / 8))
M["nDev"] = str(int(e["writer_stats"]["dev"]["slots"] / 8))
M["nTest"] = str(int(e["writer_stats"]["test"]["slots"] / 8))

w = rd(A / "warmstart_convs" / "manifest.json")
M["bigGen"] = w["models"]["big_gen"].replace("_", "\\_")
M["warmN"] = str(w["extra"]["headroom"]["n"])
M["warmGood"] = str(w["extra"]["n_good"])
M["warmBboth"] = pct(w["extra"]["headroom"]["prompted_B_both"])
M["warmCboth"] = pct(w["extra"]["headroom"]["prompted_C_both"])
M["warmLeak"] = pct(w["extra"]["headroom"]["a_leak_rate"])
M["tau"] = f"{rd(A / 'judge_calibration' / 'calibration.json')['tau']:.2f}"
jc = rd(A / "judge_calibration" / "calibration.json")
jrow = next(r for r in jc["table"] if abs(r["tau"] - jc["tau"]) < 1e-9)
M["judgeKeepGood"], M["judgeRejectBad"] = pct(jrow["keep_good"]), pct(jrow["reject_bad"])
M["judgeNGood"], M["judgeNBad"] = str(jc["n_good"]), str(jc["n_bad"])

g = A / "speaker_r1_lam1.0"
gm = rd(g / "manifest.json")
M["grpoSteps"] = str(gm["extra"]["steps"])
M["grpoHours"] = f"{gm['wall_hours']:.2f}"
M["baseFrozen"] = "yes" if gm["extra"]["base_frozen"] else "NO"
met = [json.loads(l) for l in (g / "metrics.jsonl").open()]
for k, name in (("reward", "Reward"), ("pB", "PB"), ("pC", "PC"), ("leak_rate", "Leak"), ("sense_rate", "Sense"), ("kl", "KL")):
    M[f"grpoFirst{name}"] = f3(np.mean([x[k] for x in met[:20]]))
    M[f"grpoLast{name}"] = f3(np.mean([x[k] for x in met[-20:]]))
dev = [json.loads(l) for l in (g / "dev_metrics.jsonl").open()]
M["grpoDevFirst"] = f3(dev[0]["reward"])
M["grpoDevLast"] = f3(dev[-1]["reward"])
M["grpoDevFirstStep"] = str(dev[0]["step"])

s0 = rd(R / "step0" / "summary.json")["reward_decomposition"]
t = s0["total"]
M["penLeakPct"] = pct(t["frac_penalised_leak"])
M["penSensePct"] = pct(t["frac_penalised_sense"])
gap = t["gap_unpenalised_mean_ci"]
M["gapUnpen"] = f"{gap[0]:.3f} [{gap[1]:.3f}, {gap[2]:.3f}]"
per = s0["per_step"]
M["penSenseFirstPct"] = pct(np.mean([p["frac_sense"] for p in per[:20]]))
M["penSenseLastPct"] = pct(np.mean([p["frac_sense"] for p in per[-20:]]))
M["penLeakFirstPct"] = pct(np.mean([p["frac_leak"] for p in per[:20]]))
M["penLeakLastPct"] = pct(np.mean([p["frac_leak"] for p in per[-20:]]))
M["gapFirst"] = f3(np.nanmean([p["gap_unpenalised"] for p in per[:20] if p["gap_unpenalised"] is not None]))
M["gapLast"] = f3(np.nanmean([p["gap_unpenalised"] for p in per[-20:] if p["gap_unpenalised"] is not None]))

l0 = rd(A / "listeners_r0" / "manifest.json")["extra"]
M["trainedLzeroB"] = pct(l0["dev_both"]["B"])
M["trainedLzeroC"] = pct(l0["dev_both"]["C"])
l1 = rd(A / "listeners_r1" / "manifest.json")["extra"]
M["promptedDevB"] = pct(l1["dev_both"]["B"])
M["promptedDevC"] = pct(l1["dev_both"]["C"])

d1 = rd(R / "step1" / "diag.json")
M["bViewMaxTok"] = str(d1["truncation"]["len_max"])
M["bViewMedTok"] = f"{d1['truncation']['len_median']:.0f}"
M["maxLen"] = str(d1["max_len"])
M["initLoss"] = f"{d1['grad']['loss']:.1f}"
M["collapseDistinct"] = str(d1["pred_B"]["n_distinct_pred_helper"])

for lab in ("S0", "S1", "S3"):
    p = A / f"evalpair_{lab}_trained" / "B" / "info.json"
    if p.exists():
        M[f"trainedEvalBestDevB{lab}"] = pct(rd(p)["dev_both"])
        M[f"trainedEvalBestDevC{lab}"] = pct(rd(A / f"evalpair_{lab}_trained" / "C" / "info.json")["dev_both"])

bn = R / "b_turn_naming.json"          # from scripts/b_turn_naming.py
if bn.exists():
    for lab, d in rd(bn).items():
        M[f"bNames{lab}"] = pct(d["b_names_item"])
        M[f"bNamesTrue{lab}"] = pct(d["b_names_true_slot"])
        M[f"cBothNamed{lab}"] = pct(d["C_both_when_named"])
        M[f"cBothNotNamed{lab}"] = pct(d["C_both_when_not_named"])

M["bestDevStep"] = str(max(dev, key=lambda x: x["reward"])["step"])
M["bestDevReward"] = f3(max(x["reward"] for x in dev))
for lab in ("S0", "S1", "S3"):
    mm = rd(A / f"evalpair_{lab}" / "metrics.json")["metrics"]
    M[f"jointStrict{lab}"] = pct(mm["joint_strict"]["mean"])
    M[f"joint{lab}"] = pct(mm["joint_success"]["mean"])
sft = rd(A / "speaker_sft" / "manifest.json")["extra"]
M["sftExamples"] = str(sft["n_examples"])
M["nRollouts"] = str(t["n_rollouts"])

hours = {}
for p in sorted(A.glob("*/manifest.json")):
    m = rd(p)
    hours[p.parent.name] = m.get("wall_hours", 0)
M["gpuHoursArtifacts"] = f"{sum(hours.values()):.1f}"

out = ROOT / "paper" / "tables" / "extra_macros.tex"
out.write_text("% generated by scripts/make_extra_macros.py from run logs; do not edit\n" +
               "\n".join(f"\\expandafter\\def\\csname x:{k}\\endcsname{{{v}}}" for k, v in M.items()) + "\n")
(R / "extra_macros.json").write_text(json.dumps({"values": M, "wall_hours_per_artifact": hours}, indent=2))
print(f"wrote {out} ({len(M)} macros)")
for k, v in M.items():
    print(f"  {k:28s} {v}")
