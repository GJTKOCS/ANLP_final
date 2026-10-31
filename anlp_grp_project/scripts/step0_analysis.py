"""Step 0: free checks from existing artifacts (CPU only, no model loading).
Usage (on the VM, in the repo, venv active):  python3 scripts/step0_analysis.py
Writes results/step0/summary.json and results/step0/reward_decomposition.{png,pdf}."""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOY = "--toy" in sys.argv
ART = ROOT / "artifacts" / ("toy" if TOY else "")
OUT = ROOT / "results" / ("toy" if TOY else "") / "step0"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from covert.eval.stats import bootstrap_ci  # noqa: E402

summary = {}


def fmt(m):
    return f"{m['mean']*100:5.1f} [{m['lo']*100:5.1f}, {m['hi']*100:5.1f}]"


# 1. equal-treatment audit: every rollout() call site and its a_resample argument
print("== 1. A-turn leak re-sampling call sites ==")
sites = []
for p in sorted((ROOT / "src" / "covert").rglob("*.py")):
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if re.search(r"\brollout\(", line) and "def rollout" not in line:
            m = re.search(r"a_resample\s*=\s*([^,)]+(?:\([^)]*\))?)", line)
            sites.append({"file": str(p.relative_to(ROOT)), "line": i, "a_resample": m.group(1) if m else "0 (default)"})
for s in sites:
    print(f"  {s['file']}:{s['line']}  a_resample={s['a_resample']}")
bad = [s for s in sites if s["a_resample"] != "0 (default)" and "warmstart" not in s["file"]]
print("  RESULT:", "OK - only warm start re-samples A turns" if not bad else f"PROBLEM - re-sampling outside warm start: {bad}")
summary["resample_sites"] = sites
summary["resample_ok"] = not bad

# 2 + 4. per-slot metrics, sense and leak per condition (from eval-pair metrics)
print("\n== 2. Per-slot accuracy per condition (% , 95% bootstrap CI over test episodes) ==")
rows = {}
for d in sorted(ART.glob("evalpair_*")):
    mp = d / "metrics.json"
    man = d / "manifest.json"
    if not mp.exists() or json.loads(man.read_text())["status"] != "complete":
        continue
    M = json.loads(mp.read_text())
    lab, m = M["label"], M["metrics"]
    rows[lab] = {k: m[k] for k in ("B_helper", "B_venue", "B_both", "C_helper", "C_venue", "C_both", "insider_adv",
                                   "joint_success", "leak_rate", "sense_pass", "sanity_B_wrong_memories_both") if k in m}
    rows[lab]["n_test"] = M["n_test"]
    rows[lab]["condition"] = M["condition"]
hdr = ["B_helper", "B_venue", "B_both", "C_helper", "C_venue", "C_both"]
print("  cond  " + "  ".join(f"{h:>21s}" for h in hdr))
for lab, r in rows.items():
    print(f"  {lab:5s} " + "  ".join(f"{fmt(r[h]):>21s}" for h in hdr) + f"   (n={r['n_test']}, {r['condition']})")
print("\n== 4. Sense pass and A-turn leak rate on test ==")
for lab, r in rows.items():
    print(f"  {lab:5s} sense pass {fmt(r['sense_pass'])}   A leak {fmt(r['leak_rate'])}   "
          f"B with wrong memories (both) {fmt(r['sanity_B_wrong_memories_both']) if 'sanity_B_wrong_memories_both' in r else 'n/a'}")
summary["per_condition"] = rows

# 3. reward decomposition for the GRPO round
print("\n== 3. Reward decomposition (GRPO round 1, lambda=1) ==")
tau = json.loads((ART / "judge_calibration" / "calibration.json").read_text())["tau"]
rp = ART / "speaker_r1_lam1.0" / "rollouts.jsonl"
if rp.exists():
    ro = [json.loads(l) for l in rp.open()]
    steps = sorted({r["step"] for r in ro})
    per = []
    for s in steps:
        rs = [r for r in ro if r["step"] == s]
        leak = [r["a_leak"] for r in rs]
        sense_fail = [(not r["a_leak"]) and r["sense"] < tau for r in rs]
        ok = [r for r in rs if not r["a_leak"] and r["sense"] >= tau]
        gap = [np.mean(list(r["pB"].values())) - np.mean(list(r["pC"].values())) for r in ok]
        per.append({"step": s, "n": len(rs), "frac_leak": float(np.mean(leak)), "frac_sense": float(np.mean(sense_fail)),
                    "frac_penalised": float(np.mean(leak) + np.mean(sense_fail)), "gap_unpenalised": float(np.mean(gap)) if gap else None})
    allok = [r for r in ro if not r["a_leak"] and r["sense"] >= tau]
    gaps = np.array([np.mean(list(r["pB"].values())) - np.mean(list(r["pC"].values())) for r in allok])
    tot = {"n_rollouts": len(ro), "tau": tau, "frac_penalised_leak": float(np.mean([r["a_leak"] for r in ro])),
           "frac_penalised_sense": float(np.mean([(not r["a_leak"]) and r["sense"] < tau for r in ro])),
           "gap_unpenalised_mean_ci": bootstrap_ci(gaps, 1000) if len(gaps) else None}
    for part, sl in (("first 20 steps", per[:20]), ("last 20 steps", per[-20:])):
        print(f"  {part}: penalised leak {np.mean([p['frac_leak'] for p in sl]):.3f}  sense {np.mean([p['frac_sense'] for p in sl]):.3f}  "
              f"mean(pB)-mean(pC) on unpenalised {np.nanmean([p['gap_unpenalised'] if p['gap_unpenalised'] is not None else np.nan for p in sl]):.3f}")
    print(f"  overall (n={tot['n_rollouts']}, tau={tau}): penalised by leak {tot['frac_penalised_leak']:.3f}, by sense {tot['frac_penalised_sense']:.3f}; "
          f"gap on unpenalised = {tot['gap_unpenalised_mean_ci']}")
    summary["reward_decomposition"] = {"total": tot, "per_step": per}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    x = [p["step"] for p in per]
    ax[0].plot(x, [p["frac_leak"] for p in per], label="-1 from leak")
    ax[0].plot(x, [p["frac_sense"] for p in per], label="-1 from sense judge")
    ax[0].set_title("fraction of rollouts penalised")
    ax[0].legend(fontsize=7)
    ax[1].plot(x, [p["gap_unpenalised"] if p["gap_unpenalised"] is not None else np.nan for p in per])
    ax[1].axhline(0, c="gray", lw=0.6)
    ax[1].set_title("mean(pB) - mean(pC), unpenalised rollouts")
    for a in ax:
        a.set_xlabel("GRPO step")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"reward_decomposition.{ext}", dpi=150)
    print(f"  plot: {OUT}/reward_decomposition.png")
else:
    print("  rollouts.jsonl not found")

# 5. dev metrics of the prompted pair
print("\n== 5. Prompted pair on dev (listeners_r0, listeners_r1) ==")
for r in (0, 1):
    mp = ART / f"listeners_r{r}" / "manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        print(f"  listeners_r{r}: status {m['status']}  {m.get('extra', {})}")
        summary[f"listeners_r{r}"] = m.get("extra", {})
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nwrote {OUT}/summary.json")
