"""Slide figures from run artifacts (no hand-typed numbers).
Usage:  python3 scripts/make_slide_figs.py ~/Documents/ANLP_Final/vm_run   -> slides/figures/{listeners,leak_sense}.png"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else ROOT
A = V / "artifacts"
OUT = ROOT / "slides" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
rd = lambda p: json.loads(Path(p).read_text())
calt = rd(A / "calt" / "calt.json")
CALT_KEY = {"S0": "S0", "S1": "S1", "S3": next(k for k in calt if k in ("S3", "A1"))}
conds = ["S0", "S1", "S3"]
names = {"S0": "S0 base", "S1": "S1 SFT", "S3": "S3 GRPO"}
m = {c: rd(A / f"evalpair_{c}" / "metrics.json")["metrics"] for c in conds}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED})


def bars(ax, series, ylabel):
    x = np.arange(len(conds))
    w = 0.8 / len(series)
    for k, (label, color, get) in enumerate(series):
        vals = [get(c) for c in conds]
        mean = np.array([100 * v["mean"] for v in vals])
        err = np.array([[100 * (v["mean"] - v["lo"]) for v in vals], [100 * (v["hi"] - v["mean"]) for v in vals]])
        pos = x - 0.4 + w * (k + 0.5)
        ax.bar(pos, mean, w - 0.02, color=color, label=label, zorder=2)
        ax.errorbar(pos, mean, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        for p, v, hi in zip(pos, mean, err[1]):
            ax.text(p, v + hi + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [names[c] for c in conds])
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=len(series), loc="upper right")


fig, ax = plt.subplots(figsize=(7, 3.4))
bars(ax, [("B (insider)", "#2a78d6", lambda c: m[c]["B_both"]),
          ("C (outsider, in reward)", "#eb6834", lambda c: m[c]["C_both"]),
          ("C$_{alt}$ (held-out outsider)", "#1baf7a", lambda c: calt[CALT_KEY[c]]["C_alt_both"])], "both slots correct (%)")
ax.set_ylim(0, 32)
fig.tight_layout()
fig.savefig(OUT / "listeners.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.4))
bars(ax, [("A leak", "#e34948", lambda c: m[c]["leak_rate"]),
          ("Sense pass", "#2a78d6", lambda c: m[c]["sense_pass"])], "% of test conversations")
ax.set_ylim(0, 85)
fig.tight_layout()
fig.savefig(OUT / "leak_sense.png", dpi=200)
plt.close(fig)
for f in ("reward_decomposition.png",):
    shutil.copy(ROOT / "paper" / "figures" / f, OUT / f)
print("wrote", sorted(p.name for p in OUT.iterdir()))
