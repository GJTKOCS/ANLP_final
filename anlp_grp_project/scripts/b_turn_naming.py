"""How often the frozen B *speaker* (who does not know the secret) names an inventory item in its turns on test,
after its 3 leak re-samples, and whether that helps C. CPU only, reads saved transcripts and predictions.
Usage:  python3 scripts/b_turn_naming.py ~/Documents/ANLP_Final/vm_run      -> results/b_turn_naming.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from covert.common.leak import find_inventory, has_leak  # noqa: E402

V = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else ROOT
A = V / "artifacts"
sec = {}
for line in (A / "episodes" / "private" / "test_private.jsonl").open():
    r = json.loads(line)
    sec[r["episode_id"]] = r["secret"] if "secret" in r else r
out = {}
for lab, conv in (("S0", "convs_S0"), ("S1", "convs_S1"), ("S3", "convs_A1")):
    rows = [json.loads(l) for l in (A / conv / "test.jsonl").open()]
    preds = {}
    for line in (A / f"evalpair_{lab}" / "preds_test.jsonl").open():
        p = json.loads(line)
        preds[(p["episode_id"], p["view"])] = p["pred"]
    named, true_named, c_named, c_not = [], [], [], []
    for r in rows:
        s = sec[r["episode_id"]]
        btxt = " ".join(t for sp, t in r["post_switch"] if sp == "B")
        ppl, plc = find_inventory(btxt)
        n = has_leak(btxt)
        named.append(n)
        true_named.append(s["helper"] in ppl or s["venue"] in plc)
        c = preds[(r["episode_id"], "C")]
        (c_named if n else c_not).append(c["helper"] == s["helper"] and c["venue"] == s["venue"])
    out[lab] = {"n": len(rows), "b_names_item": float(np.mean(named)), "b_names_true_slot": float(np.mean(true_named)),
                "C_both_when_named": float(np.mean(c_named)) if c_named else None, "n_named": len(c_named),
                "C_both_when_not_named": float(np.mean(c_not)) if c_not else None}
    print(lab, out[lab])
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "b_turn_naming.json").write_text(json.dumps(out, indent=2))
print("wrote results/b_turn_naming.json")
