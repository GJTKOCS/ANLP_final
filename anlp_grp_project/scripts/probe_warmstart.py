"""Quick probe (~5 min): 24 train episodes x 2 conversations with the current A prompt, scored by prompted B and C.
Run BEFORE regenerating the full warm start:  python3 scripts/probe_warmstart.py"""
import random

from covert.common.config import load_config
from covert.data import episodes as E
from covert.data import warmstart as W
from covert import stages as ST

cfg = load_config(["data.n_train=1000", "data.n_dev=150", "data.n_test=200", "hw.big_gen_bs=48"], None, False)
eps = E.load_split(cfg, "train")[:24]
by = {e["episode_id"]: e for e in eps}
lm, name = ST.big_lm(cfg)
convs = W.generate_convs(cfg, lm, eps, 2, "train", name)
ex, excluded = W.pick_examples(convs, by, cfg["seed"])
for c, p in zip(convs, W.prompted_scores(lm, convs, by, ex, cfg.hw["score_bs"])):
    c["prompted_scores"] = p
    c["sense"] = 1.0
    c["good"] = W.mark_good(c, 0.0, excluded)
print("PROBE HEADROOM", W.headroom(convs), "good:", sum(bool(c["good"]) for c in convs), "of", len(convs))
for c in random.Random(2).sample(convs, 3):
    e = by[c["episode_id"]]
    print("=" * 60, "\nSECRET", e["secret"], "| B:", c["prompted_scores"]["B"]["helper_ok"], c["prompted_scores"]["B"]["venue_ok"])
    for i in sorted(set(e["anchors"]["helper"] + e["anchors"]["venue"])):
        print("  MEM", i, e["memories"][i][:160])
    for s, t in c["post_switch"]:
        print(f"  {s}: {t}")
