"""Step 1: diagnose the trained-listener failure. Reports only; trains nothing.
(a) token lengths of the rendered B view vs max_len, truncation side, and what survives truncation;
(b) prediction distribution of the saved trained B (and C) listener on dev;
(c) gradient flow: requires_grad and grad norms of LoRA params and heads after one backward pass.
Usage (VM, venv active):  python3 scripts/step1_listener_diag.py      (~3-5 min on the GPU, 1.5B model)"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from covert import stages as ST  # noqa: E402
from covert.common.config import load_config  # noqa: E402
from covert.common.inventory import PEOPLE, PLACES  # noqa: E402
from covert.listeners.model import build_listener  # noqa: E402
from covert.listeners.train import _encode, _texts, make_items, predict_probs  # noqa: E402

t0 = time.time()
OUT = ROOT / "results" / ("toy" if "--toy" in sys.argv else "") / "step1"
OUT.mkdir(parents=True, exist_ok=True)
TOY = "--toy" in sys.argv
cfg = load_config([], None, TOY)
max_len = cfg.listeners.max_len
res = {"max_len": max_len}
print("estimated runtime: 3-5 min (loads Qwen2.5-1.5B twice)")

train, dev = ST.eps_map(cfg, "train"), ST.eps_map(cfg, "dev")
tr_items = make_items(ST.load_convs(cfg, "warmstart_convs", "train")[:200], train)
dv_items = make_items(ST.load_convs(cfg, "warmstart_convs", "dev"), dev)

model, tok, device = build_listener(cfg)
print(f"\n== (a) truncation ==  tokenizer truncation_side={tok.truncation_side!r} padding_side={tok.padding_side!r} max_len={max_len}")
texts = _texts("B", tr_items)
lens = np.array([len(tok(t)["input_ids"]) for t in texts])
over = lens > max_len
res["truncation"] = {"side": tok.truncation_side, "len_min": int(lens.min()), "len_median": float(np.median(lens)),
                     "len_p90": float(np.percentile(lens, 90)), "len_max": int(lens.max()), "frac_over": float(over.mean())}
print(f"  B-view length: min {lens.min()}  median {np.median(lens):.0f}  p90 {np.percentile(lens, 90):.0f}  max {lens.max()}  "
      f"fraction > {max_len}: {over.mean():.3f}")
lc = np.array([len(tok(t)["input_ids"]) for t in _texts("C", tr_items)])
print(f"  C-view length: median {np.median(lc):.0f}  max {lc.max()}")
for k, i in enumerate(np.where(over)[0][:3]):
    ids = tok(texts[i], truncation=True, max_length=max_len)["input_ids"]
    kept = tok.decode(ids)
    print(f"  --- truncated example {k+1} (orig {lens[i]} tokens) ---")
    print(f"  STARTS: {kept[:200]!r}")
    print(f"  ENDS:   {kept[-200:]!r}")
    print(f"  chat survives: {'<chat>' in kept}   question survives: {kept.rstrip().endswith('venue?')}")

print("\n== (b) prediction distribution of the saved trained listeners on dev ==")
for view in ("B", "C"):
    p = ROOT / "artifacts" / ("toy" if TOY else "") / "listeners_r0" / view / "model.pt"
    if not p.exists():
        print(f"  {view}: no saved trained model at {p}")
        continue
    model.load_state_dict(torch.load(p, map_location="cpu"), strict=False)
    h, v = predict_probs(model, tok, device, view, dv_items, cfg)
    ph, pv = Counter(h.argmax(1).tolist()), Counter(v.argmax(1).tolist())
    gold_h, gold_v = Counter(i["helper"] for i in dv_items), Counter(i["venue"] for i in dv_items)
    both = float(np.mean((h.argmax(1) == np.array([i["helper"] for i in dv_items])) & (v.argmax(1) == np.array([i["venue"] for i in dv_items]))))
    res[f"pred_{view}"] = {"helper_pred": {PEOPLE[k]: c for k, c in ph.most_common()}, "venue_pred": {PLACES[k]: c for k, c in pv.most_common()},
                           "helper_gold": {PEOPLE[k]: c for k, c in gold_h.most_common()}, "venue_gold": {PLACES[k]: c for k, c in gold_v.most_common()},
                           "mean_max_prob_helper": float(h.max(1).mean()), "mean_max_prob_venue": float(v.max(1).mean()), "dev_both": both,
                           "n_distinct_pred_helper": len(ph), "n_distinct_pred_venue": len(pv)}
    print(f"  {view}: dev both {both:.3f}; distinct predicted helpers {len(ph)}/10, venues {len(pv)}/10; "
          f"mean max-prob helper {h.max(1).mean():.2f} venue {v.max(1).mean():.2f}")
    print(f"     predicted helpers: {dict((PEOPLE[k], c) for k, c in ph.most_common(4))}  gold: {dict((PEOPLE[k], c) for k, c in gold_h.most_common(4))}")
    print(f"     predicted venues:  {dict((PLACES[k], c) for k, c in pv.most_common(4))}  gold: {dict((PLACES[k], c) for k, c in gold_v.most_common(4))}")
del model
torch.cuda.empty_cache()

print("\n== (c) gradient flow (fresh model, one batch of 8 B-view examples) ==")
model, tok, device = build_listener(cfg)
model.train()
groups = {"lora_A": [], "lora_B": [], "head_helper": [], "head_venue": [], "other_trainable": []}
for n, p in model.named_parameters():
    if p.requires_grad:
        g = next((k for k in ("lora_A", "lora_B", "head_helper", "head_venue") if k in n), "other_trainable")
        groups[g].append((n, p))
batch = tr_items[:8]
ids, att = _encode(tok, _texts("B", batch), max_len, device)
hh, vv = model(ids, att)
loss = F.cross_entropy(hh, torch.tensor([i["helper"] for i in batch], device=device)) + F.cross_entropy(vv, torch.tensor([i["venue"] for i in batch], device=device))
loss.backward()
res["grad"] = {"loss": float(loss.detach())}
for g, ps in groups.items():
    norms = [float(p.grad.norm()) if p.grad is not None else None for _, p in ps]
    nz = [x for x in norms if x]
    res["grad"][g] = {"n_params_tensors": len(ps), "n_with_grad": sum(x is not None for x in norms), "n_nonzero": len(nz),
                      "total_norm": float(np.sqrt(sum(x * x for x in nz))) if nz else 0.0}
    print(f"  {g:16s} tensors {len(ps):4d}  with grad {sum(x is not None for x in norms):4d}  non-zero {len(nz):4d}  total norm {res['grad'][g]['total_norm']:.4g}")
print("  note: lora_B starts at zero, so lora_A gradients are exactly zero at step 1 by construction; lora_B and heads must be non-zero.")
print(f"  loss at init: {float(loss.detach()):.3f} (2*ln(10) = {2*np.log(10):.3f} for uniform predictions)")
res["elapsed_min"] = round((time.time() - t0) / 60, 2)
(OUT / "diag.json").write_text(json.dumps(res, indent=2))
print(f"\nwrote {OUT}/diag.json  ({res['elapsed_min']} min)")
