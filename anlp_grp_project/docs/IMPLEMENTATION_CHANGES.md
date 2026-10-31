# What changed in the implementation, and what the run showed

Team 5MinsBoss — "Speaking Privately in Public". This document records every place where the Jarvislabs run departs
from the original build prompt, the evidence that forced each change, and the final results. All numbers are copied
from the run logs (copied to `~/Documents/ANLP_Final/vm_run/`); the paper reads the same numbers through generated
macros (`paper/tables/macros.tex`, `paper/tables/extra_macros.tex`), never by hand.

## 1. Summary of changes

| Area | Original design | What actually ran | Why |
|---|---|---|---|
| Listeners B and C | Qwen2.5-1.5B + LoRA, **trained**; C **retrained every round** | The **same 14B model prompted zero-shot** (`listeners.mode=prompted`): B sees memories + chat; C sees chat + rules + 3 solved examples. Used for the GRPO reward and the evaluation | Trained listeners never beat a guessing baseline (Section 3) |
| Held-out outsider | — | **C_alt**: same 14B model, reworded instructions, 3 different solved examples, never used in the reward | So the B-up/C-down result is not true by construction |
| Arms race | C adapts each round; arms-race table | **Not run** (listeners are fixed) | Consequence of the first row |
| A prompt | Spec role prompt | Spec prompt **plus** memory pointers ("memories involving the helper: #3 ...") and a hinting example; identical for all conditions | The 14B warm-start speaker referenced the wrong memories (Section 4) |
| Warm-start generation | One sample per A turn | Leaking A turns re-sampled up to 8 times (**warm start only**; verified at every call site) | 67% of A turns leaked |
| Option scoring | As written | First letter capitalised | Lowercase venues depressed B's venue accuracy |
| Big generator | 32B-AWQ via vLLM | Qwen2.5-14B-Instruct bf16, plain HF `generate` | No `autoawq`/vLLM dependency |
| Go/no-go | G2, G3 hard; G3 bar 30% | G2 warning only; G3 bar 10% | See Sections 3–4 |
| Scope | 2 rounds, S2, S4, lambda sweep | One GRPO round; S0, S1, S3 only | Compute credit budget (single H100) |
| Scale | 1500/200/300 episodes, K=4, 150 steps | 1000/150/200 episodes, K=3, 100 steps | Budget |

## 2. What stayed the same

Memory banks written before the secret is drawn; validation, splits, held-out topics and themes for test, showcase ids
fixed before training; information boundaries in `views.py` (audit passes); reward `-1` on leak or sense < tau (tau =
0.50, calibrated), else `mean(pB) - lambda * mean(pC)` with lambda = 1; custom multi-turn GRPO (group-normalised
advantages, clipped surrogate, k3 KL to the SFT adapter, A-token loss, LoRA r = 16); base weights verified frozen
(checksum identical before and after GRPO); identical B speaker, judge, leak checker and episodes across conditions.

## 3. Trained listeners: failure, diagnosis, fix, still not usable

- Pipeline runs: B dev both-correct **2.7% at every epoch** in two runs (2 epochs at 1e-4; 5 epochs at 2e-4).
- **Diagnosis (Step 1):** B's predictions were non-finite (NaN) and collapsed to one class (Ravi, rooftop cafe) for
  all 150 dev items. This is why the score was identical every epoch. Initial loss 13.2 (uniform = 4.6) and head gradient
  norms of about 190 point to a very large scale of the pooled features, so training diverged. Truncation was ruled out:
  B views have a median of 545 and a maximum of 635 tokens, under the 768 limit, and truncation is from the left anyway.
  Gradients flowed (LoRA-B and heads non-zero at step 1; LoRA-A zero by construction).
- **Fix:** LayerNorm on the pooled features, skip non-finite updates, log the train loss, and fail loudly on NaN outputs.
- **Step 3, evaluation-only listeners after the fix** (trained on each condition's 1000 training transcripts, 5
  epochs, lr 1e-4, 18 min): losses decreased (B on S0: 4.74 to 3.87), but the best B dev both-correct was 4.7% / 3.3% / 4.0%
  for S0 / S1 / S3, no better than guessing among the names in the memory bank (~3.5%). Test B both 4.0 / 2.0 / 1.5.
  C still had occasional skipped non-finite steps. **Abandoned**; reported in the paper appendix as a negative result.

## 4. Warm start: evidence for the prompt changes

| Warm start | Prompted B both | Prompted C both | A leak | Good / 3000 |
|---|---|---|---|---|
| Spec prompt | 13.2% | 10.3% | 67% | — |
| + leak re-sampling | 9.5% | 4.6% | 5.8% | 164 |
| + memory pointers and hint example (used) | **20.0%** | **4.5%** | 2.2% | **295** |

## 5. Results (200 test episodes, %, 95% bootstrap CI)

| | S0 base | S1 SFT | S3 GRPO |
|---|---|---|---|
| A leak | 60.5 [54.0, 67.0] | 6.5 [3.5, 10.0] | 6.5 [3.5, 10.0] |
| Sense pass | 41.5 [34.5, 48.5] | 51.5 [44.5, 58.0] | 59.0 [52.0, 65.5] |
| B both | 17.5 [12.5, 23.5] | 13.5 [9.0, 18.0] | 10.5 [6.5, 14.5] |
| C both (in-loop) | 20.0 [15.0, 25.5] | 2.5 [0.5, 5.0] | 3.5 [1.0, 6.0] |
| C_alt both (held-out) | 19.0 [14.0, 25.0] | 2.5 [0.5, 5.0] | 4.5 [2.0, 7.5] |
| Insider advantage | -2.5 [-7.0, 2.0] | **11.0 [5.5, 16.5]** | 7.0 [2.0, 12.0] |
| B with wrong memories, both | 13.5 | 3.5 | 2.0 |
| B helper / C helper | 40.5 / 36.5 | 37.0 / 10.0 | 29.5 / 10.5 |
| B venue / C venue | 36.5 / 36.5 | 35.5 / 31.0 | 36.0 / 30.0 |

What the results support:
1. The untrained speaker leaks (60.5%), so C does as well as B.
2. **The SFT warm start creates the insider advantage** (11.0 points), and B's edge disappears when it is given another
   episode's memories (3.5%), so the edge comes from the shared memories.
3. **GRPO gives no measurable gain over SFT** on the insider advantage (7.0 vs 11.0, intervals overlap); sense pass rises
   from 51.5 to 59.0 (intervals overlap). The paper says this plainly.
4. **Reward decomposition** (4800 GRPO rollouts): penalised by the sense judge 32.3% overall (42.5% in the first 20 steps,
   23.1% in the last 20), by leaks 5.3% (6.7% to 4.2%); mean(pB) − mean(pC) on unpenalised rollouts flat (0.191 to 0.186; overall
   0.184 [0.173, 0.193]). Training reward went from −0.396 to −0.137, dev reward from −0.428 (step 25) to −0.068 (step 100). Both pB (0.393
   to 0.342) and pC (0.214 to 0.161) fell. GRPO learned to avoid penalties, not to hint better.
5. **Held-out outsider:** C_alt matches the in-loop C on every condition, so there is no sign the speaker exploited the
   outsider prompt it was trained against. B has no held-out counterpart.
6. **The advantage is in the helper slot**; venue accuracy is similar for B and C (venue hints are readable public clues).
7. **S3 checkpoint:** the evaluated S3 adapter is `best_adapter.pt`, the checkpoint with the best dev reward (step 50 of 100, dev reward -0.034).
8. **B speaker naming items:** the frozen B speaker still names an inventory item after its 3 re-samples in 24.0% / 19.0% / 11.5% of test
   conversations (S0 / S1 / S3), the true slot in 14.0% / 8.5% / 4.5%. For S1/S3 this does not raise C (C both 2.6% vs 2.5%; 0.0% vs 4.0%).
   The reward penalises only A leaks. Source: `scripts/b_turn_naming.py` -> `results/b_turn_naming.json`.
9. **Judge calibration:** tau = 0.50 kept 100% of 50 good and rejected 80% of 50 corrupted conversations (the good ones were pre-filtered at 0.5).
- Go/no-go: G1 FAIL (warning: 37.3% template fallback), G2 FAIL (warning), G3 FAIL for the trained listeners
  (superseded by prompted mode; the prompted pair scores 20.0% B vs 2.7% C on S3's dev transcripts), G4 PASS, G5 FAIL (final sense
  rate 0.65 < 0.8, leak 0.052 > 0.05; dev reward rose).

## 6. Data quality finding

37.3% of memory slots fell back to the template sentence after three failed LLM attempts, and only 2.2% of banks are fully
LLM-written. Template memories are generic and alike, which weakens the shared history hints can point to. This is
reported in Setup and Limitations.

## 7. Bugs fixed during the run (record only)

- argparse read `G2` as an override, so the check never ran → fixed, with a regression test.
- YAML 1.1 parsed `2e-4` as a string → numeric coercion of overrides.
- GRPO reference log-probs are computed before the policy forward pass (keeps policy gradients on under adapter switching).
- Listener NaN divergence → LayerNorm + non-finite guard (Section 3).
- Generated outputs (paper tables, viewer bundle) were regenerated from raw logs after copying results off the VM.
- Follow-up GPU use: Step 1 0.2 min, Step 2 (C_alt) 2 min, Step 3 18 min.

## 8. Limitations to state (no results)

Fixed, prompted listeners (no adaptive outsider, no arms race); in-loop insider without a held-out counterpart; template
fallbacks in memories; a penalty-dominated reward; one seed, one round, 200 test episodes, small fictional inventories;
memory pointers in A's prompt make the task easier for all speakers equally. Never claim the channel is secure.

## 9. Status (31 Oct)

Done: paper (`paper/main.pdf`, compiled locally with tectonic, 9 pages incl. references, no TODOs), new Section "From the Planned
Design to the Executed Study", showcase + bloopers in Analysis, references checked and corrected (Motwani et al. and Mathew et al. author
lists; Mathew et al. now cited as IJCNLP-AACL 2025; LoRA as ICLR 2022), slides rewritten from real results (`slides/slides.pdf`, figures from
`scripts/make_slide_figs.py`). Human ratings were not collected; the paper says so. Optional: fill `results/human/human_form.csv` and run
`covert eval human-score <filled.csv>`.
