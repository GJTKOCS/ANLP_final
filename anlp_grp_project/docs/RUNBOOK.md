# Runbook (one machine, one GPU)

Estimates assume one A100 40 GB; T4 is ~3-4x slower. `covert pipeline run` executes these in order, skipping finished stages.

| # | Stage (`--only`) | Priority | Est. h | Command | Success check / typical errors |
|---|---|---|---|---|---|
| 2 | data | must | 1-1.5 | `covert data build` | G1 PASS. generator is 14B bf16 (7B on small GPUs); model recorded in the manifest |
| 3 | warmstart | must | 1-2 | `covert data warmstart` | G2 PASS (prompted B>=50%, C<=10%, leak<=20%) |
| 4 | judge | must | 0.2 | `covert judge calibrate` | `runs/judge_calibration.json` (tau) |
| 5 | listeners_r0 | must | 0.5-1 | `covert listeners train --round 0` | G3 PASS (B dev >=30% and >=2x C) |
| 6 | sft | must | 0.5 | `covert speaker sft` | needs >=1 good warm-start conv |
| 7 | s0 | must | 0.5 | `covert transcripts --cond S0` | G4 |
| 8 | grpo_r1 | must | 2-4 | `covert speaker grpo --round 1 --lam 1.0` | G5; resumes from `ckpt.pt` after a crash |
| 9 | trans_r1, listeners_r1 | must | 1-1.5 | `transcripts --cond A1`, `listeners train --round 1` | |
| 10-11 | grpo_r2 ... listeners_r2 | should | 3-5 | as above, round 2 | |
| 12 | eval_s1/s0/s3 | must | 2-3 | `covert eval run --cond S1` ... | `artifacts/evalpair_*/metrics.json` |
| 13 | s2, eval_s2 | should | 1-1.5 | best-of-8 from S0 | |
| 14-15 | s4, strongc, lam05, lam2 | optional | 2-6 | | dropped first when time is short |
| 16 | report | must | CPU | `covert eval report` | `results/results.md`, figures, `viewer/showcase.json` |

Flags: `--until STAGE`, `--only STAGE`, `--force STAGE`, `--skip-optional`, `--profile NAME`, `--deadline ISO` (default 2026-10-31T06:00+05:30), `--continue-on-fail`.
Must stages are never skipped; optional, then should, stages that cannot finish before the deadline are skipped and logged.

Order of safety: run `make smoke` first. After data, run `covert audit`. Before the paper, run `covert audit` again.

Troubleshooting: CUDA OOM -> lower `hw` batch sizes via overrides (`hw.gen_bs=16 hw.grpo_mb=4`), or use `--profile t4`.
AWQ models need `autoawq` (requirements-gpu.txt). Slow generation: expected without vLLM. G-check FAIL: read `runs/checks/G*.json` for the fallback.
Jarvislabs: stop the instance when the pipeline ends.
Resume after any crash: just re-run `covert pipeline run` (GRPO resumes from its last 25-step checkpoint; listeners restart their stage).

## Quick commands (Jarvislabs)
`bash scripts/setup.sh` once, then `bash scripts/run.sh`. Everything else is automatic. Extra tools:
`covert status` (progress) · `covert check G2` (re-run a go/no-go) · `covert audit` · `covert eval report` · `covert eval human-form` / `covert eval human-score FILLED.csv`.
Overrides go after the command, e.g. `bash scripts/run.sh speaker.rounds=1 hw.grpo.steps=100`.
