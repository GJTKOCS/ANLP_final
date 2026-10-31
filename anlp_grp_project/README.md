# Speaking Privately in Public — 5MinsBoss (ANLP final)

Multi-turn GRPO: agent A (Qwen2.5-3B + LoRA) learns to hide a plan (helper + venue) in references to a *fresh shared
history* while chatting about a cover topic; insider B (has the memories) should decode it, outsider C (chat only) should not.
In the reported run B and C are Qwen2.5-14B-Instruct prompted zero-shot, plus a held-out outsider C_alt never used in training.
Paper: `paper/main.pdf` · slides: `slides/slides.pdf` · repo: https://github.com/GJTKOCS/ANLP_final

![scoreboard](paper/figures/scoreboard.png)

## Quickstart (laptop, CPU)
```bash
make env            # pip install requirements + package (Python 3.10+)
make smoke          # unit tests + full toy pipeline on CPU (~8 min; tiny random model; needs the Qwen tokenizer in the HF cache)
```
## Train on Jarvislabs (A100 40/80 GB or A6000; give the instance >= 100 GB storage)
```bash
git clone https://github.com/GJTKOCS/ANLP_final.git && cd ANLP_final/anlp_grp_project   # or upload this folder
bash scripts/setup.sh                       # installs deps, checks the GPU, runs the unit tests (~2 min)
export HF_TOKEN=<hugging-face WRITE token>  # optional: backs up artifacts to private repos under GJTKOCS
export WANDB_API_KEY=<wandb key>            # optional: logs to gjai2006-iiit-hyderabad/ANLP_final
bash scripts/run.sh                         # runs EVERYTHING in tmux, unattended; safe to re-run, it resumes
```
Watch: `tmux attach -t covert` (detach with Ctrl-b d) · progress: `covert status` · log: `runs/pipeline.out`, `runs/pipeline_log.md`.
If the instance restarts or a stage crashes, run `bash scripts/run.sh` again: finished stages are skipped, GRPO resumes from its last checkpoint.
Limited GPU budget: `bash scripts/run.sh --skip-optional`. Out of GPU memory: `bash scripts/run.sh hw.gen_bs=16 hw.grpo_mb=4`.
When it ends: **stop the Jarvislabs instance** (hourly billing). Then on any machine: `covert eval human-form`, `make paper`, `make slides`, `make submission`.

## Results
<!-- RESULTS:START -->
Values in % with 95% bootstrap CI (1000 resamples over test episodes); chance: 10% per slot, 1% both. Reward is in [-1,1] (mean).

| Cond | B both | C both | Insider adv | Joint succ | Leak | Sense pass | Reward | strong-C both | C_alt both (held-out) |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 17.5 [12.5, 23.5] | 20.0 [15.0, 25.5] | -2.5 [-7.0, 2.0] | 3.5 [1.0, 6.0] | 60.5 [54.0, 67.0] | 41.5 [34.5, 48.5] | -0.794 | TODO | 19.0 [14.0, 25.0] |
| S0_trained | 4.0 [1.5, 7.0] | 5.0 [2.5, 8.5] | -1.0 [-5.0, 3.0] | 4.0 [1.5, 7.0] | 60.5 [54.0, 67.0] | 41.5 [34.5, 48.5] | -0.798 | TODO | TODO |
| S1 | 13.5 [9.0, 18.0] | 2.5 [0.5, 5.0] | 11.0 [5.5, 16.5] | 13.5 [9.0, 18.0] | 6.5 [3.5, 10.0] | 51.5 [44.5, 58.0] | -0.429 | TODO | 2.5 [0.5, 5.0] |
| S1_trained | 2.0 [0.5, 4.0] | 1.0 [0.0, 2.5] | 1.0 [-1.0, 3.5] | 2.0 [0.5, 4.0] | 6.5 [3.5, 10.0] | 51.5 [44.5, 58.0] | -0.507 | TODO | TODO |
| S3 | 10.5 [6.5, 14.5] | 3.5 [1.0, 6.0] | 7.0 [2.0, 12.0] | 10.5 [6.5, 14.5] | 6.5 [3.5, 10.0] | 59.0 [52.0, 65.5] | -0.381 | TODO | 4.5 [2.0, 7.5] |
| S3_trained | 1.5 [0.0, 3.5] | 1.0 [0.0, 2.5] | 0.5 [-1.5, 2.5] | 1.5 [0.0, 3.5] | 6.5 [3.5, 10.0] | 59.0 [52.0, 65.5] | -0.448 | TODO | TODO |
<!-- RESULTS:END -->

`S0/S1/S3` = base / SFT / GRPO speaker scored by the prompted 14B listeners (the reported results). `*_trained` rows score the same
transcripts with the trained 1.5B listeners, which stayed at guessing level (reported as a failed attempt, see paper Sec. 6).
`TODO` cells were not run: the strong-C evaluation and C_alt on the trained-listener rows did not fit the compute budget.
GRPO S3 uses `best_adapter.pt` (best dev reward, step 50).

### Commands that produced the results
```bash
bash scripts/run.sh data.n_train=1000 data.n_dev=150 data.n_test=200 data.warm_k_train=3 hw.grpo.steps=100 \
     hw.big_gen_bs=48 hw.grpo_mb=16 hw.sft_mb=16 listeners.mode=prompted   # full pipeline on one H100 80 GB
covert eval report                                   # tables/macros above and in paper/tables/
python3 scripts/step0_analysis.py                    # GRPO reward decomposition (results/step0/)
python3 scripts/step1_listener_diag.py               # trained-listener diagnosis (results/step1/)
python3 scripts/b_turn_naming.py <run_dir>           # B-speaker naming stats (results/b_turn_naming.json)
python3 scripts/make_extra_macros.py <run_dir>       # paper/tables/extra_macros.tex
python3 scripts/make_slide_figs.py <run_dir>         # slides/figures/
make paper slides submission
```

## Links (required by the course)
- Weights & Biases, GRPO run `speaker_r1_lam1.0`: https://forge.coreweave.com/wandb/gjai2006-iiit-hyderabad/ANLP_final/runs/otqruoj8
- Hugging Face (private; access on request):
  - episodes, warm-start conversations, transcripts: https://huggingface.co/datasets/GJTKOCS/covert-data
  - speaker adapters (SFT + GRPO): https://huggingface.co/GJTKOCS/covert-speaker
  - trained listeners: https://huggingface.co/GJTKOCS/covert-listeners
- Code: https://github.com/GJTKOCS/ANLP_final

## Notes / ASSUMPTIONs
- **The reported run used prompted listeners** (`listeners.mode=prompted`: the same 14B model zero-shot as B and C), not trained ones, plus a held-out outsider `covert eval calt`. Trained listeners did not beat a guessing baseline; see `docs/IMPLEMENTATION_CHANGES.md`.
- ASSUMPTION: the big generator (memories, warm start, prompted listeners, S4) is Qwen2.5-14B-Instruct in bf16 on >=38 GB GPUs (7B otherwise), not 32B-AWQ, to avoid the autoawq dependency and keep plain-HF generation fast. For 32B: `pip install autoawq` and see `configs/hardware/a100.yaml`. The model used is recorded in manifests; state it in the paper.
- ASSUMPTION: no vLLM path (plain HF `generate` everywhere); versions are ranges, code avoids version-fragile APIs.
- Option scoring uses the first branching token when option first tokens are distinct (fast path), exact continuation scoring otherwise.
- B speaker and sense judge = the profile's base policy model with adapters disabled; `tau` comes from `covert judge calibrate`.
- Go/no-go: G2 and G3 stop the pipeline on FAIL (later stages would be meaningless; override with `--continue-on-fail`); G1, G4, G5 warn and continue. G5 evaluates the first 50 GRPO steps after the round-1 stage.
- Toy runs write only under `*/toy/` and `results/toy/`; they never touch `paper/` or `viewer/`.
- To redo a finished GRPO stage from scratch, delete `artifacts/speaker_rR_lamX/` (re-running resumes from its checkpoint).

## Ethics
All content is fictional and benign. The project studies covert channels between AI agents as a safety-relevant phenomenon.
Speaker adapters go only to private HF repos; data, listeners, code and aggregate results may be public.
