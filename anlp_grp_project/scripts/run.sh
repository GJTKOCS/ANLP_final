#!/usr/bin/env bash
# Starts the WHOLE pipeline unattended in a tmux session (survives disconnects). Safe to re-run: it resumes.
#   bash scripts/run.sh                 # full run
#   bash scripts/run.sh --skip-optional # skip S4 / lambda sweeps
#   tmux attach -t covert               # watch (Ctrl-b d to detach);   covert status   # progress
set -uo pipefail
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
if [[ "${1:-}" != "--inner" ]]; then
  ARGS="$*"
  if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t covert 2>/dev/null || true
    tmux new-session -d -s covert "bash scripts/run.sh --inner $ARGS; echo; echo PIPELINE ENDED - stop the Jarvislabs instance; exec bash"
    echo "Started in tmux session 'covert'.  Watch: tmux attach -t covert   Progress: covert status   Log: runs/pipeline.out"
  else
    nohup bash scripts/run.sh --inner $ARGS > runs_nohup.out 2>&1 &
    echo "tmux not found; running with nohup. Log: runs_nohup.out and runs/pipeline.out"
  fi
  exit 0
fi
shift
mkdir -p runs
python3 -m covert.cli preflight || { echo "preflight failed; fix the issue above"; exit 1; }
python3 -m covert.cli pipeline run "$@" 2>&1 | tee -a runs/pipeline.out
python3 -m covert.cli audit 2>&1 | tee -a runs/pipeline.out
echo "DONE. Results: results/results.md   Showcase: viewer/showcase.json   Next: covert eval human-form"
echo "REMINDER: stop the Jarvislabs instance now to stop billing."
