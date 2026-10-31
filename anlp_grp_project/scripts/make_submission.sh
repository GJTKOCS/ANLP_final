#!/usr/bin/env bash
# Builds <team_name>-Final.zip: code, data/sample, results, paper+slides PDFs (if built), viewer. Fails on secret patterns.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$(python3 -c "from covert.common.config import load_config; print(load_config(profile='cpu_debug').team_name.replace(' ',''))")
OUT="../${NAME}-Final.zip"
rm -f "$OUT"
zip -qr "$OUT" . -x '.venv/*' 'artifacts/*' 'runs/*' 'data/*' '!data/sample/*' '*.pt' '*.safetensors' '.env' 'wandb/*' '__pycache__/*' '*/__pycache__/*' '*.egg-info/*' '.git/*' '*.zip' 'results/toy/*' '*.DS_Store' '.pytest_cache/*' '*key_DO_NOT_SHARE.json'
if unzip -p "$OUT" -x scripts/make_submission.sh | grep -aE '(hf_[A-Za-z0-9]{30,}|wandb_v1_|sk-[A-Za-z0-9]{20,})' >/dev/null; then
  echo "SECRET PATTERN FOUND in archive; aborting"; rm -f "$OUT"; exit 1
fi
ls -la "$OUT"; unzip -l "$OUT" | tail -40
