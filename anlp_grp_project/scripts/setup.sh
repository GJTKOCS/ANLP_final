#!/usr/bin/env bash
# One-time setup on a fresh Jarvislabs VM (plain Ubuntu or PyTorch image). Usage:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if ! python3 -m pip --version >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1; then
  sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv
fi
command -v tmux >/dev/null 2>&1 || sudo apt-get install -y tmux || true
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
python - <<'PY' || pip install -q torch
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
pip install -q -r requirements.txt
pip install -q -e . --no-deps
python - <<'PY'
import torch
assert torch.cuda.is_available(), "No GPU visible to PyTorch. Check `nvidia-smi` and the instance type."
p = torch.cuda.get_device_properties(0)
print(f"GPU OK: {p.name}, {p.total_memory/2**30:.0f} GB")
PY
python -m pytest -q --ignore=tests/test_smoke.py
echo
echo "Setup done. Next (same terminal, or run 'source .venv/bin/activate' first):"
echo "  export HF_TOKEN=<hugging face WRITE token>"
echo "  export WANDB_API_KEY=<wandb key>"
echo "  bash scripts/run.sh ..."
echo "REMINDER: Jarvislabs bills hourly. Stop the instance when the run is finished."
