#!/usr/bin/env bash
# One-time setup on a fresh Jarvislabs instance (A100 or A6000). Usage:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install -q -r requirements.txt
python3 -m pip install -q -e . --no-deps
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "No GPU visible. Pick a GPU instance on Jarvislabs."
p = torch.cuda.get_device_properties(0)
print(f"GPU OK: {p.name}, {p.total_memory/2**30:.0f} GB")
PY
python3 -m pytest -q --ignore=tests/test_smoke.py
echo
echo "Setup done. Next:"
echo "  export HF_TOKEN=<your hugging face WRITE token>"
echo "  export WANDB_API_KEY=<your wandb key>"
echo "  bash scripts/run.sh"
echo "REMINDER: Jarvislabs bills hourly. Stop the instance when the run is finished."
