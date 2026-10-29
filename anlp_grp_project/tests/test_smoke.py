"""End-to-end toy pipeline on CPU (tiny random model). Needs the Qwen tokenizer in the HF cache."""
import subprocess
import sys


def test_toy_pipeline():
    r = subprocess.run([sys.executable, "-m", "covert.cli", "pipeline", "run", "--toy"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    log = open("runs/toy/pipeline_log.md").read()
    assert "FAILED" not in log, log
    a = subprocess.run([sys.executable, "-m", "covert.cli", "audit", "--toy"], capture_output=True, text=True)
    assert a.returncode == 0, a.stdout
