"""GRPO end-to-end with MockListener/MockJudge on the tiny CPU model: runs, updates only LoRA, resumes."""
import torch

from covert.common.config import load_config
from covert.common.judge import Judge
from covert.common.llm import LM
from covert.common.scorer import OptionScorer
from covert.common.inventory import PEOPLE
from covert.data.episodes import build_split
from covert.mocks import MockJudge, MockListener
from covert.speaker.grpo import run_grpo


def _eps(cfg, n):
    pub, priv, _ = build_split(cfg, "train", n, None)
    return [{**p, "secret": q["secret"], "anchors": q["anchors"]} for p, q in zip(pub, priv)]


def test_grpo_runs_and_only_lora_changes(tmp_path):
    cfg = load_config(toy=True)
    lm = LM(cfg.hw["policy_model"], cfg.hw)
    base0 = {n: p.detach().clone() for n, p in lm.base.named_parameters()}
    eps = _eps(cfg, 6)
    res = run_grpo(cfg, lm, eps, eps[:2], MockListener("B"), MockListener("C"), MockJudge(), tmp_path, 1.0, 0.5, 2, None)
    assert (tmp_path / "metrics.jsonl").exists() and (tmp_path / "best_adapter.pt").exists()
    for n, p in lm.model.named_parameters():
        if "lora_" not in n:
            assert torch.equal(p, base0[n.replace("base_model.model.", "").replace(".base_layer", "")]), n
    # resume: calling again with the same steps is a no-op that does not crash
    run_grpo(cfg, lm, eps, eps[:2], MockListener("B"), MockListener("C"), MockJudge(), tmp_path, 1.0, 0.5, 2, None)


def test_generate_scorer_judge_toy():
    cfg = load_config(toy=True)
    lm = LM("toy-tiny", cfg.hw)
    out = lm.generate([lm.chat_prompt([{"role": "user", "content": "hi"}])] * 3, None, 0.9, 0.95, 8, 2)
    assert len(out) == 3 and all("\n" not in o["text"] for o in out)
    pr = OptionScorer(lm, 4).probs(["hello\nHelper:", "yo\nHelper:"], PEOPLE)
    assert pr.shape == (2, 10) and abs(pr.sum(1) - 1).max() < 1e-4
    s = Judge(lm, 4).score([("cricket", [["A", "hi"], ["B", "yo"]])])
    assert 0 <= s[0] <= 1
