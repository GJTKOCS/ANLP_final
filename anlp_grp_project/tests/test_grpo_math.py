import numpy as np
import torch

from covert.speaker.grpo import group_advantages, grpo_loss


def test_advantages():
    a = group_advantages(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))
    assert abs(a[0].mean()) < 1e-9 and a[0, 2] > 0 > a[0, 0]
    assert (a[1] == 0).all()


def test_kl_zero_when_equal_and_loss_mask():
    lp = torch.randn(2, 5)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    loss, st = grpo_loss(lp, lp.clone(), lp.clone(), torch.tensor([1.0, -1.0]), mask, 0.2, 0.04)
    assert st["kl"] == 0.0
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)         # ratio 1: mean(-adv) over equal-sized-weights = 0


def test_clipping_and_mask_only():
    lp_old = torch.zeros(1, 3)
    lp_new = torch.tensor([[0.5, 0.5, 5.0]])                         # third token masked out
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    loss, _ = grpo_loss(lp_new, lp_old, lp_new.clone(), torch.tensor([1.0]), mask, 0.2, 0.0)
    assert loss.item() == -1.2 * 1.0 + 0.0 or abs(loss.item() + 1.2) < 1e-6   # clipped at 1+eps for A>0
