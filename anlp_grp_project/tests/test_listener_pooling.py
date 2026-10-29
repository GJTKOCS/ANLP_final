import torch

from covert.listeners.model import last_token_pool


def test_pool_left_and_right_padding():
    hs = torch.arange(2 * 4 * 1, dtype=torch.float).reshape(2, 4, 1)
    right = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    left = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    assert last_token_pool(hs, right).squeeze(-1).tolist() == [2.0, 5.0]
    assert last_token_pool(hs, left).squeeze(-1).tolist() == [3.0, 7.0]
