import pytest

from covert.speaker.reward import reward_from


def test_leak_and_sense():
    assert reward_from(True, 0.9, (1, 1), (0, 0), 1.0, 0.5) == -1
    assert reward_from(False, 0.4, (1, 1), (0, 0), 1.0, 0.5) == -1


def test_prob_formula():
    r = reward_from(False, 0.9, (0.91, 0.88), (0.10, 0.18), 1.0, 0.5)
    assert r == pytest.approx(0.895 - 0.14)
    assert reward_from(False, 0.9, (0.8, 0.6), (0.2, 0.2), 2.0, 0.5) == pytest.approx(0.7 - 0.4)


def test_logprob_mode():
    r = reward_from(False, 0.9, (1.0, 1.0), (0.1, 0.1), 1.0, 0.5, "logprob")
    assert r == pytest.approx(1.0 - 0.0)
    assert reward_from(False, 0.9, (0.1, 0.1), (1.0, 1.0), 1.0, 0.5, "logprob") == pytest.approx(-1.0)
