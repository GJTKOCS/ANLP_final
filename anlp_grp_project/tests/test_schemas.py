import pytest

from covert.common.schemas import validate


def test_roundtrip_and_invalid():
    ep = {"episode_id": "a", "split": "train", "birthday_person": "Ravi", "memories": ["x"] * 8, "cover_topic": "c", "pre_switch": [], "theme_ids": []}
    validate("episode_public", ep)
    with pytest.raises(ValueError):
        validate("episode_public", {**ep, "memories": ["x"]})
    with pytest.raises(ValueError):
        validate("episode_private", {"episode_id": "a", "secret": {"helper": "x"}, "anchors": {}, "seed": 1})
    with pytest.raises(ValueError):
        validate("pred", {"episode_id": "a", "view": "B", "pred": {}, "probs": {"helper": [0.1], "venue": [0.1] * 10}})
