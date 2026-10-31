from covert.cli import build_parser


def test_check_with_overrides_keeps_positional():
    a = build_parser().parse_args(["check", "G2", "--toy", "data.n_train=5", "hw.grpo.steps=3"])
    assert a.which == "G2" and a.overrides == ["data.n_train=5", "hw.grpo.steps=3"] and a.toy


def test_check_plain_and_human_score():
    assert build_parser().parse_args(["check", "G1"]).which == "G1"
    a = build_parser().parse_args(["eval", "human-score", "f.csv", "speaker.lr=1e-5"])
    assert a.csv == "f.csv" and a.overrides == ["speaker.lr=1e-5"]


def test_other_commands_unchanged():
    a = build_parser().parse_args(["speaker", "grpo", "--round", "1", "--lam", "1.0", "hw.grpo.steps=3"])
    assert a.round == 1 and a.overrides == ["hw.grpo.steps=3"]
