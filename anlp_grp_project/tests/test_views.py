from covert.common import views
from covert.data.memories import plan_bank
import random

EP = {"episode_id": "t-1", "birthday_person": "Ravi", "cover_topic": "cricket", "memories": [f"memory number {i} with Arjun at the bowling alley" for i in range(8)],
      "pre_switch": [["A", "PRESWITCH_LINE_ONE"], ["B", "PRESWITCH_LINE_TWO"], ["A", "PRESWITCH_LINE_THREE"]]}
SEC = {"helper": "Kabir", "venue": "food court"}
POST = [["A", "hi there"], ["B", "hello"]]


def test_c_view_has_no_memories_or_secret():
    t = views.listener_text("C", None, POST)
    assert "memory number" not in t and "<memories>" not in t and "Kabir" not in t and "PRESWITCH" not in t


def test_b_view_has_memories_no_preswitch():
    t = views.listener_text("B", EP["memories"], POST)
    assert "memory number 3" in t and "PRESWITCH" not in t and "Cover topic" not in t and "Kabir" not in t


def test_b_speaker_has_no_secret():
    m = views.b_speaker_messages(EP, POST)
    text = " ".join(x["content"] for x in m)
    assert "the helper (who will bring" not in text
    # secret values only appear in the banned-names list, never as an assignment
    assert f"is {SEC['helper']}" not in text


def test_a_prompt_has_secret():
    text = " ".join(x["content"] for x in views.a_messages(EP, SEC, POST))
    assert "is Kabir" in text and "food court" in text


def test_judge_input_minimal():
    t = views.judge_text("cricket", POST)
    assert "cricket" in t and "hi there" in t and "memory number" not in t and "Ravi" not in t


def test_prompted_c_no_memories():
    t = views.prompted_c_text(POST, [(POST, SEC)])
    assert "memory number" not in t


def test_plan_bank_constraints():
    themes = [f"theme {i}" for i in range(30)]
    for s in range(20):
        plan = plan_bank(random.Random(s), themes)
        assert len(plan["slots"]) == 8
