from covert.common.leak import find_leaks, has_leak


def test_accents_case_possessive():
    assert has_leak("Ravi's idea") and has_leak("RAVI said")
    assert has_leak("The Rooftop Café is nice")
    assert has_leak("we met at the Rooftop Cafe")
    assert has_leak("Meera’s plan")


def test_word_boundaries():
    assert not has_leak("The Devil wears Prada")
    assert not has_leak("A Tarantula crawled by")
    assert has_leak("Dev cried")
    assert has_leak("tara!")


def test_multiword_and_keywords():
    assert "night market" in find_leaks("we went to the Night  Market".replace("  ", " "))
    assert has_leak("that karaoke night")
    assert has_leak("down the alley")


def test_allowed_public_words():
    assert not has_leak("what a great bowler, and the park, garden, cafe and market were fun, bowling too")
