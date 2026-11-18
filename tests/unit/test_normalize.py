# ruff: noqa: RUF001 — full-width look-alikes are the input these tests exist to fold.
"""Normalisation is part of the cache contract, so its output is pinned exactly."""

from __future__ import annotations

import pytest

from air_classifier_service.preprocess.normalize import normalize


def test_nfkc_folds_compatibility_forms() -> None:
    result = normalize("ｇｒｅａｔ ﬁnish", max_chars=100)

    assert result.text == "great finish"


def test_strips_zero_width_and_bidi_characters() -> None:
    text = "gre\u200bat\u200e su\ufeffpport"
    result = normalize(text, max_chars=100)

    assert result.text == "great support"


def test_control_characters_become_a_space_rather_than_glue() -> None:
    result = normalize("great\x00support\x07here", max_chars=100)

    assert result.text == "great support here"


def test_collapses_whitespace_runs_and_trims() -> None:
    result = normalize("  the   battery\n\nlife \t is  great  ", max_chars=100)

    assert result.text == "the battery life is great"


def test_preserves_emoji_and_sentence_punctuation() -> None:
    result = normalize("Superb service!! Would I buy again? \U0001f60d\U0001f44d", max_chars=100)

    assert result.text == "Superb service!! Would I buy again? \U0001f60d\U0001f44d"


def test_short_text_is_not_truncated() -> None:
    result = normalize("great product", max_chars=100)

    assert result.truncated is False
    assert result.text == "great product"


def test_original_length_is_measured_before_cleaning() -> None:
    result = normalize("  hello  ", max_chars=100)

    assert result.original_length == 9
    assert result.text == "hello"


def test_truncates_on_a_word_boundary() -> None:
    result = normalize("The battery life is great but the camera is awful", max_chars=20)

    assert result.truncated is True
    assert result.text == "The battery life is"
    assert len(result.text) <= 20


def test_falls_back_to_a_hard_cut_when_there_is_no_usable_boundary() -> None:
    result = normalize("a" + "b" * 50, max_chars=10)

    assert result.truncated is True
    assert len(result.text) == 10


def test_word_boundary_search_does_not_gut_the_text() -> None:
    # The only space sits early on, so honouring it would throw away most of
    # the budget; the hard cut is the lesser evil.
    result = normalize("ok " + "z" * 60, max_chars=40)

    assert len(result.text) == 40


@pytest.mark.parametrize("max_chars", [0, -1])
def test_rejects_a_nonsensical_budget(max_chars: int) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        normalize("anything", max_chars=max_chars)
