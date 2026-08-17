"""Complexity feature tests.

These features are what the escalation policy reads instead of the text, so the
computed properties — polarity_flip and emoji_text_conflict — are the ones with
real money attached to them.
"""

from __future__ import annotations

import pytest

from air_classifier.pipeline.features import extract_features
from air_classifier.preprocess.lexicon import LexiconScore, score_text


def test_every_field_is_populated_from_one_realistic_review() -> None:
    features = extract_features("Great battery BUT awful camera!! Very disappointing.")

    assert features.token_count == 7
    assert features.char_count == 52
    assert features.contrastive_count == 1
    assert features.intensifier_count == 1
    assert features.exclamation_count == 2
    assert features.positive_hits >= 1
    assert features.negative_hits >= 1
    assert features.negation_count == 0
    assert 0.0 < features.uppercase_ratio < 1.0
    assert features.question_only is False
    assert features.has_url_or_code is False


def test_counts_come_from_the_lexicon_pass() -> None:
    text = "I would not say it is great, however it is not terrible either"
    features = extract_features(text)
    score = score_text(text)

    assert features.negation_count == score.negation_count
    assert features.contrastive_count == score.contrastive_count
    assert features.positive_hits == score.positive_hits
    assert features.negative_hits == score.negative_hits
    assert features.lexical_polarity == score.polarity


def test_a_precomputed_score_is_reused_rather_than_recomputed() -> None:
    supplied = LexiconScore(
        polarity=-0.9,
        positive_hits=3,
        negative_hits=4,
        negation_count=5,
        intensifier_count=6,
        contrastive_count=7,
        emoji_polarity=0.8,
        matched_terms=("supplied",),
    )

    features = extract_features("nothing in this text supports any of that", lexicon_score=supplied)

    assert features.lexical_polarity == -0.9
    assert features.positive_hits == 3
    assert features.negation_count == 5
    assert features.contrastive_count == 7
    assert features.emoji_polarity == 0.8


def test_out_of_range_scores_are_clamped_to_the_schema_bounds() -> None:
    supplied = LexiconScore(polarity=5.0, emoji_polarity=-5.0, matched_terms=("x",))

    features = extract_features("anything", lexicon_score=supplied)

    assert features.lexical_polarity == 1.0
    assert features.emoji_polarity == -1.0


# ── Computed triggers ─────────────────────────────────────────────────────────


def test_polarity_flip_fires_when_both_polarities_are_present() -> None:
    assert extract_features("great battery but awful camera").polarity_flip is True


def test_polarity_flip_stays_off_for_a_single_sided_text() -> None:
    assert extract_features("great battery and a great screen").polarity_flip is False


def test_emoji_text_conflict_fires_when_the_emoji_contradict_the_prose() -> None:
    features = extract_features("Fantastic support \U0001f621\U0001f621")

    assert features.emoji_text_conflict is True


def test_emoji_text_conflict_stays_off_when_they_agree() -> None:
    features = extract_features("Fantastic support \U0001f60d\U0001f60d")

    assert features.emoji_text_conflict is False


def test_emoji_text_conflict_needs_both_sides_to_be_strong() -> None:
    # A neutral face is not a contradiction, however positive the prose is.
    features = extract_features("Fantastic support \U0001f610")

    assert features.emoji_text_conflict is False


# ── Surface features ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("THIS IS AWFUL", 1.0),
        ("this is awful", 0.0),
        ("", 0.0),
        ("1234 !!!", 0.0),
    ],
)
def test_uppercase_ratio_counts_cased_letters_only(text: str, expected: float) -> None:
    assert extract_features(text).uppercase_ratio == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Where is my order?", True),
        ("Is the battery any good? Does it last?", True),
        ("Where is my order? It is very late.", False),
        ("It never arrived!", False),
        ("Great product", False),
    ],
)
def test_question_only_requires_the_whole_text_to_be_a_question(text: str, expected: bool) -> None:
    assert extract_features(text).question_only is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See https://example.com/help for the fix", True),
        ("Go to www.example.com and read it", True),
        ('It throws {"error": 500} on every call', True),
        ("```\nimport foo\n```", True),
        ("The battery lasts all day and I love it", False),
    ],
)
def test_url_and_code_are_flagged(text: str, expected: bool) -> None:
    assert extract_features(text).has_url_or_code is expected


def test_exclamations_are_counted_across_scripts() -> None:
    assert extract_features("Awful!! Just awful\uff01").exclamation_count == 3


def test_cjk_text_is_tokenised_per_character() -> None:
    """Without this, a Japanese review is one token and never looks long."""
    features = extract_features("この製品はとても良いです")

    assert features.token_count == len("この製品はとても良いです")


def test_empty_text_produces_a_neutral_feature_set() -> None:
    features = extract_features("")

    assert features.token_count == 0
    assert features.char_count == 0
    assert features.lexical_polarity == 0.0
    assert features.polarity_flip is False
    assert features.emoji_text_conflict is False
