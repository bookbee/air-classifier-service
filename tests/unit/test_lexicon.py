"""Lexicon scoring tests.

Negation is the property this file exists for. Everything else here is table
hygiene and sanity; the negation cases are the ones that would make the whole
Tier-0/Tier-1 stack confidently wrong if they regressed.
"""

from __future__ import annotations

import pytest

from air_classifier.preprocess.lexicon import (
    CONTRASTIVE_MARKERS,
    DIMINISHERS,
    EMOJI_POLARITY,
    INTENSIFIERS,
    NEGATIONS,
    NEGATIVE_TERMS,
    POSITIVE_TERMS,
    score_text,
)

# ── Table hygiene ─────────────────────────────────────────────────────────────


def test_tables_are_large_enough_to_be_useful() -> None:
    assert len(POSITIVE_TERMS) >= 120
    assert len(NEGATIVE_TERMS) >= 120
    assert len(EMOJI_POLARITY) >= 40


def test_weights_carry_the_sign_of_their_table() -> None:
    assert all(0.0 < weight <= 1.0 for weight in POSITIVE_TERMS.values())
    assert all(-1.0 <= weight < 0.0 for weight in NEGATIVE_TERMS.values())
    assert all(-1.0 <= weight <= 1.0 for weight in EMOJI_POLARITY.values())


def test_emoji_table_covers_both_directions() -> None:
    assert sum(1 for weight in EMOJI_POLARITY.values() if weight > 0) >= 20
    assert sum(1 for weight in EMOJI_POLARITY.values() if weight < 0) >= 15


def test_tables_are_disjoint() -> None:
    """A word in two tables is a word whose second meaning is silently dropped."""
    terms = set(POSITIVE_TERMS) | set(NEGATIVE_TERMS)
    modifiers = set(INTENSIFIERS) | set(DIMINISHERS) | set(NEGATIONS) | set(CONTRASTIVE_MARKERS)

    assert not set(POSITIVE_TERMS) & set(NEGATIVE_TERMS)
    assert not set(INTENSIFIERS) & set(DIMINISHERS)
    assert not terms & modifiers


def test_intensifiers_amplify_and_diminishers_soften() -> None:
    assert all(multiplier > 1.0 for multiplier in INTENSIFIERS.values())
    assert all(0.0 < multiplier < 1.0 for multiplier in DIMINISHERS.values())


# ── Realistic sentences ───────────────────────────────────────────────────────

SENTENCES: list[tuple[str, int]] = [
    ("Absolutely superb build quality and a flawless finish", 1),
    ("The staff were friendly and the room was spotless", 1),
    ("Works perfectly, exactly as described", 1),
    ("I love this product, best purchase this year", 1),
    ("Delivery was quick and the packaging was immaculate", 1),
    ("Support resolved my issue in minutes, very helpful", 1),
    ("Terrible experience, the app crashes constantly", -1),
    ("Rude staff and an overpriced, laggy product", -1),
    ("The screen cracked after a week and support ignored me", -1),
    ("Waste of money, do not recommend", -1),
    ("Buggy, slow and completely unreliable", -1),
    ("Arrived damaged and the refund process was a nightmare", -1),
    ("The package arrived on Tuesday afternoon", 0),
    ("I ordered two of them for the office", 0),
]


@pytest.mark.parametrize(("text", "expected_sign"), SENTENCES)
def test_polarity_sign_matches_a_human_reading(text: str, expected_sign: int) -> None:
    polarity = score_text(text).polarity

    if expected_sign == 0:
        assert polarity == pytest.approx(0.0)
    else:
        assert polarity * expected_sign > 0


def test_polarity_stays_inside_the_schema_bounds() -> None:
    piled_on = "superb flawless perfect amazing excellent " * 20

    assert -1.0 <= score_text(piled_on).polarity <= 1.0
    assert -1.0 <= score_text(piled_on.replace("superb", "terrible")).polarity <= 1.0


def test_stronger_wording_scores_higher_than_milder_wording() -> None:
    assert score_text("superb service").polarity > score_text("fine service").polarity
    assert score_text("atrocious service").polarity < score_text("mediocre service").polarity


# ── Negation ──────────────────────────────────────────────────────────────────

NEGATION_CASES: list[tuple[str, int]] = [
    ("not great", -1),
    ("this is not great at all", -1),
    ("not very good", -1),
    ("never had a problem", 1),
    ("isn't bad", 1),
    ("the checkout isn\u2019t bad", 1),
    ("no issues at all", 1),
    ("I have never been disappointed", 1),
    ("the setup was not painless", -1),
    ("nothing broken in the box", 1),
]


@pytest.mark.parametrize(("text", "expected_sign"), NEGATION_CASES)
def test_negation_flips_the_terms_that_follow_it(text: str, expected_sign: int) -> None:
    polarity = score_text(text).polarity

    assert polarity * expected_sign > 0


def test_negation_is_counted_even_when_it_flips_nothing() -> None:
    score = score_text("I did not order this yet")

    assert score.negation_count == 1


def test_negated_positive_is_milder_than_an_outright_negative() -> None:
    """ "not great" is a complaint, not the same complaint as "terrible"."""
    assert score_text("terrible").polarity < score_text("not great").polarity < 0


def test_hits_are_counted_after_the_flip() -> None:
    score = score_text("never had a problem")

    assert score.positive_hits == 1
    assert score.negative_hits == 0


def test_negation_does_not_cross_a_clause_boundary() -> None:
    score = score_text("Not good. Amazing service.")

    assert score.polarity > 0
    assert score.positive_hits == 1
    assert score.negative_hits == 1


def test_negation_does_not_reach_beyond_its_window() -> None:
    score = score_text("I would not describe the sort of thing anyone calls great")

    assert score.polarity > 0


# ── Intensifiers, diminishers, contrast ───────────────────────────────────────


def test_intensifiers_scale_magnitude() -> None:
    plain = score_text("good")
    boosted = score_text("very good")

    assert boosted.polarity > plain.polarity
    assert boosted.intensifier_count == 1


def test_diminishers_reduce_magnitude() -> None:
    assert abs(score_text("slightly annoying").polarity) < abs(score_text("annoying").polarity)


@pytest.mark.parametrize(
    "text",
    [
        "great battery but awful camera",
        "great battery, however the camera is awful",
        "although the battery is great the camera is awful",
    ],
)
def test_contrastive_markers_are_detected_with_both_polarities(text: str) -> None:
    score = score_text(text)

    assert score.contrastive_count == 1
    assert score.positive_hits >= 1
    assert score.negative_hits >= 1


def test_matched_terms_report_what_was_seen() -> None:
    score = score_text("great battery but awful camera")

    assert score.matched_terms == ("great", "awful")


# ── Emoji ─────────────────────────────────────────────────────────────────────


def test_emoji_are_scored_separately_from_prose() -> None:
    score = score_text("Fantastic support \U0001f621\U0001f621")

    assert score.polarity > 0.3
    assert score.emoji_polarity < -0.3


def test_emoji_only_text_has_no_prose_polarity() -> None:
    score = score_text("\U0001f60d\U0001f60d")

    assert score.polarity == 0.0
    assert score.emoji_polarity > 0.5
    assert score.overall_polarity == score.emoji_polarity


def test_variation_selectors_do_not_hide_an_emoji() -> None:
    assert score_text("❤️ the service").emoji_polarity > 0.5


def test_text_without_emoji_reports_zero() -> None:
    assert score_text("good service").emoji_polarity == 0.0


def test_overall_polarity_lets_emoji_qualify_prose_without_overruling_it() -> None:
    score = score_text("Good service \U0001f60d")

    assert score.overall_polarity > score.polarity
    assert -1.0 <= score.overall_polarity <= 1.0


# ── Degenerate input ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["", "   ", "!!!", "1234"])
def test_empty_and_symbol_only_input_scores_neutral(text: str) -> None:
    score = score_text(text)

    assert score.polarity == 0.0
    assert score.positive_hits == 0
    assert score.negative_hits == 0
    assert score.matched_terms == ()
