"""Tier 0 tests.

Two things are being checked throughout, and only the second one is interesting.
The first is that each shortcut fires on the input it exists for. The second is
that it declines on the input *next to* that one — because a T0 verdict ends the
ladder, every near-miss below is a case where committing would have produced a
confident wrong answer that no later tier could correct.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from air_classifier.config import Settings
from air_classifier.constants import EscalationReason, Label, ResourceKind, Tier
from air_classifier.pipeline.features import extract_features
from air_classifier.pipeline.tiers.base import Tier as TierProtocol
from air_classifier.pipeline.tiers.base import TierContext
from air_classifier.pipeline.tiers.t0_rules import MODEL_VERSION, RulesTier, Shortcut
from air_classifier.preprocess.language import detect_language
from air_classifier.preprocess.lexicon import score_text
from air_classifier.schemas.common import LanguageInfo, Options, SafetyReport
from air_classifier.schemas.verdict import TierVerdict


def make_ctx(
    text: str,
    *,
    resource: ResourceKind = ResourceKind.SENTIMENT,
    language: LanguageInfo | None = None,
    **resource_fields: Any,
) -> TierContext:
    """A context assembled exactly the way the orchestrator assembles one."""
    score = score_text(text)
    return TierContext(
        request_id="req-t0",
        resource=resource,
        text=text,
        original_length=len(text),
        language=language or detect_language(text),
        features=extract_features(text, lexicon_score=score),
        safety=SafetyReport(),
        resource_fields=resource_fields,
        options=Options(),
        deadline_at=time.monotonic() + 5.0,
    )


def review_ctx(text: str, rating: Any, scale: Any = 5.0) -> TierContext:
    return make_ctx(text, resource=ResourceKind.REVIEWS, rating=rating, rating_scale_max=scale)


@pytest.fixture
def tier(settings: Settings) -> RulesTier:
    return RulesTier(settings)


def assert_abstained(verdict: TierVerdict) -> None:
    """An abstention: ran fine, judged nothing, escalates unconditionally."""
    assert verdict.succeeded is True
    assert verdict.is_usable is False
    assert verdict.result.label is Label.UNKNOWN
    assert verdict.result.confidence == 0.0
    assert verdict.detail == {"reason": EscalationReason.NO_SHORTCUT_MATCHED.value}


def assert_fired(verdict: TierVerdict, rule: Shortcut, label: Label) -> None:
    assert verdict.detail["rule"] == rule.value
    assert verdict.result.label is label
    assert verdict.is_usable is True


# ── Contract ──────────────────────────────────────────────────────────────────


async def test_satisfies_the_tier_protocol(tier: RulesTier) -> None:
    assert isinstance(tier, TierProtocol)
    assert tier.tier is Tier.T0_RULES
    assert tier.model_version == MODEL_VERSION
    assert await tier.is_available() is True


@pytest.mark.parametrize("code", ["en", "es", "ja", "und", "xx"])
def test_supports_every_language(tier: RulesTier, code: str) -> None:
    # Language-bound rules gate themselves; the tier as a whole always serves.
    assert tier.supports_language(code) is True


async def test_verdicts_are_stamped_with_tier_and_version(tier: RulesTier) -> None:
    fired = await tier.analyse(make_ctx("Excellent!"))
    abstained = await tier.analyse(make_ctx("it broke immediately"))

    for verdict in (fired, abstained):
        assert verdict.tier is Tier.T0_RULES
        assert verdict.model_version == MODEL_VERSION
        assert verdict.latency_ms >= 0.0


# ── 1. Contentless ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["", "   ", "\n\t ", "...", "!!!", "?!", "-- --", "¿?"])
async def test_contentless_text_is_neutral(tier: RulesTier, text: str) -> None:
    verdict = await tier.analyse(make_ctx(text))

    assert_fired(verdict, Shortcut.CONTENTLESS, Label.NEUTRAL)
    assert verdict.result.polarity == 0.0
    assert verdict.result.confidence >= 0.95


@pytest.mark.parametrize("text", ["5", "2026", "😡", "ok"])
async def test_near_miss_content_is_not_treated_as_empty(tier: RulesTier, text: str) -> None:
    # Digits, emoji and words are content: whatever happens to them, it is not
    # the punctuation rule's answer.
    verdict = await tier.analyse(make_ctx(text))

    assert verdict.detail.get("rule") != Shortcut.CONTENTLESS.value


# ── 2. Explicit rating ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rating", "scale", "label"),
    [
        (5.0, 5.0, Label.POSITIVE),
        (4.0, 5.0, Label.POSITIVE),
        (9.0, 10.0, Label.POSITIVE),
        (0.0, 5.0, Label.NEGATIVE),
        (1.0, 5.0, Label.NEGATIVE),
        (2.0, 10.0, Label.NEGATIVE),
    ],
)
async def test_unambiguous_rating_with_short_prose_decides(
    tier: RulesTier, rating: float, scale: float, label: Label
) -> None:
    verdict = await tier.analyse(review_ctx("Delivered Tuesday.", rating, scale))

    assert_fired(verdict, Shortcut.RATING, label)
    assert verdict.result.confidence >= 0.95


async def test_five_stars_with_a_long_negative_body_does_not_shortcut(tier: RulesTier) -> None:
    # The whole justification for the rating rule is that short prose has no room
    # to reverse the stars. This body has plenty of room, and uses it.
    body = (
        "Arrived broken and the seller ignored three emails about the damaged "
        "screen, so the replacement never shipped and I am still waiting."
    )
    verdict = await tier.analyse(review_ctx(body, 5.0))

    assert_abstained(verdict)


async def test_five_stars_with_a_short_contradicting_body_does_not_shortcut(
    tier: RulesTier,
) -> None:
    # "Broken." would fire the lexicon rule on its own. With five stars beside
    # it, answering from either source alone would be taking a side in a
    # disagreement, so a rating on the table suppresses the prose rules too.
    verdict = await tier.analyse(review_ctx("Broken.", 5.0))

    assert_abstained(verdict)


async def test_mid_scale_rating_does_not_shortcut(tier: RulesTier) -> None:
    verdict = await tier.analyse(review_ctx("Delivered Tuesday.", 3.0))

    assert_abstained(verdict)


async def test_rating_outside_a_review_is_ignored(tier: RulesTier) -> None:
    # Only the reviews endpoint populates `rating`; a key of that name arriving
    # on another resource did not come from a validated request model.
    verdict = await tier.analyse(make_ctx("Delivered Tuesday.", rating=5.0))

    assert_abstained(verdict)


@pytest.mark.parametrize("rating", [True, "5", None, [5]])
async def test_non_numeric_ratings_are_refused(tier: RulesTier, rating: Any) -> None:
    verdict = await tier.analyse(review_ctx("Delivered Tuesday.", rating))

    assert_abstained(verdict)


async def test_rating_out_of_scale_is_refused(tier: RulesTier) -> None:
    verdict = await tier.analyse(review_ctx("Delivered Tuesday.", 9.0, 5.0))

    assert_abstained(verdict)


async def test_negation_in_short_prose_blocks_the_rating(tier: RulesTier) -> None:
    verdict = await tier.analyse(review_ctx("Would not buy again.", 5.0))

    assert_abstained(verdict)


async def test_stars_speak_for_a_review_with_no_prose(tier: RulesTier) -> None:
    # Contentless is checked first but declines here: for a rated review, empty
    # prose means the stars are the whole input.
    verdict = await tier.analyse(review_ctx(".", 5.0))

    assert_fired(verdict, Shortcut.RATING, Label.POSITIVE)


async def test_no_prose_and_an_ambiguous_rating_falls_back_to_neutral(tier: RulesTier) -> None:
    verdict = await tier.analyse(review_ctx(".", 3.0))

    assert_fired(verdict, Shortcut.CONTENTLESS, Label.NEUTRAL)


# ── 3. Emoji-only ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("😍", Label.POSITIVE),
        ("😍😍😍", Label.POSITIVE),
        ("😡", Label.NEGATIVE),
        ("👎 👎", Label.NEGATIVE),
    ],
)
async def test_emoji_only_text_with_one_clear_polarity(
    tier: RulesTier, text: str, label: Label
) -> None:
    verdict = await tier.analyse(make_ctx(text))

    assert_fired(verdict, Shortcut.EMOJI, label)
    assert verdict.result.confidence >= 0.95


async def test_mixed_emoji_do_not_shortcut(tier: RulesTier) -> None:
    # The mean of three hearts and one rage face is respectably positive, which
    # is exactly why the mean alone is not enough to commit on.
    verdict = await tier.analyse(make_ctx("😍😍😍😡"))

    assert_abstained(verdict)


@pytest.mark.parametrize("text", ["😐", "🥱", "🤷"])
async def test_weak_or_unknown_emoji_do_not_shortcut(tier: RulesTier, text: str) -> None:
    verdict = await tier.analyse(make_ctx(text))

    assert_abstained(verdict)


async def test_emoji_beside_words_is_not_the_emoji_rule(tier: RulesTier) -> None:
    verdict = await tier.analyse(make_ctx("this product 😍"))

    assert_abstained(verdict)


async def test_emoji_do_not_fire_outside_the_lexicon_languages(tier: RulesTier) -> None:
    # Gesture emoji are not universal — 👍 is an insult in parts of the world —
    # so the rule only commits where the lexicon also has coverage.
    verdict = await tier.analyse(make_ctx("😍", language=LanguageInfo(code="ar", confidence=0.9)))

    assert_abstained(verdict)


# ── 4. Pure lexicon ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Excellent!", Label.POSITIVE),
        ("Works great", Label.POSITIVE),
        ("Superb service", Label.POSITIVE),
        ("Absolutely terrible.", Label.NEGATIVE),
        ("Very slow and buggy", Label.NEGATIVE),
        ("A total scam", Label.NEGATIVE),
    ],
)
async def test_short_unambiguous_phrases_are_read_off_the_lexicon(
    tier: RulesTier, text: str, label: Label
) -> None:
    verdict = await tier.analyse(make_ctx(text))

    assert_fired(verdict, Shortcut.LEXICON, label)
    assert verdict.result.confidence >= 0.95
    assert (verdict.result.polarity > 0) is (label is Label.POSITIVE)


async def test_negation_blocks_the_lexicon_rule(tier: RulesTier) -> None:
    # Polarity magnitude alone would clear the floor here; the negator is the
    # only thing standing between this and a confident wrong label.
    verdict = await tier.analyse(make_ctx("absolutely not terrible"))

    assert_abstained(verdict)


async def test_a_contrastive_marker_blocks_the_lexicon_rule(tier: RulesTier) -> None:
    verdict = await tier.analyse(make_ctx("Terrible though"))

    assert_abstained(verdict)


async def test_a_sentence_is_too_long_for_the_lexicon_rule(tier: RulesTier) -> None:
    verdict = await tier.analyse(make_ctx("The battery life is consistently excellent here"))

    assert_abstained(verdict)


@pytest.mark.parametrize("text", ["Good.", "Fine", "decent enough", "meh"])
async def test_mild_polarity_does_not_shortcut(tier: RulesTier, text: str) -> None:
    verdict = await tier.analyse(make_ctx(text))

    assert_abstained(verdict)


async def test_a_question_asserts_nothing(tier: RulesTier) -> None:
    verdict = await tier.analyse(make_ctx("Is it excellent?"))

    assert_abstained(verdict)


async def test_emoji_contradicting_the_prose_blocks_the_lexicon_rule(tier: RulesTier) -> None:
    verdict = await tier.analyse(make_ctx("Excellent 😡"))

    assert_abstained(verdict)


async def test_lexicon_does_not_fire_outside_its_languages(tier: RulesTier) -> None:
    verdict = await tier.analyse(
        make_ctx("Very slow and buggy", language=LanguageInfo(code="es", confidence=0.9))
    )

    assert_abstained(verdict)


async def test_undetermined_language_still_reaches_the_lexicon_rules(tier: RulesTier) -> None:
    # Detection reports `und` below its minimum length, which is most of what
    # these rules see. Excluding `und` would switch both of them off.
    ctx = make_ctx("Excellent!")

    assert ctx.language.code == "und"
    assert_fired(await tier.analyse(ctx), Shortcut.LEXICON, Label.POSITIVE)


# ── Cross-cutting ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text", ["", "😡", "Excellent!", "Very slow and buggy", "it broke immediately"]
)
async def test_every_shortcut_clears_the_configured_confidence_floor(
    tier: RulesTier, settings: Settings, text: str
) -> None:
    # A shortcut whose confidence sits below the floor is work the policy throws
    # away — it would escalate anyway, and the tier would be pure overhead.
    verdict = await tier.analyse(make_ctx(text))

    if verdict.is_usable:
        assert verdict.result.confidence >= settings.tiers.t0_rules.confidence_floor
        assert verdict.label_margin >= verdict.result.confidence


async def test_an_unreachable_confidence_floor_is_reported(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    tuned = settings.model_copy(deep=True)
    tuned.tiers.t0_rules.confidence_floor = 1.0

    RulesTier(tuned)

    assert "t0.confidence_floor_unreachable" in capsys.readouterr().out


async def test_a_hostile_input_abstains_rather_than_raising(tier: RulesTier) -> None:
    verdict = await tier.analyse(make_ctx("\x00" * 10 + "excellent " * 500))

    assert_abstained(verdict)
