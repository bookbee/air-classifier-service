"""Tier-1 classifier tests.

Two things are load-bearing and everything else here is plumbing around them.
The tier must hand the policy *honest* confidence and margin — high on text a
human would not hesitate over, low and narrow on text carrying two judgements —
and it must never do its arithmetic on the event loop, because a synchronous
model call there stalls every other in-flight request, not just this one.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping

import pytest

from air_sentiment.config import ClassifierSettings, Settings
from air_sentiment.constants import Label, ResourceKind
from air_sentiment.constants import Tier as TierId
from air_sentiment.pipeline.features import extract_features
from air_sentiment.pipeline.tiers.base import Tier as TierProtocol
from air_sentiment.pipeline.tiers.base import TierContext, TierError
from air_sentiment.pipeline.tiers.scorers import LEXICON_SCORER_NAME, ScoreResult
from air_sentiment.pipeline.tiers.t1_classifier import ClassifierTier
from air_sentiment.schemas.common import LanguageInfo, Options, SafetyReport

CLEAR_POSITIVE = "Absolutely fantastic, superb service, I love it."
CLEAR_NEGATIVE = "Absolutely terrible, broken on arrival, a total waste of money."
CONTRASTIVE = "The battery life is great but the camera is awful."
CONTRASTIVE_LONG = "Great screen, however the software is buggy and slow."


def make_context(text: str, *, language: str = "en") -> TierContext:
    """A context shaped the way the orchestrator assembles one."""
    return TierContext(
        request_id="req_test",
        resource=ResourceKind.SENTIMENT,
        text=text,
        original_length=len(text),
        language=LanguageInfo(code=language, confidence=0.9),
        features=extract_features(text),
        safety=SafetyReport(),
        options=Options(),
        deadline_at=time.monotonic() + 5.0,
    )


class StubScorer:
    """A backend with a scripted distribution.

    Real enough to satisfy the protocol, and it records what it was asked so the
    tier's truncation and blending can be observed from outside.
    """

    name: str = "stub-1.0"
    available: bool = True

    def __init__(
        self,
        probabilities: Mapping[Label, float],
        *,
        delay_s: float = 0.0,
        fail: bool = False,
    ) -> None:
        self._probabilities = dict(probabilities)
        self._delay_s = delay_s
        self._fail = fail
        self.seen: list[str] = []

    def score(self, text: str) -> ScoreResult:
        self.seen.append(text)
        if self._delay_s:
            time.sleep(self._delay_s)
        if self._fail:
            raise RuntimeError("the model exploded")
        return ScoreResult.from_probabilities(self._probabilities)


def with_classifier(settings: Settings, **kw: object) -> Settings:
    return settings.model_copy(update={"classifier": ClassifierSettings(**kw)})


# ── Confidence and margin ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [(CLEAR_POSITIVE, Label.POSITIVE), (CLEAR_NEGATIVE, Label.NEGATIVE)],
)
async def test_unambiguous_text_is_decided_confidently(
    settings: Settings, text: str, expected: Label
) -> None:
    floors = settings.tiers.t1_classifier
    verdict = await ClassifierTier(settings).analyse(make_context(text))

    assert verdict.result.label is expected
    assert verdict.result.confidence >= floors.confidence_floor
    assert verdict.label_margin >= floors.margin_floor
    assert verdict.succeeded is True


@pytest.mark.parametrize("text", [CONTRASTIVE, CONTRASTIVE_LONG])
async def test_contrastive_text_yields_low_confidence_and_a_narrow_margin(
    settings: Settings, text: str
) -> None:
    """The tier's most important behaviour, asserted against the live thresholds.

    These are the two numbers rules 7 and 8 of the escalation policy read. If
    either one came back padded, the ladder would stop here and answer a
    coin-flip with a straight face.
    """
    floors = settings.tiers.t1_classifier
    verdict = await ClassifierTier(settings).analyse(make_context(text))

    assert verdict.result.confidence < floors.confidence_floor
    assert verdict.label_margin < floors.margin_floor
    assert verdict.succeeded is True


async def test_the_margin_is_measured_rather_than_a_constant(settings: Settings) -> None:
    tier = ClassifierTier(settings)

    decided = await tier.analyse(make_context(CLEAR_POSITIVE))
    torn = await tier.analyse(make_context(CONTRASTIVE_LONG))

    assert decided.label_margin > torn.label_margin
    assert 0.0 <= torn.label_margin <= 1.0


async def test_polarity_follows_the_distribution_not_just_the_label(settings: Settings) -> None:
    tier = ClassifierTier(settings)

    positive = await tier.analyse(make_context(CLEAR_POSITIVE))
    negative = await tier.analyse(make_context(CLEAR_NEGATIVE))
    torn = await tier.analyse(make_context(CONTRASTIVE))

    assert positive.result.polarity > 0.5
    assert negative.result.polarity < -0.5
    assert abs(torn.result.polarity) < 0.2


# ── Verdict shape ─────────────────────────────────────────────────────────────


async def test_the_verdict_reports_the_tier_the_backend_and_the_language(
    settings: Settings,
) -> None:
    verdict = await ClassifierTier(settings).analyse(make_context(CLEAR_POSITIVE, language="en"))

    assert verdict.tier is TierId.T1_CLASSIFIER
    assert verdict.model_version == LEXICON_SCORER_NAME
    assert verdict.language_code == "en"
    assert verdict.latency_ms > 0.0
    assert verdict.usage.cost_usd == 0.0
    assert verdict.is_usable is True


async def test_the_live_backend_is_reported_as_the_model_version(settings: Settings) -> None:
    """A degraded deployment has to be visible in the response, not only in a log."""
    stub = StubScorer({Label.POSITIVE: 1.0})

    verdict = await ClassifierTier(settings, scorer=stub).analyse(make_context(CLEAR_POSITIVE))

    assert ClassifierTier(settings, scorer=stub).model_version == "stub-1.0"
    assert verdict.model_version == "stub-1.0"
    assert verdict.detail["backend"] == "stub-1.0"


async def test_text_is_truncated_to_the_configured_maximum(settings: Settings) -> None:
    stub = StubScorer({Label.NEUTRAL: 1.0})
    tier = ClassifierTier(with_classifier(settings, max_input_chars=64), scorer=stub)

    await tier.analyse(make_context("wonderful " * 200))

    assert len(stub.seen[0]) == 64


# ── Ensemble and calibration ──────────────────────────────────────────────────


async def test_the_lexicon_is_blended_into_a_neural_backend(settings: Settings) -> None:
    stub = StubScorer({Label.POSITIVE: 1.0})
    tier = ClassifierTier(with_classifier(settings, lexicon_weight=1.0), scorer=stub)

    verdict = await tier.analyse(make_context(CLEAR_NEGATIVE))

    # At full weight the lexicon owns the answer, so the stub's certainty about
    # the wrong label has to lose to text this unambiguous.
    assert verdict.result.label is Label.NEGATIVE
    assert verdict.detail["blended"] == "true"


async def test_a_zero_lexicon_weight_leaves_the_backend_untouched(settings: Settings) -> None:
    stub = StubScorer({Label.POSITIVE: 1.0})
    tier = ClassifierTier(with_classifier(settings, lexicon_weight=0.0), scorer=stub)

    verdict = await tier.analyse(make_context(CLEAR_NEGATIVE))

    assert verdict.result.label is Label.POSITIVE
    assert verdict.result.confidence == pytest.approx(1.0)


async def test_the_lexicon_backend_is_not_blended_with_itself(settings: Settings) -> None:
    tier = ClassifierTier(with_classifier(settings, lexicon_weight=0.5))

    verdict = await tier.analyse(make_context(CLEAR_POSITIVE))

    assert verdict.detail["blended"] == "false"


async def test_a_fitted_temperature_flattens_the_reported_confidence(settings: Settings) -> None:
    uncalibrated = ClassifierTier(with_classifier(settings, calibration_temperature=1.0))
    calibrated = ClassifierTier(with_classifier(settings, calibration_temperature=4.0))

    raw = await uncalibrated.analyse(make_context(CLEAR_POSITIVE))
    scaled = await calibrated.analyse(make_context(CLEAR_POSITIVE))

    assert scaled.result.confidence < raw.result.confidence
    assert scaled.result.label is raw.result.label


# ── Availability, language, failure ───────────────────────────────────────────


async def test_the_tier_is_available_with_no_model_artifact_at_all(settings: Settings) -> None:
    tier = ClassifierTier(settings)

    assert settings.classifier.model_path is None
    assert await tier.is_available() is True


@pytest.mark.parametrize(
    ("code", "supported"),
    [("en", True), ("EN", True), ("en-GB", True), ("fr", False), ("ja", False), ("und", False)],
)
def test_supported_languages_come_from_configuration(
    settings: Settings, code: str, supported: bool
) -> None:
    assert ClassifierTier(settings).supports_language(code) is supported


def test_a_configured_language_set_widens_the_tier(settings: Settings) -> None:
    tier = ClassifierTier(with_classifier(settings, supported_languages=["en", "es"]))

    assert tier.supports_language("es") is True
    assert tier.supports_language("de") is False


async def test_a_failing_backend_becomes_a_tier_error(settings: Settings) -> None:
    tier = ClassifierTier(settings, scorer=StubScorer({Label.POSITIVE: 1.0}, fail=True))

    with pytest.raises(TierError, match="scoring failed"):
        await tier.analyse(make_context(CLEAR_POSITIVE))


def test_the_tier_satisfies_the_ladder_protocol(settings: Settings) -> None:
    assert isinstance(ClassifierTier(settings), TierProtocol)


# ── Concurrency ───────────────────────────────────────────────────────────────


async def test_scoring_never_blocks_the_event_loop(settings: Settings) -> None:
    """A blocking backend must not stop the loop serving everything else.

    The ticker is the whole test: it can only advance if the 50ms of synchronous
    work inside the scorer happened somewhere other than this thread.
    """
    tier = ClassifierTier(settings, scorer=StubScorer({Label.POSITIVE: 1.0}, delay_s=0.05))
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.001)
            ticks += 1

    ticker = asyncio.create_task(tick())
    try:
        verdict = await tier.analyse(make_context(CLEAR_POSITIVE))
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker

    assert verdict.succeeded is True
    assert ticks >= 5


async def test_concurrent_analyses_do_not_interfere(settings: Settings) -> None:
    tier = ClassifierTier(settings)
    texts = [CLEAR_POSITIVE, CLEAR_NEGATIVE, CONTRASTIVE] * 4

    verdicts = await asyncio.gather(*(tier.analyse(make_context(text)) for text in texts))

    labels = [verdict.result.label for verdict in verdicts]
    assert labels[0::3] == [Label.POSITIVE] * 4
    assert labels[1::3] == [Label.NEGATIVE] * 4
