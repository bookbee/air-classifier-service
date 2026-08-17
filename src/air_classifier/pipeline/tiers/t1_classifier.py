"""Tier 1 — the statistical classifier.

The first rung that actually models anything. Tier 0 answers the inputs a
shortcut can settle; this one answers the broad middle — ordinary prose with
ordinary sentiment — for a few milliseconds and no tokens, which is what keeps
the LLM rungs for the inputs that need them.

Two properties make it useful to the ladder rather than merely accurate:

* **It runs off the event loop.** Scoring is CPU work, and a synchronous model
  call on the loop stalls every other in-flight request, not just this one.
  Everything blocking happens in one :func:`asyncio.to_thread` hop.
* **Its confidence and margin are real.** The policy escalates on
  ``confidence < floor`` and ``label_margin < margin_floor``; both numbers come
  from the scorer's distribution, so a genuinely ambiguous input produces a
  genuinely small pair and climbs. See the calibration note in
  :mod:`~air_classifier.pipeline.tiers.scorers` for why that is fragile.

There is no breaker or bulkhead here on purpose: the tier has no remote
dependency to protect, and its deadline is the orchestrator's ``wait_for``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Final

from air_classifier.config import Settings
from air_classifier.constants import Label
from air_classifier.constants import Tier as TierId
from air_classifier.observability.logging import get_logger, safe_text_fields
from air_classifier.observability.tracing import span
from air_classifier.pipeline.tiers.base import Tier as TierProtocol
from air_classifier.pipeline.tiers.base import TierContext, TierError
from air_classifier.pipeline.tiers.scorers import (
    LexiconScorer,
    ScorerBackend,
    ScoreResult,
    apply_temperature,
    blend,
    build_scorer,
)
from air_classifier.schemas.common import SentimentResult
from air_classifier.schemas.verdict import TierVerdict

__all__ = ["ClassifierTier"]

_log = get_logger(__name__)


class ClassifierTier:
    """Statistical sentiment classification over a pluggable backend."""

    tier: TierId = TierId.T1_CLASSIFIER

    def __init__(self, settings: Settings, *, scorer: ScorerBackend | None = None) -> None:
        self._settings = settings
        self._config = settings.classifier
        self._scorer = build_scorer(settings) if scorer is None else scorer
        self._lexicon = LexiconScorer()
        # Blending the lexicon with itself is arithmetic that changes nothing, so
        # the ensemble only engages behind a backend that is a different model.
        self._blend_weight = (
            0.0 if isinstance(self._scorer, LexiconScorer) else self._config.lexicon_weight
        )
        # Naming the live backend here is what makes a degraded deployment
        # visible in every response's `model_versions`, not only in a boot log
        # nobody reads three weeks later.
        self.model_version = self._scorer.name
        self._languages = frozenset(
            code.strip().lower() for code in self._config.supported_languages
        )

    async def is_available(self) -> bool:
        """True whenever any backend loaded — which the lexicon always does."""
        return self._scorer.available or self._lexicon.available

    def supports_language(self, code: str) -> bool:
        """Whether the live backend was trained for `code`.

        Region subtags are ignored, so ``en-GB`` is English. ``und`` is
        deliberately *not* a match: an undetermined language is not evidence of
        English, and paying one rung to find out is the cheaper mistake. Short
        text dominates the undetermined bucket and Tier 0 has already had its
        shot at it.
        """
        primary = code.split("-", 1)[0].strip().lower()
        return primary in self._languages

    async def analyse(self, ctx: TierContext) -> TierVerdict:
        started = time.perf_counter()
        # Truncation is a cost bound, not a correctness one: sentiment is carried
        # by the whole text, but the tail of a 20k-character rant does not change
        # the label enough to justify tokenising it.
        text = ctx.text[: self._config.max_input_chars]

        async with span(
            "tier.t1_classifier",
            tier=self.tier.value,
            model_version=self.model_version,
            text_len=len(text),
        ):
            try:
                score = await asyncio.to_thread(self._score_blocking, text)
            except Exception as exc:
                raise TierError(f"{self.model_version} scoring failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        positive = score.probabilities[Label.POSITIVE]
        negative = score.probabilities[Label.NEGATIVE]

        _log.debug(
            "tier.t1.scored",
            tier=self.tier.value,
            backend=self._scorer.name,
            label=score.top_label.value,
            confidence=round(score.confidence, 4),
            margin=round(score.margin, 4),
            latency_ms=round(latency_ms, 3),
            **safe_text_fields(text, self._settings),
        )

        return TierVerdict(
            tier=self.tier,
            model_version=self.model_version,
            result=SentimentResult(
                label=score.top_label,
                # Valence as the distribution states it: the mass behind
                # "positive" less the mass behind "negative".
                polarity=min(1.0, max(-1.0, positive - negative)),
                confidence=score.confidence,
            ),
            label_margin=score.margin,
            language_code=ctx.language.code,
            latency_ms=latency_ms,
            detail={
                "backend": self._scorer.name,
                "blended": "true" if self._blend_weight > 0.0 else "false",
            },
        )

    def _score_blocking(self, text: str) -> ScoreResult:
        """The whole synchronous pipeline, run once in a worker thread.

        Kept as a single blocking function rather than three awaited hops: the
        arithmetic between the stages is measured in microseconds and a thread
        handoff is not.
        """
        primary = self._scorer.score(text)
        probabilities = primary.probabilities
        if self._blend_weight > 0.0:
            probabilities = blend(
                probabilities, self._lexicon.score(text).probabilities, self._blend_weight
            )
        # Temperature is fitted against this tier's *published* output, so it is
        # applied to the ensemble rather than to either component of it.
        return ScoreResult.from_probabilities(
            apply_temperature(probabilities, self._config.calibration_temperature)
        )


#: Structural conformance, checked by mypy rather than discovered at runtime by
#: an orchestrator that only ever holds the protocol.
_PROTOCOL_CHECK: Final[type[TierProtocol]] = ClassifierTier
