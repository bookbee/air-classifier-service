"""Tier 0 — deterministic shortcuts.

This rung is not a classifier. It is an escape hatch for the inputs where a model
call would be waste: an empty comment box, a one-word review, a thumbs-up emoji,
five stars with nothing said. Everything else is somebody else's problem.

That framing decides every threshold below. A confident answer here ends the
ladder before any model reads the text, so a wrong one is unrecoverable, while an
unnecessary abstention costs a single hop to a tier that already exists and is
already budgeted for. The rules are therefore tuned so far toward abstention that
several of them decline inputs a human would call obvious — "good" on its own
escalates — and that is the intended trade, not a gap to be closed later.

Scoring is not reimplemented here. :func:`~air_classifier.preprocess.lexicon.score_text`
owns polarity, negation and emoji; :mod:`~air_classifier.pipeline.features` owns
tokenisation. This module only decides whether what they report is unambiguous
enough to act on.
"""

from __future__ import annotations

import time
import unicodedata
from enum import StrEnum
from typing import Final

from air_classifier.cache.base import CacheBackend
from air_classifier.config import Settings
from air_classifier.constants import EscalationReason, Label, Tier
from air_classifier.observability.logging import get_logger, safe_text_fields
from air_classifier.pipeline.tiers.base import TierContext
from air_classifier.preprocess.lexicon import EMOJI_POLARITY, LexiconScore, score_text
from air_classifier.schemas.common import SentimentResult
from air_classifier.schemas.verdict import TierVerdict

__all__ = ["MODEL_VERSION", "RulesTier", "Shortcut"]

_log = get_logger(__name__)

#: Bumped whenever a rule or a threshold below changes, because a cached verdict
#: produced by the old set is not reproducible under the new one.
MODEL_VERSION: Final[str] = "rules-1"


class Shortcut(StrEnum):
    """Which rule produced a verdict. Diagnostics only — never control flow."""

    CONTENTLESS = "contentless"
    RATING = "rating"
    EMOJI = "emoji"
    LEXICON = "lexicon"


# ── Thresholds ────────────────────────────────────────────────────────────────

# Confidences are flat per rule rather than derived from the score: a rule either
# matched or it did not, and interpolating a number between those two states
# would be inventing precision the rule does not have. All of them must clear
# `tiers.t0_rules.confidence_floor` (0.95 by default) or the shortcut is computed
# and then thrown away by the policy — see the warning in `__init__`.
CONTENTLESS_CONFIDENCE: Final[float] = 0.99
RATING_CONFIDENCE: Final[float] = 0.97
EMOJI_CONFIDENCE: Final[float] = 0.96
LEXICON_CONFIDENCE: Final[float] = 0.96

_MAX_SHORTCUT_CONFIDENCE: Final[float] = max(
    CONTENTLESS_CONFIDENCE, RATING_CONFIDENCE, EMOJI_CONFIDENCE, LEXICON_CONFIDENCE
)

#: Fractions of the rating scale outside which the stars are taken as decisive:
#: 1/5 and below, 4/5 and above.
RATING_LOW_FRACTION: Final[float] = 0.20
RATING_HIGH_FRACTION: Final[float] = 0.80

#: Assumed when a request carries a rating but no scale, matching `ClassifyRequest`.
DEFAULT_RATING_SCALE: Final[float] = 5.0

#: Longest prose the rating rule tolerates. This bound is the *entire* safety
#: argument for that rule: five stars say the author was satisfied, but a review
#: with room for a sentence has room to say "arrived broken, seller replaced it"
#: — and then the stars describe the resolution while the text describes the
#: product. Below a dozen tokens there is no space for that reversal, so the
#: stars cannot be contradicted by something we failed to read.
RATING_MAX_PROSE_TOKENS: Final[int] = 12

#: Prose polarity that is allowed to veto the stars. Deliberately low: the point
#: is to catch disagreement, not to measure it.
PROSE_CONTRADICTION_FLOOR: Final[float] = 0.25

#: Mean polarity an emoji-only text needs before it counts as a judgement. Above
#: the neutral faces (😐 at -0.20, 😬 at -0.40) so a shrug does not become a verdict.
EMOJI_POLARITY_FLOOR: Final[float] = 0.45

#: The lexicon rule is for phrases, not sentences. Anything longer has structure
#: — clauses, comparisons, conditionals — that a bag of terms cannot see.
LEXICON_MAX_TOKENS: Final[int] = 5

#: Squashed polarity the lexicon rule needs. `_squash` maps a raw term weight w
#: to w/sqrt(w²+2), so 0.45 corresponds to a term weighted ≥ 0.71 in isolation:
#: "excellent", "terrible", "broken" qualify; "good", "bad", "fine" do not.
LEXICON_POLARITY_FLOOR: Final[float] = 0.45

#: Languages the lexicon can be trusted on. English is the only one it encodes.
#: `und` is included on purpose: `detect_language` reports it for anything under
#: MIN_DETECTION_CHARS, which is most of what the emoji and lexicon rules see, so
#: excluding it would make both rules unreachable on exactly their target inputs.
LEXICON_LANGUAGES: Final[frozenset[str]] = frozenset({"en", "und"})


class RulesTier:
    """Tier 0. No dependency, no I/O, no failure mode."""

    tier: Tier = Tier.T0_RULES
    model_version: str = MODEL_VERSION

    def __init__(self, settings: Settings, cache: CacheBackend | None = None) -> None:
        self._settings = settings
        # Held for constructor uniformity across rungs. The verdict cache is
        # consulted by the orchestrator before the ladder starts, and a rung that
        # costs microseconds has nothing to gain from a second lookup of its own.
        self._cache = cache

        floor = settings.tiers.t0_rules.confidence_floor
        if floor > _MAX_SHORTCUT_CONFIDENCE:
            # Not fatal — every request simply escalates — but silently is the
            # wrong way for a whole tier to stop counting.
            _log.warning(
                "t0.confidence_floor_unreachable",
                confidence_floor=floor,
                max_shortcut_confidence=_MAX_SHORTCUT_CONFIDENCE,
            )

    async def is_available(self) -> bool:
        """Always true: there is nothing here that can be down."""
        return True

    async def unavailable_reason(self) -> str | None:
        """Never called for real — `is_available` never reports False."""
        return None

    def supports_language(self, code: str) -> bool:
        """True for every language.

        The rules that *are* language-bound gate themselves on
        :data:`LEXICON_LANGUAGES` internally; the rest (empty text, an explicit
        star rating) hold in any language. Reporting False here would instead
        tell the policy that Tier 0 cannot serve the request at all, which is
        both untrue and worse than the abstention it would replace.
        """
        return True

    async def analyse(self, ctx: TierContext) -> TierVerdict:
        started = time.perf_counter()
        # Microseconds of pure-Python work on a few hundred characters: handing
        # this to a thread would cost more than it saves.
        score = score_text(ctx.text)
        matched = self._first_match(ctx, score)
        latency_ms = (time.perf_counter() - started) * 1000.0

        if matched is None:
            return _abstain(latency_ms)

        rule, result = matched
        _log.debug(
            "t0.shortcut",
            rule=rule.value,
            label=result.label.value,
            **safe_text_fields(ctx.text, self._settings),
        )
        return TierVerdict(
            tier=self.tier,
            model_version=MODEL_VERSION,
            result=result,
            # A rule matched or it did not; there is no runner-up label to be
            # close to, so the margin is the confidence itself.
            label_margin=result.confidence,
            language_code=ctx.language.code,
            latency_ms=latency_ms,
            detail={"rule": rule.value},
        )

    # ── Rules, in order ───────────────────────────────────────────────────────

    def _first_match(
        self, ctx: TierContext, score: LexiconScore
    ) -> tuple[Shortcut, SentimentResult] | None:
        # An unambiguous rating takes the whole decision: either it decides, or
        # nothing here does. Falling through to the prose rules would mean
        # answering from the text while a star rating we chose not to trust says
        # the opposite — picking a side in a conflict is what the tiers above
        # are for. It is also why this is checked before the empty-text rule:
        # for a rated review, empty prose means the stars are the entire input.
        polarity = _decisive_rating(ctx)
        if polarity is not None:
            rated = self._rating_result(ctx, score, polarity)
            return (Shortcut.RATING, rated) if rated is not None else None

        if _is_contentless(ctx.text):
            return Shortcut.CONTENTLESS, SentimentResult(
                label=Label.NEUTRAL,
                polarity=0.0,
                confidence=CONTENTLESS_CONFIDENCE,
                rationale="No sentiment-bearing content.",
            )

        # Both remaining rules read the English lexicon's tables, so neither may
        # fire on text the lexicon does not cover.
        if ctx.language.code not in LEXICON_LANGUAGES:
            return None

        emoji = self._emoji_result(ctx, score)
        if emoji is not None:
            return Shortcut.EMOJI, emoji

        lexical = self._lexicon_result(ctx, score)
        if lexical is not None:
            return Shortcut.LEXICON, lexical

        return None

    def _rating_result(
        self, ctx: TierContext, score: LexiconScore, polarity: float
    ) -> SentimentResult | None:
        """Stars, when the prose is too short and too plain to argue with them."""
        features = ctx.features
        if features.token_count > RATING_MAX_PROSE_TOKENS:
            return None
        # A contrastive marker is a reversal in progress ("5 stars, but ..."),
        # and a negation is the cheapest way to say the opposite of the stars.
        if features.contrastive_count > 0 or features.negation_count > 0:
            return None

        prose = score.overall_polarity
        if prose * polarity < 0.0 and abs(prose) >= PROSE_CONTRADICTION_FLOOR:
            return None

        return SentimentResult(
            label=_label_of(polarity),
            polarity=polarity,
            confidence=RATING_CONFIDENCE,
            rationale="Rating is unambiguous and the prose is too short to contradict it.",
        )

    def _emoji_result(self, ctx: TierContext, score: LexiconScore) -> SentimentResult | None:
        """Emoji standing alone, with nothing for them to qualify."""
        if ctx.features.token_count > 0:
            return None

        # Unanimity rather than the mean alone: three hearts and one rage face
        # average to something respectably positive, and that average is exactly
        # the conflict worth escalating instead of resolving.
        signs = {_sign(EMOJI_POLARITY[char]) for char in ctx.text if char in EMOJI_POLARITY}
        if len(signs) != 1:
            return None

        polarity = score.emoji_polarity
        if abs(polarity) < EMOJI_POLARITY_FLOOR:
            return None

        return SentimentResult(
            label=_label_of(polarity),
            polarity=polarity,
            confidence=EMOJI_CONFIDENCE,
            rationale="Emoji-only text with a single, strong polarity.",
        )

    def _lexicon_result(self, ctx: TierContext, score: LexiconScore) -> SentimentResult | None:
        """A short phrase whose polarity the lexicon can read off directly."""
        features = ctx.features
        if not 1 <= features.token_count <= LEXICON_MAX_TOKENS:
            return None
        # Each of these means the surface reading is not the intended one, and
        # resolving them is precisely what the tiers above are for.
        if score.negation_count > 0 or score.contrastive_count > 0:
            return None
        if score.positive_hits > 0 and score.negative_hits > 0:
            return None
        if features.question_only:
            # "Is it any good?" carries a positive term while asserting nothing.
            return None

        polarity = score.polarity
        if not score.matched_terms or abs(polarity) < LEXICON_POLARITY_FLOOR:
            return None
        if score.emoji_polarity * polarity < 0.0:
            return None

        return SentimentResult(
            label=_label_of(polarity),
            polarity=polarity,
            confidence=LEXICON_CONFIDENCE,
            rationale="Short unambiguous lexicon match with no negation or contrast.",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _abstain(latency_ms: float) -> TierVerdict:
    """Confidence 0.0, which the policy always escalates past."""
    verdict = TierVerdict.abstained(
        Tier.T0_RULES, model_version=MODEL_VERSION, latency_ms=latency_ms
    )
    return verdict.model_copy(
        update={"detail": {"reason": EscalationReason.NO_SHORTCUT_MATCHED.value}}
    )


def _decisive_rating(ctx: TierContext) -> float | None:
    """The caller's star rating on -1..+1, when one was given and is extreme."""
    rating = _as_number(ctx.resource_fields.get("rating"))
    scale = _as_number(ctx.resource_fields.get("rating_scale_max"))
    if scale is None:
        scale = DEFAULT_RATING_SCALE
    if rating is None or scale <= 0.0 or not 0.0 <= rating <= scale:
        return None

    fraction = rating / scale
    if RATING_LOW_FRACTION < fraction < RATING_HIGH_FRACTION:
        return None
    return fraction * 2.0 - 1.0


def _is_contentless(text: str) -> bool:
    """Whether the text is empty, blank, or nothing but punctuation.

    Unicode category P covers every punctuation class. Category S — where emoji
    live — deliberately does not count: "😡" is a judgement, "!!!" is not.
    """
    stripped = text.strip()
    if not stripped:
        return True
    return all(char.isspace() or unicodedata.category(char).startswith("P") for char in stripped)


def _as_number(value: object) -> float | None:
    """Read a numeric resource field, refusing anything that is not plainly one."""
    if isinstance(value, bool):
        # bool subclasses int, and a `rating` of True must never read as 1 star.
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _label_of(polarity: float) -> Label:
    return Label.POSITIVE if polarity > 0.0 else Label.NEGATIVE


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0
