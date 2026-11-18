"""The seam between HTTP and the ladder.

Everything above this module speaks HTTP; everything below it speaks
:class:`~air_classifier_service.pipeline.tiers.base.TierContext`. Keeping the join in one
place buys three things that are easy to lose if the route handler did this
work itself:

* **Redaction happens exactly once, before anything sees the text.** The tiers
  receive :attr:`TierContext.text`, and there is no other way in. A handler that
  built its own context could forget, and nothing would fail loudly.
* **One preprocessing order, one cache key.** Normalise, redact, detect, score,
  extract — in that sequence — so two identical requests agree on what "the
  text" is and therefore on their cache key.
* **Per-item isolation is a property of the batch, not of the caller.**
  :meth:`AnalysisService.analyse_batch` converts one item's failure into a
  problem in that slot; the other ninety-nine still return answers.

The response shaping lives in a module-level function rather than a method so
it can be exercised against a hand-built ladder result, with no orchestrator
and no cache in the picture.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from pydantic import ValidationError

from air_classifier_service import __version__
from air_classifier_service.audit.base import AuditEntry, AuditSink
from air_classifier_service.cache.base import CacheBackend, verdict_cache_key
from air_classifier_service.config import Settings
from air_classifier_service.constants import (
    Actionability,
    EscalationReason,
    Intent,
    Label,
    RatingAgreement,
    RoutingPriority,
    Tone,
    Topic,
    Urgency,
)
from air_classifier_service.observability.logging import current_api_key_id, get_logger
from air_classifier_service.observability.metrics import (
    CacheEvent,
    record_cache_event,
    record_degraded,
    record_pii_redaction,
)
from air_classifier_service.pipeline.features import extract_features
from air_classifier_service.pipeline.orchestrator import NoVerdictError, Orchestrator
from air_classifier_service.pipeline.tiers.base import TierContext
from air_classifier_service.preprocess.language import detect_language
from air_classifier_service.preprocess.lexicon import score_text
from air_classifier_service.preprocess.normalize import normalize
from air_classifier_service.preprocess.pii import redact
from air_classifier_service.schemas.classify import (
    MAX_TEXT_CHARS,
    ClassificationResponse,
    ClassifyBatchItem,
    ClassifyRequest,
    ModelInfo,
    RoutingInfo,
)
from air_classifier_service.schemas.common import (
    Aspect,
    ComplexityFeatures,
    LabeledScore,
    LanguageInfo,
    Options,
    RatingConsistency,
    RequestUsage,
    SafetyReport,
    SentimentResult,
)
from air_classifier_service.schemas.errors import FieldError, ProblemDetail
from air_classifier_service.schemas.verdict import OrchestrationResult

__all__ = ["AnalysisService"]

_log = get_logger(__name__)


# ── Heuristic tables ─────────────────────────────────────────────────────────

#: Ladder used by the urgency heuristic. Index arithmetic below relies on it
#: being ordered least-urgent first.
_URGENCY_LADDER: Final[tuple[Urgency, ...]] = (
    Urgency.LOW,
    Urgency.MEDIUM,
    Urgency.HIGH,
    Urgency.CRITICAL,
)

#: Text that means someone is losing money, data, or access right now.
_CRITICAL_CUES: Final[frozenset[str]] = frozenset(
    {
        "breach",
        "cannot access",
        "cannot log in",
        "can't access",
        "can't log in",
        "charged twice",
        "chargeback",
        "data loss",
        "double charged",
        "fraud",
        "hacked",
        "lawsuit",
        "locked out",
        "outage",
        "unauthorised charge",
        "unauthorized charge",
    }
)

#: Text that needs a person today, but is not an emergency.
_HIGH_CUES: Final[frozenset[str]] = frozenset(
    {
        "asap",
        "blocked",
        "blocker",
        "broken",
        "crash",
        "crashed",
        "crashes",
        "deadline",
        "escalate",
        "immediately",
        "not working",
        "refund",
        "unacceptable",
        "unusable",
        "urgent",
        "urgently",
        "worst",
    }
)

#: Text worth a ticket, not a page.
_MEDIUM_CUES: Final[frozenset[str]] = frozenset(
    {
        "annoying",
        "bug",
        "confusing",
        "disappointed",
        "error",
        "glitch",
        "missing",
        "slow",
        "unreliable",
    }
)

#: Channels where a human already chose to reach out — the effort itself is a
#: signal. A survey response is the opposite: it was solicited. Read from
#: `ClassifyRequest.channel` when the caller supplied one; most callers do not,
#: and an absent channel simply skips this adjustment.
_ESCALATING_CHANNELS: Final[frozenset[str]] = frozenset({"call", "chat", "phone", "support"})
_DAMPENING_CHANNELS: Final[frozenset[str]] = frozenset({"nps", "research", "survey"})

#: Polarity at or below which a negative verdict starts one rung up.
_STRONG_NEGATIVE: Final[float] = -0.7

#: Keyword families mapped to the queue that owns them, and the fallback
#: `routing.queue` uses when neither `intent` nor `topics` narrowed it down
#: (see `_derive_routing`) — which happens exactly when the ladder stopped
#: below an LLM tier. Order is the tie-break, so the families that cost most
#: to route wrongly come first.
_ROUTE_FAMILIES: Final[tuple[tuple[str, frozenset[str]], ...]] = (
    (
        "security",
        frozenset(
            {
                "breach",
                "gdpr",
                "hacked",
                "password",
                "phishing",
                "privacy",
                "unauthorised",
                "unauthorized",
            }
        ),
    ),
    (
        "billing",
        frozenset(
            {
                "billing",
                "charge",
                "charged",
                "chargeback",
                "invoice",
                "overcharged",
                "payment",
                "price",
                "pricing",
                "refund",
                "subscription",
            }
        ),
    ),
    (
        "reliability",
        frozenset(
            {
                "crash",
                "crashed",
                "crashes",
                "downtime",
                "outage",
                "timeout",
                "unavailable",
                "unreliable",
            }
        ),
    ),
    (
        "performance",
        frozenset({"freezes", "lag", "laggy", "latency", "slow", "sluggish"}),
    ),
    (
        "product-quality",
        frozenset({"broken", "damaged", "defective", "faulty", "quality"}),
    ),
    (
        "shipping",
        frozenset({"arrived", "courier", "delivery", "package", "parcel", "shipping"}),
    ),
    (
        "usability",
        frozenset({"confusing", "layout", "navigation", "onboarding", "unclear", "ux"}),
    ),
    (
        "customer-support",
        frozenset({"agent", "helpdesk", "no reply", "rude", "ticket", "unhelpful"}),
    ),
)

#: How far prose polarity may sit from the normalised rating before the two are
#: called inconsistent. Reviewers round stars, so a small gap is noise.
_RATING_TOLERANCE: Final[float] = 0.4


def _compile_cues(phrases: Iterable[str]) -> re.Pattern[str]:
    """Word-boundary alternation over `phrases`.

    Boundaries rather than substrings because ``down`` must not fire on
    ``download`` and ``error`` must not fire on ``terrorise``. Longest first so
    a multi-word phrase wins over the single word nested inside it.
    """
    ordered = sorted(phrases, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in ordered) + r")\b", re.IGNORECASE)


_CRITICAL_RE: Final = _compile_cues(_CRITICAL_CUES)
_HIGH_RE: Final = _compile_cues(_HIGH_CUES)
_MEDIUM_RE: Final = _compile_cues(_MEDIUM_CUES)
_ROUTE_RES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (name, _compile_cues(terms)) for name, terms in _ROUTE_FAMILIES
)

#: `routing.queue`, preferred over the keyword match above whenever `intent`
#: or `topics` exist: they are a direct answer to "what team owns this",
#: where the keyword tables only ever approximate it from prose. Own naming
#: (underscores) rather than `_ROUTE_FAMILIES`'s hyphenated queue names —
#: unrelated tables serving the same field, not a shared vocabulary.
_INTENT_QUEUES: Final[dict[Intent, str]] = {
    Intent.COMPLAINT: "customer_support",
    Intent.INFORMATION: "customer_support",
    Intent.REFUND_REQUEST: "billing",
    Intent.RETURN_REQUEST: "returns",
    Intent.ORDER_TRACKING: "shipping",
    Intent.REPLACEMENT_REQUEST: "returns",
    Intent.CANCELLATION: "billing",
    Intent.RECOMMENDATION: "customer_support",
    Intent.PRAISE: "customer_support",
}

_TOPIC_QUEUES: Final[dict[Topic, str]] = {
    Topic.DELIVERY: "shipping",
    Topic.PAYMENT: "billing",
    Topic.PRODUCT_QUALITY: "product_quality",
    Topic.RETURN: "returns",
    Topic.REFUND: "billing",
    Topic.PRICING: "billing",
}

#: Direct rank mapping — see `RoutingPriority`'s docstring for why this is not
#: a second judgement call.
_URGENCY_PRIORITY: Final[dict[Urgency, RoutingPriority]] = {
    Urgency.CRITICAL: RoutingPriority.P1,
    Urgency.HIGH: RoutingPriority.P2,
    Urgency.MEDIUM: RoutingPriority.P3,
    Urgency.LOW: RoutingPriority.P4,
}

#: Intents that imply someone is waiting on an outcome a heuristic cannot
#: deliver — money moving, an order changing, a relationship already strained.
_REQUIRES_HUMAN_INTENTS: Final[frozenset[Intent]] = frozenset(
    {
        Intent.COMPLAINT,
        Intent.REFUND_REQUEST,
        Intent.RETURN_REQUEST,
        Intent.REPLACEMENT_REQUEST,
        Intent.CANCELLATION,
    }
)

_REQUIRES_HUMAN_TONES: Final[frozenset[Tone]] = frozenset(
    {Tone.ANGRY, Tone.FRUSTRATED, Tone.ANXIOUS}
)


# ── Preprocessing result ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Prepared:
    """What preprocessing produced, before any tier ran."""

    text: str
    original_length: int
    language: LanguageInfo
    features: ComplexityFeatures
    safety: SafetyReport
    resource_fields: dict[str, Any]
    cache_key: str | None


# ── Service ───────────────────────────────────────────────────────────────────


class AnalysisService:
    """Turns a request into a response by way of the ladder."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        settings: Settings,
        cache: CacheBackend | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._settings = settings
        self._cache = cache
        self._audit = audit

    # ── Single ────────────────────────────────────────────────────────────────

    async def analyse(
        self,
        request: ClassifyRequest,
        *,
        request_id: str,
        options: Options,
    ) -> ClassificationResponse:
        """Preprocess, consult the cache, climb the ladder, shape the answer.

        ``options`` is passed separately from ``request.options`` because the API
        key's policy has already been applied to it by the time this is called —
        a caller's requested ceiling is a proposal, not a decision.

        Raises :class:`NoVerdictError` when no rung produced a usable answer;
        that is the service's only path to a 5xx.
        """
        started = time.perf_counter()
        prepared = self._prepare(request, options=options)

        if prepared.cache_key is not None:
            hit = await self._cached(prepared.cache_key)
            if hit is not None:
                response = _shape(
                    hit,
                    request_id=request_id,
                    prepared=prepared,
                    metadata=request.metadata,
                    include_trace=options.include_trace,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    cached=True,
                )
                await self._record_audit(request_id, prepared.text, response)
                return response

        ctx = TierContext(
            request_id=request_id,
            text=prepared.text,
            original_length=prepared.original_length,
            language=prepared.language,
            features=prepared.features,
            safety=prepared.safety,
            context=dict(request.context),
            metadata=dict(request.metadata),
            resource_fields=prepared.resource_fields,
            options=options,
            # The budget is a wall-clock deadline rather than a per-tier
            # allowance: the orchestrator hands each rung whatever is left.
            deadline_at=time.monotonic() + options.latency_budget_ms / 1000.0,
        )

        result = await self._orchestrator.run(ctx)
        if result.degraded:
            # Two distinct causes share the one `degraded` flag (orchestrator.py's
            # module docstring): an outright tier failure, or the ladder wanting
            # to climb with nothing available to climb to. Report whichever
            # actually happened rather than assuming the former — a dashboard
            # reading every degraded response as "a tier failed" would send
            # DevOps chasing an error that, for a capacity gap, was never raised.
            any_tier_failed = any(not step.succeeded for step in result.steps)
            reason = (
                EscalationReason.TIER_FAILED
                if any_tier_failed
                else EscalationReason.NEXT_TIER_UNAVAILABLE
            )
            record_degraded(reason)

        if prepared.cache_key is not None:
            await self._store(prepared.cache_key, result)

        response = _shape(
            result,
            request_id=request_id,
            prepared=prepared,
            metadata=request.metadata,
            include_trace=options.include_trace,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cached=False,
        )
        await self._record_audit(request_id, prepared.text, response)
        return response

    # ── Batch ─────────────────────────────────────────────────────────────────

    async def analyse_batch(
        self,
        items: Sequence[ClassifyRequest],
        *,
        request_id: str,
        options: Options | None = None,
        clamp: Callable[[Options], Options] | None = None,
        instance: str | None = None,
    ) -> list[ClassifyBatchItem]:
        """Analyse `items` concurrently, isolating each one's failures.

        ``options`` is the batch-level default; an item carrying its own wins.
        ``clamp`` is applied to whichever of the two is chosen, so an item can
        never use its own options to climb past the API key's ceiling — the
        per-item override is a way to ask for less, not for more.

        Concurrency is bounded by ``app.batch_concurrency``: a thousand-item
        batch must not open a thousand simultaneous LLM calls and take the
        provider — and every other caller — down with it.
        """
        semaphore = asyncio.Semaphore(self._settings.app.batch_concurrency)

        async def run(index: int, item: ClassifyRequest) -> ClassifyBatchItem:
            # `clamp` is inside the guard as well: a policy refusal on one item's
            # own options is that item's failure, not the batch's.
            try:
                effective = _effective_options(item, options)
                if clamp is not None:
                    effective = clamp(effective)
                async with semaphore:
                    response = await self.analyse(
                        item, request_id=request_id, options=effective
                    )
            except Exception as exc:
                return ClassifyBatchItem(
                    index=index,
                    ok=False,
                    error=self._problem_for(
                        exc, index=index, request_id=request_id, instance=instance
                    ),
                )
            return ClassifyBatchItem(index=index, ok=True, result=response)

        return list(await asyncio.gather(*(run(i, item) for i, item in enumerate(items))))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _prepare(self, request: ClassifyRequest, *, options: Options) -> _Prepared:
        """Normalise, redact, and derive every signal the ladder needs.

        The redaction step sits here, above the orchestrator, so that no tier can
        be reached with unredacted text even by accident.
        """
        normalised = normalize(request.text, max_chars=self._max_chars())
        text = normalised.text
        safety = SafetyReport(truncated=normalised.truncated)

        if options.redact_pii:
            redaction = redact(text)
            if redaction.redacted:
                text = redaction.text
                safety = safety.model_copy(
                    update={"pii_redacted": True, "pii_findings": list(redaction.findings)}
                )
                for finding in redaction.findings:
                    record_pii_redaction(finding.kind, finding.count)

        language = detect_language(text)
        lexicon = score_text(text)
        features = extract_features(text, lexicon_score=lexicon)
        resource_fields = _resource_fields(request)

        cache_key: str | None = None
        if self._cache is not None and self._settings.cache.ttl_seconds > 0:
            cache_key = verdict_cache_key(
                prefix=self._settings.cache.key_prefix,
                text=text,
                options=options,
                resource_fields=resource_fields,
            )

        return _Prepared(
            text=text,
            original_length=normalised.original_length,
            language=language,
            features=features,
            safety=safety,
            resource_fields=resource_fields,
            cache_key=cache_key,
        )

    def _max_chars(self) -> int:
        """The input cap, in characters.

        The Tier-1 window governs even though Tiers 2 and 3 could take more: a
        verdict that changes depending on which rung answered would make the
        ladder's cost/accuracy trade visible to callers as inconsistency.
        """
        return max(1, min(self._settings.classifier.max_input_chars, MAX_TEXT_CHARS))

    async def _cached(self, key: str) -> OrchestrationResult | None:
        """Look up a stored ladder result, counting the outcome."""
        if self._cache is None:
            return None
        raw = await self._cache.get(key)
        if raw is None:
            record_cache_event(CacheEvent.MISS)
            return None
        try:
            result = OrchestrationResult.model_validate_json(raw)
        except ValidationError:
            # An entry written by an older shape outlived its schema. Treat it
            # as absent rather than failing a request over a stale optimisation.
            _log.warning("cache.unreadable_entry", key=key)
            record_cache_event(CacheEvent.MISS)
            return None
        record_cache_event(CacheEvent.HIT)
        return result

    async def _store(self, key: str, result: OrchestrationResult) -> None:
        """Persist a ladder result for reuse.

        A degraded result the ladder was *stuck with* is deliberately not
        stored: a provider blip lasts seconds, and pinning the fallback answer
        for the whole TTL would turn a momentary outage into an hour of worse
        verdicts.

        A degraded result that is nonetheless `settled` — a dead rung was
        hopped, but the rung that decided met its own floor — is stored. Its
        answer is no worse for the outage, and refusing it meant that while T2
        was down every repeat paid for T3 again (SIT CL-006i regression). It is
        stored with `degraded` cleared: `degraded` describes the response that
        lived through the outage, and a later hit that touches no dependency
        must not report one. The trace still shows the hop.
        """
        if self._cache is None:
            return
        if result.degraded:
            if not result.settled:
                return
            result = result.model_copy(update={"degraded": False})
        await self._cache.set(key, result.model_dump_json(), self._settings.cache.ttl_seconds)

    async def _record_audit(
        self, request_id: str, text: str, response: ClassificationResponse
    ) -> None:
        """Best-effort audit entry for one classification job.

        `text` is `prepared.text` — already redacted, the same text every tier
        saw. Wrapped in its own try/except as belt-and-suspenders on top of
        `AuditSink`'s own never-raise contract: a defect in a sink implementation
        must not turn a support log into a failed request.
        """
        if self._audit is None:
            return
        try:
            await self._audit.record(
                AuditEntry(
                    request_id=request_id,
                    text=text,
                    api_key_id=current_api_key_id(),
                    decided_by=response.decided_by,
                    degraded=response.degraded,
                    latency_ms=response.latency_ms,
                    timestamp=datetime.now(UTC),
                )
            )
        except Exception as exc:
            _log.warning("audit.record_failed", request_id=request_id, error=type(exc).__name__)
        record_cache_event(CacheEvent.STORE)

    def _problem_for(
        self,
        exc: Exception,
        *,
        index: int,
        request_id: str,
        instance: str | None,
    ) -> ProblemDetail:
        """Map one item's failure onto the catalogue in docs/02-lld.md §9.

        An exception that already knows its own problem document renders itself.
        That duck-typed hook is how a policy refusal raised by the caller-supplied
        ``clamp`` — a 403, decided by the HTTP layer this module may not import —
        lands in its own slot with its own status instead of being flattened into
        a 500 that tells the caller nothing.
        """
        rendered = _self_rendered(exc, instance=instance, request_id=request_id)
        if rendered is not None:
            return rendered
        if isinstance(exc, NoVerdictError):
            # Mirrors the single-call mapping in api/errors.py (SIT CL-003): a
            # ladder stopped by the item's own limits is a non-retryable 422.
            if not exc.retryable:
                return ProblemDetail.build(
                    slug="no-verdict-within-limits",
                    title="No verdict within the request's limits",
                    status=422,
                    detail=exc.limit_detail(),
                    instance=instance,
                    request_id=request_id,
                )
            return ProblemDetail.build(
                slug="no-verdict",
                title="No tier produced a verdict",
                status=503,
                detail=str(exc),
                instance=instance,
                request_id=request_id,
                retryable=True,
            )
        if isinstance(exc, ValidationError):
            return ProblemDetail.build(
                slug="validation-error",
                title="Request validation failed",
                status=422,
                detail=_first_error(exc),
                instance=instance,
                request_id=request_id,
                errors=_field_errors(exc),
            )
        if isinstance(exc, ValueError):
            return ProblemDetail.build(
                slug="validation-error",
                title="Request validation failed",
                status=422,
                detail=str(exc),
                instance=instance,
                request_id=request_id,
            )
        # Unexpected: logged with a traceback, but the caller is told nothing
        # about our internals.
        _log.error(
            "batch.item_defect",
            request_id=request_id,
            index=index,
            error=str(exc),
            exc_info=True,
        )
        return ProblemDetail.build(
            slug="internal-error",
            title="Internal error",
            status=500,
            detail="The item could not be analysed.",
            instance=instance,
            request_id=request_id,
        )


def _effective_options(item: ClassifyRequest, fallback: Options | None) -> Options:
    """The item's own options when it supplied any, else the batch default.

    ``model_fields_set`` rather than a truthiness check, because an item that
    sent ``"options": {}`` meant the defaults, not the batch's overrides.
    """
    if fallback is None or "options" in item.model_fields_set:
        return item.options
    return fallback


def _resource_fields(request: ClassifyRequest) -> dict[str, Any]:
    """Flatten the caller's optional fields for the tiers and prompts to read.

    Keys match what ``providers/prompts/loader.py`` and ``t0_rules.py`` look up;
    they are part of the internal contract, not free-form.
    """
    return {
        "channel": request.channel,
        "user_segment": request.user_segment,
        "subject": request.subject,
        "rating": request.rating,
        "rating_scale_max": request.rating_scale_max,
        "product_id": request.product_id,
        "verified_purchase": request.verified_purchase,
        "title": request.title,
    }


def _shape(
    result: OrchestrationResult,
    *,
    request_id: str,
    prepared: _Prepared,
    metadata: dict[str, Any],
    include_trace: bool,
    latency_ms: float,
    cached: bool,
) -> ClassificationResponse:
    """The verdict, in full — see schemas/classify.py.

    `tone`, `intent` and each entry of `topics` come straight off the deciding
    verdict: present only when an LLM tier answered and was asked to fill
    them, absent otherwise (`providers/prompts/loader.py`). `urgency`,
    `actionability`, `requires_human` and `routing` are heuristics computed
    here regardless of which rung decided — always present, and explicitly
    not a model prediction. `rating_consistency` is `None` unless the caller
    supplied a `rating`.
    """
    best = result.best
    total = result.total_usage
    fields = prepared.resource_fields
    channel = fields.get("channel")
    channel = channel if isinstance(channel, str) else None

    urgency_label = _derive_urgency(sentiment=best.result, text=prepared.text, channel=channel)
    routing = _derive_routing(
        text=prepared.text,
        aspects=best.aspects,
        intent=best.intent,
        topics=best.topics,
        urgency=urgency_label,
    )
    requires_human = _derive_requires_human(
        urgency=urgency_label,
        intent=best.intent.label if best.intent is not None else None,
        tone=best.tone.label if best.tone is not None else None,
    )
    actionability = _derive_actionability(
        sentiment=best.result,
        text=prepared.text,
        urgency=urgency_label,
        aspects=best.aspects,
    )

    return ClassificationResponse(
        request_id=request_id,
        sentiment=best.result,
        tone=best.tone,
        # Confidence 1.0: a deterministic rule fired, not a probability estimate.
        urgency=LabeledScore(label=urgency_label, confidence=1.0),
        topics=list(best.topics),
        intent=best.intent,
        routing=routing,
        requires_human=requires_human,
        actionability=actionability,
        aspects=list(best.aspects),
        rating_consistency=_rating_consistency(
            rating=_as_float(fields.get("rating")),
            rating_scale_max=_as_float(fields.get("rating_scale_max")) or 5.0,
            polarity=best.result.polarity,
        ),
        emotions=list(best.emotions),
        language=prepared.language,
        model=ModelInfo(name=best.model_version, version=__version__),
        decided_by=best.tier,
        degraded=result.degraded,
        # Suppressible because the trace is diagnostic weight most callers do
        # not want on every response — never because it would reveal anything.
        escalation_trace=list(result.steps) if include_trace else [],
        safety=prepared.safety,
        # `usage` is what *this* request ran and spent. A cache hit ran no tier
        # and spent nothing; reporting the original call's usage again made a
        # caller summing `est_cost_usd` count every hit as a fresh spend (SIT
        # CL-006h). The trace still shows how the cached verdict was reached.
        usage=RequestUsage()
        if cached
        else RequestUsage(
            tiers_run=len(result.verdicts),
            tokens_in=total.tokens_in,
            tokens_out=total.tokens_out,
            cache_read_tokens=total.cache_read_tokens,
            est_cost_usd=total.cost_usd,
        ),
        cached=cached,
        latency_ms=max(0.0, round(latency_ms, 3)),
        model_versions={verdict.tier.value: verdict.model_version for verdict in result.verdicts},
        metadata=dict(metadata),
    )


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


# ── Derivations ───────────────────────────────────────────────────────────────
#
# Everything below is a heuristic, not a model prediction: a deterministic
# reading of the verdict's polarity and label, a fixed cue vocabulary, and
# whatever optional fields the caller supplied. No tier asserts these and no
# LLM is asked for them — the structured-output contract in schemas/verdict.py
# carries none of them. Treat a disagreement between these functions and a
# human triager as a bug in the tables below, not as a model error.


def _derive_urgency(*, sentiment: SentimentResult, text: str, channel: str | None) -> Urgency:
    """How soon this text wants a human.

    Starts from the verdict — negative outranks positive — then lifts on
    lexical cues and nudges on the channel. Cues dominate polarity on purpose:
    "the checkout is down" delivered calmly still means the checkout is down.
    """
    level = 0
    if sentiment.label is Label.NEGATIVE:
        level = 2 if sentiment.polarity <= _STRONG_NEGATIVE else 1
    elif sentiment.label is Label.MIXED and sentiment.polarity < 0.0:
        level = 1

    if _CRITICAL_RE.search(text):
        level = 3
    elif _HIGH_RE.search(text):
        level = max(level, 2)
    elif _MEDIUM_RE.search(text):
        level = max(level, 1)

    normalised_channel = (channel or "").strip().lower()
    if normalised_channel in _ESCALATING_CHANNELS and level >= 1:
        level += 1
    elif normalised_channel in _DAMPENING_CHANNELS:
        level -= 1

    return _URGENCY_LADDER[max(0, min(len(_URGENCY_LADDER) - 1, level))]


def _derive_actionability(
    *,
    sentiment: SentimentResult,
    text: str,
    urgency: Urgency,
    aspects: Sequence[Aspect],
) -> Actionability:
    """Whether this text implies work for someone.

    ``BLOCKING`` is reserved for text that says someone is stuck; praise with
    named facets is ``INFORMATIONAL`` because it tells a team something
    without asking for anything.
    """
    if sentiment.label is Label.UNKNOWN:
        return Actionability.NONE
    if urgency is Urgency.CRITICAL or _CRITICAL_RE.search(text):
        return Actionability.BLOCKING
    if _HIGH_RE.search(text) or _MEDIUM_RE.search(text):
        return Actionability.ACTIONABLE
    if sentiment.label is Label.NEGATIVE:
        return Actionability.ACTIONABLE
    # Praise implies no work but is still worth knowing, so it is informational.
    # Neutral text that named no facet and matched no cue is the one case with
    # nothing on either side of the ledger.
    if aspects or sentiment.label is not Label.NEUTRAL:
        return Actionability.INFORMATIONAL
    return Actionability.NONE


def _derive_route(*, text: str, aspects: Sequence[Aspect]) -> str | None:
    """The queue this text probably belongs to, from keywords alone, or None.

    The fallback `_derive_routing` uses when neither `intent` nor `topics`
    narrowed it down. Aspect names count alongside the text because a tier
    that named "delivery" as a facet has done the same matching work with
    more context than these tables have. None rather than a default queue: a
    wrong route costs more than an unrouted item.
    """
    haystack = " ".join([text, *(aspect.name for aspect in aspects)])
    best_name: str | None = None
    best_hits = 0
    for name, pattern in _ROUTE_RES:
        hits = len(pattern.findall(haystack))
        # Strictly greater keeps declaration order as the tie-break.
        if hits > best_hits:
            best_name, best_hits = name, hits
    return best_name


def _derive_routing(
    *,
    text: str,
    aspects: Sequence[Aspect],
    intent: LabeledScore[Intent] | None,
    topics: Sequence[LabeledScore[Topic]],
    urgency: Urgency,
) -> RoutingInfo:
    """Queue and priority. Both heuristic, both deterministic.

    `queue` prefers `intent` — the most decisive "what do they want" signal —
    then the highest-confidence `topic`, and only falls back to the keyword
    match in `_derive_route` when neither exists, which happens exactly when
    the ladder stopped below an LLM tier. `priority` is `urgency` read through
    `_URGENCY_PRIORITY`.
    """
    queue: str | None = None
    if intent is not None:
        queue = _INTENT_QUEUES.get(intent.label)
    if queue is None and topics:
        best_topic = max(topics, key=lambda t: t.confidence)
        queue = _TOPIC_QUEUES.get(best_topic.label)
    if queue is None:
        queue = _derive_route(text=text, aspects=aspects)
    return RoutingInfo(queue=queue, priority=_URGENCY_PRIORITY[urgency])


def _derive_requires_human(*, urgency: Urgency, intent: Intent | None, tone: Tone | None) -> bool:
    """Whether a person needs to look at this.

    High or critical urgency is sufficient on its own; failing that, an intent
    that implies an outcome a rule cannot deliver, or a tone that says the
    author is already past patience.
    """
    if urgency in (Urgency.HIGH, Urgency.CRITICAL):
        return True
    if intent is not None and intent in _REQUIRES_HUMAN_INTENTS:
        return True
    return tone is not None and tone in _REQUIRES_HUMAN_TONES


def _rating_consistency(
    *,
    rating: float | None,
    rating_scale_max: float,
    polarity: float,
) -> RatingConsistency | None:
    """Compare prose polarity against the stars the caller gave, if any.

    Both are put on -1..+1 and subtracted. A positive delta means the prose is
    warmer than the rating — the classic "4 stars, glowing review" — and a
    negative one means the opposite, which is the more interesting case: it is
    what a mis-clicked rating or a sarcastic five stars looks like.
    """
    if rating is None:
        return None
    scale = rating_scale_max if rating_scale_max > 0 else 5.0
    normalised = _clamp(rating / scale * 2.0 - 1.0)
    delta = polarity - normalised

    if abs(delta) <= _RATING_TOLERANCE:
        agreement = RatingAgreement.AGREES
    elif delta > 0:
        agreement = RatingAgreement.PROSE_MORE_POSITIVE
    else:
        agreement = RatingAgreement.PROSE_MORE_NEGATIVE

    return RatingConsistency(
        agreement=agreement,
        delta=round(delta, 4),
        normalised_rating=round(normalised, 4),
    )


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _self_rendered(
    exc: Exception, *, instance: str | None, request_id: str
) -> ProblemDetail | None:
    """The exception's own problem document, when it carries one.

    Structural rather than nominal on purpose: ``api.errors.AppError`` satisfies
    it, but this module sits below the HTTP layer and must not import it.
    """
    render = getattr(exc, "to_problem", None)
    if not callable(render):
        return None
    try:
        problem = render(instance=instance, request_id=request_id)
    except TypeError:
        # Something else named `to_problem` with a different signature. Fall
        # through to the catalogue rather than guessing at its shape.
        return None
    return problem if isinstance(problem, ProblemDetail) else None


def _field_errors(exc: ValidationError) -> list[FieldError]:
    return [
        FieldError(
            field=".".join(str(part) for part in error["loc"]) or "body",
            code=str(error["type"]),
            message=str(error["msg"]),
        )
        for error in exc.errors()
    ]


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "validation failed"
    first = errors[0]
    location = ".".join(str(part) for part in first["loc"]) or "body"
    return f"{location}: {first['msg']}"
