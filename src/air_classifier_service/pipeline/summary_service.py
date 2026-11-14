"""Stateless customer-summary refresh.

This service holds no customer data at all. `refresh` takes a caller-supplied
`existing_summary` (or `None`, for a customer's first-ever summary) plus a
batch of new feedback/review items, and returns the updated state for the
caller to store in its place — see `schemas/summary.py`.

Two independent things happen per call:

* **The rollup merge is pure, deterministic arithmetic.** Every histogram,
  count and rating window is recomputed from `existing_summary.rollup` plus
  this batch's *successful* classifications. No LLM involved, cannot fail.
* **The narrative refresh is one LLM call**, guarded the same way Tier 2 and
  Tier 3 guard theirs (`pipeline/tiers/t2_local_llm.py`): its own bulkhead and
  circuit breaker, one bounded repair retry on a schema violation. Unlike a
  ladder rung, there is nothing above this to escalate to — so any failure,
  or the provider simply not being configured, degrades to the caller's prior
  narrative unchanged rather than failing the request. `degraded=true` is the
  only way that shows up.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Final

from air_classifier_service.config import Settings
from air_classifier_service.observability.logging import get_logger
from air_classifier_service.pipeline.service import AnalysisService
from air_classifier_service.providers.base import (
    LLMProvider,
    ProviderBudgetExceededError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderResult,
    ProviderSchemaError,
    ProviderUnavailableError,
)
from air_classifier_service.providers.prompts.summary_loader import (
    summary_repair_instruction,
    summary_system_prompt,
    summary_user_content,
)
from air_classifier_service.resilience.breaker import CircuitBreaker, CircuitOpenError
from air_classifier_service.resilience.bulkhead import Bulkhead, BulkheadFullError
from air_classifier_service.resilience.guard import guarded
from air_classifier_service.schemas.classify import (
    ClassificationResponse,
    ClassifyBatchItem,
    ClassifyRequest,
)
from air_classifier_service.schemas.common import Options, RequestUsage
from air_classifier_service.schemas.summary import (
    RECENT_RATINGS_WINDOW,
    CustomerSummary,
    LLMSummaryOutput,
    SummaryItem,
    SummaryRefreshRequest,
    SummaryRefreshResponse,
    SummaryRollup,
)

__all__ = ["SummaryService"]

logger = get_logger(__name__)

#: A dependency name outside the `Provider` enum, deliberately — see
#: `main.py`'s comment at the call site that builds this provider. It keeps
#: this bulkhead/breaker's fault state independent of Tier 3's, at the cost of
#: no `air_classifier_breaker_state` gauge visibility for this path
#: specifically (`resilience/breaker.py::_publish_state` skips names it can't
#: parse as a `Provider`).
_DEPENDENCY_NAME: Final[str] = "model_gateway_summary"

#: Items with no `occurred_at` sort after every dated one, and (fixed rather
#: than `None`) never compare two absent timestamps to each other, which
#: `None < None` cannot do.
_UNDATED_SORTS_LAST: Final = datetime.max.replace(tzinfo=UTC)


class SummaryService:
    """Turns a batch of new items plus a prior summary into an updated one."""

    def __init__(
        self,
        analysis_service: AnalysisService,
        provider: LLMProvider | None,
        settings: Settings,
        *,
        bulkhead: Bulkhead | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._analysis = analysis_service
        self._provider = provider
        self._settings = settings
        self._bulkhead = bulkhead or Bulkhead.from_settings(_DEPENDENCY_NAME, settings.resilience)
        self._breaker = breaker or CircuitBreaker.from_settings(
            _DEPENDENCY_NAME, settings.resilience
        )

    async def refresh(
        self,
        request: SummaryRefreshRequest,
        *,
        request_id: str,
        options: Options | None,
        clamp: Callable[[Options], Options] | None = None,
    ) -> SummaryRefreshResponse:
        started = time.perf_counter()

        classify_requests = [_to_classify_request(item) for item in request.items]
        batch_items = await self._analysis.analyse_batch(
            classify_requests,
            request_id=request_id,
            options=options,
            clamp=clamp,
        )

        pairs = _succeeded_pairs(request.items, batch_items)
        existing = request.existing_summary
        rollup = _merge_rollup(existing.rollup if existing is not None else None, pairs)
        narrative, degraded_reason = await self._refresh_narrative(
            existing=existing, pairs=pairs, request_id=request_id
        )

        return SummaryRefreshResponse(
            request_id=request_id,
            customer_id=request.customer_id,
            summary=CustomerSummary(
                narrative=narrative,
                rollup=rollup,
                version=existing.version if existing is not None else 1,
            ),
            items=batch_items,
            degraded=degraded_reason is not None,
            degraded_reason=degraded_reason,
            usage=_sum_usage(batch_items),
            latency_ms=max(0.0, round((time.perf_counter() - started) * 1000.0, 3)),
        )

    # ── Narrative refresh ────────────────────────────────────────────────────

    async def _refresh_narrative(
        self,
        *,
        existing: CustomerSummary | None,
        pairs: Sequence[tuple[SummaryItem, ClassificationResponse]],
        request_id: str,
    ) -> tuple[str, str | None]:
        """The updated narrative, and why the refresh degraded (`None` if it did not).

        Never raises. A missing/unconfigured provider, a timeout, an outage, or
        two consecutive schema violations all land the same way: the prior
        narrative comes back unchanged and the response is `degraded`. The
        reason travels with it (SIT CL-006g): "the narrative is empty" reads the
        same whether the gateway is unconfigured, its local model is down, or
        today's budget is spent, and each of those has a different fix.
        """
        prior = existing.narrative if existing is not None else ""
        if self._provider is None or not self._provider.configured:
            return prior, "model_gateway_not_configured"

        system = summary_system_prompt()
        user = summary_user_content(existing=existing, items=pairs)
        try:
            result = await self._classify_with_one_repair(system, user, request_id=request_id)
        except (ProviderError, TimeoutError, CircuitOpenError, BulkheadFullError) as exc:
            reason = _degraded_reason(exc)
            logger.warning(
                "summary.narrative_refresh_failed", error=type(exc).__name__, reason=reason
            )
            return prior, reason

        parsed = result.parsed
        if not isinstance(parsed, LLMSummaryOutput):
            logger.warning("summary.narrative_refresh_failed", error="unexpected_output_type")
            return prior, "invalid_model_output"
        return parsed.narrative, None

    async def _classify_with_one_repair(
        self, system: str, user: str, *, request_id: str
    ) -> ProviderResult:
        """One call, then at most one more with the validation errors appended.

        Mirrors `LocalLLMTier._classify_with_one_repair` exactly — a small
        model's schema violations are not usually self-correcting, so a second
        failure is left to propagate rather than looped on.
        """
        try:
            return await self._call(system, user, request_id=request_id)
        except ProviderSchemaError as first:
            errors = str(first)
        return await self._call(
            system, user + summary_repair_instruction(errors), request_id=request_id
        )

    async def _call(self, system: str, user: str, *, request_id: str) -> ProviderResult:
        assert self._provider is not None  # narrowed by _refresh_narrative's own check
        timeout_s = self._settings.model_gateway.summary_timeout_s
        return await guarded(
            self._provider.classify,
            timeout_s=timeout_s,
            breaker=self._breaker,
            bulkhead=self._bulkhead,
            system_prompt=system,
            user_content=user,
            output_model=LLMSummaryOutput,
            model=self._settings.model_gateway.summary_model,
            max_tokens=self._settings.model_gateway.summary_max_tokens,
            cache_prefix=True,
            remaining_ms=timeout_s * 1000.0,
            request_id=request_id,
        )


# ── Pure derivations ─────────────────────────────────────────────────────────


def _to_classify_request(item: SummaryItem) -> ClassifyRequest:
    """The subset of `SummaryItem` the ladder itself needs.

    `source_type` and `occurred_at` have no home on `ClassifyRequest` — they
    are folded into the rollup separately, from `SummaryItem` directly.
    """
    return ClassifyRequest(
        text=item.text,
        channel=item.channel,
        subject=item.subject,
        rating=item.rating,
        rating_scale_max=item.rating_scale_max,
        product_id=item.product_id,
        metadata=item.metadata,
    )


def _succeeded_pairs(
    items: Sequence[SummaryItem], batch_items: Sequence[ClassifyBatchItem]
) -> list[tuple[SummaryItem, ClassificationResponse]]:
    """Items that actually classified, oldest first.

    A failed item consumed no tier and produced nothing to fold in — the same
    "usage rolls up succeeded items only" reasoning `api/v1/classify.py`'s
    batch route already applies. Ordered by `occurred_at` for the rollup's own
    trend/window fields, independent of the response's `items`, which stays in
    request order (matching `ClassifyBatchItem.index`'s existing semantics).
    """
    pairs = [
        (item, batch.result)
        for item, batch in zip(items, batch_items, strict=True)
        if batch.ok and batch.result is not None
    ]
    pairs.sort(key=lambda pair: pair[0].occurred_at or _UNDATED_SORTS_LAST)
    return pairs


def _degraded_reason(exc: Exception) -> str:
    """The caller-facing reason code for a failed narrative call.

    Most specific class first: a budget refusal is also a rate limit, and every
    provider error is a `ProviderError`.
    """
    for kind, reason in _DEGRADED_REASONS:
        if isinstance(exc, kind):
            return reason
    return "model_unavailable"


_DEGRADED_REASONS: Final[tuple[tuple[type[Exception], str], ...]] = (
    (ProviderBudgetExceededError, "budget_exceeded"),
    (ProviderRateLimitError, "rate_limited"),
    (ProviderRequestError, "request_rejected"),
    (ProviderRefusalError, "model_refused"),
    (ProviderSchemaError, "invalid_model_output"),
    (TimeoutError, "timeout"),
    (ProviderUnavailableError, "model_unavailable"),
    (CircuitOpenError, "model_unavailable"),
    (BulkheadFullError, "model_unavailable"),
)


def _merge_rollup(
    existing: SummaryRollup | None,
    pairs: Sequence[tuple[SummaryItem, ClassificationResponse]],
) -> SummaryRollup:
    """Fold `pairs` into `existing` (or a fresh rollup). Pure and deterministic."""
    rollup = existing.model_copy(deep=True) if existing is not None else SummaryRollup()

    for item, result in pairs:
        rollup.total_items += 1
        rollup.counts_by_source_type[item.source_type] = (
            rollup.counts_by_source_type.get(item.source_type, 0) + 1
        )
        rollup.sentiment_counts[result.sentiment.label] = (
            rollup.sentiment_counts.get(result.sentiment.label, 0) + 1
        )
        rollup.urgency_counts[result.urgency.label] = (
            rollup.urgency_counts.get(result.urgency.label, 0) + 1
        )
        for topic in result.topics:
            rollup.topic_counts[topic.label] = rollup.topic_counts.get(topic.label, 0) + 1

        if item.rating is not None:
            rollup.rating_count += 1
            rollup.rating_sum += item.rating
            rollup.average_rating = rollup.rating_sum / rollup.rating_count
            recent = [*rollup.recent_ratings, item.rating][-RECENT_RATINGS_WINDOW:]
            rollup.recent_ratings = recent
            rollup.recent_average_rating = sum(recent) / len(recent)

        occurred = item.occurred_at or datetime.now(UTC)
        if rollup.first_seen_at is None or occurred < rollup.first_seen_at:
            rollup.first_seen_at = occurred
        if rollup.last_seen_at is None or occurred > rollup.last_seen_at:
            rollup.last_seen_at = occurred

    return rollup


def _sum_usage(items: Sequence[ClassifyBatchItem]) -> RequestUsage:
    """Usage across the items that succeeded only — mirrors the batch route."""
    results = [item.result for item in items if item.result is not None]
    return RequestUsage(
        tiers_run=sum(r.usage.tiers_run for r in results),
        tokens_in=sum(r.usage.tokens_in for r in results),
        tokens_out=sum(r.usage.tokens_out for r in results),
        cache_read_tokens=sum(r.usage.cache_read_tokens for r in results),
        est_cost_usd=sum(r.usage.est_cost_usd for r in results),
    )
