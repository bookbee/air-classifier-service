"""``POST /v1/classify`` and ``POST /v1/classify/batch`` — the whole surface.

The single entry point: sentiment, tone, topics, intent, urgency, routing,
requires-human, aspects and rating consistency, in one call. See
``schemas/classify.py`` for the response shape and why it carries everything
three separate resource routes used to.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status

from air_classifier.api.deps import (
    AppStateDep,
    RateLimitedKeyDep,
    apply_key_policy,
    lookup_idempotent,
    store_idempotent,
)
from air_classifier.api.v1.router import (
    ANALYSIS_ERROR_RESPONSES,
    AnalysisServiceDep,
    ensure_batch_size,
    key_clamp,
    replay_response,
    request_id_of,
)
from air_classifier.schemas.classify import (
    ClassificationResponse,
    ClassifyBatchRequest,
    ClassifyBatchResponse,
    ClassifyRequest,
)
from air_classifier.schemas.common import RequestUsage

__all__ = ["router"]

router = APIRouter(tags=["classify"])


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse one piece of text, in full detail",
    description=(
        "Sentiment, tone, topics, intent, urgency, routing and a requires-human flag, in one "
        "call. `tone`, `topics` and `intent` are populated only when the ladder reached an LLM "
        "tier — set `options.min_tier` to guarantee that. Supply `rating` for a review-shaped "
        "text to also get `rating_consistency`, or `channel` for a support-shaped one to "
        "sharpen the urgency read; both are optional and independent."
    ),
    responses=ANALYSIS_ERROR_RESPONSES,
)
async def classify(
    request: Request,
    payload: ClassifyRequest,
    key: RateLimitedKeyDep,
    service: AnalysisServiceDep,
) -> ClassificationResponse | Response:
    """One text, one detailed verdict."""
    body = await request.body()
    replayed = await lookup_idempotent(request, key, body=body)
    if replayed is not None:
        return replay_response(replayed)

    result = await service.analyse(
        payload,
        request_id=request_id_of(request),
        options=apply_key_policy(payload.options, key),
    )
    await store_idempotent(request, key, body=body, response=result.model_dump_json())
    return result


@router.post(
    "/classify/batch",
    response_model=ClassifyBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse many texts in one call, in full detail",
    description=(
        "Analyses every item concurrently, bounded by `app.batch_concurrency`. "
        "A failing item yields a problem document in its own slot while its "
        "siblings still return results, so the call is never all-or-nothing."
    ),
    responses=ANALYSIS_ERROR_RESPONSES,
)
async def classify_batch(
    request: Request,
    payload: ClassifyBatchRequest,
    key: RateLimitedKeyDep,
    state: AppStateDep,
    service: AnalysisServiceDep,
) -> ClassifyBatchResponse | Response:
    """Many texts, one call, per-item isolation."""
    body = await request.body()
    replayed = await lookup_idempotent(request, key, body=body)
    if replayed is not None:
        return replay_response(replayed)

    ensure_batch_size(len(payload.items), state.settings)
    request_id = request_id_of(request)
    started = time.perf_counter()

    items = await service.analyse_batch(
        payload.items,
        request_id=request_id,
        options=apply_key_policy(payload.options, key) if payload.options is not None else None,
        clamp=key_clamp(key),
        instance=request.url.path,
    )

    # `usage` rolls up the items that succeeded only. A failed item consumed no
    # tier, and counting a slot that produced nothing would overstate what the
    # caller is billed for.
    succeeded = sum(1 for item in items if item.ok)
    results = [item.result for item in items if item.result is not None]
    result = ClassifyBatchResponse(
        request_id=request_id,
        items=items,
        succeeded=succeeded,
        failed=len(items) - succeeded,
        usage=RequestUsage(
            tiers_run=sum(r.usage.tiers_run for r in results),
            tokens_in=sum(r.usage.tokens_in for r in results),
            tokens_out=sum(r.usage.tokens_out for r in results),
            cache_read_tokens=sum(r.usage.cache_read_tokens for r in results),
            est_cost_usd=sum(r.usage.est_cost_usd for r in results),
        ),
        latency_ms=max(0.0, round((time.perf_counter() - started) * 1000.0, 3)),
    )
    await store_idempotent(request, key, body=body, response=result.model_dump_json())
    return result
