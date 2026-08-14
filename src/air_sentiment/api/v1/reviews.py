"""``POST /v1/reviews`` and ``POST /v1/reviews/batch``.

The same ladder as ``/v1/sentiment``, plus the prose-versus-stars cross-check.
When the request carries a ``rating``, the response's ``rating_consistency``
reports the signed gap between the verdict's polarity and the rating normalised
onto the same -1..+1 scale — which is what a mis-clicked star, or a sarcastic
five, looks like from outside. With no rating there is nothing to compare, and
the field is null rather than a guess.

``options.aspects`` is worth knowing about here: it is the only resource where
callers routinely name the facets they care about, and the tiers score those
rather than inventing their own.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status

from air_sentiment.api.deps import (
    AppStateDep,
    RateLimitedKeyDep,
    apply_key_policy,
    lookup_idempotent,
    store_idempotent,
)
from air_sentiment.api.v1.router import (
    ANALYSIS_ERROR_RESPONSES,
    AnalysisServiceDep,
    batch_response,
    ensure_batch_size,
    key_clamp,
    replay_response,
    request_id_of,
)
from air_sentiment.constants import ResourceKind
from air_sentiment.schemas.requests import ReviewBatchRequest, ReviewRequest
from air_sentiment.schemas.responses import ReviewBatchResponse, ReviewResponse

__all__ = ["router"]

router = APIRouter(tags=["reviews"])

_RATING_NOTE = (
    "`rating_consistency` is null when the request carried no `rating`; "
    "otherwise it reports `agrees`, `prose_more_positive` or "
    "`prose_more_negative` with the signed delta on the -1..+1 scale."
)


@router.post(
    "/reviews",
    response_model=ReviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse one product review",
    description=f"Sentiment and aspects for a single review. {_RATING_NOTE}",
    responses=ANALYSIS_ERROR_RESPONSES,
)
async def analyse_review(
    request: Request,
    payload: ReviewRequest,
    key: RateLimitedKeyDep,
    service: AnalysisServiceDep,
) -> ReviewResponse | Response:
    """One review, one verdict, plus the rating cross-check."""
    body = await request.body()
    replayed = await lookup_idempotent(request, key, body=body)
    if replayed is not None:
        return replay_response(replayed)

    result = await service.analyse(
        payload,
        resource=ResourceKind.REVIEWS,
        request_id=request_id_of(request),
        options=apply_key_policy(payload.options, key),
    )
    await store_idempotent(request, key, body=body, response=result.model_dump_json())
    return result


@router.post(
    "/reviews/batch",
    response_model=ReviewBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse many reviews in one call",
    description=(
        "Analyses every item concurrently, bounded by `app.batch_concurrency`. "
        "A failing item yields a problem document in its own slot while its "
        f"siblings still return results. {_RATING_NOTE}"
    ),
    responses=ANALYSIS_ERROR_RESPONSES,
)
async def analyse_reviews_batch(
    request: Request,
    payload: ReviewBatchRequest,
    key: RateLimitedKeyDep,
    state: AppStateDep,
    service: AnalysisServiceDep,
) -> ReviewBatchResponse | Response:
    """Many reviews, one call, per-item isolation."""
    body = await request.body()
    replayed = await lookup_idempotent(request, key, body=body)
    if replayed is not None:
        return replay_response(replayed)

    ensure_batch_size(len(payload.items), state.settings)
    request_id = request_id_of(request)
    started = time.perf_counter()

    items = await service.analyse_batch(
        payload.items,
        resource=ResourceKind.REVIEWS,
        request_id=request_id,
        options=apply_key_policy(payload.options, key) if payload.options is not None else None,
        clamp=key_clamp(key),
        instance=request.url.path,
    )

    result = batch_response(
        ReviewBatchResponse,
        request_id=request_id,
        items=items,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )
    await store_idempotent(request, key, body=body, response=result.model_dump_json())
    return result
