"""``POST /v1/sentiment`` and ``POST /v1/sentiment/batch``.

Generic text with no domain assumptions — the resource the other two specialise.
Both handlers are deliberately thin: authenticate, reconcile the caller's options
with their key, check for a replay, hand the work to
:class:`~air_sentiment.pipeline.service.AnalysisService`, remember the answer.
Everything deciding *what the verdict is* lives below the service; everything
deciding *whether the caller may ask* lives in ``api/deps.py``.
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
from air_sentiment.schemas.requests import SentimentBatchRequest, SentimentRequest
from air_sentiment.schemas.responses import SentimentBatchResponse, SentimentResponse

__all__ = ["router"]

router = APIRouter(tags=["sentiment"])


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse one piece of text",
    description=(
        "Runs the escalation ladder over a single text and returns the verdict, "
        "the rung that decided it, and — unless suppressed via `options."
        "include_trace` — why the ladder climbed as far as it did."
    ),
    responses=ANALYSIS_ERROR_RESPONSES,
)
async def analyse_sentiment(
    request: Request,
    payload: SentimentRequest,
    key: RateLimitedKeyDep,
    service: AnalysisServiceDep,
) -> SentimentResponse | Response:
    """One text, one verdict."""
    body = await request.body()
    replayed = await lookup_idempotent(request, key, body=body)
    if replayed is not None:
        return replay_response(replayed)

    result = await service.analyse(
        payload,
        resource=ResourceKind.SENTIMENT,
        request_id=request_id_of(request),
        options=apply_key_policy(payload.options, key),
    )
    await store_idempotent(request, key, body=body, response=result.model_dump_json())
    return result


@router.post(
    "/sentiment/batch",
    response_model=SentimentBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse many texts in one call",
    description=(
        "Analyses every item concurrently, bounded by `app.batch_concurrency`. "
        "A failing item yields a problem document in its own slot while its "
        "siblings still return results, so the call is never all-or-nothing."
    ),
    responses=ANALYSIS_ERROR_RESPONSES,
)
async def analyse_sentiment_batch(
    request: Request,
    payload: SentimentBatchRequest,
    key: RateLimitedKeyDep,
    state: AppStateDep,
    service: AnalysisServiceDep,
) -> SentimentBatchResponse | Response:
    """Many texts, one call, per-item isolation."""
    body = await request.body()
    replayed = await lookup_idempotent(request, key, body=body)
    if replayed is not None:
        return replay_response(replayed)

    ensure_batch_size(len(payload.items), state.settings)
    request_id = request_id_of(request)
    started = time.perf_counter()

    # The batch-level default is reconciled with the key here; a per-item
    # override is reconciled inside the service, which is the only place that
    # knows which of the two an item ended up with.
    items = await service.analyse_batch(
        payload.items,
        resource=ResourceKind.SENTIMENT,
        request_id=request_id,
        options=apply_key_policy(payload.options, key) if payload.options is not None else None,
        clamp=key_clamp(key),
        instance=request.url.path,
    )

    result = batch_response(
        SentimentBatchResponse,
        request_id=request_id,
        items=items,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )
    await store_idempotent(request, key, body=body, response=result.model_dump_json())
    return result
