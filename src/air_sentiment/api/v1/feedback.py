"""``POST /v1/feedback`` and ``POST /v1/feedback/batch``.

The same ladder as ``/v1/sentiment``, plus triage: ``urgency``,
``actionability`` and ``suggested_route``. Those three are **derived by the
service from the verdict, a fixed cue vocabulary and the feedback channel** —
no tier and no model asserts them. The derivation and the reasoning behind it
live in ``pipeline/service.py``; the OpenAPI descriptions below say so too,
because a caller reading only the schema would otherwise take them for model
output.
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
from air_sentiment.schemas.requests import FeedbackBatchRequest, FeedbackRequest
from air_sentiment.schemas.responses import FeedbackBatchResponse, FeedbackResponse

__all__ = ["router"]

router = APIRouter(tags=["feedback"])

_TRIAGE_NOTE = (
    "`urgency`, `actionability` and `suggested_route` are deterministic "
    "heuristics computed from the verdict, a fixed cue vocabulary and the "
    "`channel` — they are not model output. Use them to order a queue, not as "
    "evidence."
)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse one piece of user feedback",
    description=f"Sentiment plus triage signals for a single feedback item. {_TRIAGE_NOTE}",
    responses=ANALYSIS_ERROR_RESPONSES,
)
async def analyse_feedback(
    request: Request,
    payload: FeedbackRequest,
    key: RateLimitedKeyDep,
    service: AnalysisServiceDep,
) -> FeedbackResponse | Response:
    """One feedback item, one verdict, plus triage."""
    body = await request.body()
    replayed = await lookup_idempotent(request, key, body=body)
    if replayed is not None:
        return replay_response(replayed)

    result = await service.analyse(
        payload,
        resource=ResourceKind.FEEDBACK,
        request_id=request_id_of(request),
        options=apply_key_policy(payload.options, key),
    )
    await store_idempotent(request, key, body=body, response=result.model_dump_json())
    return result


@router.post(
    "/feedback/batch",
    response_model=FeedbackBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse many feedback items in one call",
    description=(
        "Analyses every item concurrently, bounded by `app.batch_concurrency`. "
        "A failing item yields a problem document in its own slot while its "
        f"siblings still return results. {_TRIAGE_NOTE}"
    ),
    responses=ANALYSIS_ERROR_RESPONSES,
)
async def analyse_feedback_batch(
    request: Request,
    payload: FeedbackBatchRequest,
    key: RateLimitedKeyDep,
    state: AppStateDep,
    service: AnalysisServiceDep,
) -> FeedbackBatchResponse | Response:
    """Many feedback items, one call, per-item isolation."""
    body = await request.body()
    replayed = await lookup_idempotent(request, key, body=body)
    if replayed is not None:
        return replay_response(replayed)

    ensure_batch_size(len(payload.items), state.settings)
    request_id = request_id_of(request)
    started = time.perf_counter()

    items = await service.analyse_batch(
        payload.items,
        resource=ResourceKind.FEEDBACK,
        request_id=request_id,
        options=apply_key_policy(payload.options, key) if payload.options is not None else None,
        clamp=key_clamp(key),
        instance=request.url.path,
    )

    result = batch_response(
        FeedbackBatchResponse,
        request_id=request_id,
        items=items,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )
    await store_idempotent(request, key, body=body, response=result.model_dump_json())
    return result
