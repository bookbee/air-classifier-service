"""``POST /v1/summary/refresh`` — the stateless customer-summary endpoint.

This service holds no customer data. The caller supplies its own stored
`existing_summary` alongside a batch of new feedback/review items and gets an
updated one back to store in its place. See ``schemas/summary.py`` for the
contract and ``pipeline/summary_service.py`` for how the two are reconciled.
"""

from __future__ import annotations

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
    SummaryServiceDep,
    ensure_batch_size,
    key_clamp,
    replay_response,
    request_id_of,
)
from air_classifier.schemas.summary import SummaryRefreshRequest, SummaryRefreshResponse

__all__ = ["router"]

router = APIRouter(tags=["summary"])


@router.post(
    "/summary/refresh",
    response_model=SummaryRefreshResponse,
    status_code=status.HTTP_200_OK,
    summary="Fold new feedback/reviews into a customer's rolling summary",
    description=(
        "Stateless: classifies every new item, deterministically merges the rollup onto "
        "`existing_summary` (or starts one, when omitted), and refreshes the narrative "
        "through an LLM call. This service stores nothing — persist the returned `summary` "
        "and send it back as `existing_summary` next time. A narrative-refresh failure "
        "returns `degraded: true` with the prior narrative unchanged; the rollup and "
        "per-item verdicts are unaffected."
    ),
    responses=ANALYSIS_ERROR_RESPONSES,
)
async def refresh_summary(
    request: Request,
    payload: SummaryRefreshRequest,
    key: RateLimitedKeyDep,
    state: AppStateDep,
    service: SummaryServiceDep,
) -> SummaryRefreshResponse | Response:
    """Many new items, one updated summary, per-item isolation."""
    body = await request.body()
    replayed = await lookup_idempotent(request, key, body=body)
    if replayed is not None:
        return replay_response(replayed)

    ensure_batch_size(len(payload.items), state.settings)
    result = await service.refresh(
        payload,
        request_id=request_id_of(request),
        options=apply_key_policy(payload.options, key) if payload.options is not None else None,
        clamp=key_clamp(key),
    )
    await store_idempotent(request, key, body=body, response=result.model_dump_json())
    return result
