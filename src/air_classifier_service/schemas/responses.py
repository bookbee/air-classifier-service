"""Operational response models — liveness and readiness.

`/v1/classify`'s request/response shapes live in `schemas/classify.py`; this
file is only the system-status surface `api/v1/system.py` serves.
"""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "air-classifier-service"
    version: str


class ReadyComponent(BaseModel):
    name: str
    ready: bool
    detail: str | None = None


class ReadyResponse(BaseModel):
    """Readiness.

    ``status`` is ``degraded`` — not ``not_ready`` — when an upper tier is down
    but a lower one can still answer. Taking the pod out of rotation for a cloud
    outage would be exactly the wrong reaction.
    """

    status: str
    components: list[ReadyComponent]
