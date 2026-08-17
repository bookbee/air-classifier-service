"""The tier contract.

Every rung implements :class:`Tier`. The orchestrator holds only this protocol,
never a concrete class — which is what lets the whole ladder be exercised with
scripted fakes and no network at all.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from air_classifier.constants import ResourceKind
from air_classifier.constants import Tier as TierId
from air_classifier.schemas.common import (
    ComplexityFeatures,
    LanguageInfo,
    Options,
    SafetyReport,
)
from air_classifier.schemas.verdict import TierVerdict


class TierError(Exception):
    """A rung could not produce a verdict.

    Raised by tier implementations; caught by the orchestrator and converted into
    a failed verdict. It never escapes to the HTTP layer.
    """


class TierUnavailableError(TierError):
    """The rung's dependency is down, or its circuit breaker is open."""


class TierContext(BaseModel):
    """Everything a rung needs, assembled once by the orchestrator.

    Threaded immutably: each rung receives a copy carrying its predecessors'
    verdicts, so a tier cannot corrupt state for the ones that follow it.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    resource: ResourceKind

    #: Normalised and, when enabled, PII-redacted. This is what tiers analyse.
    text: str
    #: Length of the caller's original text, before normalisation or truncation.
    original_length: int

    language: LanguageInfo
    features: ComplexityFeatures
    safety: SafetyReport

    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    #: Resource-specific inputs (rating, channel, subject, ...) as a flat mapping.
    resource_fields: dict[str, Any] = Field(default_factory=dict)

    options: Options
    prior_verdicts: list[TierVerdict] = Field(default_factory=list)

    #: `time.monotonic()` value past which the ladder must stop.
    deadline_at: float
    cost_spent_usd: float = Field(default=0.0, ge=0.0)

    @property
    def remaining_ms(self) -> float:
        """Milliseconds left in the request's latency budget; never negative."""
        return max(0.0, (self.deadline_at - time.monotonic()) * 1000.0)

    @property
    def budget_exhausted(self) -> bool:
        return time.monotonic() >= self.deadline_at

    def with_verdict(self, verdict: TierVerdict) -> TierContext:
        """Return a copy carrying one more verdict and its accumulated cost."""
        return self.model_copy(
            update={
                "prior_verdicts": [*self.prior_verdicts, verdict],
                "cost_spent_usd": self.cost_spent_usd + verdict.usage.cost_usd,
            }
        )

    def last_usable_label(self) -> TierVerdict | None:
        """Most recent verdict that actually committed to a label."""
        for verdict in reversed(self.prior_verdicts):
            if verdict.is_usable:
                return verdict
        return None


@runtime_checkable
class Tier(Protocol):
    """One rung of the ladder."""

    #: Which rung this is. Must be unique across registered tiers.
    tier: TierId
    #: Version string reported in `model_versions` and `/v1/capabilities`.
    model_version: str

    async def is_available(self) -> bool:
        """Whether this rung can currently serve.

        Must be cheap and must not raise — a probe that throws is treated as
        unavailable. Used by readiness and by the policy's lookahead.
        """
        ...

    async def analyse(self, ctx: TierContext) -> TierVerdict:
        """Produce a verdict.

        Implementations may raise :class:`TierError`; anything else escaping is
        also caught by the orchestrator but is logged as a defect.
        """
        ...

    def supports_language(self, code: str) -> bool:
        """Whether this rung is trained or prompted for the given language."""
        ...
