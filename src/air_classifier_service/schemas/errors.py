"""RFC 9457 problem details.

Kept in `schemas` rather than `api` so that response models (notably batch item
failures) can embed a problem without importing the HTTP layer — dependencies
run downward only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from air_classifier_service.constants import ERROR_TYPE_BASE


class FieldError(BaseModel):
    """One field-level validation failure."""

    field: str
    code: str
    message: str | None = None


class ProblemDetail(BaseModel):
    """An RFC 9457 `application/problem+json` body."""

    type: str = Field(description="Absolute URI identifying the problem class.")
    title: str = Field(description="Short, human-readable, stable across occurrences.")
    status: int = Field(ge=100, le=599)
    detail: str | None = Field(default=None, description="Explanation specific to this occurrence.")
    instance: str | None = Field(default=None, description="URI of the specific occurrence.")
    request_id: str | None = None
    errors: list[FieldError] = Field(default_factory=list)
    retryable: bool = Field(
        default=False,
        description=(
            "An RFC 9457 extension member, not part of the base spec. True when the same "
            "request, unmodified, might succeed on retry (a transient dependency failure); "
            "false when it would not (a client error, or a defect retrying will not fix)."
        ),
    )

    @classmethod
    def build(
        cls,
        *,
        slug: str,
        title: str,
        status: int,
        detail: str | None = None,
        instance: str | None = None,
        request_id: str | None = None,
        errors: list[FieldError] | None = None,
        retryable: bool = False,
    ) -> ProblemDetail:
        return cls(
            type=f"{ERROR_TYPE_BASE}/{slug}",
            title=title,
            status=status,
            detail=detail,
            instance=instance,
            request_id=request_id,
            errors=errors or [],
            retryable=retryable,
        )
