"""Per-request dependencies: identity, quota, key policy, idempotency.

Everything a route needs *before* it may touch the ladder lives here, so a
handler reads as "analyse this text" rather than as a preamble of security
checks. Three things are worth knowing about the shapes below:

* **Singletons arrive through :class:`AppState`, not through ``getattr``.**
  ``app.state`` is an untyped bag; touching it from every route would spread
  ``Any`` across the whole HTTP layer and let a renamed attribute reach
  production as an ``AttributeError`` under load. It is unpacked exactly once,
  here, into a typed container.
* **Key policy clamps quietly but refuses loudly.** Lowering a caller's tier
  ceiling is a cost guardrail and costs them nothing but accuracy; ignoring an
  explicit ``redact_pii=false`` would leave them believing redaction was off.
  The first is silent, the second is a 403. See :func:`apply_key_policy`.
* **Nothing here logs a raw key.** The store never renders one, and the only
  identifier that reaches a log line or a metric label is ``ApiKeyRecord.id``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated, Final

from fastapi import Depends, FastAPI, Request, Response
from pydantic import BaseModel, ValidationError

from air_classifier.api.errors import (
    IdempotencyConflictError,
    InsufficientScopeError,
    MissingApiKeyError,
    RateLimitedError,
)
from air_classifier.api.middleware import API_KEY_ID_ATTR
from air_classifier.cache.base import CacheBackend, idempotency_cache_key
from air_classifier.config import Settings
from air_classifier.constants import HEADER_API_KEY, HEADER_IDEMPOTENCY_KEY
from air_classifier.observability.logging import bind_request_context, get_logger
from air_classifier.pipeline.orchestrator import Orchestrator
from air_classifier.pipeline.tiers.base import Tier
from air_classifier.schemas.common import Options
from air_classifier.security.api_keys import ApiKeyRecord, ApiKeyStore
from air_classifier.security.quotas import RateLimiter, RateLimitResult

__all__ = [
    "ApiKeyDep",
    "AppState",
    "AppStateDep",
    "IdempotencyRecord",
    "RateLimitedKeyDep",
    "apply_key_policy",
    "enforce_rate_limit",
    "get_app_state",
    "idempotency_key_of",
    "lookup_idempotent",
    "rate_limited_key",
    "require_api_key",
    "require_scope",
    "store_idempotent",
]

logger = get_logger(__name__)

#: Attribute on ``app.state`` holding the :class:`AppState`. Named once so
#: startup and the routes cannot drift apart on a string literal.
STATE_ATTR: Final[str] = "app_state"

#: Attribute on ``request.state`` holding the ``RateLimit-*`` headers for this
#: request.
#:
#: The two outcomes that matter already carry the headers without reading this:
#: a served request gets them from the injected ``Response``, and a refusal
#: carries them on :class:`RateLimitedError` itself. It is published here for
#: the remaining case — a handler that fails *after* quota was charged, whose
#: error response is built somewhere that never saw the limiter — so that a
#: future response hook can attach them without re-checking the quota.
RATE_LIMIT_HEADERS_ATTR: Final[str] = "rate_limit_headers"


# ── Application state ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AppState:
    """The singletons built once at startup and shared by every request.

    Frozen because none of these are per-request: swapping one mid-flight would
    give two concurrent requests two different views of the service.
    """

    settings: Settings
    cache: CacheBackend
    api_keys: ApiKeyStore
    rate_limiter: RateLimiter
    orchestrator: Orchestrator
    #: Every rung the app built, in ladder order. Readiness and
    #: ``/v1/capabilities`` need the tiers themselves, not just the orchestrator
    #: they were handed to.
    tiers: tuple[Tier, ...] = field(default_factory=tuple)

    def attach(self, app: FastAPI) -> None:
        """Publish this state on ``app.state`` for :func:`get_app_state`."""
        setattr(app.state, STATE_ATTR, self)


def get_app_state(request: Request) -> AppState:
    """The typed singleton container for this app.

    A missing state means the app was constructed without a lifespan having
    run — a wiring bug, not a caller error, so it raises rather than 4xx.
    """
    state = getattr(request.app.state, STATE_ATTR, None)
    if not isinstance(state, AppState):
        raise RuntimeError(
            f"app.state.{STATE_ATTR} is not an AppState; "
            "the application lifespan did not run or did not call AppState.attach()"
        )
    return state


# ── Authentication ────────────────────────────────────────────────────────────


async def require_api_key(request: Request) -> ApiKeyRecord:
    """Resolve the caller's identity from ``X-API-Key``.

    The 401/403 split is deliberate and is what the store's ``None`` cannot
    express on its own: *no* credential is a 401 (send one), while a credential
    that is unknown, malformed or revoked is a 403 (this one will never work).
    Neither branch reveals which of those it was, and neither logs the key.
    """
    state = get_app_state(request)
    raw = request.headers.get(HEADER_API_KEY)
    presented = bool(raw and raw.strip())

    # Passing None when nothing was presented lets the store apply
    # `allow_unauthenticated` itself, so the development identity is minted in
    # exactly one place rather than in every dependency that needs one.
    record = state.api_keys.lookup(raw if presented else None)
    if record is not None:
        bind_request_context(api_key_id=record.id)
        # Also stamped on `request.state`, and not only on the logging context:
        # `AccessLogMiddleware` is raw ASGI running *outside* this dependency's
        # task, so it reads the scope rather than the contextvars — without this
        # line every authenticated request is logged and counted as anonymous,
        # and the per-key request and cost series would be permanently empty.
        setattr(request.state, API_KEY_ID_ATTR, record.id)
        return record

    if not presented:
        logger.info("auth.missing_key", path=request.url.path)
        raise MissingApiKeyError(detail=f"{HEADER_API_KEY} header is required")

    logger.warning("auth.rejected", path=request.url.path)
    raise InsufficientScopeError(detail="The provided API key is not valid for this service")


def require_scope(scope: str) -> Callable[[Request], Awaitable[ApiKeyRecord]]:
    """Dependency factory: authenticate, then demand ``scope``.

    A factory rather than a parameterised dependency so the scope is fixed at
    import time and shows up in the route declaration, where a reviewer reading
    the router can see which endpoints need which grant.
    """

    async def dependency(request: Request) -> ApiKeyRecord:
        record = await require_api_key(request)
        if not record.has_scope(scope):
            logger.warning("auth.scope_denied", key_id=record.id, scope=scope)
            raise InsufficientScopeError(detail=f"API key lacks the '{scope}' scope")
        return record

    return dependency


# ── Rate limiting ─────────────────────────────────────────────────────────────


async def enforce_rate_limit(
    request: Request,
    key: ApiKeyRecord,
    response: Response | None = None,
) -> RateLimitResult:
    """Charge one request against ``key``'s quota.

    The ``RateLimit-*`` headers are published on every outcome, not only on a
    rejection: a client that can only discover its remaining budget by being
    refused has to hit the wall to find it. They are stashed on ``request.state``
    so the middleware can attach them even when the response is an error body
    built somewhere else entirely.
    """
    state = get_app_state(request)
    result = await state.rate_limiter.check_key(key)

    headers = result.headers()
    setattr(request.state, RATE_LIMIT_HEADERS_ATTR, headers)
    if response is not None:
        response.headers.update(headers)

    if not result.allowed:
        # The headers ride on the exception too, so the 429 carries the budget
        # even if nothing else touches the response on the way out.
        raise RateLimitedError(
            detail="Request quota exhausted for this API key",
            retry_after=result.retry_after_s,
            headers=headers,
        )
    return result


async def rate_limited_key(
    request: Request,
    response: Response,
    key: Annotated[ApiKeyRecord, Depends(require_api_key)],
) -> ApiKeyRecord:
    """Authenticate and charge quota in one dependency.

    Takes ``Response`` because that is the only injection point from which a
    dependency can put headers on a successful response.
    """
    await enforce_rate_limit(request, key, response=response)
    return key


# ── Key policy ────────────────────────────────────────────────────────────────


def apply_key_policy(options: Options, key: ApiKeyRecord) -> Options:
    """Reconcile a caller's options with the ceilings carried by their key.

    Returns a new :class:`Options`; the caller's is never mutated, so the
    original request body still reflects what was actually asked for when it is
    echoed back or logged.

    Two different answers to "the key disagrees with the request", and the
    difference is what the caller ends up believing:

    * **Tier ceiling — clamp silently.** A key capped below what was requested
      gets a cheaper, possibly less accurate verdict, and the response says
      which tier decided. Nothing the caller believes is false, so a 403 here
      would only make a cost guardrail look like a broken client.
    * **Forced redaction — refuse loudly.** A caller that explicitly sent
      ``redact_pii=false`` and got redaction anyway would be wrong about where
      their data went. ``redact_pii_was_explicit`` exists exactly to tell that
      apart from a caller who said nothing and took the safe default, and only
      the explicit opt-out is rejected.
    """
    if key.force_pii_redaction and options.redact_pii_was_explicit and not options.redact_pii:
        logger.warning("policy.redaction_opt_out_denied", key_id=key.id)
        raise InsufficientScopeError(
            detail="This API key requires PII redaction; redact_pii cannot be disabled"
        )

    updates: dict[str, object] = {}

    ceiling = key.clamp_tier(options.max_tier)
    if ceiling is not options.max_tier:
        logger.info(
            "policy.tier_clamped",
            key_id=key.id,
            requested=options.max_tier.value,
            granted=ceiling.value,
        )
        updates["max_tier"] = ceiling
        # `min_tier <= max_tier` is an Options invariant, and model_copy does not
        # re-validate — a floor left above the new ceiling would reach the policy
        # as an impossible request rather than as a validation error.
        if options.min_tier is not None and options.min_tier.rank > ceiling.rank:
            updates["min_tier"] = ceiling

    if key.force_pii_redaction and not options.redact_pii:
        updates["redact_pii"] = True

    return options.model_copy(update=updates)


# ── Idempotency ───────────────────────────────────────────────────────────────


class IdempotencyRecord(BaseModel):
    """What a replayable response is stored as.

    The body digest travels with the response because an idempotency key only
    means "this is the same request" if the request was in fact the same. Without
    it, a client that reused a key by accident would be served an answer to
    someone else's question and have no way to notice.
    """

    body_sha256: str
    response: str


def idempotency_key_of(request: Request) -> str | None:
    """The caller's ``Idempotency-Key``, or ``None`` when they sent none.

    Idempotency is opt-in: an absent header means every request is processed,
    which is the correct default for an analysis endpoint with no side effects.
    """
    raw = request.headers.get(HEADER_IDEMPOTENCY_KEY)
    stripped = raw.strip() if raw else ""
    return stripped or None


async def lookup_idempotent(
    request: Request, key: ApiKeyRecord, *, body: bytes | str
) -> str | None:
    """The stored response for this ``Idempotency-Key``, if there is one.

    Raises :class:`IdempotencyConflictError` when the key has been seen before
    with a *different* body. Returning the stored answer in that case would be
    the one failure mode idempotency is supposed to prevent, and a 409 tells the
    caller which of the two requests is the mistake.
    """
    idem_key = idempotency_key_of(request)
    if idem_key is None:
        return None

    state = get_app_state(request)
    cache_key = idempotency_cache_key(
        prefix=state.settings.cache.key_prefix,
        api_key_id=key.id,
        idempotency_key=idem_key,
    )
    raw = await state.cache.get(cache_key)
    if raw is None:
        return None

    try:
        record = IdempotencyRecord.model_validate_json(raw)
    except ValidationError:
        # An entry written by an older schema, or a truncated value. A replay is
        # an optimisation, so the request simply runs again rather than failing.
        logger.warning("idempotency.entry_unreadable", key_id=key.id)
        return None

    if record.body_sha256 != _body_digest(body):
        logger.warning("idempotency.conflict", key_id=key.id)
        raise IdempotencyConflictError(
            detail=(f"{HEADER_IDEMPOTENCY_KEY} has already been used with a different request body")
        )

    logger.info("idempotency.replay", key_id=key.id)
    return record.response


async def store_idempotent(
    request: Request,
    key: ApiKeyRecord,
    *,
    body: bytes | str,
    response: str,
) -> None:
    """Remember ``response`` so a replay of this key returns it verbatim.

    Stored verbatim rather than re-serialised from a model: a replay must be
    byte-identical to the original answer, including any field the response
    model would render differently after a deploy.
    """
    idem_key = idempotency_key_of(request)
    if idem_key is None:
        return

    state = get_app_state(request)
    ttl = state.settings.cache.idempotency_ttl_seconds
    if ttl <= 0:
        # A zero TTL is how idempotency storage is switched off; writing an entry
        # that expires on arrival would only cost a round trip per request.
        return

    cache_key = idempotency_cache_key(
        prefix=state.settings.cache.key_prefix,
        api_key_id=key.id,
        idempotency_key=idem_key,
    )
    record = IdempotencyRecord(body_sha256=_body_digest(body), response=response)
    await state.cache.set(cache_key, record.model_dump_json(), ttl)


def _body_digest(body: bytes | str) -> str:
    """sha256 of the raw request body.

    Hashed rather than stored so that a cache entry never holds a second copy of
    the caller's text, and compared in full rather than truncated because this is
    the check that keeps two different requests apart.
    """
    payload = body.encode("utf-8") if isinstance(body, str) else body
    return hashlib.sha256(payload).hexdigest()


# ── FastAPI aliases ───────────────────────────────────────────────────────────

#: Typed singletons.
AppStateDep = Annotated[AppState, Depends(get_app_state)]
#: Authenticated caller, no quota charged — for cheap, unmetered endpoints.
ApiKeyDep = Annotated[ApiKeyRecord, Depends(require_api_key)]
#: Authenticated caller with one request charged against their quota. This is
#: what an analysis route wants.
RateLimitedKeyDep = Annotated[ApiKeyRecord, Depends(rate_limited_key)]
