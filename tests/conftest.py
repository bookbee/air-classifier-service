"""Fixtures shared across the whole suite.

There is no ``anyio_backend`` fixture here on purpose: ``pyproject.toml`` sets
``asyncio_mode = "auto"``, so pytest-asyncio already runs every ``async def
test_*`` without a marker or a backend parameter.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import structlog

# Private, but it is the type structlog's own `get_logger` returns, and the
# alternative — duck-typing on an attribute name — would be no more stable and
# considerably less obvious.
from structlog._config import BoundLoggerLazyProxy

from air_classifier_service.cache.memory import MemoryCache
from air_classifier_service.config import AppSettings, Settings, get_settings
from air_classifier_service.observability import logging as obs_logging


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's machine out of the test run.

    ``Settings`` is 12-factor, which means an exported ``AIR_CLASSIFIER__*``
    variable or a ``.env`` sitting in the checkout silently rewrites the object
    under test. Both are stripped for the duration of every test, so a suite that
    passes on a laptop passes in CI for the same reasons.

    ``get_settings`` is cached, so its cache is dropped on the way in *and* on
    the way out — a test that populated it must not hand a stale singleton to
    the next one.
    """
    for name in list(os.environ):
        if name.upper().startswith("AIR_CLASSIFIER__"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@dataclass(frozen=True)
class _LoggingBaseline:
    """Enough global logging state to put the process back as it was found."""

    handler: logging.Handler | None
    level: int
    config: dict[str, Any]


@pytest.fixture(scope="session", autouse=True)
def _pristine_logging() -> _LoggingBaseline:
    """The process's logging state before any test touched it.

    Session-scoped and autouse so it is captured before *any* other fixture
    runs. That ordering is the whole point: a module- or session-scoped fixture
    that boots the service is set up ahead of every function-scoped one, so a
    per-test snapshot would already contain the service's handler and would
    happily "restore" the pollution it was meant to undo.
    """
    return _LoggingBaseline(
        handler=obs_logging._handler,
        level=logging.getLogger().level,
        config=structlog.get_config(),
    )


@pytest.fixture(autouse=True)
def _restore_global_logging(_pristine_logging: _LoggingBaseline) -> Iterator[None]:
    """Return process-wide logging to its pristine state after each test.

    Bringing the service up reconfigures structlog and installs a root handler.
    That is ``create_app``'s job rather than an accident — but stdlib logging is
    a process singleton, so a handler left behind by one suite is still attached
    when ``tests/unit/test_logging.py`` runs later, and every line it asserts on
    then arrives twice or not at all.

    This lives in the root conftest rather than beside the one suite that boots
    the service, because "the one suite that boots the service" stopped being
    true the moment a second one did — and the resulting failure lands in a
    *third*, unrelated file, which is close to the worst place to debug it from.

    Only *our* handler is added and removed. ``pytest``'s own capture handlers
    come and go on the root logger between tests, and rewriting the whole list
    would tear ``caplog`` out from under the next one.
    """
    yield

    baseline = _pristine_logging
    current = obs_logging._handler
    if current is baseline.handler:
        return

    root = logging.getLogger()
    if current is not None:
        root.removeHandler(current)
    if baseline.handler is not None and baseline.handler not in root.handlers:
        root.addHandler(baseline.handler)
    root.setLevel(baseline.level)
    obs_logging._handler = baseline.handler
    structlog.configure(**baseline.config)
    _release_memoised_loggers()


def _release_memoised_loggers() -> None:
    """Undo structlog's first-use memoisation of every module-level logger.

    ``configure_logging`` enables ``cache_logger_on_first_use``, which makes a
    lazy proxy overwrite its own ``bind`` with a closure over whatever
    configuration was live the first time it was used. In a process that
    configures logging once at startup that is exactly right, and it is why the
    setting is on.

    In a test *run* it is not: a logger memoised while the booted service's
    configuration was live would go on ignoring ``structlog.testing.capture_logs``
    for every suite that runs afterwards. That failure is silent — it arrives as
    an empty list of captured records rather than as an error — so it is cleaned
    up at the source instead of being left for the next author to find.
    """
    for name, module in list(sys.modules.items()):
        if not name.startswith("air_classifier_service"):
            continue
        for value in vars(module).values():
            if isinstance(value, BoundLoggerLazyProxy):
                # The memo is an instance attribute shadowing the class's
                # `bind`; dropping it restores the lazy behaviour.
                vars(value).pop("bind", None)


@pytest.fixture
def settings() -> Settings:
    """Default settings with ``app.env`` set to ``test``.

    Deliberately everything-else-default: a fixture that pre-tuned knobs would
    make every test that depends on a default assert against this file rather
    than against ``config.py``. Tests needing a different shape build their own
    with ``settings.model_copy(update=...)`` or a fresh ``Settings(...)``.
    """
    return Settings(app=AppSettings(env="test"))


@pytest.fixture
async def memory_cache(settings: Settings) -> AsyncIterator[MemoryCache]:
    """A live in-process cache backend, sized from ``settings``.

    Real rather than a fake because :class:`MemoryCache` *is* the default
    backend — a test that passes against a hand-rolled stub proves less than one
    that passes against the thing production runs.
    """
    cache = MemoryCache(max_entries=settings.cache.max_entries)
    try:
        yield cache
    finally:
        await cache.aclose()
