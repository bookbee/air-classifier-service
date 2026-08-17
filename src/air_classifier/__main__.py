"""Console entry point — ``air-classifier``, or ``python -m air_classifier``.

Thin by design. Everything that decides what the service *is* lives in
:func:`air_classifier.main.create_app`; this module only decides how the process
is served, and takes even that from settings so a container never needs a
different command line.

The app is handed to uvicorn as an import string with ``factory=True`` rather
than as an object: ``workers`` forks after import, and a pre-built app would
have to be picklable and would share its connection pools across the fork.
"""

from __future__ import annotations

import uvicorn

from air_classifier.config import get_settings

__all__ = ["main"]


def main() -> None:
    """Serve the application with uvicorn, configured from the environment."""
    settings = get_settings()
    uvicorn.run(
        "air_classifier.main:create_app",
        factory=True,
        host=settings.app.host,
        port=settings.app.port,
        workers=settings.app.workers,
        # `None`, not a dict: uvicorn's default config installs its own handlers
        # and would print an unstructured line next to every structured one.
        # `configure_logging` adopts uvicorn's loggers instead, so its output
        # arrives as JSON carrying the same request context as ours.
        log_config=None,
        # `AccessLogMiddleware` already emits one line per request, with the
        # route template, the api key id and the request id. uvicorn's version
        # has none of those and would double the volume.
        access_log=False,
        timeout_graceful_shutdown=int(settings.app.shutdown_grace_seconds),
        # The version banner is free reconnaissance for anything scanning.
        server_header=False,
    )


if __name__ == "__main__":
    main()
