#!/usr/bin/env python3
"""Fetch the pinned Tier-1 ONNX sentiment artifacts into a target directory.

This runs during the Docker build, *before* the application virtualenv exists,
so it may use nothing outside the standard library.

A failed download is deliberately not a failed build. The classifier tier is
written against a pluggable scorer backend and falls back to the pure-lexicon
scorer when the artifact is absent (docs/00-plan.md §4, resolved item 1), so an
image built without a model is a degraded image, not a broken one — and that
beats a release train blocked by someone else's CDN. Pass ``--strict`` to invert
that for pipelines where the artifact is mandatory.

Note for anyone reading this expecting it to produce today's actual Tier-1
default: it does not. `MODEL_REPO` below is the older sequence-classification
artifact this service shipped with first; the current default backend is the
zero-shot NLI model `export_zeroshot_model.py` produces instead
(`ClassifierSettings.scorer_kind = "zero_shot_nli"` in `config.py`). This
stage's build args default to empty and it never actually runs in the current
build (see the Dockerfile's `model` stage comment) — it is kept as a working,
documented escape hatch for baking a different artifact in, not as the
producer of what ships today.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

#: Pinned to a commit, not a branch: `main` moving under us would silently
#: change the model an image was built with while the tag stayed the same.
MODEL_REPO = "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "0b6928efcb76139cae2c6881d49cda67fe119f42"
_REVISION_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}"

DEFAULT_MODEL_URL = f"{_REVISION_BASE}/onnx/model_quantized.onnx"
DEFAULT_MODEL_SHA256 = "79b45c7fd8c2673aa53615aba07caeb2de798f63a95a2393a365982429c94f7f"
DEFAULT_TOKENIZER_URL = f"{_REVISION_BASE}/tokenizer.json"
DEFAULT_TOKENIZER_SHA256 = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"

DEFAULT_TARGET = Path("/opt/models")
DEFAULT_TIMEOUT_S = 120.0

#: Names are fixed regardless of source URL so the image's
#: AIR_CLASSIFIER__CLASSIFIER__* paths stay valid when the URL is overridden.
MODEL_FILENAME = "model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"

_CHUNK_BYTES = 1 << 20
_ARTIFACT_MODE = 0o644
_ALLOWED_SCHEMES = frozenset({"https"})
_USER_AGENT = "air-classifier-service-fetch-model/1"
_LOG_PREFIX = "[fetch-model]"


class FetchError(RuntimeError):
    """One artifact could not be placed on disk."""


@dataclass(frozen=True, slots=True)
class Artifact:
    """A single file to retrieve, and how to recognise the right one."""

    name: str
    url: str
    sha256: str
    filename: str


def _log(message: str) -> None:
    print(f"{_LOG_PREFIX} {message}")


def _warn(message: str) -> None:
    print(f"{_LOG_PREFIX} {message}", file=sys.stderr)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, timeout: float) -> None:
    """Stream ``url`` to ``destination``, atomically."""
    scheme = urlsplit(url).scheme
    if scheme not in _ALLOWED_SCHEMES:
        allowed = ", ".join(sorted(_ALLOWED_SCHEMES))
        raise FetchError(f"refusing to fetch {url!r}: scheme must be one of {allowed}")

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310 — scheme checked above
    handle, temp_name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    temp_path = Path(temp_name)
    try:
        with (
            os.fdopen(handle, "wb") as sink,
            urllib.request.urlopen(request, timeout=timeout) as response,  # noqa: S310 — as above
        ):
            shutil.copyfileobj(response, sink, _CHUNK_BYTES)
        # mkstemp opens 0600 and rename preserves the mode; the image bakes
        # these in as root and serves as an unprivileged user, who would then
        # get EACCES on the model it was told to load.
        temp_path.chmod(_ARTIFACT_MODE)
        # Rename last so a half-written file is never mistaken for a complete
        # one by a later build, a restarted container, or the skip check below.
        temp_path.replace(destination)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise FetchError(f"{url}: {exc}") from exc


def _fetch(artifact: Artifact, target: Path, *, timeout: float, force: bool) -> Path:
    destination = target / artifact.filename

    if destination.exists() and not force:
        if not artifact.sha256 or _digest(destination) == artifact.sha256:
            _log(f"{artifact.name}: already present at {destination}")
            return destination
        _warn(f"{artifact.name}: on-disk checksum does not match, re-downloading")

    _log(f"{artifact.name}: downloading {artifact.url}")
    _download(artifact.url, destination, timeout=timeout)

    if artifact.sha256:
        actual = _digest(destination)
        if actual != artifact.sha256:
            destination.unlink(missing_ok=True)
            raise FetchError(
                f"{artifact.name}: sha256 mismatch — expected {artifact.sha256}, got {actual}"
            )
        _log(f"{artifact.name}: sha256 verified")
    else:
        _warn(f"{artifact.name}: no checksum supplied, integrity unverified")

    _log(f"{artifact.name}: wrote {destination} ({destination.stat().st_size} bytes)")
    return destination


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fetch_model.py",
        description="Download the pinned Tier-1 ONNX sentiment model and tokenizer.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Directory to write artifacts into (default: {DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_MODEL_URL,
        help="ONNX model URL (default: the pinned revision).",
    )
    parser.add_argument(
        "--sha256",
        default=DEFAULT_MODEL_SHA256,
        help="Expected sha256 of the model; empty string skips verification.",
    )
    parser.add_argument(
        "--tokenizer-url",
        default=DEFAULT_TOKENIZER_URL,
        help="Tokenizer JSON URL; empty string skips the tokenizer entirely.",
    )
    parser.add_argument(
        "--tokenizer-sha256",
        default=DEFAULT_TOKENIZER_SHA256,
        help="Expected sha256 of the tokenizer; empty string skips verification.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-connection timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when a verified artifact is already present.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when an artifact could not be fetched.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    target: Path = args.target
    strict: bool = args.strict

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _warn(f"cannot create {target}: {exc}")
        return 1 if strict else 0

    artifacts = [
        Artifact("model", args.url, args.sha256.strip().lower(), MODEL_FILENAME),
    ]
    if args.tokenizer_url:
        artifacts.append(
            Artifact(
                "tokenizer",
                args.tokenizer_url,
                args.tokenizer_sha256.strip().lower(),
                TOKENIZER_FILENAME,
            )
        )

    failed = False
    for artifact in artifacts:
        try:
            _fetch(artifact, target, timeout=args.timeout, force=args.force)
        except FetchError as exc:
            failed = True
            _warn(f"{exc}")

    if failed:
        _warn(
            "one or more artifacts are missing; the Tier-1 classifier will run on its "
            "lexicon fallback and /v1/capabilities will report the degraded backend"
        )
        return 1 if strict else 0

    _log(f"all artifacts present in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
