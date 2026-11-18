"""The golden set — docs/02-lld.md §10's "Accuracy" layer, previously named
and never built (00-plan.md §9 said so explicitly).

Loads `evals/golden_set.jsonl` and runs every entry through the *real* app —
not stubs — via `TestClient`, the same way `tests/integration/test_app_boot.py`
does. Two things this catches that no other suite does:

* **A prompt or threshold change that breaks classification.** Every other
  test either fakes the tiers entirely or scripts a provider's response; this
  is the one place a genuine regression in T0/T1's real output, or in what
  counts as "confident enough," shows up as a failing assertion instead of a
  demo surprise.
* **Confidence calibration drifting into decorative territory.** If a
  classifier returns high confidence on everything, the escalation policy
  never escalates and the whole point of the ladder is gone. See
  `test_confident_cases_score_higher_than_ambiguous_ones` below.

Every entry's expected label and confidence band was verified against the
live service before being committed here (see the PR/commit that added this
file) — not guessed. Where a "clear" case actually resolves the fooled way
that is deliberately part of the fixture (`t1-pos-5`, `amb-*`): the ladder
existing at all is the fix for exactly that class of error, and pretending T1
gets everything right would make this suite lie about what T1 for.

Two groups only, both CI-safe, no external service: `shortcut`/`clear`/
`clear_loose`/`ambiguous` pin to T0 or T1 and run with air-model-gateway switched off
entirely, matching `test_app_boot.py`'s "no Redis, no air-model-gateway" deployment. The
`live` group needs a reachable air-model-gateway and is marked accordingly — run it with
`make test-eval-live` once air-model-gateway is up, not on every commit.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from air_classifier_service.config import (
    AppSettings,
    CacheSettings,
    ModelGatewaySettings,
    SecuritySettings,
    Settings,
)
from air_classifier_service.constants import HEADER_API_KEY
from air_classifier_service.main import create_app
from air_classifier_service.security.api_keys import ALL_SCOPES, hash_key

_GOLDEN_SET_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "evals" / "golden_set.jsonl"
)

RAW_API_KEY: Final[str] = "air-classifier-service-eval-key"
_SALT: Final[str] = "eval-test-salt"
AUTH: Final[dict[str, str]] = {HEADER_API_KEY: RAW_API_KEY}


def _load_golden_set() -> list[dict[str, Any]]:
    entries = [
        json.loads(line)
        for line in _GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise RuntimeError(f"{_GOLDEN_SET_PATH} loaded zero entries — check the path")
    return entries


_ENTRIES: Final[list[dict[str, Any]]] = _load_golden_set()


def _by_group(*groups: str) -> list[dict[str, Any]]:
    return [entry for entry in _ENTRIES if entry["group"] in groups]


def _api_keys_inline() -> str:
    return json.dumps(
        [
            {
                "id": "eval-test",
                "name": "golden set test key",
                "key_hash": hash_key(RAW_API_KEY, _SALT),
                "scopes": sorted(ALL_SCOPES),
            }
        ]
    )


def _settings(*, gateway_enabled: bool, gateway_base_url: str | None = None) -> Settings:
    """No Redis, no air-model-gateway by default — same "awkward on purpose" deployment
    `test_app_boot.py` uses, so the CI-safe groups never depend on the machine
    running them having anything more than Python installed.

    The token comes from `MODEL_GATEWAY_TOKEN`, not `AIR_CLASSIFIER__MODEL_GATEWAY__TOKEN`
    — the latter is stripped by the suite's own hermetic-environment fixture
    before every test, on purpose, so a real air-model-gateway token is never one
    accidentally-exported variable away from leaking into an assertion.
    `demo-tiers.sh` reads the same variable the same way.
    """
    token = os.environ.get("MODEL_GATEWAY_TOKEN", "air-classifier-dev")
    return Settings(
        app=AppSettings(env="test"),
        model_gateway=ModelGatewaySettings(
            enabled=gateway_enabled,
            base_url=gateway_base_url or "http://127.0.0.1:8083",
            token=SecretStr(token) if gateway_enabled else None,
        ),
        cache=CacheSettings(backend="memory"),
        security=SecuritySettings(
            api_keys_inline=_api_keys_inline(),
            hash_salt=SecretStr(_SALT),
        ),
    )


def _client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as started:
        yield started


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """T0/T1 only — every CI-safe group needs nothing more, and pinning a
    `live` case to a tier this app never built would fail confusingly."""
    yield from _client(create_app(_settings(gateway_enabled=False)))


@pytest.fixture(scope="module")
def live_client() -> Iterator[TestClient]:
    """The full ladder, against whatever air-model-gateway `MODEL_GATEWAY_BASE_URL` names —
    same override the demo script and doctor.sh already respect."""
    base_url = os.environ.get("MODEL_GATEWAY_BASE_URL", "http://127.0.0.1:8083")
    yield from _client(
        create_app(_settings(gateway_enabled=True, gateway_base_url=base_url))
    )


def _classify(client: TestClient, entry: dict[str, Any]) -> dict[str, Any] | None:
    """POST one golden entry. Returns the parsed body, or None on the
    expected-503 path — callers branch on `expect_status` to know which."""
    options: dict[str, Any] = {"latency_budget_ms": 30_000}
    pin = entry.get("pin_tier")
    if pin is not None:
        options["min_tier"] = pin
        options["max_tier"] = pin
    response = client.post(
        "/v1/classify", json={"text": entry["text"], "options": options}, headers=AUTH
    )
    expected_status = entry.get("expect_status", 200)
    assert response.status_code == expected_status, (
        f"{entry['id']}: expected HTTP {expected_status}, got {response.status_code} "
        f"— {response.text[:300]}"
    )
    return response.json() if expected_status == 200 else None


# ── CI-safe: T0/T1 only, no external service ────────────────────────────────


@pytest.mark.parametrize(
    "entry", _by_group("shortcut", "clear", "clear_loose"), ids=lambda e: e["id"]
)
def test_golden_case_gets_the_expected_label(client: TestClient, entry: dict[str, Any]) -> None:
    body = _classify(client, entry)
    if body is None:
        return  # the expected-503 shortcut case; _classify already asserted the status

    label = body["sentiment"]["label"]
    confidence = body["sentiment"]["confidence"]
    expected = entry["expect_label"]
    accepted = {expected} if isinstance(expected, str) else set(expected)
    assert label in accepted, (
        f"{entry['id']}: expected one of {sorted(accepted)}, got {label!r} at confidence "
        f"{confidence:.3f} — {entry.get('note', '')}"
    )
    floor = entry.get("min_confidence")
    if floor is not None:
        assert confidence >= floor, (
            f"{entry['id']}: correct label but confidence {confidence:.3f} fell below "
            f"the {floor} floor this case is pinned at — a real calibration regression, "
            f"not a fluke, if this starts failing."
        )


@pytest.mark.parametrize("entry", _by_group("ambiguous"), ids=lambda e: e["id"])
def test_ambiguous_case_stays_appropriately_uncertain(
    client: TestClient, entry: dict[str, Any]
) -> None:
    """No label assertion — several of these are cases T1 gets wrong on
    purpose (see the module docstring). What must hold is that T1 does not
    claim confidence it has not earned, because that confidence is what the
    escalation policy gates on."""
    body = _classify(client, entry)
    assert body is not None
    confidence = body["sentiment"]["confidence"]
    ceiling = entry["max_confidence"]
    assert confidence < ceiling, (
        f"{entry['id']}: confidence {confidence:.3f} rose to or past {ceiling} — "
        f"this case is deliberately hard ({entry.get('note', '')}); if T1 now scores it "
        f"this confidently, either T1 genuinely improved (update the fixture) or "
        f"confidence calibration just got worse, not better (investigate before merging)."
    )


def test_confident_cases_score_higher_than_ambiguous_ones(client: TestClient) -> None:
    """The calibration check the second hardening prompt asked for, made
    concrete: if 'clear' and 'ambiguous' cases score about the same on
    average, confidence is not tracking difficulty and the escalation policy
    it gates is decorative — exactly the failure mode `00-plan.md`'s own risk
    table already names."""
    clear = [_classify(client, e)["sentiment"]["confidence"] for e in _by_group("clear")]
    ambiguous = [
        _classify(client, e)["sentiment"]["confidence"] for e in _by_group("ambiguous")
    ]
    mean_clear = sum(clear) / len(clear)
    mean_ambiguous = sum(ambiguous) / len(ambiguous)
    assert mean_clear > mean_ambiguous + 0.15, (
        f"mean confident-case confidence ({mean_clear:.3f}) is not meaningfully above "
        f"mean ambiguous-case confidence ({mean_ambiguous:.3f}) — confidence has stopped "
        f"being a useful escalation signal."
    )


# ── Live: needs a reachable air-model-gateway ──────────────────────────────────────────


@pytest.mark.live
@pytest.mark.parametrize("entry", _by_group("live"), ids=lambda e: e["id"])
def test_live_case_resolves_correctly_through_the_full_ladder(
    live_client: TestClient, entry: dict[str, Any]
) -> None:
    """The other half of the story `test_golden_case_gets_the_expected_label`
    tells: T1 alone gets several of these wrong (see `amb-sarcasm-1` and
    friends above) — the ladder's job is to get them right anyway by
    escalating. This is the test that actually proves it does."""
    body = _classify(live_client, entry)
    assert body is not None
    label = body["sentiment"]["label"]
    confidence = body["sentiment"]["confidence"]
    expected = entry["expect_label"]
    accepted = {expected} if isinstance(expected, str) else set(expected)
    assert label in accepted, (
        f"{entry['id']}: expected one of {sorted(accepted)}, got {label!r} — "
        f"decided_by={body.get('decided_by')}, degraded={body.get('degraded')}"
    )
    assert confidence >= entry["min_confidence"]
