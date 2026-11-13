"""The ladder against air-model-gateway's wire contract, end to end.

Every other test of Tiers 2 and 3 runs behind a fake provider, and the adapter's
own tests stop at the adapter. Neither can see the failure that matters most in
this integration: the gateway rejecting every call (a leftover body field, a
retired alias, a revoked token), which the ladder is *designed* to absorb. The
service keeps answering 200 from Tier 1 and nothing looks wrong from outside.
So this file boots the real app with the real adapter, puts a gateway double
behind ``respx`` at the HTTP layer, and asserts on what a caller actually gets:
which rung decided, and what the trace says when the LLM rungs fail.

Nothing opens a socket: ``respx`` intercepts every request to the gateway's
base URL, and an unmocked route fails the test rather than reaching the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Final

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from air_classifier_service.config import (
    AirLlmSettings,
    AppSettings,
    CacheSettings,
    SecuritySettings,
    Settings,
)
from air_classifier_service.constants import HEADER_API_KEY, HEADER_REQUEST_ID
from air_classifier_service.constants import Tier as TierId
from air_classifier_service.main import create_app
from air_classifier_service.security.api_keys import ALL_SCOPES, hash_key

pytestmark = pytest.mark.integration

#: Real auth, as in test_app_boot.py: the deployment's own key-store path.
_RAW_API_KEY: Final[str] = "gateway-wiring-test-key"
_SALT: Final[str] = "gateway-wiring-test-salt"
_AUTH: Final[dict[str, str]] = {HEADER_API_KEY: _RAW_API_KEY}

_GENERATE_URL: Final[str] = "http://localhost:8083/v1/generate"
_READY_URL: Final[str] = "http://localhost:8083/v1/ready"

#: Two strong opposite signals joined by "but": climbs from Tier 1 on structure
#: alone, whatever Tier 1's confidence (see test_ladder_wiring.py).
_CONTRASTIVE: Final[str] = "The battery life is great but the camera is awful."

#: Which concrete model answers each class, as this service's routing override
#: in air-model-gateway's deploy/routing.yaml has it today.
_ANSWERED_BY: Final[dict[str, tuple[str, str]]] = {
    "local": ("ollama", "llama3.2:3b"),
    "auto": ("anthropic", "claude-haiku-4-5"),
    "cloud": ("anthropic", "claude-sonnet-5"),
}


def _answer(request: httpx.Request) -> httpx.Response:
    """A gateway double: a confident, schema-valid verdict from whichever
    model serves the requested class."""
    model_class = json.loads(request.content)["model"]
    provider, model = _ANSWERED_BY[model_class]
    verdict = {
        "label": "mixed",
        "polarity": 0.1,
        "confidence": 0.92,
        "rationale": "Praises the battery, criticises the camera.",
    }
    return httpx.Response(
        200,
        json={
            "id": f"gen_{model_class}",
            "model_class": model_class,
            "model": model,
            "provider": provider,
            "content": json.dumps(verdict),
            "usage": {"prompt_tokens": 900, "completion_tokens": 60, "total_tokens": 960},
            "cost_usd": 0.0 if provider == "ollama" else 0.0012,
            "cached": False,
            "refusal": False,
            "finish_reason": "stop",
            "provider_finish_reason": "stop",
        },
    )


@pytest.fixture
def gateway() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.get(_READY_URL).mock(return_value=httpx.Response(200, json={"status": "ready"}))
        yield router


def _settings() -> Settings:
    """Both LLM rungs on, pointed at the gateway's default local address, which
    `respx` intercepts. Memory cache so no test depends on Redis."""
    keys = [
        {
            "id": "gateway-wiring",
            "name": "gateway wiring test key",
            "key_hash": hash_key(_RAW_API_KEY, _SALT),
            "scopes": sorted(ALL_SCOPES),
        }
    ]
    return Settings(
        app=AppSettings(env="test"),
        air_llm=AirLlmSettings(enabled=True, token=SecretStr("test-token")),
        cache=CacheSettings(backend="memory"),
        security=SecuritySettings(api_keys_inline=json.dumps(keys), hash_salt=SecretStr(_SALT)),
    )


def _classify(text: str) -> httpx.Response:
    with TestClient(create_app(_settings())) as client:
        return client.post("/v1/classify", headers=_AUTH, json={"text": text})


def _steps(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {step["tier"]: step for step in body["escalation_trace"]}


def test_a_contrastive_review_is_decided_by_an_llm_rung(gateway: respx.MockRouter) -> None:
    """The check that the migration actually works: the decision comes from a
    model the gateway served, not from Tier 1 with the LLM rungs quietly out."""
    route = gateway.post(_GENERATE_URL).mock(side_effect=_answer)

    response = _classify(_CONTRASTIVE)

    assert response.status_code == 200
    body = response.json()
    assert body["decided_by"] == TierId.T3_CLOUD_LLM.value
    assert body["degraded"] is False
    steps = _steps(body)
    assert steps[TierId.T2_LOCAL_LLM.value]["succeeded"] is True
    assert steps[TierId.T3_CLOUD_LLM.value]["succeeded"] is True

    sent = [json.loads(call.request.content) for call in route.calls]
    # Tier 2 asks for `local`, Tier 3 for `auto`; confident enough that the
    # `cloud` arbiter is never called.
    assert [payload["model"] for payload in sent] == ["local", "auto"]
    assert all("task" not in payload for payload in sent)
    # One id across both services' logs: the gateway sees the id the caller got.
    assert {call.request.headers[HEADER_REQUEST_ID] for call in route.calls} == {body["request_id"]}


@pytest.mark.parametrize(
    ("status", "code"),
    [(422, "validation_error"), (403, "policy_denied"), (401, "unauthorized")],
)
def test_a_rejected_request_shows_up_as_failed_rungs_not_a_quiet_degrade(
    gateway: respx.MockRouter, status: int, code: str
) -> None:
    """The ladder still answers (a tier failure is never a 5xx), but the trace
    must record both LLM rungs as *failed*: that is the only signal a caller or
    a dashboard has that the gateway is refusing this service's calls."""
    gateway.post(_GENERATE_URL).mock(
        return_value=httpx.Response(
            status,
            headers={"content-type": "application/problem+json"},
            json={"status": status, "code": code, "request_id": "gw"},
        )
    )

    response = _classify(_CONTRASTIVE)

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["decided_by"] == TierId.T1_CLASSIFIER.value
    steps = _steps(body)
    assert steps[TierId.T2_LOCAL_LLM.value]["succeeded"] is False
    assert steps[TierId.T3_CLOUD_LLM.value]["succeeded"] is False
