"""PII redaction tests.

Two failure modes are being guarded here, and they pull in opposite directions:
letting an identifier through to a cloud provider, and mangling an order number
or a date so badly that the sentiment computed on the leftovers is wrong.
"""

from __future__ import annotations

import pytest

from air_sentiment.constants import PIIKind
from air_sentiment.preprocess.pii import luhn_valid, redact


def _kinds(text: str) -> set[PIIKind]:
    return {finding.kind for finding in redact(text).findings}


# ── Nothing to do ─────────────────────────────────────────────────────────────

CLEAN_TEXTS: list[str] = [
    "The delivery was late and the box was crushed",
    "Order 1234567890123456 never arrived",
    "Meeting at 12:30:45 on 2024-01-15 went fine",
    "Version 1.2.3 fixed the crash, thanks!",
    "It cost 19.99 and arrived in 3 days",
    "",
]


@pytest.mark.parametrize("text", CLEAN_TEXTS)
def test_text_without_pii_comes_back_byte_identical(text: str) -> None:
    result = redact(text)

    assert result.text == text
    assert result.redacted is False
    assert result.findings == []


# ── Detection ─────────────────────────────────────────────────────────────────


def test_email_is_redacted() -> None:
    result = redact("Contact me at john.doe@example.co.uk please")

    assert result.text == "Contact me at [REDACTED:email] please"
    assert result.redacted is True
    assert _kinds("Contact me at john.doe@example.co.uk please") == {PIIKind.EMAIL}


def test_repeated_findings_are_counted_once_per_occurrence() -> None:
    result = redact("Write to a@example.com or b@example.com")

    (finding,) = result.findings
    assert finding.kind is PIIKind.EMAIL
    assert finding.count == 2
    assert "@" not in result.text


@pytest.mark.parametrize(
    "text",
    [
        "Call +44 20 7946 0958 about the order",
        "Call +1 (555) 123-4567 about the order",
        "Call 555-0123 about the order",
        "Call 020 7946 0958 about the order",
    ],
)
def test_phone_numbers_are_redacted(text: str) -> None:
    result = redact(text)

    assert PIIKind.PHONE in _kinds(text)
    assert result.text.startswith("Call [REDACTED:phone]")


def test_ipv4_and_ipv6_are_redacted() -> None:
    result = redact("Server 192.168.1.10 and 2001:db8::1 are both down")

    assert result.text == "Server [REDACTED:ip_address] and [REDACTED:ip_address] are both down"
    assert result.findings[0].count == 2


def test_national_id_shapes_are_redacted() -> None:
    assert _kinds("SSN 123-45-6789 was leaked") == {PIIKind.NATIONAL_ID}
    assert _kinds("My NI number is AB123456C") == {PIIKind.NATIONAL_ID}


def test_credentials_in_a_url_take_the_whole_url() -> None:
    result = redact("Try https://admin:hunter2@internal.example.com/panel and see")

    assert result.text == "Try [REDACTED:url_credentials] and see"
    assert result.findings[0].kind is PIIKind.URL_CREDENTIALS


def test_a_plain_url_is_not_pii() -> None:
    text = "Docs are at https://example.com/help and they are useless"

    assert redact(text).redacted is False


# ── Luhn ──────────────────────────────────────────────────────────────────────

LUHN_VALID: list[str] = [
    "4111111111111111",
    "5500000000000004",
    "378282246310005",
    "6011111111111117",
    "1234567812345670",
]
LUHN_INVALID: list[str] = [
    "4111111111111112",
    "1234567890123456",
    "9876543210987654",
    "0000000000000001",
]


@pytest.mark.parametrize("digits", LUHN_VALID)
def test_luhn_accepts_valid_check_digits(digits: str) -> None:
    assert luhn_valid(digits) is True


@pytest.mark.parametrize("digits", LUHN_INVALID)
def test_luhn_rejects_invalid_check_digits(digits: str) -> None:
    assert luhn_valid(digits) is False


@pytest.mark.parametrize("digits", LUHN_VALID)
def test_card_shaped_runs_that_pass_luhn_are_redacted(digits: str) -> None:
    text = f"They charged {digits} twice"

    assert redact(text).text == "They charged [REDACTED:credit_card] twice"


@pytest.mark.parametrize("digits", LUHN_INVALID)
def test_card_shaped_runs_that_fail_luhn_are_left_alone(digits: str) -> None:
    """Order numbers and internal IDs are card-shaped; mangling them is a bug."""
    text = f"Reference {digits} is still open"

    assert PIIKind.CREDIT_CARD not in _kinds(text)
    assert digits in redact(text).text


@pytest.mark.parametrize("separator", [" ", "-"])
def test_separated_card_numbers_are_still_recognised(separator: str) -> None:
    grouped = separator.join(["4111", "1111", "1111", "1111"])
    text = f"Card {grouped} was declined"

    assert redact(text).text == "Card [REDACTED:credit_card] was declined"


# ── Sentence integrity ────────────────────────────────────────────────────────


def test_redaction_leaves_the_sentence_analysable() -> None:
    result = redact("Charged my card 4111111111111111 twice, awful service!")

    assert result.text == "Charged my card [REDACTED:credit_card] twice, awful service!"


def test_surrounding_punctuation_survives() -> None:
    result = redact("Mail me (john@example.com), please!")

    assert result.text == "Mail me ([REDACTED:email]), please!"


def test_mixed_findings_are_reported_per_kind() -> None:
    text = "Reach me at a@example.com or +44 20 7946 0958, card 4111111111111111"
    result = redact(text)

    counts = {finding.kind: finding.count for finding in result.findings}
    assert counts == {PIIKind.EMAIL: 1, PIIKind.CREDIT_CARD: 1, PIIKind.PHONE: 1}
    assert result.redacted is True


def test_placeholders_are_not_themselves_redacted_again() -> None:
    once = redact("Email a@example.com and call +44 20 7946 0958")

    assert redact(once.text).redacted is False
