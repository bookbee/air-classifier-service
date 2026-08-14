"""PII detection and redaction.

This runs before any tier sees the text, which sets two constraints that pull in
opposite directions. It has to catch identifiers before they reach a cloud
provider's logs, and it has to leave a *sentence* behind: the redacted text is
the text sentiment is computed on, so a replacement that swallows the
surrounding punctuation, or that eats an order number and leaves a fragment,
degrades the answer for every downstream rung.

Precision therefore matters as much as recall. Card-shaped digit runs are
validated with the Luhn checksum rather than trusted on length alone, dates are
excluded from the phone pattern, and every replacement is a single bracketed
token that tokenises cleanly and carries no sentiment of its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from air_sentiment.constants import REDACTION_TEMPLATE, PIIKind
from air_sentiment.observability.logging import get_logger
from air_sentiment.schemas.common import PIIFinding

logger = get_logger(__name__)

# ── Patterns ──────────────────────────────────────────────────────────────────

_EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}"
)

#: Any scheme with a ``user:password@`` userinfo section. The whole URL goes,
#: not just the credentials: a URL that needed embedded credentials is pointing
#: at something private, and the host is part of what leaks.
_URL_CREDENTIALS_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s/:@]+:[^\s/@]*@\S+"
)

#: 13 to 19 digits with optional single spaces or hyphens. Deliberately loose:
#: :func:`luhn_valid` is what decides, not the shape.
_CARD_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")

#: US SSN and UK National Insurance shapes. Both are checksum-free, so the
#: patterns are kept tight rather than generous.
_SSN_RE: Final[re.Pattern[str]] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_NINO_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[A-CEGHJ-PR-TW-Z]{2} ?\d{2} ?\d{2} ?\d{2} ?[A-D]\b"
)

_IPV4_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)

#: Full and ``::``-compressed forms. The lookarounds keep it away from clock
#: times, which are the classic false positive for a colon-separated pattern.
_IPV6_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![:.\w])(?:"
    r"(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}"
    r"|(?:[0-9A-Fa-f]{1,4}:){1,7}:(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){0,6})?"
    r"|::(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){0,6})?"
    r")(?![:.\w])"
)

#: Optional country code, optional area-code parentheses, then 6 to 15 digits
#: separated by spaces, dots or hyphens.
_PHONE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w+.-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,4}\)[ .-]?)?\d(?:[ .-]?\d){5,14}(?![\w-])"
)

_DIGITS_RE: Final[re.Pattern[str]] = re.compile(r"\D")
_SEPARATOR_RE: Final[re.Pattern[str]] = re.compile(r"[ .()-]")
#: ISO and common little/big-endian date shapes, which the phone pattern would
#: otherwise happily claim.
_DATE_LIKE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\d{4}[-.]\d{1,2}[-.]\d{1,2}$|^\d{1,2}[-.]\d{1,2}[-.]\d{2,4}$"
)

_MIN_PHONE_DIGITS: Final[int] = 7
_MAX_PHONE_DIGITS: Final[int] = 15
#: A bare digit run with no punctuation and no country code needs to be at least
#: this long to be a phone number rather than an order or ticket reference.
_BARE_PHONE_DIGITS: Final[int] = 10

_MIN_CARD_DIGITS: Final[int] = 13
_MAX_CARD_DIGITS: Final[int] = 19
_MIN_IPV6_HEX_DIGITS: Final[int] = 4


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PIIRedactionResult:
    """Redacted text plus a tally of what was found, per kind."""

    text: str
    findings: list[PIIFinding]
    redacted: bool


# ── Validators ────────────────────────────────────────────────────────────────


def luhn_valid(digits: str) -> bool:
    """Luhn (mod-10) checksum, the gate on every card-shaped digit run.

    Roughly nine in ten random 16-digit numbers fail it, which is what keeps
    order numbers, ticket references and SKUs out of the redactor.
    """
    if not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _is_card(candidate: str) -> bool:
    digits = _DIGITS_RE.sub("", candidate)
    return _MIN_CARD_DIGITS <= len(digits) <= _MAX_CARD_DIGITS and luhn_valid(digits)


def _is_phone(candidate: str) -> bool:
    if _DATE_LIKE_RE.match(candidate):
        return False
    digits = _DIGITS_RE.sub("", candidate)
    if not _MIN_PHONE_DIGITS <= len(digits) <= _MAX_PHONE_DIGITS:
        return False
    if candidate.startswith("+") or _SEPARATOR_RE.search(candidate):
        return True
    return len(digits) >= _BARE_PHONE_DIGITS


def _is_ipv6(candidate: str) -> bool:
    """Reject degenerate matches like ``a::b`` that are almost certainly prose."""
    hex_digits = len(candidate.replace(":", ""))
    return hex_digits >= _MIN_IPV6_HEX_DIGITS and ("::" in candidate or candidate.count(":") >= 4)


# ── Detectors ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Detector:
    kind: PIIKind
    pattern: re.Pattern[str]
    validate: Callable[[str], bool] | None = None


#: Order is behaviour, not taste. Credentialed URLs run before e-mail so that
#: ``user:pass@host`` is not filed as an address; cards and national IDs run
#: before phone numbers because every one of them is also phone-shaped.
_DETECTORS: Final[tuple[_Detector, ...]] = (
    _Detector(PIIKind.URL_CREDENTIALS, _URL_CREDENTIALS_RE),
    _Detector(PIIKind.EMAIL, _EMAIL_RE),
    _Detector(PIIKind.CREDIT_CARD, _CARD_RE, _is_card),
    _Detector(PIIKind.NATIONAL_ID, _SSN_RE),
    _Detector(PIIKind.NATIONAL_ID, _NINO_RE),
    _Detector(PIIKind.IP_ADDRESS, _IPV6_RE, _is_ipv6),
    _Detector(PIIKind.IP_ADDRESS, _IPV4_RE),
    _Detector(PIIKind.PHONE, _PHONE_RE, _is_phone),
)


def redact(text: str) -> PIIRedactionResult:
    """Replace recognised identifiers with ``[REDACTED:<kind>]`` placeholders.

    Returns the input untouched, with ``redacted=False``, when nothing matched —
    the common case, and one that must not pay for a rebuilt string or produce a
    text that differs from the original by so much as a byte.
    """
    working = text
    counts: dict[PIIKind, int] = {}

    for detector in _DETECTORS:
        working, hits = _apply(detector, working)
        if hits:
            counts[detector.kind] = counts.get(detector.kind, 0) + hits

    if not counts:
        return PIIRedactionResult(text=text, findings=[], redacted=False)

    findings = [PIIFinding(kind=kind, count=count) for kind, count in counts.items()]
    logger.debug("pii.redacted", kinds={kind.value: count for kind, count in counts.items()})
    return PIIRedactionResult(text=working, findings=findings, redacted=True)


def _apply(detector: _Detector, text: str) -> tuple[str, int]:
    placeholder = REDACTION_TEMPLATE.format(kind=detector.kind.value)
    validate = detector.validate
    hits = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal hits
        candidate = match.group()
        if validate is not None and not validate(candidate):
            return candidate
        hits += 1
        return placeholder

    return detector.pattern.sub(_replace, text), hits
