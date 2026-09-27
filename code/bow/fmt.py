"""Exact output formatting.

The published samples use three *different* numeric renderings, and each output
column must match its own. Verified against every amount string in
dataset/sample_requests.csv by tests/test_fmt.py.

  payment_plan / reduce_to  -> 2dp when fractional, bare integer otherwise, no commas
                               (620.4 -> "620.40", 25256 -> "25256")
  amount_safe_to_pay        -> minimal representation, trailing zeros stripped
                               (17229139.2 -> "17229139.2", 462 -> "462")
  explanation text          -> thousands separators, 2dp when fractional
                               (25256 -> "25,256", 15952906.67 -> "15,952,906.67")
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _quantized(value: Decimal) -> Decimal:
    """Round to 2dp, the maximum precision any dataset amount carries."""
    return value.quantize(Decimal("0.01"))


def is_integral(value: Decimal) -> bool:
    return _quantized(value) == _quantized(value).to_integral_value()


def plan_amount(value: Decimal) -> str:
    """Amount as rendered inside payment_plan and reduce_to actions."""
    q = _quantized(value)
    if is_integral(q):
        return str(int(q))
    return f"{q:.2f}"


def safe_amount(value: Decimal) -> str:
    """Amount as rendered in the amount_safe_to_pay column.

    Trailing zeros are stripped: the samples carry 17229139.2 and 603.3, not
    17229139.20 / 603.30.
    """
    q = _quantized(value)
    if is_integral(q):
        return str(int(q))
    normalized = q.normalize()
    # normalize() can yield exponent notation for large round numbers.
    sign, digits, exponent = normalized.as_tuple()
    if isinstance(exponent, int) and exponent > 0:
        normalized = normalized.quantize(Decimal(1))
    return f"{normalized:f}"


def money_text(currency: str, value: Decimal) -> str:
    """Amount as rendered inside decision_explanation, e.g. 'ZAR 25,256'."""
    q = _quantized(value)
    if is_integral(q):
        return f"{currency} {int(q):,}"
    return f"{currency} {q:,.2f}"


def date_text(day: date) -> str:
    """Date as rendered inside decision_explanation, e.g. '8 August 2025'."""
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def iso(day: date) -> str:
    return day.isoformat()


def payment_plan(payments: list[tuple[date, Decimal]]) -> str:
    """Chronological 'YYYY-MM-DD:amount' entries joined by '|', or 'none'."""
    if not payments:
        return "none"
    ordered = sorted(payments, key=lambda p: p[0])
    return "|".join(f"{iso(d)}:{plan_amount(a)}" for d, a in ordered)
