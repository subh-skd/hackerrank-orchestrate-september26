"""Dated currency conversion.

Every foreign-currency event in the dataset is salary, and all five required
pairs (USD->INR, USD->IDR, USD->EUR, EUR->ZAR, EUR->USD) have direct rates, so
no chaining is needed. A rate is looked up for the event's settlement date in
the stated from->to direction, per AGENTS.md 6.1.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal


class RateUnavailable(LookupError):
    """Raised when no rate can be found for a pair. Never silently ignored."""


class FxTable:
    def __init__(self, rates: dict[tuple[date, str, str], Decimal]) -> None:
        self._rates = rates
        self._dates_by_pair: dict[tuple[str, str], list[date]] = {}
        for rate_date, src, dst in rates:
            self._dates_by_pair.setdefault((src, dst), []).append(rate_date)
        for bucket in self._dates_by_pair.values():
            bucket.sort()
        # Records every non-exact date match so the trace can disclose it.
        self.approximations: list[tuple[str, str, date, date]] = []

    def rate(self, src: str, dst: str, on: date) -> Decimal:
        if src == dst:
            return Decimal(1)
        exact = self._rates.get((on, src, dst))
        if exact is not None:
            return exact
        available = self._dates_by_pair.get((src, dst))
        if not available:
            raise RateUnavailable(f"no {src}->{dst} rate in exchange_rates.csv")
        # Nearest rate on or before the date; otherwise the earliest known.
        earlier = [d for d in available if d <= on]
        chosen = earlier[-1] if earlier else available[0]
        self.approximations.append((src, dst, on, chosen))
        return self._rates[(chosen, src, dst)]

    def convert(self, amount: Decimal, src: str, dst: str, on: date) -> Decimal:
        """Convert amount from src to dst using the rate dated `on`."""
        if src == dst:
            return amount
        return (amount * self.rate(src, dst, on)).quantize(Decimal("0.01"))
