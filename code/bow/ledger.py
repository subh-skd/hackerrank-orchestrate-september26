"""Ledger normalization: decide what each event row means for cash.

Every event is classified into exactly one CashRole with a recorded reason, so
the per-request trace can explain any inclusion or exclusion. Rules come from
problem_statement.md ("Reserve pending debits. Do not count pending credits,
bonuses, commissions, refunds, lottery proceeds, or investment gains until they
settle" / "Ignore pending credits, failed or cancelled transactions, duplicate
records, and unrealized investments") and AGENTS.md 6.1 and 6.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum

from .fx import FxTable
from .models import (
    CANCELLED,
    CREDIT,
    FAILED,
    NON_CASH,
    PENDING,
    SCHEDULED,
    SETTLED,
    UNREALIZED,
    Event,
    Profile,
)


class CashRole(str, Enum):
    HISTORY = "history"  # settled and in the past: feeds recurrence detection
    RESERVED = "reserved"  # future or pending outflow that must be honoured
    EXPECTED_INCOME = "expected_income"  # confirmed future credit
    EXCLUDED = "excluded"  # does not affect cash at all


@dataclass(frozen=True)
class LedgerEntry:
    """One classified event, with its amount converted to the home currency."""

    event: Event
    role: CashRole
    reason: str
    amount_home: Decimal | None  # signed: negative for debits, positive for credits
    cash_date: date

    @property
    def event_id(self) -> str:
        return self.event.event_id


# Descriptions the generator uses for the authorization half of an
# authorization/settlement pair, and for reversal bookkeeping. Matched
# case-insensitively as substrings.
_AUTHORIZATION_HINTS = (
    "card authorization",
    "pending fuel authorization",
    "card charge later reversed",
    "cancelled card authorization",
)
_REVERSAL_HINTS = (
    "settled card charge reversal",
    "card charge later reversed",
)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


class Ledger:
    """Normalized view of one user's financial events as of a request date."""

    def __init__(
        self,
        profile: Profile,
        events: list[Event],
        fx: FxTable,
        as_of: date,
        image_amounts: dict[str, Decimal] | None = None,
    ) -> None:
        self.profile = profile
        self.as_of = as_of
        self.fx = fx
        self.image_amounts = image_amounts or {}
        self.entries: list[LedgerEntry] = []
        self.unresolved_amounts: list[str] = []
        self._classify(events)

    # -- public views ----------------------------------------------------

    @property
    def history(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.role is CashRole.HISTORY]

    @property
    def reserved(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.role is CashRole.RESERVED]

    @property
    def expected_income(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.role is CashRole.EXPECTED_INCOME]

    @property
    def excluded(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.role is CashRole.EXCLUDED]

    def by_id(self, event_id: str) -> LedgerEntry | None:
        for entry in self.entries:
            if entry.event_id == event_id:
                return entry
        return None

    # -- classification --------------------------------------------------

    def _amount(self, event: Event) -> Decimal | None:
        """Resolve the event amount, falling back to an image-extracted value."""
        if event.amount is not None:
            return event.amount
        resolved = self.image_amounts.get(event.event_id)
        if resolved is None:
            self.unresolved_amounts.append(event.event_id)
            return None
        return resolved

    def _to_home(self, event: Event, amount: Decimal) -> Decimal:
        return self.fx.convert(
            amount, event.currency, self.profile.home_currency, event.cash_date
        )

    def _signed(self, event: Event, amount: Decimal) -> Decimal:
        home = self._to_home(event, amount)
        return home if event.direction == CREDIT else -home

    def _classify(self, events: list[Event]) -> None:
        superseded = self._superseded_ids(events)

        for event in events:
            role, reason = self._role_for(event, superseded)

            amount_home: Decimal | None = None
            if role is not CashRole.EXCLUDED:
                raw = self._amount(event)
                if raw is None:
                    # An unresolvable amount must never be read as zero.
                    role, reason = (
                        CashRole.EXCLUDED,
                        "amount blank and no image value available",
                    )
                else:
                    amount_home = self._signed(event, raw)

            self.entries.append(
                LedgerEntry(
                    event=event,
                    role=role,
                    reason=reason,
                    amount_home=amount_home,
                    cash_date=event.cash_date,
                )
            )

    def _superseded_ids(self, events: list[Event]) -> dict[str, str]:
        """Event ids replaced by a later record, mapped to the reason.

        Two generator patterns are handled:
          * failed debit -> scheduled retry, joined by linked_event_id. The
            failed row is dropped; the retry is the live obligation.
          * authorization -> settlement duplicate pair: same user, same amount,
            same category, within a few days, where one description marks it as
            an authorization. Only the settled row counts.
        """
        out: dict[str, str] = {}

        by_id = {e.event_id: e for e in events}
        for event in events:
            if event.linked_event_id and event.linked_event_id in by_id:
                parent = by_id[event.linked_event_id]
                if parent.status == FAILED and event.status in (SCHEDULED, PENDING):
                    out[parent.event_id] = f"superseded by retry {event.event_id}"

        # Authorization/settlement duplicates.
        for event in events:
            if _norm(event.description) not in (_norm(h) for h in _AUTHORIZATION_HINTS):
                continue
            for other in events:
                if other.event_id == event.event_id:
                    continue
                if (
                    other.amount is not None
                    and event.amount is not None
                    and other.amount == event.amount
                    and other.category == event.category
                    and other.status == SETTLED
                    and abs((other.event_date - event.event_date).days) <= 5
                ):
                    out.setdefault(
                        event.event_id,
                        f"duplicate authorization of settled {other.event_id}",
                    )
                    break

        # Reversal pairs net to zero: drop both halves.
        reversals = [
            e
            for e in events
            if _norm(e.description) in (_norm(h) for h in _REVERSAL_HINTS)
        ]
        for reversal in reversals:
            out.setdefault(reversal.event_id, "reversed charge, nets to zero")

        return out

    def _role_for(
        self, event: Event, superseded: dict[str, str]
    ) -> tuple[CashRole, str]:
        if event.event_id in superseded:
            return CashRole.EXCLUDED, superseded[event.event_id]

        if event.direction == NON_CASH or event.status == UNREALIZED:
            return CashRole.EXCLUDED, "non-cash / unrealized investment value"

        if event.status == CANCELLED:
            return CashRole.EXCLUDED, "cancelled"

        if event.status == FAILED:
            return CashRole.EXCLUDED, "failed, no cash movement"

        if event.status == PENDING:
            if event.direction == CREDIT:
                return CashRole.EXCLUDED, "pending credit, not counted until settled"
            return CashRole.RESERVED, "pending debit, reserved"

        if event.status == SCHEDULED:
            if event.direction == CREDIT:
                return CashRole.EXPECTED_INCOME, "scheduled confirmed credit"
            return CashRole.RESERVED, "scheduled debit, reserved"

        if event.status == SETTLED:
            if event.cash_date > self.as_of:
                # Settled but dated after the request: treat by cash direction.
                if event.direction == CREDIT:
                    return CashRole.EXPECTED_INCOME, "settled future credit"
                return CashRole.RESERVED, "settled future debit"
            return CashRole.HISTORY, "settled history"

        return CashRole.EXCLUDED, f"unrecognized status {event.status!r}"


def opening_balance(profile: Profile) -> Decimal:
    """Balance available on the request date, before any reservations.

    current_available_balance is taken as stated; pending debits are applied by
    the forecast as reservations rather than being assumed already deducted.
    """
    return profile.current_available_balance


def window_end(as_of: date, days: int = 90) -> date:
    """Last day of the forecast window (problem_statement.md: 90-Day Safety Check)."""
    return as_of + timedelta(days=days)
