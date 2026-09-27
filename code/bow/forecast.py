"""The 90-day balance walk and the quantities derived from it.

problem_statement.md: "Forecast the user's balance for the next 90 days ...
A plan is safe only if the balance never falls below minimum_balance_to_keep."

A payment of X on date p lowers the balance on every day >= p by exactly X and
leaves earlier days untouched. So safety for any single-payment plan is decided
by two arrays computed once from the baseline curve:

    prefix_min[d] = min balance over [start, d]
    suffix_min[d] = min balance over [d, end]

    paying X on p is safe  <=>  prefix_min[p-1] >= minimum
                           and  suffix_min[p] - X >= minimum

which makes amount_safe_to_pay a closed form rather than a search:

    amount_safe_to_pay = clamp(global_trough - minimum, 0, requested_amount)

Multi-payment plans (installments, partial) are verified by re-simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from .ledger import Ledger, opening_balance, window_end
from .recurrence import (
    VARIABLE_CATEGORIES,
    CashFlow,
    ForecastRule,
    Projector,
)

FORECAST_DAYS = 90


@dataclass
class Curve:
    """Daily closing balances across the forecast window."""

    start: date
    days: list[date]
    balances: list[Decimal]
    minimum: Decimal

    def __post_init__(self) -> None:
        self._prefix: list[Decimal] = []
        self._suffix: list[Decimal] = []
        running = None
        for value in self.balances:
            running = value if running is None else min(running, value)
            self._prefix.append(running)
        running = None
        for value in reversed(self.balances):
            running = value if running is None else min(running, value)
            self._suffix.append(running)
        self._suffix.reverse()

    def index(self, day: date) -> int:
        return (day - self.start).days

    @property
    def trough(self) -> Decimal:
        return min(self.balances)

    @property
    def trough_date(self) -> date:
        return self.days[self.balances.index(self.trough)]

    def prefix_min(self, day: date) -> Decimal | None:
        """Minimum balance from the window start through `day`, or None if before."""
        i = self.index(day)
        if i < 0:
            return None
        return self._prefix[min(i, len(self._prefix) - 1)]

    def suffix_min(self, day: date) -> Decimal:
        i = max(0, min(self.index(day), len(self._suffix) - 1))
        return self._suffix[i]

    def balance_on(self, day: date) -> Decimal:
        i = max(0, min(self.index(day), len(self.balances) - 1))
        return self.balances[i]

    def is_safe(self) -> bool:
        return self.trough >= self.minimum

    def can_pay_on(self, day: date, amount: Decimal) -> bool:
        """Is a single payment of `amount` on `day` safe across the whole window?"""
        if day < self.start or day > self.days[-1]:
            return False
        before = self.prefix_min(day - timedelta(days=1))
        if before is not None and before < self.minimum:
            return False
        return self.suffix_min(day) - amount >= self.minimum

    def headroom(self) -> Decimal:
        """How much can leave today without breaching the minimum."""
        return self.trough - self.minimum


@dataclass
class SpendingChange:
    """A permitted adjustment to a flexible recurring expense.

    The cited event_id is the most recent occurrence of the series, which is how
    the solved samples cite them (e.g. stop:event_476 is user_06's December
    "Family streaming plan" row). The change applies to the whole series going
    forward, which is what series_key targets.
    """

    action: str  # "stop" | "reduce_to"
    event_id: str
    category: str
    description: str
    series_key: str
    new_amount: Decimal | None = None

    def render(self) -> str:
        from . import fmt

        if self.action == "stop":
            return f"stop:{self.event_id}"
        assert self.new_amount is not None
        return f"reduce_to:{self.event_id}:{fmt.plan_amount(self.new_amount)}"

    @property
    def monthly_relief(self) -> Decimal:
        """Placeholder filled in by Forecaster.available_changes()."""
        return getattr(self, "_relief", Decimal(0))


class Forecaster:
    """Builds curves for one request under a given rule."""

    def __init__(
        self,
        ledger: Ledger,
        rule: ForecastRule,
        amendments: list | None = None,
    ) -> None:
        self.ledger = ledger
        self.rule = rule
        self.start = ledger.as_of
        self.end = window_end(ledger.as_of, FORECAST_DAYS)
        self.minimum = ledger.profile.minimum_balance_to_keep
        self.opening = opening_balance(ledger.profile)
        self.amendments = amendments or []

        projector = Projector(ledger, rule)
        self.flows: list[CashFlow] = projector.project(self.end)
        self.series = projector.series
        self.notes = projector.notes

        self.applied_amendments: list[str] = []
        if self.amendments:
            from .evidence.apply import apply_amendments

            self.flows, self.applied_amendments = apply_amendments(
                self.flows,
                self.amendments,
                self.start,
                self.end,
                ledger.profile.home_currency,
                ledger.fx,
            )

    # -- curve construction ----------------------------------------------

    def curve(
        self,
        payments: list[tuple[date, Decimal]] | None = None,
        changes: list[SpendingChange] | None = None,
        extra_flows: list[CashFlow] | None = None,
    ) -> Curve:
        """Daily balances with optional payments and spending changes applied."""
        stops = {c.series_key for c in (changes or []) if c.action == "stop"}
        reductions = {
            c.series_key: c.new_amount
            for c in (changes or [])
            if c.action == "reduce_to" and c.new_amount is not None
        }

        deltas: dict[date, Decimal] = {}

        def add(when: date, amount: Decimal) -> None:
            if when < self.start:
                when = self.start
            if when > self.end:
                return
            deltas[when] = deltas.get(when, Decimal(0)) + amount

        for flow in self.flows + list(extra_flows or []):
            amount = flow.amount
            if flow.is_outflow and flow.series_key:
                if flow.series_key in stops:
                    continue
                if flow.series_key in reductions:
                    capped = reductions[flow.series_key]
                    # A reduction caps each occurrence; it never increases spend.
                    amount = -min(-amount, capped)
            add(flow.when, amount)

        for when, amount in payments or []:
            add(when, -amount)

        days: list[date] = []
        balances: list[Decimal] = []
        running = self.opening
        cursor = self.start
        while cursor <= self.end:
            running += deltas.get(cursor, Decimal(0))
            days.append(cursor)
            balances.append(running)
            cursor += timedelta(days=1)

        return Curve(
            start=self.start, days=days, balances=balances, minimum=self.minimum
        )

    # -- derived output quantities ---------------------------------------

    def amount_safe_to_pay(self, requested: Decimal) -> Decimal:
        """Largest amount payable on the request date, before spending changes.

        problem_statement.md: "the most the user can pay today before optional
        spending changes without breaking the 90-day safety check, capped at
        requested_amount."
        """
        headroom = self.curve().headroom()
        if headroom <= 0:
            return Decimal(0)
        return min(headroom, requested).quantize(Decimal("0.01"))

    def earliest_full_payment(self, requested: Decimal) -> date | None:
        """First date one full payment passes the check, ignoring spending changes.

        problem_statement.md: "the first date the full amount passes the safety
        check without optional spending changes." Empty when never safe.
        """
        base = self.curve()
        cursor = self.start
        while cursor <= self.end:
            if base.can_pay_on(cursor, requested):
                return cursor
            cursor += timedelta(days=1)
        return None

    # -- spending-change candidates --------------------------------------

    def available_changes(self) -> list[SpendingChange]:
        """Permitted stop/reduce actions, one per eligible recurring series.

        Only non-protected flexible events in a category the user allows may be
        changed (AGENTS.md 6.2). Actions are ordered by how much relief they
        deliver, so the plan search tries the most effective ones first.
        """
        profile = self.ledger.profile

        # Most recent flexible occurrence per series that we actually project.
        projected_keys = {f.series_key for f in self.flows if f.is_outflow}
        latest: dict[str, object] = {}
        for entry in self.ledger.history:
            event = entry.event
            if not event.is_flexible or entry.amount_home is None:
                continue
            if event.category in VARIABLE_CATEGORIES:
                key = f"debit:{event.category}:"
            else:
                key = f"debit:{event.category}:{event.description}"
            if key not in projected_keys:
                continue
            prior = latest.get(key)
            if prior is None or event.cash_date >= prior.event.cash_date:  # type: ignore[union-attr]
                latest[key] = entry

        monthly = self._monthly_outflow_by_series()

        out: list[SpendingChange] = []
        for key, entry in sorted(latest.items()):
            event = entry.event  # type: ignore[union-attr]
            category = event.category
            spend = monthly.get(key, Decimal(0))

            if event.is_stoppable and profile.may_stop(category):
                change = SpendingChange(
                    action="stop",
                    event_id=event.event_id,
                    category=category,
                    description=event.description,
                    series_key=key,
                )
                change._relief = spend  # type: ignore[attr-defined]
                out.append(change)

            if event.is_reducible and profile.may_reduce(category):
                floor = event.minimum_allowed_amount
                if floor is not None:
                    change = SpendingChange(
                        action="reduce_to",
                        event_id=event.event_id,
                        category=category,
                        description=event.description,
                        series_key=key,
                        new_amount=floor,
                    )
                    occurrences = self._occurrences_per_month(key)
                    relief = spend - floor * occurrences
                    change._relief = max(relief, Decimal(0))  # type: ignore[attr-defined]
                    out.append(change)

        out.sort(key=lambda c: (-c.monthly_relief, c.event_id))
        return out

    def window_relief(self, change: SpendingChange) -> Decimal:
        """Total projected spending a change removes across the whole window."""
        total = Decimal(0)
        for flow in self.flows:
            if not flow.is_outflow or flow.series_key != change.series_key:
                continue
            magnitude = -flow.amount
            if change.action == "stop":
                total += magnitude
            elif change.new_amount is not None:
                total += max(magnitude - change.new_amount, Decimal(0))
        return total

    def _monthly_outflow_by_series(self) -> dict[str, Decimal]:
        """Projected outflow per series over the first 30 days of the window."""
        horizon = self.start + timedelta(days=30)
        totals: dict[str, Decimal] = {}
        for flow in self.flows:
            if not flow.is_outflow or not flow.series_key:
                continue
            if flow.when > horizon:
                continue
            totals[flow.series_key] = (
                totals.get(flow.series_key, Decimal(0)) - flow.amount
            )
        return totals

    def _occurrences_per_month(self, series_key: str) -> Decimal:
        horizon = self.start + timedelta(days=30)
        count = sum(
            1
            for f in self.flows
            if f.series_key == series_key and f.is_outflow and f.when <= horizon
        )
        return Decimal(max(count, 1))
