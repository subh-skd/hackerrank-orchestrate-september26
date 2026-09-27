"""Recurrence detection and forward projection.

Measured structure across all 275 users:

  * groceries 3.58/mo, transport 3.44/mo, dining 2.32/mo - rotating
    descriptions, jittered amounts. These are the "essential variable
    spending" the statement asks us to forecast conservatively, and they are
    aggregated per category.
  * every other expense category is exactly 1.00/mo with a stable description -
    a monthly fixed commitment, keyed by (category, description).
  * income hides two different things inside category "salary": stable payroll
    series ("Payroll credit", "Base salary", ...) and variable ones
    ("Monthly sales commission", "Quarterly performance bonus", "Delivery
    platform payout", ...). 140 of 275 users have more than one. Projecting the
    variable ones would invent income the statement forbids counting
    ("Do not count pending credits, bonuses, commissions ... until they
    settle"; "Do not invent unsupported future income"), so only stable
    recurring income series are carried forward.

Every free choice in how to project is a field on ForecastRule so calibrate.py
can identify the generator's rule against the solved samples instead of us
guessing.
"""

from __future__ import annotations

import calendar
import statistics
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import Decimal

from .ledger import CashRole, Ledger

VARIABLE_CATEGORIES = frozenset({"groceries", "transport", "dining"})


@dataclass(frozen=True)
class ForecastRule:
    """The parameterized projection rule. Defaults are the pre-calibration guess."""

    # Essential variable spending (groceries / transport / dining).
    # Measured across all users these recur on a fixed per-user interval
    # (7, 10, 14, 5 or 21 days), so "cadence" projects one occurrence per
    # interval from the last one, sized by the estimator over recent occurrences.
    variable_estimator: str = "mean"  # mean | median | max | last
    variable_lookback: int = 3  # complete months (monthly placements)
    variable_placement: str = (
        "cadence"  # cadence | replay | prorata | lump_start | lump_end
    )
    variable_cadence: str = "mode"  # mode | min  (interval between occurrences)
    variable_occurrences: int = 4  # recent occurrences feeding the estimator

    # Monthly fixed commitments.
    fixed_estimator: str = "last"  # last | mean | median | max
    fixed_lookback: int = 3

    # Income.
    income_require_stable: bool = True
    income_stability: str = "last2eq"  # constant | last2eq | any
    income_estimator: str = "last"  # last | mean | median | min

    # A series is projected only when it recurs at least this often within the
    # lookback window, and was last seen within `live_within_months`.
    min_occurrences: int = 2
    live_within_months: int = 1
    # Income needs its own floor: a single "Final employer payroll" or
    # "Prorated first salary" row must never become a recurring stream.
    income_min_occurrences: int = 2

    # History-supported variable income (freelance invoices, platform payouts,
    # commissions) that fails the stability test. "none" never counts it;
    # otherwise it is aggregated into one stream and projected on its cadence
    # at a conservative per-occurrence estimate. A message saying such income is
    # still pending suppresses it (evidence/apply.py).
    variable_income: str = "none"  # none | min | median | mean
    variable_income_occurrences: int = 6

    def with_(self, **kwargs: object) -> "ForecastRule":
        return replace(self, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CashFlow:
    """One projected or booked movement of cash, signed."""

    when: date
    amount: Decimal  # negative = outflow, positive = inflow
    label: str
    source: str  # event id, "projected:<key>", or "amendment:<id>"
    category: str = ""
    series_key: str = ""

    @property
    def is_outflow(self) -> bool:
        return self.amount < 0


@dataclass
class Series:
    """A detected recurring series and the evidence behind it."""

    key: str
    category: str
    description: str
    direction: str  # "debit" | "credit"
    occurrences: list[tuple[date, Decimal]] = field(default_factory=list)
    projected_amount: Decimal | None = None
    anchor: date | None = None
    skipped_reason: str = ""

    @property
    def amounts(self) -> list[Decimal]:
        return [a for _d, a in self.occurrences]

    @property
    def is_constant(self) -> bool:
        return len(set(self.amounts)) == 1

    @property
    def last_two_equal(self) -> bool:
        amounts = self.amounts
        return len(amounts) >= 2 and amounts[-1] == amounts[-2]


def _estimate(values: list[Decimal], how: str) -> Decimal:
    if not values:
        return Decimal(0)
    if how == "last":
        return values[-1]
    if how == "max":
        return max(values)
    if how == "min":
        return min(values)
    if how == "median":
        return Decimal(str(statistics.median(sorted(values))))
    total = sum(values, Decimal(0))
    return (total / Decimal(len(values))).quantize(Decimal("0.01"))


def _month_key(day: date) -> tuple[int, int]:
    return (day.year, day.month)


def _add_months(day: date, months: int) -> date:
    """Shift by whole months, clamping the day to the target month's length."""
    total = (day.year * 12 + day.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last))


def _complete_months(as_of: date, count: int) -> list[tuple[int, int]]:
    """The `count` whole calendar months immediately before as_of's month."""
    out = []
    cursor = date(as_of.year, as_of.month, 1)
    for _ in range(count):
        cursor = _add_months(cursor, -1)
        out.append(_month_key(cursor))
    return list(reversed(out))


class Projector:
    """Builds the forward cash-flow series for one user."""

    def __init__(self, ledger: Ledger, rule: ForecastRule) -> None:
        self.ledger = ledger
        self.rule = rule
        self.as_of = ledger.as_of
        self.notes: list[str] = []
        self.series: list[Series] = []

    # -- history ---------------------------------------------------------

    def _history(self) -> list[tuple[date, Decimal, str, str, str]]:
        """(cash_date, magnitude, direction, category, description) for settled past."""
        out = []
        for entry in self.ledger.entries:
            if entry.role is not CashRole.HISTORY or entry.amount_home is None:
                continue
            if entry.amount_home == 0:
                continue
            direction = "credit" if entry.amount_home > 0 else "debit"
            out.append(
                (
                    entry.cash_date,
                    abs(entry.amount_home),
                    direction,
                    entry.event.category,
                    entry.event.description,
                )
            )
        return sorted(out, key=lambda r: r[0])

    # -- series detection ------------------------------------------------

    def detect(self) -> list[Series]:
        """Group history into recurring series, keyed per §module docstring."""
        history = self._history()
        buckets: dict[tuple[str, str, str], Series] = {}

        for when, amount, direction, category, description in history:
            if direction == "debit" and category in VARIABLE_CATEGORIES:
                key = (direction, category, "")  # aggregate the variable ones
                label = category
            else:
                key = (direction, category, description)
                label = description
            series = buckets.get(key)
            if series is None:
                series = Series(
                    key=f"{direction}:{category}:{key[2]}",
                    category=category,
                    description=label,
                    direction=direction,
                )
                buckets[key] = series
            series.occurrences.append((when, amount))

        self.series = [buckets[k] for k in sorted(buckets)]
        return self.series

    # -- projections -----------------------------------------------------

    def _live_months(self) -> set[tuple[int, int]]:
        return set(_complete_months(self.as_of, self.rule.live_within_months))

    def fixed_flows(self, until: date) -> list[CashFlow]:
        """Project the one-per-month fixed expense commitments."""
        months_fixed = _complete_months(self.as_of, self.rule.fixed_lookback)
        live = self._live_months()
        flows: list[CashFlow] = []

        for series in self.series:
            if series.direction != "debit":
                continue
            if series.category in VARIABLE_CATEGORIES:
                continue  # handled by variable_flows

            window = [o for o in series.occurrences if _month_key(o[0]) in months_fixed]
            if len(window) < self.rule.min_occurrences:
                series.skipped_reason = (
                    f"only {len(window)} occurrence(s) in lookback; recurrence "
                    "not supported by history"
                )
                continue
            if not any(_month_key(o[0]) in live for o in window):
                series.skipped_reason = "series ended before the request"
                continue

            amount = _estimate([o[1] for o in window], self.rule.fixed_estimator)
            if amount <= 0:
                continue

            series.projected_amount = amount
            anchor = window[-1][0]
            series.anchor = anchor

            step = 1
            while True:
                when = _add_months(anchor, step)
                if when > until:
                    break
                if when >= self.as_of:
                    flows.append(
                        CashFlow(
                            when=when,
                            amount=-amount,
                            label=f"projected {series.description}",
                            source=f"projected:{series.key}",
                            category=series.category,
                            series_key=series.key,
                        )
                    )
                step += 1

        return flows

    # -- income ----------------------------------------------------------

    def income_flows(self, until: date) -> list[CashFlow]:
        """Confirmed future credits, plus continuation of recurring income.

        A confirmed `scheduled` credit is counted on its settlement date
        (AGENTS.md 6.3). Beyond it, income continues only when the history
        supports a *stable* recurring series - never from a commission, bonus or
        gig payout, which the statement forbids counting before settlement.
        """
        flows: list[CashFlow] = []
        booked_dates: set[date] = set()

        for entry in self.ledger.expected_income:
            if entry.amount_home is None or entry.amount_home <= 0:
                continue
            if not (self.as_of <= entry.cash_date <= until):
                continue
            flows.append(
                CashFlow(
                    when=entry.cash_date,
                    amount=entry.amount_home,
                    label=entry.event.description,
                    source=entry.event_id,
                    category=entry.event.category,
                    series_key=f"booked:{entry.event_id}",
                )
            )
            booked_dates.add(entry.cash_date)

        primary = self.primary_income_series()
        level, anchor = self._income_level_and_anchor(primary, flows)
        if level is None or anchor is None or level <= 0:
            if primary is None and not flows:
                flows.extend(self.variable_income_flows(until))
            return flows

        step = 1
        while True:
            when = _add_months(anchor, step)
            if when > until:
                break
            if when >= self.as_of and when not in booked_dates:
                flows.append(
                    CashFlow(
                        when=when,
                        amount=level,
                        label="projected recurring income",
                        source="projected:income",
                        category="salary",
                        series_key="projected:income",
                    )
                )
            step += 1
        return flows

    def variable_income_flows(self, until: date) -> list[CashFlow]:
        """Conservative projection of recurring-but-variable income."""
        if self.rule.variable_income == "none":
            return []
        months = set(_complete_months(self.as_of, self.rule.fixed_lookback))
        live = self._live_months()
        occurrences: list[tuple[date, Decimal]] = []
        for series in self.series:
            if series.direction != "credit" or series.category != "salary":
                continue
            occurrences.extend(
                o for o in series.occurrences if _month_key(o[0]) in months
            )
        occurrences.sort()
        if len(occurrences) < 3 or not any(
            _month_key(o[0]) in live for o in occurrences
        ):
            return []

        gaps = [
            (b[0] - a[0]).days
            for a, b in zip(occurrences, occurrences[1:])
            if (b[0] - a[0]).days > 0
        ]
        if not gaps:
            return []
        counts: dict[int, int] = {}
        for gap in gaps:
            counts[gap] = counts.get(gap, 0) + 1
        interval = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]

        recent = [a for _d, a in occurrences[-self.rule.variable_income_occurrences :]]
        amount = _estimate(recent, self.rule.variable_income)
        if amount <= 0:
            return []
        self.notes.append(
            f"variable income projected every {interval}d at {amount} "
            f"({self.rule.variable_income} of last {len(recent)} payments)"
        )
        out = []
        when = occurrences[-1][0] + timedelta(days=interval)
        while when <= until:
            if when >= self.as_of:
                out.append(
                    CashFlow(
                        when,
                        amount,
                        "projected variable income",
                        "projected:variable_income",
                        "salary",
                        "projected:variable_income",
                    )
                )
            when += timedelta(days=interval)
        return out

    def primary_income_series(self) -> Series | None:
        """The stable recurring income series, if the history supports one."""
        months = _complete_months(self.as_of, self.rule.fixed_lookback)
        live = self._live_months()
        best: Series | None = None
        best_total = Decimal(0)

        for series in self.series:
            if series.direction != "credit":
                continue
            window = [o for o in series.occurrences if _month_key(o[0]) in months]
            if len(window) < self.rule.income_min_occurrences:
                series.skipped_reason = (
                    f"only {len(window)} occurrence(s) in lookback; recurring "
                    "income not supported by history"
                )
                continue
            if not any(_month_key(o[0]) in live for o in window):
                series.skipped_reason = "income series ended before the request"
                continue

            scoped = Series(
                key=series.key,
                category=series.category,
                description=series.description,
                direction=series.direction,
                occurrences=window,
            )
            if self.rule.income_require_stable and not self._income_stable(scoped):
                series.skipped_reason = (
                    "variable income (commission / bonus / gig payout); not "
                    "counted as confirmed future income"
                )
                continue

            total = sum(scoped.amounts, Decimal(0))
            if total > best_total:
                best, best_total = series, total

        return best

    def _income_level_and_anchor(
        self, primary: Series | None, booked: list[CashFlow]
    ) -> tuple[Decimal | None, date | None]:
        """A confirmed scheduled credit is the strongest evidence of the level."""
        salary_booked = [f for f in booked if f.category == "salary"]
        if salary_booked:
            latest = max(salary_booked, key=lambda f: f.when)
            return latest.amount, latest.when

        if primary is None:
            self.notes.append(
                "no stable recurring income series and no confirmed scheduled "
                "credit; no future income projected"
            )
            return None, None

        months = _complete_months(self.as_of, self.rule.fixed_lookback)
        window = [o for o in primary.occurrences if _month_key(o[0]) in months]
        level = _estimate([o[1] for o in window], self.rule.income_estimator)
        primary.projected_amount = level
        primary.anchor = window[-1][0]
        return level, window[-1][0]

    def _income_stable(self, series: Series) -> bool:
        mode = self.rule.income_stability
        if mode == "any":
            return True
        if mode == "constant":
            return series.is_constant
        return series.is_constant or series.last_two_equal

    def variable_flows(self, until: date) -> list[CashFlow]:
        """Project groceries / transport / dining forward."""
        months = _complete_months(self.as_of, self.rule.variable_lookback)
        flows: list[CashFlow] = []

        for series in self.series:
            if (
                series.direction != "debit"
                or series.category not in VARIABLE_CATEGORIES
            ):
                continue

            if self.rule.variable_placement == "cadence":
                flows.extend(self._cadence_flows(series, until))
                continue

            totals: dict[tuple[int, int], Decimal] = {}
            for when, amount in series.occurrences:
                key = _month_key(when)
                if key in months:
                    totals[key] = totals.get(key, Decimal(0)) + amount
            ordered = [totals[m] for m in months if m in totals]
            if not ordered:
                series.skipped_reason = "no occurrences in lookback window"
                continue

            monthly = _estimate(ordered, self.rule.variable_estimator)
            if monthly <= 0:
                continue
            series.projected_amount = monthly
            flows.extend(self._place_variable(series, monthly, months, until))

        return flows

    def _cadence_flows(self, series: Series, until: date) -> list[CashFlow]:
        """One projected occurrence per detected interval after the last one."""
        occurrences = sorted(series.occurrences)
        if len(occurrences) < 2:
            series.skipped_reason = "fewer than two occurrences; no cadence"
            return []

        gaps = [
            (b[0] - a[0]).days
            for a, b in zip(occurrences, occurrences[1:])
            if (b[0] - a[0]).days > 0
        ]
        if not gaps:
            return []
        if self.rule.variable_cadence == "min":
            interval = min(gaps)
        else:
            counts: dict[int, int] = {}
            for gap in gaps:
                counts[gap] = counts.get(gap, 0) + 1
            # Most frequent gap; ties go to the shorter (more conservative) one.
            interval = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        first_projected = occurrences[-1][0] + timedelta(days=interval)
        if self.rule.variable_estimator.startswith("ml_q"):
            amount = self._ml_amount(series, occurrences, first_projected)
        else:
            recent = [a for _d, a in occurrences[-self.rule.variable_occurrences :]]
            amount = _estimate(recent, self.rule.variable_estimator)
        if amount <= 0:
            return []
        series.projected_amount = amount
        series.anchor = occurrences[-1][0]

        out = []
        when = occurrences[-1][0] + timedelta(days=interval)
        while when <= until:
            if when >= self.as_of:
                out.append(
                    CashFlow(
                        when=when,
                        amount=-amount,
                        label=f"projected {series.category} (every {interval}d)",
                        source=f"projected:{series.key}",
                        category=series.category,
                        series_key=series.key,
                    )
                )
            when += timedelta(days=interval)
        return out

    def _ml_amount(
        self, series: Series, occurrences: list[tuple[date, Decimal]], next_day: date
    ) -> Decimal:
        """Learned upper-quantile forecast of the next occurrence (bow/ml)."""
        from .ml.spend_model import WINDOW, get_model

        model = get_model()
        if model is None or len(occurrences) < WINDOW:
            recent = [a for _d, a in occurrences[-self.rule.variable_occurrences :]]
            return _estimate(recent, "mean")
        q = int(self.rule.variable_estimator.removeprefix("ml_q")) / 100
        history = [(d, float(a)) for d, a in occurrences]
        return model.predict_amount(history, series.category, next_day, q)

    def _place_variable(
        self,
        series: Series,
        monthly: Decimal,
        months: list[tuple[int, int]],
        until: date,
    ) -> list[CashFlow]:
        placement = self.rule.variable_placement
        category = series.category

        def flow(when: date, amount: Decimal, suffix: str = "") -> CashFlow:
            return CashFlow(
                when=when,
                amount=-amount,
                label=f"projected {category}{suffix}",
                source=f"projected:{series.key}",
                category=category,
                series_key=series.key,
            )

        if placement == "prorata":
            daily = (monthly / Decimal(30)).quantize(Decimal("0.01"))
            out = []
            cursor = self.as_of
            while cursor <= until:
                out.append(flow(cursor, daily, " (daily)"))
                cursor += timedelta(days=1)
            return out

        if placement in ("lump_start", "lump_end"):
            out = []
            cursor = date(self.as_of.year, self.as_of.month, 1)
            while cursor <= until:
                if placement == "lump_start":
                    when = cursor
                else:
                    last = calendar.monthrange(cursor.year, cursor.month)[1]
                    when = date(cursor.year, cursor.month, last)
                if self.as_of <= when <= until:
                    out.append(flow(when, monthly, " (monthly)"))
                cursor = _add_months(cursor, 1)
            return out

        # "replay": reuse the most recent complete month's day pattern, scaled
        # so each projected month totals `monthly`.
        template_month = None
        for key in reversed(months):
            if any(_month_key(o[0]) == key for o in series.occurrences):
                template_month = key
                break
        if template_month is None:
            return []

        template = [o for o in series.occurrences if _month_key(o[0]) == template_month]
        template_total = sum((o[1] for o in template), Decimal(0))
        if template_total <= 0:
            return []
        scale = monthly / template_total

        out = []
        shift = 1
        while _add_months(date(*template_month, 1), shift) <= until:
            for when, amount in template:
                target = _add_months(when, shift)
                if self.as_of <= target <= until:
                    out.append(flow(target, (amount * scale).quantize(Decimal("0.01"))))
            shift += 1
        return out

    # -- booked flows ----------------------------------------------------

    def booked_flows(self, until: date) -> list[CashFlow]:
        """Pending and scheduled obligations already on the books.

        Confirmed future credits are emitted by income_flows instead.
        """
        out = []
        for entry in self.ledger.reserved:
            if entry.amount_home is None:
                continue
            when = max(entry.cash_date, self.as_of)
            if when > until:
                continue
            out.append(
                CashFlow(
                    when=when,
                    amount=entry.amount_home,
                    label=entry.event.description,
                    source=entry.event_id,
                    category=entry.event.category,
                    series_key=f"booked:{entry.event_id}",
                )
            )
        return out

    def project(self, until: date) -> list[CashFlow]:
        """The complete forward series for the window."""
        self.detect()
        flows = (
            self.booked_flows(until)
            + self.income_flows(until)
            + self.fixed_flows(until)
            + self.variable_flows(until)
        )
        return sorted(flows, key=lambda f: (f.when, f.source))
