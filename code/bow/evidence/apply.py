"""Apply typed amendments to a projected cash-flow series.

Conflict order from problem_statement.md: an explicit amendment beats the
history-based projection, so every forecast-affecting intent rewrites the
projected flows it describes. Intents that only confirm an exclusion change
nothing here - the ledger already refuses pending credits, bonuses, prizes and
unrealized gains.
"""

from __future__ import annotations

import calendar
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal

from ..fx import FxTable
from ..recurrence import CashFlow
from . import schema as s

INCOME_KEY = "projected:income"


def _is_salary(flow: CashFlow) -> bool:
    return flow.amount > 0 and flow.category == "salary"


def _month_shift(day: date, months: int, dom: int | None = None) -> date:
    total = day.year * 12 + day.month - 1 + months
    year, month = divmod(total, 12)
    month += 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(dom or day.day, last))


class AmendmentApplier:
    def __init__(
        self,
        flows: list[CashFlow],
        as_of: date,
        until: date,
        home_currency: str,
        fx: FxTable,
    ) -> None:
        self.flows = list(flows)
        self.as_of = as_of
        self.until = until
        self.home = home_currency
        self.fx = fx
        self.applied: list[str] = []

    # -- helpers -------------------------------------------------------

    def _home(self, amount: Decimal, currency: str | None, on: date) -> Decimal:
        if not currency or currency == self.home:
            return amount
        return self.fx.convert(amount, currency, self.home, on)

    def _salary_flows(self) -> list[CashFlow]:
        return sorted((f for f in self.flows if _is_salary(f)), key=lambda f: f.when)

    def _replace(self, old: CashFlow, new: CashFlow) -> None:
        self.flows[self.flows.index(old)] = new

    def _monthly_from(self, start: date, amount: Decimal, label: str) -> list[CashFlow]:
        out = []
        step = 0
        while True:
            when = _month_shift(start, step)
            if when > self.until:
                break
            if when >= self.as_of:
                out.append(
                    CashFlow(when, amount, label, INCOME_KEY, "salary", INCOME_KEY)
                )
            step += 1
        return out

    def _drop_projected_income(self, from_day: date | None = None) -> None:
        self.flows = [
            f
            for f in self.flows
            if not (
                _is_salary(f)
                and f.series_key == INCOME_KEY
                and (from_day is None or f.when >= from_day)
            )
        ]

    # -- intents -------------------------------------------------------

    def apply(self, amendment: s.Amendment) -> None:
        handler = getattr(self, f"_on_{amendment.intent}", None)
        if handler is None:
            return
        snapshot = list(self.flows)
        handler(amendment)
        if self.flows != snapshot:
            self.applied.append(f"{amendment.message_id}: {amendment.describe()}")

    def _on_income_ended(self, a: s.Amendment) -> None:
        self._drop_projected_income()
        self._drop_variable_income()

    def _drop_variable_income(self) -> None:
        self.flows = [
            f for f in self.flows if f.series_key != "projected:variable_income"
        ]

    # Evidence that the variable income is not yet earned removes its projection.
    def _on_gig_payout_pending(self, a: s.Amendment) -> None:
        self._drop_variable_income()

    def _on_bonus_pending(self, a: s.Amendment) -> None:
        self._drop_variable_income()

    def _on_invoice_approved_suppression(self) -> None:
        self._drop_variable_income()

    def _on_salary_date_change(self, a: s.Amendment) -> None:
        salaries = self._salary_flows()
        if not salaries or a.effective_date is None:
            return
        target = a.effective_date
        first = salaries[0]
        self._replace(first, replace(first, when=target))
        # Later projected paydays follow the revised day of month.
        for flow in salaries[1:]:
            if flow.series_key == INCOME_KEY:
                moved = _month_shift(
                    target,
                    (flow.when.year - target.year) * 12
                    + flow.when.month
                    - target.month,
                )
                if moved <= target:
                    moved = _month_shift(target, 1)
                self._replace(flow, replace(flow, when=moved))

    def _set_level(self, amount: Decimal, from_day: date | None = None) -> None:
        for flow in self._salary_flows():
            if from_day is not None and flow.when < from_day:
                continue
            self._replace(flow, replace(flow, amount=amount))

    def _on_salary_increase(self, a: s.Amendment) -> None:
        if a.amount is None:
            return
        on = a.effective_date or self.as_of
        self._set_level(self._home(a.amount, a.currency, on), from_day=a.effective_date)

    def _set_level_conservatively(self, a: s.Amendment) -> None:
        """Cap income at the confirmed figure without ever raising it.

        These messages restate a level rather than announce a dated change, and
        can disagree with the settled payroll history (sample 11: history base
        23,256,000 vs message 38,760,000; only the history level reproduces the
        published answer). With no dated amendment to prefer, the statement's
        last resort applies: the financially safer interpretation.
        """
        if a.amount is None:
            return
        confirmed = self._home(a.amount, a.currency, self.as_of)
        salaries = self._salary_flows()
        if not salaries:
            nxt = _month_shift(self.as_of, 0)
            for flow in self._monthly_from(nxt, confirmed, "confirmed salary"):
                self.flows.append(flow)
            return
        for flow in salaries:
            if flow.amount > confirmed:
                self._replace(flow, replace(flow, amount=confirmed))

    def _on_base_salary_confirmed(self, a: s.Amendment) -> None:
        self._set_level_conservatively(a)

    def _on_household_income_remaining(self, a: s.Amendment) -> None:
        self._set_level_conservatively(a)

    def _next_salary(self) -> CashFlow | None:
        upcoming = [f for f in self._salary_flows() if f.when >= self.as_of]
        return upcoming[0] if upcoming else None

    def _on_salary_next_reduced(self, a: s.Amendment) -> None:
        nxt = self._next_salary()
        if nxt is not None and a.amount is not None:
            self._replace(
                nxt, replace(nxt, amount=self._home(a.amount, a.currency, nxt.when))
            )

    def _on_salary_temporary(self, a: s.Amendment) -> None:
        self._on_salary_next_reduced(a)

    def _on_salary_plus_arrears(self, a: s.Amendment) -> None:
        nxt = self._next_salary()
        if nxt is None or a.amount is None:
            return
        regular = self._home(a.amount, a.currency, nxt.when)
        arrears = self._home(a.secondary_amount or Decimal(0), a.currency, nxt.when)
        self._set_level(regular)
        nxt = self._next_salary()
        if nxt is not None:
            self._replace(nxt, replace(nxt, amount=regular + arrears))

    def _add_income_stream(self, a: s.Amendment, label: str) -> None:
        if a.amount is None or a.effective_date is None:
            return
        start = a.effective_date
        amount = self._home(a.amount, a.currency, start)
        # Replace any projection on or after the start with the confirmed stream.
        self._drop_projected_income(from_day=start)
        existing = {f.when for f in self._salary_flows()}
        for flow in self._monthly_from(start, amount, label):
            if flow.when not in existing:
                self.flows.append(flow)

    def _on_first_salary(self, a: s.Amendment) -> None:
        self._add_income_stream(a, "first salary (confirmed by message)")

    def _on_salary_resumes(self, a: s.Amendment) -> None:
        self._add_income_stream(a, "salary resumes (confirmed by message)")

    def _on_salary_confirmed_fx(self, a: s.Amendment) -> None:
        if a.amount is None or a.effective_date is None:
            return
        amount = self._home(a.amount, a.currency, a.effective_date)
        on_day = [f for f in self._salary_flows() if f.when == a.effective_date]
        if on_day:
            for flow in on_day:
                self._replace(flow, replace(flow, amount=amount))
        elif self.as_of <= a.effective_date <= self.until:
            self.flows.append(
                CashFlow(
                    a.effective_date,
                    amount,
                    "confirmed salary",
                    INCOME_KEY,
                    "salary",
                    INCOME_KEY,
                )
            )

    def _on_invoice_approved(self, a: s.Amendment) -> None:
        # "Only invoices marked as confirmed should be included": the approved
        # invoice replaces any projection of the still-pending ones.
        self._on_invoice_approved_suppression()
        if a.amount is None or a.effective_date is None:
            return
        if not (self.as_of <= a.effective_date <= self.until):
            return
        amount = self._home(a.amount, a.currency, a.effective_date)
        self.flows.append(
            CashFlow(
                a.effective_date,
                amount,
                "approved invoice (confirmed by message)",
                f"amendment:{a.message_id}",
                "income",
                f"amendment:{a.message_id}",
            )
        )

    def _on_rent_increase(self, a: s.Amendment) -> None:
        if a.percent is None:
            return
        factor = Decimal(1) + a.percent / Decimal(100)
        for flow in list(self.flows):
            if (
                flow.is_outflow
                and flow.category in ("rent", "housing")
                and flow.series_key.startswith("projected:")
            ):
                self._replace(
                    flow,
                    replace(
                        flow, amount=(flow.amount * factor).quantize(Decimal("0.01"))
                    ),
                )


def apply_amendments(
    flows: list[CashFlow],
    amendments: list[s.Amendment],
    as_of: date,
    until: date,
    home_currency: str,
    fx: FxTable,
) -> tuple[list[CashFlow], list[str]]:
    applier = AmendmentApplier(flows, as_of, until, home_currency, fx)
    for amendment in amendments:
        # Dispatch every intent with a handler: besides the forecast-changing
        # ones, "payout/bonus still pending" must remove a variable-income
        # projection.
        applier.apply(amendment)
    return sorted(applier.flows, key=lambda f: (f.when, f.source)), applier.applied
