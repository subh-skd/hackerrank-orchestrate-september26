"""Candidate plan enumeration, safety verification, and ranking.

Every eligible plan is generated, re-simulated against the 90-day forecast, and
the safe ones are ranked by the order problem_statement.md prescribes:

  1. complete the full request by desired_completion_date   (hard constraint)
  2. require no spending changes
  3. minimize the total amount paid
  4. start payment earlier
  5. use fewer payments
  6. lowest payment_option_id

Eligibility ("Choosing Between Safe Plans"): an immediate method must appear in
payment_methods_user_will_consider; installments must also fit
max_installment_months (compared with number_of_payments - consistent with all
25 samples); partial payment needs allows_partial_payment and
0 < amount_safe_to_pay < requested_amount; wait needs the user to accept
full_payment and a safe future date on or before the deadline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from itertools import combinations

from .forecast import Forecaster, SpendingChange
from .models import PaymentOption, Profile, Request

MAX_CHANGES = 3


@dataclass
class Plan:
    method: str  # full_payment | partial_payment | installments | wait
    payments: list[tuple[date, Decimal]]
    changes: list[SpendingChange] = field(default_factory=list)
    option: PaymentOption | None = None
    safe: bool = False
    rejected_reason: str = ""

    @property
    def total(self) -> Decimal:
        return sum((a for _d, a in self.payments), Decimal(0))

    @property
    def start(self) -> date:
        return min(d for d, _a in self.payments)

    @property
    def finish(self) -> date:
        return max(d for d, _a in self.payments)

    @property
    def option_id(self) -> str:
        return self.option.payment_option_id if self.option else ""

    def rank_key(self) -> tuple:
        option_order = _option_number(self.option_id)
        return (
            len(self.changes) > 0,
            self.total,
            self.start,
            len(self.payments),
            option_order,
        )

    def label(self) -> str:
        tag = self.method
        if self.option:
            tag += f" ({self.option.payment_option_id})"
        if self.changes:
            tag += " + " + "|".join(c.render() for c in self.changes)
        return tag


def _option_number(option_id: str) -> int:
    digits = "".join(ch for ch in option_id if ch.isdigit())
    return int(digits) if digits else 10**9


@dataclass
class PlanSearch:
    """All candidates considered for one request, and the winner."""

    request: Request
    amount_safe_to_pay: Decimal
    earliest: date | None
    candidates: list[Plan]
    chosen: Plan | None


def _is_safe(forecaster: Forecaster, plan: Plan) -> bool:
    curve = forecaster.curve(payments=plan.payments, changes=plan.changes)
    return curve.is_safe()


def _change_sets(
    forecaster: Forecaster, available: list[SpendingChange]
) -> list[list[SpendingChange]]:
    """Every combination of up to three changes, least disruptive first.

    Stopping and reducing the same financial event are mutually exclusive, so a
    series appears at most once per combination.

    Ordering: the smallest total cut to projected spending across the window
    that still works. Solved sample 21 pins this down - the answer stops an
    11/month backup plan and trims a 47/month streaming plan to 23.50 (34.50 of
    relief) rather than reducing shopping by 76.78, which would also have
    sufficed. Samples 06 and 11 are consistent with the same rule.
    """
    relief = {id(c): forecaster.window_relief(c) for c in available}
    out: list[list[SpendingChange]] = []
    for size in range(1, MAX_CHANGES + 1):
        for combo in combinations(available, size):
            keys = [c.series_key for c in combo]
            if len(set(keys)) != len(keys):
                continue
            out.append(list(combo))
    out.sort(
        key=lambda cs: (
            sum((relief[id(c)] for c in cs), Decimal(0)),
            len(cs),
            [_option_number(c.event_id) for c in cs],
        )
    )
    return out


def _with_changes(
    forecaster: Forecaster, base: Plan, change_sets: list[list[SpendingChange]]
) -> Plan | None:
    """The first (smallest, most effective) change set that makes `base` safe."""
    for changes in change_sets:
        trial = Plan(
            method=base.method,
            payments=base.payments,
            changes=sorted(changes, key=lambda c: _option_number(c.event_id)),
            option=base.option,
        )
        if _is_safe(forecaster, trial):
            trial.safe = True
            return trial
    return None


def search(
    request: Request,
    profile: Profile,
    options: list[PaymentOption],
    forecaster: Forecaster,
) -> PlanSearch:
    requested = request.requested_amount
    today = request.request_date
    deadline = request.desired_completion_date

    safe_today = forecaster.amount_safe_to_pay(requested)
    earliest = forecaster.earliest_full_payment(requested)
    change_sets = _change_sets(forecaster, forecaster.available_changes())

    candidates: list[Plan] = []

    def consider(plan: Plan, allow_changes: bool) -> None:
        if plan.finish > deadline:
            plan.rejected_reason = "completes after desired_completion_date"
            candidates.append(plan)
            return
        if _is_safe(forecaster, plan):
            plan.safe = True
            candidates.append(plan)
            return
        plan.rejected_reason = "breaches the minimum balance"
        candidates.append(plan)
        if allow_changes and change_sets:
            changed = _with_changes(forecaster, plan, change_sets)
            if changed is not None:
                candidates.append(changed)

    # Full payment today.
    if profile.considers_full_payment:
        full_option = next(
            (o for o in options if o.payment_method == "full_payment"), None
        )
        pay_day = full_option.first_payment_date if full_option else today
        consider(
            Plan("full_payment", [(pay_day, requested)], option=full_option),
            allow_changes=True,
        )

    # Partial payment: amount_safe_to_pay today, the rest on the earliest date.
    if (
        request.allows_partial_payment
        and profile.considers_partial_payment
        and Decimal(0) < safe_today < requested
    ):
        if earliest is not None and earliest <= deadline:
            consider(
                Plan(
                    "partial_payment",
                    [(today, safe_today), (earliest, requested - safe_today)],
                ),
                allow_changes=False,
            )
        else:
            candidates.append(
                Plan(
                    "partial_payment",
                    [(today, safe_today)],
                    rejected_reason="remainder never safe by the deadline",
                )
            )

    # Supplied installment options.
    if profile.considers_installments:
        limit = profile.max_installment_months or 0
        for option in options:
            if option.payment_method != "installments":
                continue
            plan = Plan("installments", option.schedule(), option=option)
            if option.number_of_payments > limit:
                plan.rejected_reason = (
                    f"{option.number_of_payments} payments exceeds "
                    f"max_installment_months {limit}"
                )
                candidates.append(plan)
                continue
            consider(plan, allow_changes=True)

    # Wait for the earliest safe full-payment date.
    if profile.considers_full_payment and earliest is not None and earliest > today:
        consider(Plan("wait", [(earliest, requested)]), allow_changes=False)

    safe = [p for p in candidates if p.safe]
    chosen = min(safe, key=Plan.rank_key) if safe else None

    return PlanSearch(
        request=request,
        amount_safe_to_pay=safe_today,
        earliest=earliest,
        candidates=candidates,
        chosen=chosen,
    )


def status_for(plan: Plan | None) -> tuple[str, str]:
    """(affordability_status, recommended_payment_method) for the chosen plan."""
    if plan is None:
        return "not_affordable", "not_recommended"
    if plan.method == "wait":
        return "affordable_later", "wait"
    if plan.method == "full_payment" and not plan.changes:
        return "affordable_now", "full_payment"
    return "affordable_with_plan", plan.method
