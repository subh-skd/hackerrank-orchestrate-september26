"""decision_explanation templates.

Mined from the 25 solved samples; every figure is filled from the simulator, so
an explanation can never cite a number the engine did not compute. The
"leaves at least X available" figure is the user's minimum_balance_to_keep in
every sample that uses it.
"""

from __future__ import annotations

from .fmt import date_text, money_text
from .forecast import SpendingChange
from .models import Profile, Request
from .plans import Plan


def _change_phrase(change: SpendingChange, currency: str) -> str:
    subject = change.description[:1].lower() + change.description[1:]
    if change.action == "stop":
        return f"stop the {subject}"
    assert change.new_amount is not None
    return f"reduce the {subject} to {money_text(currency, change.new_amount)}"


def _changes_prefix(changes: list[SpendingChange], currency: str) -> str:
    phrases = [_change_phrase(c, currency) for c in changes]
    if len(phrases) == 1:
        joined = phrases[0]
    else:
        joined = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return joined[:1].upper() + joined[1:]


def uses_template_b(request: Request, profile: Profile) -> bool:
    """Which not_recommended wording applies.

    Solved from the samples (7/7): the "Although X is available today" wording
    is used when partial payment was the user's only path - they accept neither
    full payment nor installments - and the request allows it.
    """
    methods = profile.payment_methods_user_will_consider
    return (
        "full_payment" not in methods
        and "installments" not in methods
        and request.allows_partial_payment
    )


def explain(
    request: Request,
    profile: Profile,
    plan: Plan | None,
    amount_safe_to_pay,
) -> str:
    cur = profile.home_currency
    minimum = money_text(cur, profile.minimum_balance_to_keep)
    requested = money_text(cur, request.requested_amount)

    if plan is None:
        if uses_template_b(request, profile) and amount_safe_to_pay > 0:
            return (
                f"Do not proceed with the {requested} request. Although "
                f"{money_text(cur, amount_safe_to_pay)} is available today, the full "
                "amount cannot be completed safely within 90 days."
            )
        return (
            f"Do not make this payment by {date_text(request.desired_completion_date)}. "
            f"None of the available options keeps the {minimum} minimum protected."
        )

    if plan.method == "wait":
        day = date_text(plan.payments[0][0])
        return (
            f"Pay {requested} in full on {day}. Paying earlier would take the "
            f"balance below the {minimum} minimum."
        )

    if plan.method == "partial_payment":
        (first_day, first), (second_day, second) = plan.payments
        return (
            f"Pay {money_text(cur, first)} today and the remaining "
            f"{money_text(cur, second)} on {date_text(second_day)}. This completes "
            f"the full request and keeps the {minimum} minimum protected."
        )

    if plan.method == "installments":
        amount = money_text(cur, plan.payments[0][1])
        count = len(plan.payments)
        body = (
            f"use {count} installments of {amount}, starting "
            f"{date_text(plan.start)}. This leaves at least {minimum} available."
        )
        if plan.changes:
            return f"{_changes_prefix(plan.changes, cur)}, then {body}"
        return body[:1].upper() + body[1:]

    # full_payment
    if plan.changes:
        return (
            f"{_changes_prefix(plan.changes, cur)}, then pay {requested} today. "
            f"This leaves at least {minimum} available."
        )
    return (
        f"Pay {requested} today. This leaves at least {minimum} available over "
        "the next 90 days."
    )
