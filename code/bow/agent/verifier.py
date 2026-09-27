"""Verification gate for the agent's decisions.

The model proposes a plan; this module decides whether that proposal is
acceptable under the problem statement:

* the method is accepted by the user and eligible for the request
  (installments within max_installment_months, partial-payment shape, wait
  only if full payment is accepted and becomes safe later)
* the plan completes by desired_completion_date
* the projected balance never falls below minimum_balance_to_keep
* no other safe plan outranks it under the statement's six ranking rules
* spending changes are permitted and the least disruptive set that works

An accepted proposal becomes the output row. A rejected one returns the
specific reason, which the graph feeds back to the model for a retry.

Each request's Workspace is found through the run config
(configurable.request_id).
"""

from __future__ import annotations

import json
import threading
from decimal import Decimal

from langchain_core.runnables import RunnableConfig

from ..forecast import SpendingChange
from ..plans import Plan, status_for
from ..writers import Decision
from .workspace import Workspace

_REGISTRY: dict[str, Workspace] = {}
_LOCK = threading.Lock()

METHODS = ("full_payment", "partial_payment", "installments", "wait", "not_recommended")
RANK_CRITERIA = (
    "requires no spending changes",
    "pays less in total",
    "starts payment earlier",
    "uses fewer payments",
    "has the lower payment_option_id",
)


def register(ws: Workspace) -> None:
    with _LOCK:
        _REGISTRY[ws.request.request_id] = ws


def workspace(config: RunnableConfig) -> Workspace:
    return _REGISTRY[config["configurable"]["request_id"]]


def _resolve_changes(
    ws: Workspace, specs: list[str]
) -> tuple[list[SpendingChange], list[str]]:
    available = {c.render(): c for c in ws.forecaster().available_changes()}
    chosen, errors = [], []
    for spec in specs:
        spec = spec.strip()
        if not spec or spec == "none":
            continue
        change = available.get(spec)
        if change is None:
            errors.append(f"'{spec}' is not a permitted spending change")
        else:
            chosen.append(change)
    keys = [c.series_key for c in chosen]
    if len(set(keys)) != len(keys):
        errors.append(
            "the same expense cannot be both stopped and reduced, or changed twice"
        )
    if len(chosen) > 3:
        errors.append("at most three spending changes are allowed")
    return chosen, errors


def _build_plan(
    ws: Workspace, method: str, option_id: str | None, changes: list[SpendingChange]
):
    """Return (plan or None, eligibility errors)."""
    req, profile = ws.request, ws.profile
    safe, earliest = ws.amount_safe_to_pay, ws.earliest
    errors: list[str] = []

    if method not in METHODS:
        return None, [f"unknown method {method!r}"]
    if method == "not_recommended":
        return None, []

    if method in ("full_payment", "partial_payment", "installments"):
        if method not in profile.payment_methods_user_will_consider:
            errors.append(f"the user does not accept {method}")
    if method == "wait" and not profile.considers_full_payment:
        errors.append("wait requires the user to accept full_payment")

    if method == "full_payment":
        option = next(
            (o for o in ws.options if o.payment_method == "full_payment"), None
        )
        day = option.first_payment_date if option else req.request_date
        plan = Plan(
            "full_payment",
            [(day, req.requested_amount)],
            changes=changes,
            option=option,
        )
    elif method == "partial_payment":
        if not req.allows_partial_payment:
            errors.append("this request does not allow partial payment")
        if not (Decimal(0) < safe < req.requested_amount):
            errors.append(
                f"partial payment needs 0 < amount_safe_to_pay ({safe}) < requested_amount"
            )
        if earliest is None:
            errors.append("the remainder never becomes safe within 90 days")
            return None, errors
        plan = Plan(
            "partial_payment",
            [(req.request_date, safe), (earliest, req.requested_amount - safe)],
            changes=changes,
        )
    elif method == "installments":
        option = next((o for o in ws.options if o.payment_option_id == option_id), None)
        if option is None or option.payment_method != "installments":
            return None, errors + [
                f"{option_id!r} is not an installment option for this request"
            ]
        limit = profile.max_installment_months or 0
        if option.number_of_payments > limit:
            errors.append(
                f"{option.number_of_payments} payments exceeds max_installment_months {limit}"
            )
        plan = Plan("installments", option.schedule(), changes=changes, option=option)
    else:  # wait
        if earliest is None:
            return None, errors + [
                "full payment never becomes safe within the 90-day forecast"
            ]
        if earliest <= req.request_date:
            errors.append(
                "full payment is already safe today; wait is not a later payment"
            )
        plan = Plan("wait", [(earliest, req.requested_amount)], changes=changes)

    if plan.finish > req.desired_completion_date:
        errors.append(
            f"finishes {plan.finish}, after desired_completion_date {req.desired_completion_date}"
        )
    return plan, errors


def _outranks(better: Plan, worse: Plan) -> str | None:
    for criterion, left, right in zip(
        RANK_CRITERIA, better.rank_key(), worse.rank_key()
    ):
        if left != right:
            return criterion if left < right else None
    return None


def verify_submission(
    ws: Workspace,
    recommended_payment_method: str,
    decision_explanation: str,
    payment_option_id: str = "",
    spending_changes: list[str] | None = None,
) -> str:
    """Return "ACCEPTED: <row>" (and set ws.decision) or "REJECTED:\\n- reasons"."""
    req, profile = ws.request, ws.profile
    reference = ws.reference_search()
    problems: list[str] = []

    explanation = " ".join((decision_explanation or "").split())
    if not explanation:
        problems.append("decision_explanation is empty")
    elif len(explanation) > 400:
        problems.append("decision_explanation must be concise (under 400 characters)")
    elif profile.home_currency not in explanation:
        problems.append(
            f"decision_explanation must cite amounts in {profile.home_currency}"
        )

    changes, change_errors = _resolve_changes(ws, spending_changes or [])
    problems += change_errors

    plan: Plan | None = None
    if recommended_payment_method == "not_recommended":
        if changes:
            problems.append("not_recommended cannot carry spending changes")
        if reference.chosen is not None:
            problems.append(
                f"a safe eligible plan exists ({reference.chosen.label()}); "
                "not_recommended is only for when no plan is SAFE"
            )
    else:
        plan, plan_errors = _build_plan(
            ws, recommended_payment_method, payment_option_id or None, changes
        )
        problems += plan_errors
        if plan is not None and not plan_errors:
            curve = ws.forecaster().curve(payments=plan.payments, changes=plan.changes)
            if not curve.is_safe():
                problems.append(
                    "this plan takes the balance below minimum_balance_to_keep"
                )
            best = reference.chosen
            if best is not None and not problems:
                reason = _outranks(best, plan)
                if reason:
                    problems.append(
                        f"a higher-ranked safe plan exists: {best.label()} - it {reason}"
                    )
                elif plan.changes and best.changes:
                    mine = sum(
                        (ws.forecaster().window_relief(c) for c in plan.changes),
                        Decimal(0),
                    )
                    least = sum(
                        (ws.forecaster().window_relief(c) for c in best.changes),
                        Decimal(0),
                    )
                    if mine > least:
                        problems.append(
                            "a less disruptive set of spending changes also makes this safe: "
                            + "|".join(c.render() for c in best.changes)
                        )

    if problems:
        ws.rejected_submissions.append("; ".join(problems))
        return "REJECTED:\n- " + "\n- ".join(problems)

    status, method = status_for(plan)
    ws.decision = Decision(
        request_id=req.request_id,
        amount_safe_to_pay=ws.amount_safe_to_pay,
        affordability_status=status,
        recommended_payment_method=method,
        payment_plan=list(plan.payments) if plan else [],
        earliest_date_for_full_payment=ws.earliest,
        spending_changes=[c.render() for c in plan.changes] if plan else [],
        decision_explanation=explanation,
    )
    ws.decided_by = "agent"
    return "ACCEPTED: " + json.dumps(ws.decision.as_row())
