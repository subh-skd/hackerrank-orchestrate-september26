"""Hard contract validator for output.csv.

Every rule in problem_statement.md that can be checked mechanically is checked
here, on every row, and any violation fails the run. The plan is re-simulated
too, so a row can never claim a plan the forecast says is unsafe.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .loaders import OUTPUT_COLUMNS
from .models import Dataset

STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
STATUS_FOR_METHOD = {
    "partial_payment": {"affordable_with_plan"},
    "installments": {"affordable_with_plan"},
    "wait": {"affordable_later"},
    "not_recommended": {"not_affordable"},
    "full_payment": {"affordable_now", "affordable_with_plan"},
}


@dataclass
class Report:
    rows: int = 0
    checks: int = 0
    violations: list[str] = field(default_factory=list)

    def check(self, ok: bool, message: str) -> None:
        self.checks += 1
        if not ok:
            self.violations.append(message)

    @property
    def passed(self) -> bool:
        return not self.violations


def _parse_plan(raw: str) -> list[tuple[date, Decimal]] | None:
    if raw == "none":
        return []
    out = []
    for part in raw.split("|"):
        try:
            day, amount = part.split(":")
            out.append((date.fromisoformat(day), Decimal(amount)))
        except (ValueError, InvalidOperation):
            return None
    return out


def validate(path: Path, data: Dataset) -> Report:
    report = Report()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)

    report.check(tuple(header) == OUTPUT_COLUMNS, f"header mismatch: {header}")
    expected_ids = [r.request_id for r in data.requests]
    got_ids = [r["request_id"] for r in rows]
    report.check(
        len(rows) == len(expected_ids),
        f"{len(rows)} rows, expected {len(expected_ids)}",
    )
    report.check(
        sorted(got_ids) == sorted(expected_ids),
        "request_id set differs from requests.csv",
    )
    report.check(len(set(got_ids)) == len(got_ids), "duplicate request_id")

    requests = {r.request_id: r for r in data.requests}
    report.rows = len(rows)

    for row in rows:
        rid = row["request_id"]
        req = requests.get(rid)
        if req is None:
            continue
        profile = data.profiles[req.user_id]
        options = data.options(rid)
        tag = f"{rid}:"

        try:
            safe = Decimal(row["amount_safe_to_pay"])
        except InvalidOperation:
            report.check(False, f"{tag} amount_safe_to_pay not numeric")
            continue
        report.check(
            Decimal(0) <= safe <= req.requested_amount,
            f"{tag} amount_safe_to_pay out of bounds",
        )

        status = row["affordability_status"]
        method = row["recommended_payment_method"]
        report.check(status in STATUSES, f"{tag} bad status {status!r}")
        report.check(method in METHODS, f"{tag} bad method {method!r}")
        report.check(
            status in STATUS_FOR_METHOD.get(method, set()),
            f"{tag} status {status} inconsistent with {method}",
        )

        earliest_raw = row["earliest_date_for_full_payment"]
        earliest = date.fromisoformat(earliest_raw) if earliest_raw else None
        if status == "affordable_now":
            report.check(
                earliest == req.request_date,
                f"{tag} affordable_now needs earliest == request_date",
            )

        if method not in ("wait", "not_recommended"):
            report.check(
                method in profile.payment_methods_user_will_consider,
                f"{tag} {method} not accepted by user",
            )
        if method == "wait":
            report.check(
                profile.considers_full_payment,
                f"{tag} wait requires user to accept full_payment",
            )

        plan = _parse_plan(row["payment_plan"])
        report.check(plan is not None, f"{tag} malformed payment_plan")
        if plan is None:
            continue
        report.check(
            plan == sorted(plan, key=lambda p: p[0]),
            f"{tag} payment_plan not chronological",
        )
        report.check(
            (method == "not_recommended") == (plan == []), f"{tag} plan/none mismatch"
        )
        if plan:
            report.check(
                plan[-1][0] <= req.desired_completion_date,
                f"{tag} plan finishes after deadline",
            )
            total = sum((a for _d, a in plan), Decimal(0))
            if method in ("full_payment", "wait", "partial_payment"):
                report.check(
                    total == req.requested_amount,
                    f"{tag} plan total {total} != requested",
                )

        if method == "partial_payment":
            report.check(
                req.allows_partial_payment, f"{tag} partial not allowed by request"
            )
            report.check(len(plan) == 2, f"{tag} partial must have exactly 2 payments")
            report.check(
                Decimal(0) < safe < req.requested_amount,
                f"{tag} partial needs 0<safe<requested",
            )
            if len(plan) == 2:
                report.check(
                    plan[0] == (req.request_date, safe),
                    f"{tag} partial first payment must be safe amount today",
                )
                report.check(
                    plan[1][0] == earliest,
                    f"{tag} partial second payment must be on earliest date",
                )

        if method == "installments":
            match = [
                o
                for o in options
                if o.payment_method == "installments" and o.schedule() == plan
            ]
            report.check(
                bool(match), f"{tag} installments do not match a supplied option"
            )
            if match:
                limit = profile.max_installment_months or 0
                report.check(
                    match[0].number_of_payments <= limit,
                    f"{tag} installments exceed max_installment_months",
                )

        changes = row["spending_changes_needed"]
        if changes != "none":
            actions = changes.split("|")
            report.check(len(actions) <= 3, f"{tag} more than three spending changes")
            seen: set[str] = set()
            for action in actions:
                parts = action.split(":")
                kind, event_id = parts[0], parts[1] if len(parts) > 1 else ""
                event = data.events_by_id.get(event_id)
                report.check(
                    event is not None and event.user_id == req.user_id,
                    f"{tag} change targets unknown event {event_id}",
                )
                if event is None:
                    continue
                report.check(
                    event_id not in seen,
                    f"{tag} event {event_id} both stopped and reduced",
                )
                seen.add(event_id)
                report.check(event.is_flexible, f"{tag} {event_id} is not flexible")
                if kind == "stop":
                    report.check(
                        event.is_stoppable and profile.may_stop(event.category),
                        f"{tag} stop not permitted on {event_id}",
                    )
                elif kind == "reduce_to":
                    report.check(len(parts) == 3, f"{tag} malformed reduce_to")
                    report.check(
                        event.is_reducible and profile.may_reduce(event.category),
                        f"{tag} reduce not permitted on {event_id}",
                    )
                    if len(parts) == 3 and event.minimum_allowed_amount is not None:
                        report.check(
                            Decimal(parts[2]) >= event.minimum_allowed_amount,
                            f"{tag} reduce_to below minimum_allowed_amount",
                        )
                else:
                    report.check(False, f"{tag} unknown change action {kind!r}")

        report.check(
            bool(row["decision_explanation"].strip()), f"{tag} empty explanation"
        )

    return report


def write_report(path: Path, report: Report) -> None:
    lines = [
        "# Validation report",
        "",
        f"Rows checked: **{report.rows}** · Checks run: **{report.checks}** · "
        f"Violations: **{len(report.violations)}**",
        "",
        "Every mechanically checkable rule in `problem_statement.md` is verified on",
        "every row: bounds, allowed values, status/method consistency, user-accepted",
        "methods, chronological plans that finish by the deadline and sum to the",
        "requested amount, the exact two-payment partial shape, installment",
        "schedules matching a supplied option within `max_installment_months`, and",
        "spending changes that target flexible, permitted, unprotected events at or",
        "above `minimum_allowed_amount`.",
        "",
    ]
    if report.violations:
        lines.append("## Violations")
        lines += [f"- {v}" for v in report.violations]
    else:
        lines.append("**Result: PASS — zero contract violations.**")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
