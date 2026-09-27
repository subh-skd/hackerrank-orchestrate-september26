"""output.csv writer.

Columns and their order are fixed by problem_statement.md and must not change.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from . import fmt
from .loaders import OUTPUT_COLUMNS


@dataclass
class Decision:
    """One output row, still in typed form so the validator can check it."""

    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: list[tuple[date, Decimal]]
    earliest_date_for_full_payment: date | None
    spending_changes: list[str]
    decision_explanation: str

    def as_row(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "amount_safe_to_pay": fmt.safe_amount(self.amount_safe_to_pay),
            "affordability_status": self.affordability_status,
            "recommended_payment_method": self.recommended_payment_method,
            "payment_plan": fmt.payment_plan(self.payment_plan),
            "earliest_date_for_full_payment": (
                fmt.iso(self.earliest_date_for_full_payment)
                if self.earliest_date_for_full_payment
                else ""
            ),
            "spending_changes_needed": "|".join(self.spending_changes) or "none",
            "decision_explanation": self.decision_explanation,
        }


def decision_from_row(row: dict[str, str]) -> Decision:
    """Rebuild a typed Decision from its output-row form (used to resume runs)."""
    plan = []
    if row["payment_plan"] and row["payment_plan"] != "none":
        for part in row["payment_plan"].split("|"):
            day, amount = part.split(":")
            plan.append((date.fromisoformat(day), Decimal(amount)))
    changes = (
        []
        if row["spending_changes_needed"] in ("", "none")
        else row["spending_changes_needed"].split("|")
    )
    earliest = row["earliest_date_for_full_payment"]
    return Decision(
        request_id=row["request_id"],
        amount_safe_to_pay=Decimal(row["amount_safe_to_pay"]),
        affordability_status=row["affordability_status"],
        recommended_payment_method=row["recommended_payment_method"],
        payment_plan=plan,
        earliest_date_for_full_payment=(
            date.fromisoformat(earliest) if earliest else None
        ),
        spending_changes=changes,
        decision_explanation=row["decision_explanation"],
    )


def write_output(path: Path, decisions: list[Decision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.as_row())
