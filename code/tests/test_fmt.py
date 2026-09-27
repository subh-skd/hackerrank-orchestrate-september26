"""Round-trip the output formatters against every string in sample_requests.csv.

This is the P0 gate: if we cannot reproduce the published samples' rendering
byte-for-byte from source values, every downstream column is wrong even when
the arithmetic is right.

Run:  python code/tests/test_fmt.py
"""

from __future__ import annotations

import datetime
import re
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bow import fmt  # noqa: E402
from bow.loaders import load_dataset  # noqa: E402

failures: list[str] = []
checks = 0


def check(label: str, got: str, want: str) -> None:
    global checks
    checks += 1
    if got != want:
        failures.append(f"{label}: got {got!r} want {want!r}")


def main() -> int:
    data = load_dataset()

    # 1. amount_safe_to_pay renders with trailing zeros stripped.
    for sample in data.samples:
        raw = sample.expected["amount_safe_to_pay"]
        check(
            f"{sample.request_id} amount_safe_to_pay",
            fmt.safe_amount(Decimal(raw)),
            raw,
        )

    # 2. payment_plan renders from source values, not from the sample string.
    for sample in data.samples:
        raw_plan = sample.expected["payment_plan"]
        if raw_plan == "none":
            check(f"{sample.request_id} payment_plan", fmt.payment_plan([]), raw_plan)
            continue

        method = sample.expected["recommended_payment_method"]
        entries = raw_plan.split("|")

        if method == "installments":
            # Amounts must come verbatim from the matching payment option.
            option = next(
                o
                for o in data.options(sample.request_id)
                if o.payment_method == "installments"
                and len(o.schedule()) == len(entries)
                and fmt.iso(o.first_payment_date) == entries[0].split(":")[0]
            )
            rendered = fmt.payment_plan(option.schedule())
        elif method == "partial_payment":
            safe = Decimal(sample.expected["amount_safe_to_pay"])
            second_date = datetime.date.fromisoformat(
                sample.expected["earliest_date_for_full_payment"]
            )
            rendered = fmt.payment_plan(
                [
                    (sample.request_date, safe),
                    (second_date, sample.requested_amount - safe),
                ]
            )
        else:  # full_payment or wait: one payment of the full requested amount
            pay_date = datetime.date.fromisoformat(entries[0].split(":")[0])
            rendered = fmt.payment_plan([(pay_date, sample.requested_amount)])

        check(f"{sample.request_id} payment_plan", rendered, raw_plan)

    # 3. Money strings inside decision_explanation round-trip.
    money = re.compile(r"\b(ZAR|IDR|EUR|INR|USD) ([\d,]+(?:\.\d+)?)\b")
    for sample in data.samples:
        text = sample.expected["decision_explanation"]
        for currency, amount in money.findall(text):
            value = Decimal(amount.replace(",", ""))
            check(
                f"{sample.request_id} money_text {amount}",
                fmt.money_text(currency, value),
                f"{currency} {amount}",
            )

    # 4. Long-form dates inside decision_explanation round-trip.
    long_date = re.compile(
        r"\b(\d{1,2}) (January|February|March|April|May|June|July|August|"
        r"September|October|November|December) (\d{4})\b"
    )
    for sample in data.samples:
        text = sample.expected["decision_explanation"]
        for day, month_name, year in long_date.findall(text):
            month = fmt.MONTHS.index(month_name) + 1
            value = datetime.date(int(year), month, int(day))
            check(
                f"{sample.request_id} date_text",
                fmt.date_text(value),
                f"{day} {month_name} {year}",
            )

    # 5. reduce_to amounts round-trip.
    for sample in data.samples:
        changes = sample.expected["spending_changes_needed"]
        if changes in ("none", ""):
            continue
        for action in changes.split("|"):
            parts = action.split(":")
            if parts[0] == "reduce_to":
                check(
                    f"{sample.request_id} reduce_to",
                    fmt.plan_amount(Decimal(parts[2])),
                    parts[2],
                )

    print(f"formatter round-trip: {checks - len(failures)}/{checks} checks passed")
    for failure in failures:
        print("  FAIL", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
