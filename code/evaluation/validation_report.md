# Validation report

Rows checked: **250** · Checks run: **2909** · Violations: **0**

Every mechanically checkable rule in `problem_statement.md` is verified on
every row: bounds, allowed values, status/method consistency, user-accepted
methods, chronological plans that finish by the deadline and sum to the
requested amount, the exact two-payment partial shape, installment
schedules matching a supplied option within `max_installment_months`, and
spending changes that target flexible, permitted, unprotected events at or
above `minimum_allowed_amount`.

**Result: PASS — zero contract violations.**
