"""Request -> Decision, plus a field-by-field scorer for the solved samples."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from . import explain as explain_mod
from .engine import Engine
from .plans import PlanSearch, search, status_for
from .recurrence import ForecastRule
from .writers import Decision

RULE_FILE = Path(__file__).resolve().parents[1] / "config" / "forecast_rule.json"


def load_rule(path: Path = RULE_FILE) -> ForecastRule:
    if not path.exists():
        return ForecastRule()
    raw = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(ForecastRule)}
    return ForecastRule(**{k: v for k, v in raw.items() if k in known})


def save_rule(rule: ForecastRule, path: Path = RULE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(rule), indent=2) + "\n", encoding="utf-8")


def decide(engine: Engine, request, rule: ForecastRule) -> tuple[Decision, PlanSearch]:
    profile = engine.data.profiles[request.user_id]
    forecaster = engine.forecaster(request, rule)
    result = search(
        request, profile, engine.data.options(request.request_id), forecaster
    )

    plan = result.chosen
    status, method = status_for(plan)
    decision = Decision(
        request_id=request.request_id,
        amount_safe_to_pay=result.amount_safe_to_pay,
        affordability_status=status,
        recommended_payment_method=method,
        payment_plan=list(plan.payments) if plan else [],
        earliest_date_for_full_payment=result.earliest,
        spending_changes=[c.render() for c in plan.changes] if plan else [],
        decision_explanation=explain_mod.explain(
            request, profile, plan, result.amount_safe_to_pay
        ),
    )
    return decision, result


SCORED_FIELDS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)


def score_samples(engine: Engine, rule: ForecastRule, verbose: bool = True) -> dict:
    totals = {name: 0 for name in SCORED_FIELDS}
    rows = []
    for sample in engine.data.samples:
        decision, _ = decide(engine, sample, rule)
        got = decision.as_row()
        want = sample.expected
        hits = {name: got[name] == want[name] for name in SCORED_FIELDS}
        for name, ok in hits.items():
            totals[name] += ok
        rows.append((sample.request_id, got, want, hits))

    if verbose:
        n = len(engine.data.samples)
        print("field accuracy on solved samples:")
        for name in SCORED_FIELDS:
            print(f"  {name:32} {totals[name]:2}/{n}")
        for rid, got, want, hits in rows:
            misses = [
                k for k, ok in hits.items() if not ok and k != "decision_explanation"
            ]
            if misses:
                print(f"\n  {rid}:")
                for k in misses:
                    print(f"     {k:30} got {got[k]!r}\n     {'':30} want {want[k]!r}")
    return {"totals": totals, "rows": rows}
