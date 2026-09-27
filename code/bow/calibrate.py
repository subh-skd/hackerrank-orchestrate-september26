"""Identify the forecast rule against the solved samples.

sample_requests.csv gives 25 fully solved requests. Those whose
amount_safe_to_pay lies strictly between 0 and requested_amount are exact
equations on the forecast; earliest_date_for_full_payment adds a date
constraint on every sample. We search the ForecastRule space by coordinate
descent (one parameter at a time, repeated until nothing improves) and keep
the rule that reproduces the most samples.

This fits *rules*, never answers: the winning ForecastRule is then applied
unchanged to all 250 evaluation requests, and no sample value is ever read on
the prediction path.

Run:  python code/main.py --calibrate
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from decimal import Decimal
from pathlib import Path

from .engine import Engine
from .loaders import load_dataset
from .models import Request
from .recurrence import ForecastRule

SEARCH_SPACE: dict[str, list[object]] = {
    "variable_estimator": [
        "mean",
        "median",
        "max",
        "last",
        "ml_q50",
        "ml_q75",
        "ml_q90",
    ],
    "variable_lookback": [1, 2, 3, 4, 5, 6],
    "variable_placement": ["cadence", "replay", "prorata", "lump_start", "lump_end"],
    "variable_cadence": ["mode", "min"],
    "variable_occurrences": [1, 2, 3, 4, 5, 6, 8, 12],
    "fixed_estimator": ["last", "mean", "median", "max"],
    "fixed_lookback": [1, 2, 3, 4, 6],
    "income_stability": ["last2eq", "constant", "any"],
    "income_estimator": ["last", "mean", "min"],
    "min_occurrences": [1, 2, 3],
    "income_min_occurrences": [2, 3],
    "variable_income": ["none", "min", "median", "mean"],
    "variable_income_occurrences": [3, 4, 6, 8, 12],
    "live_within_months": [1, 2],
}


@dataclass
class SampleResult:
    request: Request
    got_amount: Decimal
    want_amount: Decimal
    got_earliest: date | None
    want_earliest: str

    @property
    def amount_exact(self) -> bool:
        return abs(self.got_amount - self.want_amount) <= Decimal("0.01")

    @property
    def rel_error(self) -> float:
        denom = max(abs(self.want_amount), Decimal(1))
        return min(float(abs(self.got_amount - self.want_amount) / denom), 1.0)

    @property
    def earliest_exact(self) -> bool:
        got = self.got_earliest.isoformat() if self.got_earliest else ""
        return got == self.want_earliest


@dataclass
class Score:
    amount_exact: int
    earliest_exact: int
    mean_rel_error: float
    results: list[SampleResult]
    field_hits: int = 0  # exact matches across every scored output column

    @property
    def key(self) -> tuple[float, ...]:
        # Optimize what is graded: exact matches across all output columns,
        # then amount accuracy as the tiebreak.
        return (self.field_hits, self.amount_exact, -self.mean_rel_error)


def evaluate(engine: Engine, rule: ForecastRule, samples: list[Request]) -> Score:
    from .pipeline import SCORED_FIELDS, decide

    results = []
    field_hits = 0
    for sample in samples:
        decision, search = decide(engine, sample, rule)
        row = decision.as_row()
        field_hits += sum(
            row[name] == sample.expected[name]
            for name in SCORED_FIELDS
            if name != "decision_explanation"
        )
        results.append(
            SampleResult(
                request=sample,
                got_amount=search.amount_safe_to_pay,
                want_amount=Decimal(sample.expected["amount_safe_to_pay"]),
                got_earliest=search.earliest,
                want_earliest=sample.expected["earliest_date_for_full_payment"],
            )
        )
    return Score(
        amount_exact=sum(r.amount_exact for r in results),
        earliest_exact=sum(r.earliest_exact for r in results),
        mean_rel_error=sum(r.rel_error for r in results) / max(len(results), 1),
        results=results,
        field_hits=field_hits,
    )


def coordinate_search(
    engine: Engine,
    samples: list[Request],
    start: ForecastRule | None = None,
    rounds: int = 3,
    log=print,
) -> tuple[ForecastRule, Score]:
    best_rule = start or ForecastRule()
    best = evaluate(engine, best_rule, samples)
    log(f"start: {describe(best)}")

    for round_no in range(1, rounds + 1):
        improved = False
        for name, values in SEARCH_SPACE.items():
            for value in values:
                if getattr(best_rule, name) == value:
                    continue
                candidate = best_rule.with_(**{name: value})
                score = evaluate(engine, candidate, samples)
                if score.key > best.key:
                    best_rule, best, improved = candidate, score, True
                    log(f"  round {round_no}: {name}={value!r} -> {describe(best)}")
        if not improved:
            break
    return best_rule, best


def describe(score: Score) -> str:
    n = len(score.results)
    return (
        f"field hits {score.field_hits}/{n * 6}, amount exact {score.amount_exact}/{n}, earliest exact "
        f"{score.earliest_exact}/{n}, mean rel err {score.mean_rel_error:.4f}"
    )


def write_report(path: Path, rule: ForecastRule, score: Score) -> None:
    lines = [
        "# Calibration report",
        "",
        "Forecast rule identified by coordinate search against the 25 solved",
        "samples in `dataset/sample_requests.csv`. The rule below is applied",
        "unchanged to every evaluation request.",
        "",
        f"**Result:** {describe(score)}",
        "",
        "## Selected rule",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for f in fields(rule):
        lines.append(f"| `{f.name}` | `{getattr(rule, f.name)!r}` |")
    lines += [
        "",
        "## Per-sample results",
        "",
        "| Sample | Expected safe | Got | Match | Expected earliest | Got | Match |",
        "|---|---:|---:|:---:|---|---|:---:|",
    ]
    for r in score.results:
        lines.append(
            f"| {r.request.request_id} | {r.want_amount} | {r.got_amount} | "
            f"{'✓' if r.amount_exact else '✗'} | {r.want_earliest or '—'} | "
            f"{r.got_earliest or '—'} | {'✓' if r.earliest_exact else '✗'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(report_path: Path, engine: Engine | None = None) -> ForecastRule:
    if engine is None:
        engine = Engine(load_dataset())
    data = engine.data
    rule, score = coordinate_search(engine, data.samples)
    write_report(report_path, rule, score)
    print(f"final: {describe(score)}")
    print(f"rule:  {rule}")
    return rule
