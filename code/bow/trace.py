"""Per-request decision traces: evaluation/traces/<request_id>.md.

Each trace lets a reader follow one output row back to its source rows and the
rule that produced it: evidence applied, ledger rows reserved or excluded (with
reasons), recurring series projected or skipped (with reasons), the 90-day
trough, every candidate plan with its verdict, and why the winner won.
"""

from __future__ import annotations

from pathlib import Path

from . import fmt
from .engine import Engine
from .ledger import CashRole
from .plans import PlanSearch
from .recurrence import ForecastRule
from .writers import Decision

RANKING = (
    "Safe plans are ranked by: (1) complete by desired_completion_date (hard constraint), "
    "(2) no spending changes, (3) lowest total paid, (4) earliest start, "
    "(5) fewest payments, (6) lowest payment_option_id."
)


def _md(value: object) -> str:
    return str(value).replace("|", "\\|")


def render(
    engine: Engine,
    rule: ForecastRule,
    request,
    decision: Decision,
    search: PlanSearch,
    forecaster=None,
    amendments=None,
    image_amounts=None,
    agent_run=None,
) -> str:
    data = engine.data
    profile = data.profiles[request.user_id]
    cur = profile.home_currency
    forecaster = forecaster or engine.forecaster(request, rule)
    amendments = (
        amendments
        if amendments is not None
        else engine.amendments.get(request.user_id, [])
    )
    image_amounts = image_amounts if image_amounts is not None else engine.image_amounts
    curve = forecaster.curve()
    ledger = forecaster.ledger
    row = decision.as_row()

    lines = [
        f"# {request.request_id} — decision trace",
        "",
        f"> {request.request_text}",
        "",
        "## Output row",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    lines += [
        f"| `{k}` | {_md(v) or '—'} |" for k, v in row.items() if k != "request_id"
    ]

    lines += [
        "",
        "## Request and profile",
        "",
        f"- **User** `{request.user_id}` · home currency **{cur}** · request **{request.request_date}** "
        f"· deadline **{request.desired_completion_date}** · type `{request.request_type}`",
        f"- **Requested** {fmt.money_text(cur, request.requested_amount)} · partial payment allowed: "
        f"**{request.allows_partial_payment}**",
        f"- **Balance** {fmt.money_text(cur, profile.current_available_balance)} · **minimum to keep** "
        f"{fmt.money_text(cur, profile.minimum_balance_to_keep)}",
        f"- **Accepts** {', '.join(sorted(profile.payment_methods_user_will_consider)) or '—'} · "
        f"max installments: {profile.max_installment_months if profile.max_installment_months is not None else 'n/a'}",
        f"- **Protected** {', '.join(sorted(profile.expense_categories_to_protect)) or '—'} · "
        f"**may reduce** {', '.join(sorted(profile.expense_categories_user_is_willing_to_reduce)) or '—'} · "
        f"**may stop** {', '.join(sorted(profile.expense_categories_user_is_willing_to_stop)) or '—'}",
        "",
        "## Evidence applied",
        "",
    ]
    if amendments:
        for a in amendments:
            effect = (
                "applied to forecast"
                if a.affects_forecast
                else "confirms nothing new is counted"
            )
            lines.append(
                f"- `{a.message_id}` → **{a.intent}** ({a.describe()}; via {a.source}) — {effect}"
            )
    else:
        lines.append("- No messages for this user.")
    image_rows = [e for e in ledger.entries if e.event.amount is None]
    for entry in image_rows:
        value = image_amounts.get(entry.event_id)
        lines.append(
            f"- `{entry.event_id}` blank amount → "
            + (
                f"**{fmt.money_text(entry.event.currency, value)}** read from its image"
                if value is not None
                else "**unresolved** (no image value; excluded rather than read as zero)"
            )
        )
    if forecaster.applied_amendments:
        lines.append("")
        lines.append(
            "Forecast adjustments made: " + "; ".join(forecaster.applied_amendments)
        )

    lines += ["", "## Ledger rows that are not plain history", ""]
    special = [e for e in ledger.entries if e.role is not CashRole.HISTORY]
    if special:
        lines += ["| Event | Date | Amount | Role | Reason |", "|---|---|---:|---|---|"]
        for e in special:
            amount = (
                fmt.money_text(cur, e.amount_home) if e.amount_home is not None else "—"
            )
            lines.append(
                f"| `{e.event_id}` {_md(e.event.description)} | {e.cash_date} | {amount} | {e.role.value} | {_md(e.reason)} |"
            )
    else:
        lines.append("- None: every row is settled history.")
    lines.append(
        f"\n{len(ledger.history)} settled history rows feed recurrence detection."
    )

    lines += [
        "",
        "## Recurring series",
        "",
        "| Series | Direction | Projected amount | Status |",
        "|---|---|---:|---|",
    ]
    for series in forecaster.series:
        status = (
            "projected"
            if series.projected_amount is not None
            else (series.skipped_reason or "not projected")
        )
        amount = (
            fmt.money_text(cur, series.projected_amount)
            if series.projected_amount is not None
            else "—"
        )
        lines.append(
            f"| {_md(series.description)} (`{series.category}`) | {series.direction} | {amount} | {_md(status)} |"
        )
    for note in forecaster.notes:
        lines.append(f"\n_Note: {note}_")

    lines += [
        "",
        "## 90-day forecast",
        "",
        f"- Window {forecaster.start} → {forecaster.end}, {len(forecaster.flows)} projected or booked cash flows.",
        f"- Lowest balance **{fmt.money_text(cur, curve.trough)}** on **{curve.trough_date}**; "
        f"headroom above the minimum {fmt.money_text(cur, curve.headroom())}.",
        f"- **amount_safe_to_pay** = clamp(trough − minimum, 0, requested) = "
        f"{fmt.money_text(cur, search.amount_safe_to_pay)}.",
        f"- **earliest_date_for_full_payment** = first day a single full payment keeps every later day "
        f"above the minimum, without spending changes: {search.earliest or 'none within 90 days'}.",
        "",
        "## Candidate plans",
        "",
        RANKING,
        "",
        "| Plan | Payments | Total | Start → finish | Verdict |",
        "|---|---:|---:|---|---|",
    ]
    for plan in search.candidates:
        verdict = "✅ safe" if plan.safe else f"❌ {plan.rejected_reason}"
        if plan is search.chosen:
            verdict = "🏆 **chosen**"
        span = f"{plan.start} → {plan.finish}" if plan.payments else "—"
        lines.append(
            f"| {_md(plan.label())} | {len(plan.payments)} | {fmt.money_text(cur, plan.total)} | {span} | {verdict} |"
        )
    if not search.candidates:
        lines.append(
            "| — | — | — | — | no eligible payment method for this user and request |"
        )

    lines += ["", "## Verification: exhaustive ranking check", ""]
    if search.chosen is None:
        lines.append(
            "No eligible plan is safe, so `not_recommended` is the only acceptable answer."
        )
    else:
        others = [p for p in search.candidates if p.safe and p is not search.chosen]
        lines.append(f"`{search.chosen.label()}` is the highest-ranked safe plan.")
        if others:
            lines.append(
                "Other safe plans it outranks: "
                + ", ".join(f"`{p.label()}`" for p in others)
                + "."
            )

    if agent_run is not None:
        ws = agent_run.workspace
        lines += [
            "",
            "## Agent run",
            "",
            f"- **Decided by:** {'the local Qwen agent (choice accepted by the verifier)' if ws.decided_by == 'agent' else 'fallback exhaustive search - the model did not reach an accepted choice'}",
            f"- **Model calls:** {len(agent_run.usage.calls)} · tokens in/out: "
            f"{sum(c.input_tokens for c in agent_run.usage.calls):,} / {sum(c.output_tokens for c in agent_run.usage.calls):,}"
            f" · CPU time {agent_run.seconds:.0f} s",
        ]
        if agent_run.error:
            lines.append(f"- **Error:** {_md(agent_run.error[:300])}")
        if ws.image_readings:
            lines += ["", "### Image readings (OCR + local model)", ""]
            for event_id, reading in ws.image_readings.items():
                value = reading.get("amount") or "unresolved"
                lines.append(
                    f"- `{event_id}` from `{reading.get('image_id')}`: **{value}** labelled "
                    f"\"{_md(reading.get('label', ''))}\"; grounded in OCR text: {reading.get('grounded_in_ocr')}"
                )
        if ws.lean_decisions:
            lines += [
                "",
                "### Model decisions",
                "",
                "| Attempt | Model | Chose | Model's reason | Verifier | Seconds |",
                "|---:|---|---|---|---|---:|",
            ]
            for i, d in enumerate(ws.lean_decisions, 1):
                lines.append(
                    f"| {i} | `{d.get('model')}` | {d.get('plan') or '—'} | {_md(d.get('why', ''))} | "
                    f"{_md((d.get('verdict') or '')[:160])} | {d.get('seconds') or '—'} |"
                )
        if ws.rejected_submissions:
            lines += ["", "### Choices the verifier rejected", ""]
            lines += [
                f"{i}. {_md(r)}" for i, r in enumerate(ws.rejected_submissions, 1)
            ]

    return "\n".join(lines) + "\n"


def write_agent_traces(
    directory: Path, engine: Engine, rule: ForecastRule, runs
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    index = [
        "# Decision traces",
        "",
        "One file per request: evidence applied (messages and image readings), ledger reservations and "
        "exclusions, projected series, the 90-day forecast, every candidate plan, the local model's choice "
        "and stated reason, verifier verdicts, and the exhaustive ranking check.",
        "",
        "| Request | Status | Method | Decided by | Model calls |",
        "|---|---|---|---|---:|",
    ]
    for run in runs:
        ws = run.workspace
        request = ws.request
        (directory / f"{request.request_id}.md").write_text(
            render(
                engine,
                rule,
                request,
                run.decision,
                ws.reference_search(),
                forecaster=ws.forecaster(),
                amendments=ws.amendments,
                image_amounts=ws.image_amounts,
                agent_run=run,
            ),
            encoding="utf-8",
        )
        row = run.decision.as_row()
        index.append(
            f"| [{request.request_id}]({request.request_id}.md) | {row['affordability_status']} | "
            f"{row['recommended_payment_method']} | {ws.decided_by} | {len(run.usage.calls)} |"
        )
    (directory / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
