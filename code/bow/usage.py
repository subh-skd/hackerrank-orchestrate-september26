"""evaluation/usage_report.md - model usage and cost for the run that wrote output.csv."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def write_usage_report(path: Path, runs, image_calls=None) -> None:
    decision_calls = [c for run in runs for c in run.usage.calls]
    image_calls = list(image_calls or [])
    calls = decision_calls + image_calls
    requests = len(runs)

    by_model: dict[str, list] = defaultdict(list)
    for call in calls:
        by_model[call.model].append(call)

    def tokens(rows):
        return sum(c.input_tokens for c in rows), sum(c.output_tokens for c in rows)

    total_in, total_out = tokens(calls)
    total_tokens = total_in + total_out
    per_request = [len(run.usage.calls) for run in runs]
    agent_decided = sum(1 for run in runs if run.workspace.decided_by == "agent")
    cpu_seconds = sum(run.seconds for run in runs)

    lines = [
        "# Token usage and cost report",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by the full-dataset run that "
        "wrote `output.csv`.",
        "",
        "## How the model is used",
        "",
        "Each request runs a LangGraph agent whose decision step is a **local** Qwen model downloaded from the "
        "Hugging Face Hub and executed on CPU with `transformers` (no API, no GPU). The model receives a compact "
        "case file in which every candidate plan has been simulated against the 90-day forecast, and it chooses a "
        "plan by the problem statement's ranking. A verifier rejects unsafe, ineligible or outranked choices and "
        "returns the reason for one retry. Before the requests run, the same model reads the 16 document images "
        "from their OCR text.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Provider | Hugging Face Hub weights, local CPU inference (`transformers`, LangChain `BaseChatModel`) |",
        f"| Models | {', '.join(f'`{m}`' for m in sorted(by_model)) or '—'} |",
        f"| Requests | {requests} |",
        f"| Decisions accepted from the model | {agent_decided} / {requests} |",
        f"| Model calls (decisions + image readings) | {len(calls):,} ({len(decision_calls):,} + {len(image_calls):,}) |",
        f"| Decision calls per request (avg / max) | {len(decision_calls) / max(requests, 1):.2f} / {max(per_request, default=0)} |",
        f"| Input tokens | {total_in:,} |",
        f"| Output tokens | {total_out:,} |",
        f"| Total tokens | {total_tokens:,} |",
        f"| Average tokens per request | {total_tokens / max(requests, 1):,.0f} |",
        "| Estimated total cost (USD) | $0.00 (local inference, no API fees) |",
        "| Estimated cost per request (USD) | $0.00 |",
        f"| CPU inference time, decisions (total / per request) | {cpu_seconds / 60:.1f} min / {cpu_seconds / max(requests, 1):.1f} s |",
        "",
        "## Per model",
        "",
        "| Model | Calls | Input tokens | Output tokens | Cost (USD) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, rows in sorted(by_model.items()):
        t_in, t_out = tokens(rows)
        lines.append(f"| `{model}` | {len(rows):,} | {t_in:,} | {t_out:,} | $0.00 |")
    lines.append(
        f"| **Overall** | **{len(calls):,}** | **{total_in:,}** | **{total_out:,}** | **$0.00** |"
    )
    lines += [
        "",
        "## Method",
        "",
        "Token counts are exact: they come from the model's own tokenizer, attached to every response as LangChain "
        "`usage_metadata` and collected by a per-request callback. Image-reading calls are counted from their "
        "cached records. There is no per-token price for local inference. The cost is local compute, reported "
        "above as CPU time on a 6-core laptop CPU (AMD Ryzen 5 5600H).",
        "",
        "Model answers are cached (`code/cache/llm`, `code/cache/results_local`, `code/cache/image_amounts.json`), "
        "so re-running reproduces `output.csv` without new inference. The figures above are the calls that "
        "produced it.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
