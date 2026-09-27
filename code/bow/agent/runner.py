"""Run the local agent over a set of requests.

* Image evidence is extracted once, up front (OCR + local model, cached).
* Requests run sequentially (one CPU model), each through the LangGraph agent.
* Every finished request is saved to code/cache/results_local/, keyed by model
  and image evidence, so an interrupted run resumes where it stopped.
* A global DiskCache replays identical model answers on a rerun.
* A per-request callback records each model call's tokens for the usage report.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_llm_cache

from ..engine import Engine
from ..evidence import rules
from ..evidence.images_local import extract_image_amounts
from ..loaders import dataset_dir
from ..recurrence import ForecastRule
from ..writers import Decision, decision_from_row
from .local_agent import build_local_graph, fallback
from .local_llm import DiskCache, local_chat_model
from .verifier import register
from .workspace import Workspace


@dataclass
class CallRecord:
    model: str
    input_tokens: int
    output_tokens: int


class UsageTracker(BaseCallbackHandler):
    """Collects token usage from every chat-model call in one request's run."""

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []
        self._lock = threading.Lock()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                meta = getattr(message, "response_metadata", None) or {}
                with self._lock:
                    self.calls.append(
                        CallRecord(
                            model=meta.get("model_name", "unknown"),
                            input_tokens=int(usage.get("input_tokens", 0) or 0),
                            output_tokens=int(usage.get("output_tokens", 0) or 0),
                        )
                    )


@dataclass
class AgentRun:
    workspace: Workspace
    usage: UsageTracker
    error: str = ""
    seconds: float = 0.0
    state: dict = field(default_factory=dict)

    @property
    def decision(self) -> Decision:
        assert self.workspace.decision is not None
        return self.workspace.decision


def _load_saved(record: dict, ws: Workspace, run: AgentRun) -> None:
    ws.commit_evidence([rules.parse(m) for m in ws.messages], dict(ws.image_amounts))
    ws.decision = decision_from_row(record["row"])
    ws.decided_by = record["decided_by"]
    ws.lean_decisions = record["attempts"]
    ws.rejected_submissions = record.get("rejected", [])
    run.usage.calls = [
        CallRecord(c["model"], c["input_tokens"], c["output_tokens"])
        for c in record["calls"]
    ]
    run.seconds = record.get("seconds", 0.0)
    run.error = record.get("error", "")


def run_local_agent(
    engine: Engine,
    requests: list,
    rule: ForecastRule,
    cache_dir: Path,
    fresh: bool = False,
    log=print,
) -> list[AgentRun]:
    set_llm_cache(DiskCache(cache_dir / "llm"))
    results_dir = cache_dir / "results_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    graph = build_local_graph()
    model = local_chat_model()

    log("image evidence: OCR + local model")
    engine.image_amounts, engine.image_records = extract_image_amounts(
        engine.data, dataset_dir(), cache_dir / "image_amounts.json", model, log=log
    )

    runs: list[AgentRun] = []
    started = time.monotonic()
    done_live = 0
    log(
        f"local agent: {len(requests)} request(s), model {model.model_id} (CPU, transformers)"
    )

    for index, request in enumerate(requests, 1):
        ws = Workspace(engine=engine, request=request, rule=rule)
        blank_ids = [e.event_id for e, _ in ws.blank_amount_events()]
        ws.image_amounts = {
            k: v for k, v in engine.image_amounts.items() if k in blank_ids
        }
        ws.image_readings = {
            k: engine.image_records[k] for k in blank_ids if k in engine.image_records
        }
        evidence_key = {k: str(v) for k, v in sorted(ws.image_amounts.items())}
        register(ws)
        run = AgentRun(workspace=ws, usage=UsageTracker())
        saved = results_dir / f"{request.request_id}.json"

        if saved.exists() and not fresh:
            record = json.loads(saved.read_text(encoding="utf-8"))
            if (
                record.get("model") == model.model_id
                and record.get("image_amounts", {}) == evidence_key
            ):
                _load_saved(record, ws, run)
                runs.append(run)
                continue

        config = {
            "configurable": {"request_id": request.request_id},
            "callbacks": [run.usage],
            "recursion_limit": 30,
            "run_name": f"buy-or-wait:{request.request_id}",
        }
        t0 = time.monotonic()
        try:
            run.state = graph.invoke({"request_id": request.request_id}, config)
        except Exception as exc:  # never lose the request; fall back and flag it
            run.error = f"{type(exc).__name__}: {exc}"
            fallback({}, config)
        if ws.decision is None:
            fallback({}, config)
            run.error = run.error or "graph ended without a decision"
        run.seconds = round(time.monotonic() - t0, 2)
        done_live += 1

        saved.write_text(
            json.dumps(
                {
                    "model": model.model_id,
                    "image_amounts": evidence_key,
                    "decided_by": ws.decided_by,
                    "row": ws.decision.as_row(),
                    "attempts": ws.lean_decisions,
                    "rejected": ws.rejected_submissions,
                    "calls": [c.__dict__ for c in run.usage.calls],
                    "seconds": run.seconds,
                    "error": run.error,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        runs.append(run)

        elapsed = time.monotonic() - started
        remaining = (len(requests) - index) * (elapsed / max(done_live, 1))
        log(
            f"  [{index}/{len(requests)}] {request.request_id}: {ws.decision.recommended_payment_method} "
            f"by {ws.decided_by} in {run.seconds:.0f}s (~{remaining / 60:.0f} min left)"
            + (f" ERROR {run.error[:120]}" if run.error else "")
        )

    agent_count = sum(1 for r in runs if r.workspace.decided_by == "agent")
    log(
        f"local agent: {agent_count}/{len(runs)} decided by the agent, {len(runs) - agent_count} by fallback"
    )
    return runs
