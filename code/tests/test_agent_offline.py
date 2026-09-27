"""Exercise the LangGraph agent loop without loading model weights.

A scripted stand-in replaces the local Qwen model. Its first answer is
deliberately wrong (a plan that is not the best safe one), which proves the
verifier rejects it and the retry path recovers. Its second answer names the
best candidate. Everything else - case-file construction, the graph, routing,
verification, and fallback - is the production code path.

Run:  python code/tests/test_agent_offline.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402

import bow.agent.local_agent as local_agent  # noqa: E402
from bow.agent.verifier import register, workspace  # noqa: E402
from bow.agent.workspace import Workspace  # noqa: E402
from bow.engine import Engine  # noqa: E402
from bow.loaders import load_dataset  # noqa: E402
from bow.pipeline import SCORED_FIELDS, load_rule  # noqa: E402


class ScriptedModel:
    model_id = "scripted-test-model"

    def invoke(self, messages, config=None):
        ws = workspace(config)
        prompt = messages[-1].content
        # "P3: full_payment payment_option_16 total ..., changes stop:event_476 -> SAFE"
        rows = re.findall(r"^(P\d+): (\S+) (\S*) total .*?changes (\S+) ->", prompt, flags=re.M)
        best = ws.reference_search().chosen
        best_id = "NR"
        if best is not None:
            wanted = (best.method, best.option_id, "|".join(c.render() for c in best.changes) or "none")
            best_id = next(pid for pid, method, option, changes in rows if (method, option, changes) == wanted)
        if "previous choice was rejected" in prompt:
            pick = best_id
        elif best_id != "NR":
            pick = "NR"  # deliberately wrong: a safe plan exists
        else:
            pick = rows[0][0] if rows else "P99"  # deliberately wrong: unsafe or nonexistent plan
        content = json.dumps({"plan": pick, "why": "scripted"})
        return AIMessage(content=content, response_metadata={"model_name": self.model_id, "seconds": 0})


def main() -> int:
    local_agent.local_chat_model = lambda: ScriptedModel()
    data = load_dataset()
    engine = Engine(data)
    rule = load_rule()
    graph = local_agent.build_local_graph()

    rejected, agent, hits = 0, 0, {name: 0 for name in SCORED_FIELDS if name != "decision_explanation"}
    for sample in data.samples:
        ws = Workspace(engine=engine, request=sample, rule=rule)
        register(ws)
        graph.invoke({"request_id": sample.request_id}, {"configurable": {"request_id": sample.request_id}})
        rejected += bool(ws.rejected_submissions)
        agent += ws.decided_by == "agent"
        row = ws.decision.as_row()
        for name in hits:
            hits[name] += row[name] == sample.expected[name]

    n = len(data.samples)
    print(f"decided by the agent: {agent}/{n}; verifier rejected the deliberately wrong first choice: {rejected}/{n}")
    print("field accuracy through the agent path:", hits)
    ok = rejected == n and agent >= n - 1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
