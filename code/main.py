"""Buy or Wait? - AI financial decision agent (LangGraph + local Qwen via Hugging Face).

    python code/main.py                 # agent over dataset/requests.csv -> output.csv + code/evaluation/*
    python code/main.py --samples       # agent over the 25 solved samples, scored per column
    python code/main.py --limit 5       # first N requests only (smoke test)
    python code/main.py --fresh         # ignore saved per-request results
    python code/main.py --engine-only   # no LLM: exhaustive plan search (reference / fallback path)
    python code/main.py --calibrate     # re-identify the forecast rule on the samples

Model: BOW_LOCAL_MODEL (default Qwen/Qwen3-1.7B), downloaded into HF_HOME
(default <repo>/.cache/huggingface) on first run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bow.loaders import repo_root  # noqa: E402


def rule_based_engine(data):
    from bow.engine import Engine
    from bow.evidence import rules

    amendments = {uid: [rules.parse(m) for m in msgs] for uid, msgs in data.messages_by_user.items()}
    return Engine(data, amendments=amendments)


def score(rows_by_id: dict, samples) -> None:
    from bow.pipeline import SCORED_FIELDS

    print("field accuracy on solved samples:")
    for name in SCORED_FIELDS:
        hits = sum(rows_by_id[s.request_id][name] == s.expected[name] for s in samples)
        print(f"  {name:32} {hits:2}/{len(samples)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", action="store_true", help="run on the 25 solved samples and score them")
    parser.add_argument("--limit", type=int, default=0, help="only the first N requests")
    parser.add_argument("--fresh", action="store_true", help="ignore saved per-request results")
    parser.add_argument("--engine-only", action="store_true", help="no LLM; exhaustive plan search")
    parser.add_argument("--calibrate", action="store_true", help="re-identify the forecast rule on the samples")
    parser.add_argument("--output", type=Path, default=None, help="output.csv path")
    args = parser.parse_args(argv)

    from bow import calibrate, pipeline, trace, validate
    from bow.loaders import load_dataset
    from bow.writers import write_output

    root = repo_root()
    evaluation = root / "code" / "evaluation"
    data = load_dataset()
    rule = pipeline.load_rule()

    if args.calibrate:
        pipeline.save_rule(calibrate.run(evaluation / "calibration_report.md", engine=rule_based_engine(data)))
        return 0

    requests = data.samples if args.samples else data.requests
    if args.limit:
        requests = requests[: args.limit]

    if args.engine_only:
        engine = rule_based_engine(data)
        decisions = [pipeline.decide(engine, r, rule)[0] for r in requests]
        if args.samples:
            score({d.request_id: d.as_row() for d in decisions}, requests)
        else:
            out = args.output or (root / "output.csv")
            write_output(out, decisions)
            print(f"wrote {len(decisions)} rows (engine-only) -> {out}")
        return 0

    from bow.agent.local_llm import ensure_hf_home
    from bow.agent.runner import CallRecord, run_local_agent
    from bow.engine import Engine
    from bow.usage import write_usage_report

    ensure_hf_home()
    engine = Engine(data)
    runs = run_local_agent(engine, requests, rule, root / "code" / "cache", fresh=args.fresh)
    decisions = [run.decision for run in runs]

    if args.samples:
        score({d.request_id: d.as_row() for d in decisions}, requests)
        return 0

    out = args.output or (root / "output.csv")
    write_output(out, decisions)
    print(f"wrote {len(decisions)} rows -> {out}")

    trace.write_agent_traces(evaluation / "traces", engine, rule, runs)
    image_calls = [
        CallRecord(r["model"], int((r.get("usage") or {}).get("input_tokens", 0)),
                   int((r.get("usage") or {}).get("output_tokens", 0)))
        for r in engine.image_records.values()
    ]
    write_usage_report(evaluation / "usage_report.md", runs, image_calls=image_calls)
    print(f"wrote traces and usage report -> {evaluation}")

    if args.limit:
        return 0
    report = validate.validate(out, data)
    validate.write_report(evaluation / "validation_report.md", report)
    print(f"validation: {report.checks} checks, {len(report.violations)} violations")
    for violation in report.violations[:20]:
        print(f"  VIOLATION {violation}")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
