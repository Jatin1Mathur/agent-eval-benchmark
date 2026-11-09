"""Command-line entry point.

Examples
--------
Offline (CI):            agent-eval --models reference --prompts baseline --gate
Compare noisy model:     agent-eval --models reference reference-noisy
Real local models:       agent-eval --models ollama:llama3.1:8b ollama:qwen2.5:7b \
                                    --prompts baseline tool_first --judge ollama:llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from agent_eval.judge import heuristic_judge, llm_judge
from agent_eval.llms import get_llm
from agent_eval.metrics import score, summarize, to_frame
from agent_eval.prompts import PROMPT_VARIANTS
from agent_eval.runner import load_cases, run_suite

ROOT = Path(__file__).resolve().parents[2]


def check_gate(summary: pd.DataFrame, thresholds: dict[str, float]) -> list[str]:
    """Return a list of human-readable failures; empty list means the gate passes."""
    failures = []
    for _, row in summary.iterrows():
        for metric, minimum in thresholds.items():
            if row[metric] < minimum:
                failures.append(
                    f"{row['model']}/{row['prompt']}: {metric}={row[metric]:.3f} < {minimum}"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate and benchmark a LangGraph agent")
    ap.add_argument("--dataset", default=str(ROOT / "data" / "golden.jsonl"))
    ap.add_argument("--models", nargs="+", default=["reference"])
    ap.add_argument("--prompts", nargs="+", default=["baseline"], choices=list(PROMPT_VARIANTS))
    ap.add_argument("--judge", default="heuristic", help="'heuristic', 'none' or 'ollama:<model>'")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--gate", action="store_true", help="exit 1 if thresholds are not met")
    ap.add_argument("--thresholds", default=str(ROOT / "data" / "thresholds.json"))
    args = ap.parse_args(argv)

    cases = load_cases(args.dataset)
    by_id = {c.id: c for c in cases}
    judge_llm = get_llm(args.judge) if args.judge.startswith("ollama:") else None

    results = []
    for model in args.models:
        for variant in args.prompts:
            print(f"Running {len(cases)} cases  model={model}  prompt={variant}")
            for r in run_suite(cases, model, variant):
                case = by_id[r.case_id]
                score(case, r)
                if args.judge != "none":
                    verdict = (
                        llm_judge(judge_llm, case, r) if judge_llm else heuristic_judge(case, r)
                    )
                    r.judge_score = verdict.score
                results.append(r)

    runs = to_frame(results)
    summary = summarize(runs, {c.id: c.expected_risk for c in cases})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(summary.to_markdown(index=False, floatfmt=".3f") + "\n")

    failed = runs[~runs["task_success"]][
        ["model", "prompt_variant", "case_id", "tools_called", "final_answer"]
    ]
    print("\n" + summary.to_markdown(index=False, floatfmt=".3f"))
    if not failed.empty:
        print(f"\n{len(failed)} failing case(s):")
        print(failed.to_markdown(index=False))

    if args.gate:
        failures = check_gate(summary, json.loads(Path(args.thresholds).read_text()))
        if failures:
            print("\nREGRESSION GATE FAILED:\n  " + "\n  ".join(failures))
            return 1
        print("\nRegression gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
