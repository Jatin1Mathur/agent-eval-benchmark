"""Every golden case runs as its own pytest test, so a failure names the exact case."""

from pathlib import Path

import pytest

from agent_eval.agent import build_graph
from agent_eval.llms import get_llm
from agent_eval.metrics import score
from agent_eval.runner import load_cases, run_case

CASES = load_cases(Path(__file__).resolve().parents[1] / "data" / "golden.jsonl")


@pytest.fixture(scope="module")
def graph():
    return build_graph(get_llm("reference"))


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_golden_case(graph, case):
    r = score(case, run_case(graph, case, "reference", "baseline"))
    assert r.error is None, r.error
    assert r.route_correct, f"route {r.route!r} != {case.expected_route!r}"
    assert r.risk_correct, f"risk {r.risk!r} != {case.expected_risk!r}"
    assert r.tool_selection_correct, f"tools called: {[t.name for t in r.tool_calls]}"
    assert r.tool_args_correct, f"args: {[t.args for t in r.tool_calls]}"
    assert r.answer_correct, f"answer: {r.final_answer!r}"
