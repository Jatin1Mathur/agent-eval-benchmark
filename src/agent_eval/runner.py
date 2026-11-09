"""Runs golden cases through the graph and reads the outcome from the message trace."""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agent_eval.agent import build_graph
from agent_eval.llms import get_llm
from agent_eval.schemas import CaseResult, EvalCase, ExpectedToolCall


def load_cases(path: str | Path) -> list[EvalCase]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvalCase.model_validate(json.loads(line)) for line in lines if line.strip()]


def run_case(graph, case: EvalCase, model: str, prompt_variant: str) -> CaseResult:
    config = {"configurable": {"thread_id": f"{model}-{prompt_variant}-{case.id}"}}
    start = time.perf_counter()
    error = None
    try:
        state = graph.invoke({"messages": [HumanMessage(content=case.prompt)]}, config)
    except Exception as exc:  # a crash is a result to report, not a reason to stop the run
        state, error = {"messages": [], "route": "error", "risk": "unknown"}, repr(exc)
    latency = time.perf_counter() - start

    ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
    tool_calls = [
        ExpectedToolCall(name=tc["name"], args=tc["args"]) for m in ai_msgs for tc in m.tool_calls
    ]
    usage = [m.usage_metadata or {} for m in ai_msgs]

    return CaseResult(
        case_id=case.id,
        category=case.category,
        model=model,
        prompt_variant=prompt_variant,
        route=state.get("route", "error"),
        risk=state.get("risk", "unknown"),
        tool_calls=tool_calls,
        final_answer=str(ai_msgs[-1].content) if ai_msgs else "",
        latency_s=round(latency, 4),
        input_tokens=sum(u.get("input_tokens", 0) for u in usage),
        output_tokens=sum(u.get("output_tokens", 0) for u in usage),
        error=error,
    )


def run_suite(cases: list[EvalCase], model: str, prompt_variant: str) -> list[CaseResult]:
    graph = build_graph(get_llm(model), prompt_variant=prompt_variant)
    return [run_case(graph, c, model, prompt_variant) for c in cases]
