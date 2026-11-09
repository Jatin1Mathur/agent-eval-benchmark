"""Deterministic scoring of a run against its golden case, plus aggregate metrics."""

from __future__ import annotations

import ast
import math
from typing import Any

import pandas as pd

from agent_eval.schemas import CaseResult, EvalCase
from agent_eval.tools import _safe_eval


def _expr_value(expr: str) -> float | None:
    try:
        return float(_safe_eval(ast.parse(expr, mode="eval")))
    except Exception:
        return None


def args_match(tool: str, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Compare tool arguments semantically, not as raw strings.

    ``(12+8)*3`` and ``( 12 + 8 ) * 3`` are the same call; so are 100 and 100.0.
    """
    if tool == "calculator" and "expression" in expected:
        exp, act = _expr_value(expected["expression"]), _expr_value(actual.get("expression", ""))
        return exp is not None and act is not None and math.isclose(exp, act, rel_tol=1e-6)
    for key, value in expected.items():
        got = actual.get(key)
        if isinstance(value, int | float) and isinstance(got, int | float):
            if not math.isclose(float(value), float(got), rel_tol=1e-6):
                return False
        elif str(value).lower() != str(got).lower():
            return False
    return True


def score(case: EvalCase, result: CaseResult) -> CaseResult:
    result.route_correct = result.route == case.expected_route
    result.risk_correct = result.risk == case.expected_risk

    first = result.tool_calls[0] if result.tool_calls else None
    if case.expected_tool is None:
        result.tool_selection_correct = first is None
        result.tool_args_correct = first is None
    else:
        result.tool_selection_correct = first is not None and first.name == case.expected_tool.name
        result.tool_args_correct = result.tool_selection_correct and args_match(
            case.expected_tool.name, case.expected_tool.args, first.args
        )

    answer = result.final_answer.lower().replace(",", "")
    result.answer_correct = all(s.lower() in answer for s in case.answer_must_contain)
    return result


def _prf(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float, float]:
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def to_frame(results: list[CaseResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = r.model_dump(exclude={"tool_calls"})
        row["tools_called"] = ",".join(t.name for t in r.tool_calls)
        row["task_success"] = r.task_success
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, expected_risk: dict[str, str]) -> pd.DataFrame:
    """One row per (model, prompt_variant) with the headline metrics."""
    out = []
    for (model, variant), g in df.groupby(["model", "prompt_variant"], sort=False):
        true_high = g["case_id"].map(expected_risk).eq("high")
        p, r, f1 = _prf(true_high, g["risk"].eq("high"))
        out.append(
            {
                "model": model,
                "prompt": variant,
                "task_success": g["task_success"].mean(),
                "routing_acc": g["route_correct"].mean(),
                "high_risk_precision": p,
                "high_risk_recall": r,
                "high_risk_f1": f1,
                "tool_selection_acc": g["tool_selection_correct"].mean(),
                "tool_args_acc": g["tool_args_correct"].mean(),
                "answer_acc": g["answer_correct"].mean(),
                "judge_score": g["judge_score"].mean() if g["judge_score"].notna().any() else None,
                "p50_latency_s": g["latency_s"].median(),
                "p95_latency_s": g["latency_s"].quantile(0.95),
                "avg_tokens": (g["input_tokens"] + g["output_tokens"]).mean(),
                "errors": int(g["error"].notna().sum()),
            }
        )
    return pd.DataFrame(out).sort_values("task_success", ascending=False, ignore_index=True)
