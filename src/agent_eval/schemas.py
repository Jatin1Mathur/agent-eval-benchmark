"""Typed contracts for golden test cases and evaluation results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Route = Literal["agent", "safe_response"]
Risk = Literal["low", "medium", "high"]


class ExpectedToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    """One labelled prompt in the golden dataset."""

    id: str
    prompt: str
    category: str
    expected_route: Route
    expected_risk: Risk
    expected_tool: ExpectedToolCall | None = None
    answer_must_contain: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    """What actually happened when the agent ran one case, read from the message trace."""

    case_id: str
    category: str
    model: str
    prompt_variant: str
    route: str
    risk: str
    tool_calls: list[ExpectedToolCall]
    final_answer: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    error: str | None = None

    # filled in by scoring
    route_correct: bool = False
    risk_correct: bool = False
    tool_selection_correct: bool = False
    tool_args_correct: bool = False
    answer_correct: bool = False
    judge_score: float | None = None

    @property
    def task_success(self) -> bool:
        return (
            self.route_correct
            and self.tool_selection_correct
            and self.tool_args_correct
            and self.answer_correct
        )
