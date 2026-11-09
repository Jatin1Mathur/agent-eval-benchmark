from agent_eval.metrics import args_match, score
from agent_eval.schemas import CaseResult, EvalCase, ExpectedToolCall


def _case(**kw) -> EvalCase:
    base = dict(
        id="c1",
        prompt="p",
        category="calculation",
        expected_route="agent",
        expected_risk="low",
        expected_tool={"name": "calculator", "args": {"expression": "(12 + 8) * 3"}},
        answer_must_contain=["60"],
    )
    return EvalCase.model_validate({**base, **kw})


def _result(tool_calls, answer="The result is 60.", route="agent", risk="low") -> CaseResult:
    return CaseResult(
        case_id="c1",
        category="calculation",
        model="m",
        prompt_variant="baseline",
        route=route,
        risk=risk,
        tool_calls=tool_calls,
        final_answer=answer,
        latency_s=0.0,
        input_tokens=1,
        output_tokens=1,
    )


def test_args_match_is_semantic_for_expressions():
    assert args_match("calculator", {"expression": "(12+8)*3"}, {"expression": "3 * (8 + 12)"})
    assert not args_match("calculator", {"expression": "(12+8)*3"}, {"expression": "12+8*3"})


def test_args_match_numeric_tolerance():
    exp = {"value": 100, "from_unit": "km", "to_unit": "miles"}
    assert args_match(
        "unit_converter", exp, {"value": 100.0, "from_unit": "KM", "to_unit": "miles"}
    )
    assert not args_match("unit_converter", exp, {**exp, "value": 1000})


def test_perfect_run_is_task_success():
    r = score(
        _case(), _result([ExpectedToolCall(name="calculator", args={"expression": "(12 + 8) * 3"})])
    )
    assert r.task_success


def test_wrong_tool_fails_selection_and_args():
    r = score(_case(), _result([ExpectedToolCall(name="search_docs", args={"query": "x"})]))
    assert not r.tool_selection_correct and not r.tool_args_correct and not r.task_success


def test_unexpected_tool_call_on_chit_chat_is_penalised():
    case = _case(expected_tool=None, answer_must_contain=[])
    r = score(case, _result([ExpectedToolCall(name="calculator", args={"expression": "1+1"})]))
    assert not r.tool_selection_correct


def test_missing_fact_fails_answer():
    r = score(
        _case(),
        _result(
            [ExpectedToolCall(name="calculator", args={"expression": "(12+8)*3"})], answer="Done."
        ),
    )
    assert not r.answer_correct
