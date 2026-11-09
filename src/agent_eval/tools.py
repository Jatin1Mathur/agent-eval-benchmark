"""Tools the agent can call. Each tool has a Pydantic argument schema, so
invalid arguments are rejected before the tool body ever runs."""

from __future__ import annotations

import ast
import operator
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- calculator

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only numeric arithmetic is allowed")


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(12 + 8) * 3'")


@tool("calculator", args_schema=CalculatorArgs)
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression safely (no eval of arbitrary code)."""
    result = _safe_eval(ast.parse(expression, mode="eval"))
    return str(round(result, 6)).removesuffix(".0") if isinstance(result, float) else str(result)


# --------------------------------------------------------------------------- unit converter

_FACTORS: dict[tuple[str, str], float] = {
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.609344,
    ("kg", "lb"): 2.204623,
    ("lb", "kg"): 0.453592,
    ("bar", "psi"): 14.503774,
    ("psi", "bar"): 0.068948,
}

Unit = Literal["km", "miles", "kg", "lb", "bar", "psi", "celsius", "fahrenheit"]


class ConvertArgs(BaseModel):
    value: float = Field(description="Numeric value to convert")
    from_unit: Unit
    to_unit: Unit


@tool("unit_converter", args_schema=ConvertArgs)
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between supported engineering units."""
    if (from_unit, to_unit) == ("celsius", "fahrenheit"):
        out = value * 9 / 5 + 32
    elif (from_unit, to_unit) == ("fahrenheit", "celsius"):
        out = (value - 32) * 5 / 9
    elif (from_unit, to_unit) in _FACTORS:
        out = value * _FACTORS[(from_unit, to_unit)]
    else:
        raise ValueError(f"Unsupported conversion {from_unit} -> {to_unit}")
    return f"{round(out, 2)} {to_unit}"


# --------------------------------------------------------------------------- doc search

KNOWLEDGE_BASE: dict[str, tuple[list[str], str]] = {
    "travel": (
        ["travel", "trip", "business trip"],
        "Business travel must be booked via the internal portal at least 14 days ahead.",
    ),
    "remote": (
        ["remote", "home office", "work from home"],
        "Employees may work remotely up to 3 days per week by arrangement with their lead.",
    ),
    "security": (
        ["security", "password", "api key", "credential", "incident"],
        "Never share credentials. Report incidents to the security team within 24 hours.",
    ),
    "onboarding": (
        ["onboarding", "intern", "first day"],
        "New interns receive laptop, badge and Git access on their first day.",
    ),
}


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, description="Topic to look up in the internal handbook")


@tool("search_docs", args_schema=SearchArgs)
def search_docs(query: str) -> str:
    """Search the internal handbook and return the best matching passage."""
    q = query.lower()
    scores = {k: sum(kw in q for kw in kws) for k, (kws, _) in KNOWLEDGE_BASE.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "No matching document found."
    return KNOWLEDGE_BASE[best][1]


ALL_TOOLS = [calculator, unit_converter, search_docs]
