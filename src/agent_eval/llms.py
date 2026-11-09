"""Model factory.

* ``ollama:<model>``   -> a real local LLM served by Ollama (e.g. ``ollama:llama3.1:8b``).
* ``reference``        -> a deterministic rule-based chat model that speaks the same
                          tool-calling protocol. It lets the whole pipeline, CI and the
                          regression gate run offline in seconds, with no GPU.
* ``reference-noisy``  -> same, but with seeded mistakes (skipped tools, wrong args),
                          used to prove the metrics actually catch regressions.
"""

from __future__ import annotations

import random
import re
import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_UNIT_ALIASES = {
    "km": "km",
    "kilometers": "km",
    "kilometres": "km",
    "mile": "miles",
    "miles": "miles",
    "kg": "kg",
    "kilograms": "kg",
    "lb": "lb",
    "lbs": "lb",
    "pounds": "lb",
    "bar": "bar",
    "psi": "psi",
    "c": "celsius",
    "°c": "celsius",
    "celsius": "celsius",
    "f": "fahrenheit",
    "°f": "fahrenheit",
    "fahrenheit": "fahrenheit",
}
_UNIT_RE = "|".join(sorted(map(re.escape, _UNIT_ALIASES), key=len, reverse=True))
_CONVERT_RE = re.compile(
    rf"(-?\d+(?:\.\d+)?)\s*({_UNIT_RE})\b.*?\b(?:to|in|into)\s+({_UNIT_RE})\b", re.I
)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", re.I)
_EXPR_RE = re.compile(r"[\d\(][\d\s\.\+\-\*/\(\)%]*[\d\)]")
_DOC_WORDS = ("policy", "remote", "travel", "security", "onboarding", "handbook", "intern")


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ReferenceChatModel(BaseChatModel):
    """Deterministic stand-in for an LLM that emits real ``tool_calls``."""

    noise: float = 0.0
    seed: int = 7
    _rng: random.Random | None = None

    @property
    def _llm_type(self) -> str:
        return "reference"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):  # noqa: ARG002
        return self  # tool schemas are implicit in the rules below

    # -- planning ---------------------------------------------------------------------
    def _plan(self, user: str) -> tuple[str, dict] | None:
        if m := _CONVERT_RE.search(user):
            value, src, dst = m.groups()
            return "unit_converter", {
                "value": float(value),
                "from_unit": _UNIT_ALIASES[src.lower()],
                "to_unit": _UNIT_ALIASES[dst.lower()],
            }
        if m := _PERCENT_RE.search(user):
            return "calculator", {"expression": f"{m.group(2)} * {m.group(1)} / 100"}
        if any(w in user.lower() for w in _DOC_WORDS):
            return "search_docs", {"query": user}
        for cand in _EXPR_RE.findall(user):
            if any(op in cand for op in "+-*/%"):
                return "calculator", {"expression": cand.strip()}
        return None

    def _corrupt(self, plan: tuple[str, dict] | None) -> tuple[str, dict] | None:
        rng = self._rng or random.Random(self.seed)
        self._rng = rng
        if plan is None or rng.random() >= self.noise:
            return plan
        if rng.random() < 0.5:
            return None  # "forgets" to call the tool
        name, args = plan
        if name == "calculator":
            return name, {"expression": args["expression"].replace("*", "+", 1)}
        if name == "unit_converter":
            return name, {**args, "value": args["value"] * 10}
        return name, args

    # -- generation -------------------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_text = " ".join(str(m.content) for m in messages)
        last = messages[-1]

        if isinstance(last, ToolMessage):
            content = (
                f"According to the handbook: {last.content}"
                if last.name == "search_docs"
                else f"The result is {last.content}."
            )
            msg = AIMessage(content=content)
        else:
            user = next(str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage))
            plan = self._corrupt(self._plan(user))
            if plan is None:
                msg = AIMessage(
                    content="I can help with calculations, unit conversions and handbook "
                    "questions. Could you give me a bit more detail?"
                )
            else:
                name, args = plan
                msg = AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}],
                )

        msg.usage_metadata = {
            "input_tokens": _tokens(prompt_text),
            "output_tokens": _tokens(str(msg.content) or str(msg.tool_calls)),
            "total_tokens": _tokens(prompt_text) + _tokens(str(msg.content) or "x" * 40),
        }
        return ChatResult(generations=[ChatGeneration(message=msg)])


def get_llm(spec: str) -> BaseChatModel:
    if spec == "reference":
        return ReferenceChatModel()
    if spec == "reference-noisy":
        return ReferenceChatModel(noise=0.35)
    if spec.startswith("ollama:"):
        from langchain_ollama import ChatOllama

        return ChatOllama(model=spec.removeprefix("ollama:"), temperature=0)
    raise ValueError(f"Unknown model spec '{spec}'. Use 'reference' or 'ollama:<model>'.")
