"""The agent under test: a LangGraph StateGraph with a deterministic guardrail
router, a ReAct loop (agent <-> tools) and a safe fixed-response node.

    START -> router --high risk--> safe_response -> END
                    \\-otherwise-> agent <-> tools
                                     \\-> END (no more tool calls)
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent_eval.prompts import PROMPT_VARIANTS, SAFE_RESPONSE
from agent_eval.risk import RiskLevel, classify_risk
from agent_eval.tools import ALL_TOOLS

MAX_STEPS = 6  # hard cap on ReAct iterations so a confused model cannot loop forever


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    risk: str
    route: str
    steps: int


def build_graph(llm: BaseChatModel, prompt_variant: str = "baseline", checkpointer=None):
    system = SystemMessage(content=PROMPT_VARIANTS[prompt_variant])
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def router(state: AgentState) -> dict:
        user_text = state["messages"][-1].content
        risk = classify_risk(str(user_text))
        route = "safe_response" if risk == RiskLevel.HIGH else "agent"
        return {"risk": risk.value, "route": route, "steps": 0}

    def agent(state: AgentState) -> dict:
        if state.get("steps", 0) >= MAX_STEPS:
            return {"messages": [AIMessage(content="Stopping: step limit reached.")]}
        reply = llm_with_tools.invoke([system, *state["messages"]])
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    def safe_response(_: AgentState) -> dict:
        return {"messages": [AIMessage(content=SAFE_RESPONSE)]}

    g = StateGraph(AgentState)
    g.add_node("router", router)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=True))
    g.add_node("safe_response", safe_response)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["route"], ["agent", "safe_response"])
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    g.add_edge("safe_response", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
