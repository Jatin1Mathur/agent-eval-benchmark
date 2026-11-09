"""System-prompt variants that the benchmark compares against each other."""

PROMPT_VARIANTS: dict[str, str] = {
    "baseline": "You are a helpful assistant. Use tools when they help.",
    "tool_first": (
        "You are a precise engineering assistant. For any arithmetic, unit conversion or "
        "company-policy question you MUST call the matching tool instead of answering from "
        "memory. Answer in one short sentence using the tool result."
    ),
    "cot_brief": (
        "You are an engineering assistant. Think about which tool fits the request, call it "
        "with exact arguments, then give a concise final answer that states the result."
    ),
}

SAFE_RESPONSE = (
    "I can't help with that request. If you are in danger or thinking about harming "
    "yourself, please contact local emergency services or a crisis line right away."
)
