from typing import TypedDict, Any

class AgentState(TypedDict):
    question: str
    tool_results: dict[str, Any]
    answer: str
    confidence: float
    sources: list[str]
    tools_called: list[str]
    tools_failed: list[str]
