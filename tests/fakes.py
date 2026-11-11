"""A scripted stand-in for the model, so tests run offline and deterministically."""

from __future__ import annotations

from typing import Any

from app.llm import LLMResponse, ToolCall


def turn(thought: str = "", *calls: tuple[str, Any], stop_reason: str | None = None, tokens: tuple[int, int] = (100, 20)) -> LLMResponse:
    """Build one model turn. Each call is (tool_name, arguments)."""
    tool_calls = [ToolCall(id=f"call_{index}", name=name, arguments=arguments) for index, (name, arguments) in enumerate(calls)]
    content: list[dict[str, Any]] = [{"type": "text", "text": thought}] if thought else []
    content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in tool_calls]
    return LLMResponse(
        content=content,
        thought=thought,
        tool_calls=tool_calls,
        stop_reason=stop_reason or ("tool_use" if tool_calls else "end_turn"),
        input_tokens=tokens[0],
        output_tokens=tokens[1],
    )


class ScriptedLLM:
    """Returns prepared turns in order and repeats the last one when the script runs out."""

    def __init__(self, turns: list[LLMResponse]) -> None:
        self._turns = turns
        self.requests: list[dict[str, Any]] = []

    def complete(self, *, model: str, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> LLMResponse:
        self.requests.append({"model": model, "system": system, "tools": tools, "messages": list(messages)})
        return self._turns[min(len(self.requests), len(self._turns)) - 1]
