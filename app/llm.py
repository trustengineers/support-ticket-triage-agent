"""A thin wrapper over the Anthropic SDK.

The agent loop depends only on the small `LLMClient` protocol, so tests can
replace the real model with a scripted one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic


class MissingCredentialsError(RuntimeError):
    """No API key (or other credential) is configured for the model provider."""


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: Any


@dataclass(frozen=True)
class LLMResponse:
    """One model turn, reduced to what the agent loop needs."""

    content: list[Any]  # raw content blocks, sent back unchanged on the next request
    thought: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    """Anything that can produce the next model turn."""

    def complete(self, *, model: str, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> LLMResponse: ...


class AnthropicLLM:
    """Calls the Messages API. Credentials come from the environment (ANTHROPIC_API_KEY)."""

    def __init__(self, max_tokens: int = 8000, thinking_display: str = "summarized", client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic()
        self._max_tokens = max_tokens
        self._thinking_display = thinking_display

    def complete(self, *, model: str, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> LLMResponse:
        """Send the conversation and return the model's next turn."""
        extra: dict[str, Any] = {}
        if self._thinking_display != "off":
            # Returns a readable summary of the model's reasoning, shown as the step's "thought".
            extra["thinking"] = {"type": "adaptive", "display": self._thinking_display}
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=self._max_tokens,
                system=system,
                tools=tools,
                messages=messages,
                **extra,
            )
        except TypeError as exc:
            # The SDK raises a bare TypeError when it finds no credentials to send.
            if "authentication method" in str(exc):
                raise MissingCredentialsError("No model credentials found. Set ANTHROPIC_API_KEY in .env.") from exc
            raise
        return _to_response(response)


def _to_response(response: anthropic.types.Message) -> LLMResponse:
    thoughts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "thinking" and block.thinking.strip():
            thoughts.append(block.thinking.strip())
        elif block.type == "text" and block.text.strip():
            thoughts.append(block.text.strip())
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
    usage = response.usage
    input_tokens = usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
    return LLMResponse(
        content=list(response.content),
        thought="\n".join(thoughts),
        tool_calls=tool_calls,
        stop_reason=response.stop_reason,
        input_tokens=input_tokens,
        output_tokens=usage.output_tokens,
    )
