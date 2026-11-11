"""The four tools the agent can call: schemas for the model, validation and handlers.

Every handler validates its arguments. A bad call never raises into the agent loop;
it comes back to the model as an error result so it can correct itself.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from app import db
from app.config import CATEGORIES, PRIORITIES, SENTIMENTS
from app.kb import KnowledgeBase

MAX_REPLY_CHARS = 4000
MAX_REASON_CHARS = 500
MAX_QUERY_CHARS = 200

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "classify_ticket",
        "description": "Record the category, priority and customer sentiment of the current ticket. Call it once, before replying or escalating.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": list(CATEGORIES)},
                "priority": {"type": "string", "enum": list(PRIORITIES)},
                "sentiment": {"type": "string", "enum": list(SENTIMENTS)},
            },
            "required": ["category", "priority", "sentiment"],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the help-center FAQ (written in English) and return the best-matching articles with their full text. Use short English keywords, e.g. 'refund annual plan'.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A few English keywords"}},
            "required": ["query"],
        },
    },
    {
        "name": "draft_reply",
        "description": "Save a suggested reply to the customer for a support agent to review and send. It is not sent automatically. Base it on knowledge base articles; do not promise anything the articles do not support.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "reply_text": {"type": "string", "description": "The full reply, in the customer's language"},
            },
            "required": ["ticket_id", "reply_text"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Flag the ticket for a human agent. Use it when the ticket needs a decision, an account action or an investigation that a drafted reply cannot provide.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "reason": {"type": "string", "description": "One or two sentences for the human who picks it up"},
            },
            "required": ["ticket_id", "reason"],
        },
    },
]


@dataclass
class ToolContext:
    """What the tools need to act on the ticket that is being triaged."""

    conn: sqlite3.Connection
    kb: KnowledgeBase
    run_id: int
    ticket_id: str


@dataclass(frozen=True)
class ToolResult:
    """The text returned to the model for one tool call."""

    content: str
    is_error: bool = False


class ToolInputError(ValueError):
    """The model called a tool with arguments that cannot be used."""


def _text(args: dict[str, Any], key: str, max_chars: int) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f'"{key}" is required and must be a non-empty string')
    value = value.strip()
    if len(value) > max_chars:
        raise ToolInputError(f'"{key}" is too long ({len(value)} characters, the limit is {max_chars})')
    return value


def _choice(args: dict[str, Any], key: str, allowed: tuple[str, ...]) -> str:
    value = args.get(key)
    if not isinstance(value, str) or value.strip().lower() not in allowed:
        raise ToolInputError(f'"{key}" must be one of: {", ".join(allowed)} (got {value!r})')
    return value.strip().lower()


def _own_ticket(args: dict[str, Any], ctx: ToolContext) -> str:
    ticket_id = _text(args, "ticket_id", 100)
    if ticket_id != ctx.ticket_id:
        raise ToolInputError(f'ticket_id "{ticket_id}" is not the ticket being triaged ("{ctx.ticket_id}")')
    return ticket_id


def classify_ticket(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Store category, priority and sentiment for the current ticket."""
    category = _choice(args, "category", CATEGORIES)
    priority = _choice(args, "priority", PRIORITIES)
    sentiment = _choice(args, "sentiment", SENTIMENTS)
    db.save_classification(ctx.conn, ctx.run_id, category, priority, sentiment)
    return {"ticket_id": ctx.ticket_id, "category": category, "priority": priority, "sentiment": sentiment}


def search_knowledge_base(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Return the top FAQ articles for a query."""
    query = _text(args, "query", MAX_QUERY_CHARS)
    hits = ctx.kb.search(query, top_k=3)
    return {
        "query": query,
        "results": [{"file": hit.file, "title": hit.title, "score": hit.score, "text": hit.text} for hit in hits],
    }


def draft_reply(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Save a suggested reply for the current ticket."""
    ticket_id = _own_ticket(args, ctx)
    reply_text = _text(args, "reply_text", MAX_REPLY_CHARS)
    reply_id = db.save_reply(ctx.conn, ctx.run_id, ticket_id, reply_text)
    return {"saved": True, "reply_id": reply_id, "ticket_id": ticket_id}


def escalate_to_human(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Flag the current ticket for a human agent."""
    ticket_id = _own_ticket(args, ctx)
    reason = _text(args, "reason", MAX_REASON_CHARS)
    escalation_id = db.save_escalation(ctx.conn, ctx.run_id, ticket_id, reason)
    return {"escalated": True, "escalation_id": escalation_id, "ticket_id": ticket_id}


HANDLERS: dict[str, Callable[[dict[str, Any], ToolContext], dict[str, Any]]] = {
    "classify_ticket": classify_ticket,
    "search_knowledge_base": search_knowledge_base,
    "draft_reply": draft_reply,
    "escalate_to_human": escalate_to_human,
}


def _as_arguments(raw: Any) -> dict[str, Any]:
    """Accept a dict, or a JSON string that decodes to one."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolInputError(f"arguments are not valid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ToolInputError("arguments must be a JSON object")
    return raw


def execute_tool(name: str, arguments: Any, ctx: ToolContext) -> ToolResult:
    """Run one tool call. Unknown tools, bad arguments and handler failures become error results."""
    handler = HANDLERS.get(name)
    if handler is None:
        return ToolResult(f'Unknown tool "{name}". Available tools: {", ".join(HANDLERS)}.', is_error=True)
    try:
        payload = handler(_as_arguments(arguments), ctx)
    except ToolInputError as exc:
        return ToolResult(f"Invalid arguments for {name}: {exc}", is_error=True)
    except Exception as exc:  # noqa: BLE001 - a tool failure must not crash the agent loop
        return ToolResult(f"{name} failed: {type(exc).__name__}: {exc}", is_error=True)
    return ToolResult(json.dumps(payload, ensure_ascii=False))
