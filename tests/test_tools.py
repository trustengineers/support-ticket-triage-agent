from __future__ import annotations

import json
from typing import Any

import pytest

from app import db, tools
from app.tools import ToolContext, execute_tool
from tests.conftest import TICKET_ID


def run(name: str, arguments: object, ctx: ToolContext) -> tuple[Any, bool]:
    """Call a tool the way the agent loop does; decode the payload of a successful call."""
    result = execute_tool(name, arguments, ctx)
    return (result.content if result.is_error else json.loads(result.content)), result.is_error


# classify_ticket -----------------------------------------------------------


def test_classify_ticket_stores_the_classification(ctx: ToolContext) -> None:
    payload, is_error = run("classify_ticket", {"category": "billing", "priority": "high", "sentiment": "negative"}, ctx)
    assert not is_error and payload["category"] == "billing"
    saved = db.get_run(ctx.conn, ctx.run_id)
    assert saved is not None
    assert (saved["category"], saved["priority"], saved["sentiment"]) == ("billing", "high", "negative")


def test_classify_ticket_normalizes_case(ctx: ToolContext) -> None:
    payload, is_error = run("classify_ticket", {"category": " Bug ", "priority": "URGENT", "sentiment": "Neutral"}, ctx)
    assert not is_error and (payload["category"], payload["priority"]) == ("bug", "urgent")


@pytest.mark.parametrize(
    "arguments",
    [
        {"category": "refunds", "priority": "high", "sentiment": "negative"},
        {"category": "billing", "priority": "critical", "sentiment": "negative"},
        {"category": "billing", "priority": "high"},
        {"category": 3, "priority": "high", "sentiment": "negative"},
    ],
)
def test_classify_ticket_rejects_bad_values(ctx: ToolContext, arguments: dict[str, Any]) -> None:
    message, is_error = run("classify_ticket", arguments, ctx)
    assert is_error and "must be one of" in message
    saved = db.get_run(ctx.conn, ctx.run_id)
    assert saved is not None and saved["category"] is None


# search_knowledge_base -----------------------------------------------------


def test_search_knowledge_base_returns_ranked_articles(ctx: ToolContext) -> None:
    payload, is_error = run("search_knowledge_base", {"query": "refund within 30 days"}, ctx)
    assert not is_error
    assert payload["results"][0]["file"] == "refunds.md"
    assert "30 days" in payload["results"][0]["text"]
    assert len(payload["results"]) <= 3


def test_search_knowledge_base_with_no_match_is_not_an_error(ctx: ToolContext) -> None:
    payload, is_error = run("search_knowledge_base", {"query": "xylophone"}, ctx)
    assert not is_error and payload["results"] == []


@pytest.mark.parametrize("arguments", [{}, {"query": ""}, {"query": "   "}, {"query": "x" * 201}, {"query": ["refund"]}])
def test_search_knowledge_base_rejects_bad_queries(ctx: ToolContext, arguments: dict[str, Any]) -> None:
    _, is_error = run("search_knowledge_base", arguments, ctx)
    assert is_error


# draft_reply ---------------------------------------------------------------


def test_draft_reply_saves_the_reply(ctx: ToolContext) -> None:
    payload, is_error = run("draft_reply", {"ticket_id": TICKET_ID, "reply_text": "Hi, you can get a refund within 30 days."}, ctx)
    assert not is_error and payload["saved"] is True
    assert db.get_reply(ctx.conn, ctx.run_id) == "Hi, you can get a refund within 30 days."


def test_draft_reply_again_replaces_the_earlier_draft(ctx: ToolContext) -> None:
    run("draft_reply", {"ticket_id": TICKET_ID, "reply_text": "first"}, ctx)
    run("draft_reply", {"ticket_id": TICKET_ID, "reply_text": "second"}, ctx)
    rows = ctx.conn.execute("SELECT reply_text FROM replies WHERE run_id = ?", (ctx.run_id,)).fetchall()
    assert [row["reply_text"] for row in rows] == ["second"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"ticket_id": "T-999", "reply_text": "hello"},
        {"ticket_id": TICKET_ID, "reply_text": ""},
        {"ticket_id": TICKET_ID},
        {"ticket_id": TICKET_ID, "reply_text": "x" * (tools.MAX_REPLY_CHARS + 1)},
    ],
)
def test_draft_reply_rejects_bad_input(ctx: ToolContext, arguments: dict[str, Any]) -> None:
    _, is_error = run("draft_reply", arguments, ctx)
    assert is_error
    assert db.get_reply(ctx.conn, ctx.run_id) is None


# escalate_to_human ---------------------------------------------------------


def test_escalate_to_human_flags_the_ticket(ctx: ToolContext) -> None:
    payload, is_error = run("escalate_to_human", {"ticket_id": TICKET_ID, "reason": "Possible account takeover."}, ctx)
    assert not is_error and payload["escalated"] is True
    assert db.get_escalation(ctx.conn, ctx.run_id) == "Possible account takeover."


@pytest.mark.parametrize("arguments", [{"ticket_id": "T-999", "reason": "x"}, {"ticket_id": TICKET_ID, "reason": " "}, {"reason": "no id"}])
def test_escalate_to_human_rejects_bad_input(ctx: ToolContext, arguments: dict[str, Any]) -> None:
    _, is_error = run("escalate_to_human", arguments, ctx)
    assert is_error
    assert db.get_escalation(ctx.conn, ctx.run_id) is None


# execute_tool --------------------------------------------------------------


def test_unknown_tool_is_an_error_result(ctx: ToolContext) -> None:
    message, is_error = run("delete_ticket", {}, ctx)
    assert is_error and "Unknown tool" in message and "classify_ticket" in message


def test_arguments_as_a_json_string_are_accepted(ctx: ToolContext) -> None:
    payload, is_error = run("search_knowledge_base", '{"query": "cancel plan"}', ctx)
    assert not is_error and payload["results"][0]["file"] == "cancel-subscription.md"


@pytest.mark.parametrize("arguments", ['{"query": "cancel', "[1, 2]", None, 42])
def test_invalid_json_or_non_object_arguments_are_error_results(ctx: ToolContext, arguments: object) -> None:
    message, is_error = run("search_knowledge_base", arguments, ctx)
    assert is_error and "Invalid arguments" in message


def test_a_crashing_handler_becomes_an_error_result(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        raise RuntimeError("disk full")

    monkeypatch.setitem(tools.HANDLERS, "draft_reply", boom)
    message, is_error = run("draft_reply", {"ticket_id": TICKET_ID, "reply_text": "hi"}, ctx)
    assert is_error and "disk full" in message


def test_schemas_match_the_handlers() -> None:
    assert [schema["name"] for schema in tools.TOOL_SCHEMAS] == list(tools.HANDLERS)
