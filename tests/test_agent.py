from __future__ import annotations

import sqlite3
from typing import Any

from app.agent import Step, Ticket, TriageResult, run_triage
from app.config import Settings
from app.kb import KnowledgeBase
from tests.fakes import ScriptedLLM, turn

TICKET = Ticket(id="T-200", subject="Refund", body="I upgraded 10 days ago. Can I get my money back?", customer="a@example.com")
CLASSIFY = ("classify_ticket", {"category": "billing", "priority": "medium", "sentiment": "neutral"})
SEARCH = ("search_knowledge_base", {"query": "refund"})
DRAFT = ("draft_reply", {"ticket_id": "T-200", "reply_text": "You can request a refund within 30 days."})


def triage(llm: ScriptedLLM, conn: sqlite3.Connection, kb: KnowledgeBase, settings: Settings, **kwargs: Any) -> TriageResult:
    return run_triage(TICKET, model="model-a", llm=llm, conn=conn, kb=kb, settings=settings, **kwargs)


def test_happy_path_classifies_searches_and_drafts(conn, kb, settings) -> None:
    llm = ScriptedLLM([turn("Classify and look up the refund policy.", CLASSIFY, SEARCH), turn("Draft the reply.", DRAFT), turn("Reply drafted.")])
    seen: list[Step] = []
    result = triage(llm, conn, kb, settings, on_step=seen.append)

    assert (result.status, result.stop_reason) == ("replied", "finished")
    assert (result.category, result.priority, result.sentiment) == ("billing", "medium", "neutral")
    assert result.reply == "You can request a refund within 30 days."
    assert (result.model_calls, result.tool_calls) == (3, 3)
    assert [step.tool for step in result.steps] == ["classify_ticket", "search_knowledge_base", "draft_reply", None]
    assert seen == result.steps
    assert (result.input_tokens, result.output_tokens) == (300, 60)
    assert result.cost_usd == round((300 * 10 + 60 * 50) / 1_000_000, 6)


def test_every_step_is_logged_to_sqlite(conn, kb, settings) -> None:
    result = triage(ScriptedLLM([turn("Classify.", CLASSIFY), turn("Draft.", DRAFT), turn("Done.")]), conn, kb, settings)
    rows = conn.execute("SELECT step, thought, tool, arguments, is_error FROM steps WHERE run_id = ? ORDER BY id", (result.run_id,)).fetchall()
    assert [(row["step"], row["tool"]) for row in rows] == [(1, "classify_ticket"), (2, "draft_reply"), (3, None)]
    assert rows[0]["thought"] == "Classify." and '"billing"' in rows[0]["arguments"]
    run = conn.execute("SELECT status, steps, tool_calls FROM runs WHERE id = ?", (result.run_id,)).fetchone()
    assert tuple(run) == ("replied", 3, 2)


def test_parallel_tool_results_go_back_in_one_message(conn, kb, settings) -> None:
    llm = ScriptedLLM([turn("Both at once.", CLASSIFY, SEARCH), turn("Draft.", DRAFT), turn("Done.")])
    triage(llm, conn, kb, settings)
    tool_results = llm.requests[1]["messages"][-1]
    assert tool_results["role"] == "user"
    assert [block["tool_use_id"] for block in tool_results["content"]] == ["call_0", "call_1"]


def test_loop_stops_at_the_step_cap_and_escalates(conn, kb, settings) -> None:
    llm = ScriptedLLM([turn("Searching again.", SEARCH)])
    result = triage(llm, conn, kb, settings)
    assert len(llm.requests) == settings.max_steps == 6
    assert (result.status, result.stop_reason) == ("escalated", "step_limit")
    assert result.escalation_reason is not None and "step limit" in result.escalation_reason
    assert result.steps[-1].tool == "escalate_to_human"


def test_a_tool_error_is_returned_to_the_model_which_recovers(conn, kb, settings) -> None:
    bad = ("classify_ticket", {"category": "refunds", "priority": "medium", "sentiment": "neutral"})
    llm = ScriptedLLM([turn("Classify.", bad), turn("Fix the category.", CLASSIFY, DRAFT), turn("Done.")])
    result = triage(llm, conn, kb, settings)
    assert result.steps[0].is_error and "must be one of" in (result.steps[0].result or "")
    assert llm.requests[1]["messages"][-1]["content"][0]["is_error"] is True
    assert (result.status, result.category) == ("replied", "billing")


def test_unknown_tool_does_not_crash_the_loop(conn, kb, settings) -> None:
    llm = ScriptedLLM([turn("Try a tool that does not exist.", ("refund_customer", {"amount": 49})), turn("Draft.", DRAFT), turn("Done.")])
    result = triage(llm, conn, kb, settings)
    assert result.steps[0].is_error and result.status == "replied"


def test_finishing_without_an_outcome_escalates_automatically(conn, kb, settings) -> None:
    result = triage(ScriptedLLM([turn("Classify.", CLASSIFY), turn("All done.")]), conn, kb, settings)
    assert (result.status, result.stop_reason) == ("escalated", "finished")
    assert result.escalation_reason is not None and "without drafting a reply or escalating" in result.escalation_reason


def test_refusal_escalates_without_running_tools(conn, kb, settings) -> None:
    result = triage(ScriptedLLM([turn("", DRAFT, stop_reason="refusal")]), conn, kb, settings)
    assert (result.status, result.stop_reason, result.tool_calls) == ("escalated", "refusal", 0)
    assert result.reply is None


def test_truncated_output_does_not_run_partial_tool_calls(conn, kb, settings) -> None:
    result = triage(ScriptedLLM([turn("", DRAFT, stop_reason="max_tokens")]), conn, kb, settings)
    assert (result.status, result.stop_reason, result.tool_calls) == ("escalated", "truncated", 0)


def test_escalation_wins_over_a_holding_reply(conn, kb, settings) -> None:
    escalate = ("escalate_to_human", {"ticket_id": "T-200", "reason": "Refund outside policy."})
    result = triage(ScriptedLLM([turn("Escalate with a holding reply.", CLASSIFY, DRAFT, escalate), turn("Done.")]), conn, kb, settings)
    assert result.status == "escalated" and result.reply and result.escalation_reason == "Refund outside policy."


def test_unknown_model_price_gives_no_cost(conn, kb, settings) -> None:
    llm = ScriptedLLM([turn("Draft.", DRAFT), turn("Done.")])
    result = run_triage(TICKET, model="unpriced-model", llm=llm, conn=conn, kb=kb, settings=settings)
    assert result.cost_usd is None


def test_the_ticket_is_sent_as_data_with_its_id(conn, kb, settings) -> None:
    llm = ScriptedLLM([turn("Draft.", DRAFT), turn("Done.")])
    triage(llm, conn, kb, settings)
    first = llm.requests[0]
    assert "Ticket ID: T-200" in first["messages"][0]["content"]
    assert "at most 6 turns" in first["system"]
    assert len(first["tools"]) == 4
