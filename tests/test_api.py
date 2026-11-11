from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.kb import KnowledgeBase
from app.llm import LLMResponse
from app.main import create_app
from tests.fakes import ScriptedLLM, turn


def make_client(settings: Settings, kb: KnowledgeBase, turns: list[LLMResponse]) -> TestClient:
    return TestClient(create_app(settings=settings, llm=ScriptedLLM(turns), kb=kb))


def test_triage_returns_the_decision_and_the_steps(settings, kb) -> None:
    turns = [
        turn(
            "Classify and search.",
            ("classify_ticket", {"category": "account", "priority": "low", "sentiment": "neutral"}),
            ("search_knowledge_base", {"query": "invitation expired"}),
        ),
        turn("Draft.", ("draft_reply", {"ticket_id": "T-300", "reply_text": "Ask an admin to resend the invitation."})),
        turn("Reply drafted."),
    ]
    response = make_client(settings, kb, turns).post(
        "/triage", json={"ticket_id": "T-300", "subject": "Invite expired", "body": "My invite link expired.", "model": "B"}
    )
    assert response.status_code == 200
    data = response.json()
    assert (data["status"], data["category"], data["model"]) == ("replied", "account", "model-b")
    assert data["reply"] == "Ask an admin to resend the invitation."
    assert [step["tool"] for step in data["steps"]] == ["classify_ticket", "search_knowledge_base", "draft_reply", None]


def test_triage_generates_a_ticket_id_when_missing(settings, kb) -> None:
    response = make_client(settings, kb, [turn("Nothing to do.")]).post("/triage", json={"subject": "Hi", "body": "Thanks!"})
    assert response.status_code == 200
    assert response.json()["ticket_id"].startswith("T-")
    assert response.json()["status"] == "escalated"  # safety net: no reply was drafted


def test_triage_rejects_invalid_requests(settings, kb) -> None:
    client = make_client(settings, kb, [turn("Done.")])
    assert client.post("/triage", json={"subject": "No body"}).status_code == 422
    assert client.post("/triage", json={"subject": "", "body": "x"}).status_code == 422
    assert client.post("/triage", content="not json", headers={"content-type": "application/json"}).status_code == 422
