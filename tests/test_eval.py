from __future__ import annotations

from pathlib import Path

import eval as evaluation
from app.config import Price, Settings
from tests.fakes import ScriptedLLM, turn

TICKETS = [
    {"id": "T-1", "subject": "Refund", "body": "Refund please", "expected": {"category": "billing", "escalate": False}},
    {"id": "T-2", "subject": "Hacked", "body": "Someone is in my account", "expected": {"category": "account", "escalate": True}},
]


def scripted() -> ScriptedLLM:
    """Classifies every ticket as billing and drafts a reply, so T-1 is right and T-2 is wrong."""

    class PerTicket(ScriptedLLM):
        def complete(self, *, model, system, tools, messages):  # type: ignore[no-untyped-def]
            ticket_id = messages[0]["content"].split("\n")[0].removeprefix("Ticket ID: ")
            if len(messages) == 1:
                return turn(
                    "Classify and reply.",
                    ("classify_ticket", {"category": "billing", "priority": "low", "sentiment": "neutral"}),
                    ("draft_reply", {"ticket_id": ticket_id, "reply_text": "Thanks for reaching out."}),
                    tokens=(1000, 100),
                )
            return turn("Done.", tokens=(1200, 50))

    return PerTicket([])


def test_eval_scores_both_models_and_renders_the_report(kb, tmp_path: Path) -> None:
    settings = Settings(model_a="model-a", model_b="model-b", prices={"model-a": Price(10.0, 50.0)}, db_path=tmp_path / "unused.db")
    rows = evaluation.run_eval(TICKETS, settings, scripted(), kb, tmp_path / "eval.db", workers=2)

    assert list(rows) == ["model-a", "model-b"]
    assert [row.ticket_id for row in rows["model-a"]] == ["T-1", "T-2"]
    summary_a, summary_b = (evaluation.summarize(model, model_rows) for model, model_rows in rows.items())
    assert summary_a["category_accuracy"] == 0.5 and summary_a["escalation_accuracy"] == 0.5
    assert summary_a["avg_tool_calls"] == 2.0
    assert (summary_a["input_tokens"], summary_a["output_tokens"]) == (4400, 300)
    assert summary_a["cost_usd"] == (4400 * 10 + 300 * 50) / 1_000_000
    assert summary_b["cost_usd"] is None  # no price configured for model-b

    report = evaluation.render_report([summary_a, summary_b], rows, settings)
    assert "| `model-a` | 50% | 50% | 2.0 | 4,400 / 300 | $0.0590 |" in report
    assert "| `model-b` | 50% | 50% | 2.0 | 4,400 / 300 | n/a | n/a |" in report
    assert "| T-1 | billing | ✅ billing | ✅ billing |" in report
    assert "| T-2 | account, escalate | ❌ billing | ❌ billing |" in report


def test_update_readme_replaces_only_the_marked_block(tmp_path: Path, monkeypatch) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- eval:start -->\nold table\n<!-- eval:end -->\noutro\n", encoding="utf-8")
    monkeypatch.setattr(evaluation, "README", readme)
    assert evaluation.update_readme("# Evaluation report\n\nDate: today.\n\n| table |\n\n## Per ticket\n\nrows\n")
    assert readme.read_text(encoding="utf-8") == "intro\n<!-- eval:start -->\nDate: today.\n\n| table |\n<!-- eval:end -->\noutro\n"
