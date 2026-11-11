"""Run every sample ticket through MODEL_A and MODEL_B and compare them.

Usage:
    python eval.py                 # all tickets, both models
    python eval.py --limit 5       # first 5 tickets (a cheap smoke run)

Writes results/report.md and prints the same markdown.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import anthropic

from app import db
from app.agent import run_triage
from app.cli import load_tickets, to_ticket
from app.config import ROOT, Settings, load_settings
from app.kb import KnowledgeBase
from app.llm import AnthropicLLM, LLMClient, MissingCredentialsError

RESULTS_DIR = ROOT / "results"
README = ROOT / "README.md"
README_START, README_END = "<!-- eval:start -->", "<!-- eval:end -->"


@dataclass(frozen=True)
class Row:
    """The outcome of one ticket on one model."""

    ticket_id: str
    model: str
    expected_category: str
    expected_escalate: bool
    category: str | None = None
    escalated: bool = False
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: int = 0
    error: str | None = None

    @property
    def category_correct(self) -> bool:
        return self.error is None and self.category == self.expected_category

    @property
    def escalation_correct(self) -> bool:
        return self.error is None and self.escalated == self.expected_escalate


def evaluate_ticket(raw: dict[str, Any], model: str, llm: LLMClient, kb: KnowledgeBase, settings: Settings, db_path: Path) -> Row:
    """Run one ticket on one model. An API failure is recorded on the row instead of stopping the eval."""
    expected = raw["expected"]
    base = {"ticket_id": raw["id"], "model": model, "expected_category": expected["category"], "expected_escalate": expected["escalate"]}
    conn = db.connect(db_path)
    try:
        result = run_triage(to_ticket(raw), model=model, llm=llm, conn=conn, kb=kb, settings=settings)
    except anthropic.AnthropicError as exc:
        return Row(**base, error=f"{type(exc).__name__}: {exc}"[:200])
    finally:
        conn.close()
    return Row(
        **base,
        category=result.category,
        escalated=result.status == "escalated",
        tool_calls=result.tool_calls,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )


def summarize(model: str, rows: list[Row]) -> dict[str, Any]:
    """Aggregate one model's rows into the numbers shown in the summary table."""
    done = [row for row in rows if row.error is None]
    count = len(rows)
    costs = [row.cost_usd for row in done]
    return {
        "model": model,
        "tickets": count,
        "errors": count - len(done),
        "category_accuracy": sum(row.category_correct for row in rows) / count if count else 0.0,
        "escalation_accuracy": sum(row.escalation_correct for row in rows) / count if count else 0.0,
        "avg_tool_calls": sum(row.tool_calls for row in done) / len(done) if done else 0.0,
        "input_tokens": sum(row.input_tokens for row in done),
        "output_tokens": sum(row.output_tokens for row in done),
        "cost_usd": None if any(cost is None for cost in costs) else sum(costs),  # type: ignore[arg-type]
        "avg_latency_s": sum(row.latency_ms for row in done) / len(done) / 1000 if done else 0.0,
    }


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def render_report(summaries: list[dict[str, Any]], rows_by_model: dict[str, list[Row]], settings: Settings) -> str:
    """Build the markdown report: a summary table and a per-ticket table."""
    lines = [
        "# Evaluation report",
        "",
        f"Date: {date.today().isoformat()}. Tickets: {summaries[0]['tickets']} from `sample_tickets.json`. Step cap: {settings.max_steps}.",
        "",
        "| Model | Category accuracy | Escalation accuracy | Avg tool calls | Total tokens (in / out) | Total cost | Cost per ticket | Avg latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        per_ticket = None if s["cost_usd"] is None else s["cost_usd"] / max(s["tickets"] - s["errors"], 1)
        lines.append(
            f"| `{s['model']}` | {s['category_accuracy']:.0%} | {s['escalation_accuracy']:.0%} | {s['avg_tool_calls']:.1f} | "
            f"{s['input_tokens']:,} / {s['output_tokens']:,} | {_money(s['cost_usd'])} | {_money(per_ticket)} | {s['avg_latency_s']:.1f}s |"
        )
    lines.append("")
    for s in summaries:
        price = settings.price_for(s["model"])
        note = f"${price.input:g} input / ${price.output:g} output per million tokens" if price else "no price configured, so cost is n/a"
        errors = f" {s['errors']} run(s) failed and count as wrong." if s["errors"] else ""
        lines.append(f"- `{s['model']}`: {note}.{errors}")

    models = list(rows_by_model)
    lines += ["", "## Per ticket", "", "| Ticket | Expected | " + " | ".join(f"`{m}`" for m in models) + " |", "|---|---|" + "---|" * len(models)]
    for index, first in enumerate(rows_by_model[models[0]]):
        expected = f"{first.expected_category}{', escalate' if first.expected_escalate else ''}"
        cells = []
        for model in models:
            row = rows_by_model[model][index]
            if row.error:
                cells.append("error")
                continue
            got = f"{row.category}{', escalate' if row.escalated else ''}"
            cells.append(f"{'✅' if row.category_correct and row.escalation_correct else '❌'} {got}")
        lines.append(f"| {first.ticket_id} | {expected} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def update_readme(report: str) -> bool:
    """Replace the eval table in README.md (between the eval markers) with the latest summary."""
    if not README.exists():
        return False
    text = README.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start == -1 or end == -1:
        return False
    # Everything between the report's title and its per-ticket section.
    summary = report.split("## Per ticket")[0].split("\n", 2)[2].strip()
    README.write_text(text[: start + len(README_START)] + "\n" + summary + "\n" + text[end:], encoding="utf-8", newline="\n")
    return True


def run_eval(tickets: list[dict[str, Any]], settings: Settings, llm: LLMClient, kb: KnowledgeBase, db_path: Path, workers: int) -> dict[str, list[Row]]:
    """Run all tickets on both models. Tickets run in parallel; row order follows the tickets file."""
    rows_by_model: dict[str, list[Row]] = {}
    for model in dict.fromkeys((settings.model_a, settings.model_b)):
        print(f"Running {len(tickets)} tickets on {model}...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows_by_model[model] = list(pool.map(lambda raw: evaluate_ticket(raw, model, llm, kb, settings, db_path), tickets))
    return rows_by_model


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Compare MODEL_A and MODEL_B on the sample tickets.")
    parser.add_argument("--limit", type=int, help="only the first N tickets")
    parser.add_argument("--workers", type=int, default=4, help="tickets run in parallel (default 4)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    settings = load_settings()
    tickets = load_tickets(ROOT / "sample_tickets.json")[: args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)
    db_path = RESULTS_DIR / "eval.db"
    db_path.unlink(missing_ok=True)

    llm = AnthropicLLM(max_tokens=settings.max_tokens, thinking_display=settings.thinking_display)
    try:
        rows_by_model = run_eval(tickets, settings, llm, KnowledgeBase.from_dir(settings.kb_dir), db_path, args.workers)
    except MissingCredentialsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    summaries = [summarize(model, rows) for model, rows in rows_by_model.items()]
    report = render_report(summaries, rows_by_model, settings)
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved {RESULTS_DIR / 'report.md'}", file=sys.stderr)
    if args.limit is None and update_readme(report):
        print("Updated the eval table in README.md", file=sys.stderr)
    return 1 if any(s["errors"] == s["tickets"] for s in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
