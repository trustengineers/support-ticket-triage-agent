"""Run tickets through the agent from the terminal and print every step.

Examples:
    python -m app.cli --tickets T-001 T-006 T-012
    python -m app.cli --tickets T-003 --model B
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import anthropic

from app import db
from app.agent import Step, Ticket, TriageResult, run_triage
from app.config import ROOT, load_settings
from app.kb import KnowledgeBase
from app.llm import AnthropicLLM, MissingCredentialsError

RESULT_PREVIEW_CHARS = 260


def load_tickets(path: Path) -> list[dict[str, Any]]:
    """Read the sample tickets file."""
    return json.loads(path.read_text(encoding="utf-8"))


def to_ticket(raw: dict[str, Any]) -> Ticket:
    """Build a Ticket from a sample-file entry (the "expected" label is not passed to the agent)."""
    return Ticket(id=raw["id"], subject=raw["subject"], body=raw["body"], customer=raw.get("customer"))


def _indent(text: str, prefix: str = "           ") -> str:
    return textwrap.indent(textwrap.fill(text, width=96, replace_whitespace=False), prefix).lstrip()


def print_step(step: Step) -> None:
    """Print one agent step: thought, tool, arguments and result."""
    if step.thought:
        print(f"  [step {step.step}] thought: {_indent(step.thought)}")
    if step.tool:
        result = step.result or ""
        if len(result) > RESULT_PREVIEW_CHARS:
            result = result[:RESULT_PREVIEW_CHARS] + "..."
        print(f"  [step {step.step}] tool:    {step.tool}({json.dumps(step.arguments, ensure_ascii=False)})")
        print(f"  [step {step.step}] {'ERROR:  ' if step.is_error else 'result:  '}{result}")


def print_result(result: TriageResult) -> None:
    """Print the outcome and the usage of one run."""
    print(f"  => {result.status.upper()}  category={result.category}  priority={result.priority}  sentiment={result.sentiment}")
    if result.escalation_reason:
        print(f"     escalation: {_indent(result.escalation_reason, '                 ')}")
    if result.reply:
        print("     drafted reply:")
        print(textwrap.indent(result.reply, "       | "))
    cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "n/a (no price configured)"
    print(
        f"     {result.model_calls} model calls, {result.tool_calls} tool calls, "
        f"{result.input_tokens:,} in / {result.output_tokens:,} out tokens, cost {cost}, {result.latency_ms / 1000:.1f}s\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m app.cli`."""
    parser = argparse.ArgumentParser(description="Triage sample tickets and print each agent step.")
    parser.add_argument("--tickets", nargs="+", metavar="ID", help="ticket ids from the tickets file (default: all)")
    parser.add_argument("--model", default="A", help='"A", "B" or a model id (default: A)')
    parser.add_argument("--file", type=Path, default=ROOT / "sample_tickets.json", help="tickets file")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    settings = load_settings()
    model = settings.resolve_model(args.model)
    tickets = load_tickets(args.file)
    if args.tickets:
        by_id = {t["id"]: t for t in tickets}
        missing = [ticket_id for ticket_id in args.tickets if ticket_id not in by_id]
        if missing:
            parser.error(f"unknown ticket id(s): {', '.join(missing)}")
        tickets = [by_id[ticket_id] for ticket_id in args.tickets]

    kb = KnowledgeBase.from_dir(settings.kb_dir)
    conn = db.connect(settings.db_path)
    try:
        llm = AnthropicLLM(max_tokens=settings.max_tokens, thinking_display=settings.thinking_display)
        for raw in tickets:
            print(f"=== {raw['id']} | {model} " + "=" * 40)
            print(f"  subject: {raw['subject']}")
            print(f"  body:    {_indent(raw['body'])}\n")
            result = run_triage(to_ticket(raw), model=model, llm=llm, conn=conn, kb=kb, settings=settings, on_step=print_step)
            print_result(result)
    except MissingCredentialsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except anthropic.AuthenticationError:
        print("The model provider rejected the credentials. Set ANTHROPIC_API_KEY in .env.", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as exc:
        print(f"Could not reach the model provider: {exc}", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as exc:
        print(f"The model provider returned an error ({exc.status_code}): {exc.message}", file=sys.stderr)
        return 1
    except anthropic.AnthropicError as exc:
        print(f"Model client error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
