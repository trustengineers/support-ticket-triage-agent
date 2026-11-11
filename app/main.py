"""FastAPI app with one endpoint: POST /triage.

Run with: uvicorn app.main:app --reload
"""

from __future__ import annotations

import uuid
from typing import Any

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import db
from app.agent import Ticket, run_triage
from app.config import Settings, load_settings
from app.kb import KnowledgeBase
from app.llm import AnthropicLLM, LLMClient, MissingCredentialsError


class TriageRequest(BaseModel):
    """A ticket to triage."""

    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=20_000)
    ticket_id: str | None = Field(default=None, max_length=100, description="Generated when omitted")
    customer: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, description='"A" (default), "B" or a model id')


def create_app(settings: Settings | None = None, llm: LLMClient | None = None, kb: KnowledgeBase | None = None) -> FastAPI:
    """Build the app. Tests pass a scripted `llm`; production uses the Anthropic client."""
    settings = settings or load_settings()
    kb = kb or KnowledgeBase.from_dir(settings.kb_dir)
    app = FastAPI(title="Support Ticket Triage Agent", version="1.0.0")

    def get_llm() -> LLMClient:
        # Created on first use so the app can start (and serve /docs) without credentials.
        nonlocal llm
        if llm is None:
            llm = AnthropicLLM(max_tokens=settings.max_tokens, thinking_display=settings.thinking_display)
        return llm

    @app.post("/triage")
    def triage(request: TriageRequest) -> dict[str, Any]:
        """Run the agent on one ticket and return its decision and every logged step."""
        ticket = Ticket(
            id=request.ticket_id or f"T-{uuid.uuid4().hex[:8]}",
            subject=request.subject,
            body=request.body,
            customer=request.customer,
        )
        conn = db.connect(settings.db_path)
        try:
            result = run_triage(ticket, model=settings.resolve_model(request.model), llm=get_llm(), conn=conn, kb=kb, settings=settings)
        except MissingCredentialsError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise HTTPException(status_code=429, detail="The model provider is rate limiting requests. Try again shortly.") from exc
        except anthropic.APIConnectionError as exc:
            raise HTTPException(status_code=503, detail="Could not reach the model provider.") from exc
        except anthropic.APIStatusError as exc:
            raise HTTPException(status_code=502, detail=f"The model provider returned an error ({exc.status_code}).") from exc
        except anthropic.AnthropicError as exc:
            raise HTTPException(status_code=502, detail="The model client is not configured correctly (check ANTHROPIC_API_KEY).") from exc
        finally:
            conn.close()
        return result.to_dict()

    return app


app = create_app()
