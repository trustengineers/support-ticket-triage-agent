"""Settings loaded from the environment (.env). Nothing here is secret."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ("billing", "bug", "feature_request", "account", "other")
PRIORITIES = ("low", "medium", "high", "urgent")
SENTIMENTS = ("positive", "neutral", "negative")


@dataclass(frozen=True)
class Price:
    """Price in USD per million tokens."""

    input: float
    output: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return the cost in USD of one run."""
        return (input_tokens * self.input + output_tokens * self.output) / 1_000_000


# List prices per million tokens (input, output). Override or extend in .env with
# MODEL_A_PRICE / MODEL_B_PRICE, e.g. MODEL_A_PRICE=10,50.
KNOWN_PRICES: dict[str, Price] = {
    "claude-fable-5-2": Price(8.00, 40.00),
    "claude-fable-5-1": Price(10.00, 50.00),
    "claude-opus-5-5": Price(4.00, 20.00),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-haiku-5-5": Price(0.10, 0.50),
    "claude-haiku-4-5": Price(1.00, 5.00),
}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the agent, the API and the eval."""

    model_a: str
    model_b: str
    prices: dict[str, Price] = field(default_factory=dict)
    max_steps: int = 6
    max_tokens: int = 8000
    thinking_display: str = "summarized"  # "off" omits the thinking parameter
    db_path: Path = ROOT / "triage.db"
    kb_dir: Path = ROOT / "kb"

    def resolve_model(self, name: str | None) -> str:
        """Map "A"/"B" (or nothing) to a model id; any other value is used as the id."""
        if not name or name.upper() == "A":
            return self.model_a
        if name.upper() == "B":
            return self.model_b
        return name

    def price_for(self, model: str) -> Price | None:
        """Return the configured price for a model, or None when it is unknown."""
        return self.prices.get(model)


def _parse_price(raw: str | None) -> Price | None:
    if not raw:
        return None
    try:
        input_price, output_price = (float(part) for part in raw.split(","))
    except ValueError as exc:
        raise ValueError(f'price must look like "10,50" (input,output per million tokens), got "{raw}"') from exc
    return Price(input_price, output_price)


def load_settings(env_file: Path | None = None) -> Settings:
    """Read settings from the environment, loading .env first if it exists."""
    load_dotenv(env_file or ROOT / ".env")
    model_a = os.getenv("MODEL_A", "claude-fable-5-2")
    model_b = os.getenv("MODEL_B", "claude-haiku-5-5")

    prices = dict(KNOWN_PRICES)
    for model, key in ((model_a, "MODEL_A_PRICE"), (model_b, "MODEL_B_PRICE")):
        price = _parse_price(os.getenv(key))
        if price:
            prices[model] = price

    return Settings(
        model_a=model_a,
        model_b=model_b,
        prices=prices,
        max_steps=int(os.getenv("MAX_STEPS", "6")),
        max_tokens=int(os.getenv("MAX_TOKENS", "8000")),
        thinking_display=os.getenv("THINKING_DISPLAY", "summarized").lower(),
        db_path=Path(os.getenv("DB_PATH", str(ROOT / "triage.db"))),
        kb_dir=Path(os.getenv("KB_DIR", str(ROOT / "kb"))),
    )
