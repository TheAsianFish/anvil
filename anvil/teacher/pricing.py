"""Token prices, used to account for every teacher call in dollars.

Prices are USD per million tokens from Anthropic's public price list (checked 2026-10-16).
Haiku 5.5 has a second rate card above 100K prompt tokens; label prompts are far below it.
A model missing from this table is refused rather than silently costed at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvil.errors import ConfigError


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        """Five-minute cache writes are billed at 1.25x the input price."""
        return self.input * 1.25


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01),
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_response(cls, usage: dict[str, object]) -> Usage:
        def n(key: str) -> int:
            value = usage.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) else 0

        return cls(
            input_tokens=n("input_tokens"),
            output_tokens=n("output_tokens"),
            cache_read_input_tokens=n("cache_read_input_tokens"),
            cache_creation_input_tokens=n("cache_creation_input_tokens"),
        )


def price_for(model: str) -> Price:
    try:
        return PRICES[model]
    except KeyError:
        known = ", ".join(sorted(PRICES))
        raise ConfigError(
            f"no price for teacher model {model!r} (known: {known}); add it to"
            " anvil/teacher/pricing.py so its calls can be accounted"
        ) from None


def cost_usd(model: str, usage: Usage) -> float:
    """Dollar cost of one response. Output tokens include any thinking tokens."""
    p = price_for(model)
    return (
        usage.input_tokens * p.input
        + usage.output_tokens * p.output
        + usage.cache_read_input_tokens * p.cache_read
        + usage.cache_creation_input_tokens * p.cache_write
    ) / 1_000_000
