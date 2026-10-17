from __future__ import annotations

import pytest

from anvil.errors import ConfigError
from anvil.teacher.pricing import PRICES, Usage, cost_usd, price_for


@pytest.mark.parametrize("model", sorted(PRICES))
def test_cost_counts_every_token_kind(model: str) -> None:
    p = PRICES[model]
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=2_000_000,
        cache_read_input_tokens=3_000_000,
        cache_creation_input_tokens=4_000_000,
    )
    expected = p.input + 2 * p.output + 3 * p.cache_read + 4 * 1.25 * p.input
    assert cost_usd(model, usage) == pytest.approx(expected)
    assert p.cache_write == pytest.approx(1.25 * p.input)
    assert cost_usd(model, Usage()) == 0.0


def test_opus_label_price() -> None:
    usage = Usage(input_tokens=1000, output_tokens=50)
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(0.005)


def test_usage_from_response_ignores_missing_and_non_int() -> None:
    usage = Usage.from_response(
        {"input_tokens": 12, "output_tokens": "7", "cache_read_input_tokens": None, "x": 1}
    )
    assert usage == Usage(input_tokens=12)


def test_unknown_model() -> None:
    with pytest.raises(ConfigError, match=r"pricing\.py"):
        price_for("claude-unknown")
    with pytest.raises(ConfigError):
        cost_usd("gpt-whatever", Usage(input_tokens=1))
