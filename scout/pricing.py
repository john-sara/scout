"""Per-million-token prices in USD, from https://platform.claude.com/docs/en/about-claude/pricing
(checked 2026-10-20). Prices change, so verify before quoting cost numbers.

Haiku 5.5 is priced higher for prompts over 100,000 tokens. Scout's prompts stay far below
that, so only the base rate is used.
"""

PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call. Models not in the table (for example local ones) cost 0."""
    inp, out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * inp + output_tokens * out) / 1_000_000
