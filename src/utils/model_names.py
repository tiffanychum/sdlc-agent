"""
Model-name normalisation.

Different LLM providers, the Poe-compatible OpenAI gateway, and our own
config files spell the same model in subtly different ways:

    claude-sonnet-4.6                    (canonical, from .env)
    claude-sonnet-4-6                    (Poe API form — dot becomes dash)
    Claude-Sonnet-4-6                    (mixed case from response headers)
    claude-sonnet-4-6claude-sonnet-4-6   (Poe occasionally doubles the
                                          model id when it streams back
                                          a tool/thinking response)
    openai/gpt-5                         (provider-prefixed)

Without a canonicalisation step the dashboard ends up with multiple
"different" rows (with different costs, token counts and call volumes)
that should have been a single aggregated row.

The normaliser is intentionally **conservative**:
    - it only collapses names that are demonstrably the same model
    - it never invents a vendor prefix or strips information beyond
      casing / punctuation
    - it returns the input unchanged when nothing matches
"""

from __future__ import annotations

import re

# Pattern matches a trailing "-MAJOR-MINOR" version suffix, e.g. "-4-6".
# We rewrite it to "-MAJOR.MINOR" so claude-sonnet-4-6 == claude-sonnet-4.6.
_TRAILING_DASH_VERSION = re.compile(r"-(\d+)-(\d+)$")


def normalize_model_name(name: str | None) -> str:
    """Return a canonical model identifier suitable for grouping/aggregation.

    Examples:
        >>> normalize_model_name("claude-sonnet-4-6")
        'claude-sonnet-4.6'
        >>> normalize_model_name("claude-sonnet-4-6claude-sonnet-4-6")
        'claude-sonnet-4.6'
        >>> normalize_model_name("Claude-Sonnet-4-6")
        'claude-sonnet-4.6'
        >>> normalize_model_name("openai/gpt-5")
        'gpt-5'
        >>> normalize_model_name("")
        ''
        >>> normalize_model_name(None)
        ''
    """
    if not name:
        return ""
    n = name.strip().lower()

    # 1) Strip provider prefix (e.g. "openai/gpt-5" → "gpt-5") so the
    #    grouping is based on the underlying model, not the gateway.
    if "/" in n:
        n = n.split("/")[-1]

    # 2) Collapse exact duplicates ("claude-sonnet-4-6claude-sonnet-4-6"
    #    → "claude-sonnet-4-6"). We do this BEFORE the dot-rewrite so the
    #    even-length test is reliable.
    half = len(n) // 2
    if half > 0 and len(n) % 2 == 0 and n[:half] == n[half:]:
        n = n[:half]

    # 3) Convert trailing "-MAJOR-MINOR" → "-MAJOR.MINOR" so dot- and
    #    dash-separated versions of the same model collapse.
    n = _TRAILING_DASH_VERSION.sub(r"-\1.\2", n)

    return n


__all__ = ["normalize_model_name"]
