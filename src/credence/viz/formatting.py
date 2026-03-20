"""Model name formatting and color helpers."""

import re

from credence.core.util import provider, short_model
from credence.viz.constants import CHINESE_PROVIDERS, PROVIDER_COLORS

# Re-export for existing imports
__all__ = ["provider", "short_model", "provider_color", "is_chinese_model", "truncate", "natural_sort_key", "unslugify"]


def provider_color(model_id: str) -> str:
    """Get the color for a model based on its provider."""
    return PROVIDER_COLORS.get(provider(model_id), "#888888")


def is_chinese_model(model_id: str) -> bool:
    """Check if a model belongs to a Chinese provider."""
    return provider(model_id) in CHINESE_PROVIDERS


def truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if it exceeds max_len."""
    return text[:max_len] + "..." if len(text) > max_len else text


def natural_sort_key(s):
    """Sort key that handles embedded numbers naturally (1, 2, 10 not 1, 10, 2)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]


def model_sort_key(model_id: str) -> tuple[str, int, str]:
    """Sort key that groups by provider and orders smaller models before larger ones."""
    name = short_model(model_id).lower()
    prov = provider(model_id)

    # Assign a size tier within provider: 0 = small, 1 = mid, 2 = large
    if any(tok in name for tok in ("mini", "flash", "haiku", "small")):
        tier = 0
    elif any(tok in name for tok in ("sonnet", "pro",)):
        tier = 1
    else:
        tier = 2

    return (prov, tier, name)


def unslugify(slug: str) -> str:
    """Convert a snake_case slug to Title Case (e.g. 'frontier_natural_science' → 'Frontier Natural Science')."""
    return slug.replace("_", " ").title()
