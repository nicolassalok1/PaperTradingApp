"""
Domain helpers for the Options model layer (UI-neutral code).
"""

from __future__ import annotations

from app.model.options.core import shared as opt_shared


def _get_cached_iv_for(*args, **kwargs):
    try:
        return opt_shared.get_cached_iv_for(*args, **kwargs)
    except Exception:
        return None


__all__ = [
    "_get_cached_iv_for",
]
