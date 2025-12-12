"""UI-facing controller helpers for options pages."""

from __future__ import annotations

from typing import Any

from app.controller.options_controller import _get_cached_iv_for as _options_get_cached_iv_for


def get_cached_iv_for(*args: Any, **kwargs: Any):
    """Proxy cached IV lookup to the options controller."""
    return _options_get_cached_iv_for(*args, **kwargs)


__all__ = ["get_cached_iv_for"]
