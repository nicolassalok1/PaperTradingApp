"""
Minimal fallback text renderer for options panels.
Provides a short placeholder description to avoid import errors in the UI.
"""


def render_option_text(option_label: str, option_tag: str | None = None) -> str:
    label = option_label or option_tag or "Option"
    return f"Description indisponible pour {label}."


__all__ = ["render_option_text"]
