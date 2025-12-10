"""
Selector helpers for the Options tab (pure, no Streamlit).
"""

from __future__ import annotations


def choose_option(default_option: str = "Call"):
    """
    Normalize an option selection to (label, char).

    Returns:
        (label, char) where char is 'c' or 'p'.
    """
    label = "Call" if str(default_option).lower().startswith("c") else "Put"
    return label, ("c" if label == "Call" else "p")


def choose_option_select(default_option: str = "Call"):
    """
    Backward-compatible alias to choose_option.
    """
    return choose_option(default_option)
