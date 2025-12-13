"""
Small UI helpers to keep the Streamlit pages consistent.

These are View-only helpers (no business logic), intended to improve UX:
- quickstart sections
- consistent micro-copy blocks
"""

from __future__ import annotations

from typing import Iterable

import streamlit as st


def render_quickstart(title: str, bullets: Iterable[str], *, expanded: bool = False) -> None:
    items = [str(b).strip() for b in bullets if str(b).strip()]
    if not items:
        return
    with st.expander(title, expanded=expanded):
        st.markdown("\n".join([f"- {b}" for b in items]))


__all__ = ["render_quickstart"]

