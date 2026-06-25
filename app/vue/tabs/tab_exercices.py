"""
Top-level "Exercices" tab — a registry of interactive quant mini-studies.

Designed to be extended: append a new entry to ``EXERCISES`` (key, label, render
callable) and it becomes a sub-tab automatically. The first exercise is the
SPX/VIX Portfolio Allocation book.
"""
from __future__ import annotations

from typing import Callable

import streamlit as st

from app.vue.components.exercises.portfolio_allocation import render as render_portfolio_allocation
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "🧪 Exercices"


# Extensible registry — add exercises here, no other wiring needed.
EXERCISES: list[dict[str, object]] = [
    {
        "key": "portfolio_allocation",
        "label": "Portfolio Allocation",
        "render": render_portfolio_allocation,
    },
]


def render_tab() -> None:
    render_page_header(
        "Exercices",
        "Mini-études quant interactives, branchées sur les moteurs validés du repo.",
        icon="🧪",
        badge="Quant",
    )

    if not EXERCISES:
        st.info("Aucun exercice disponible pour l'instant.")
        return

    labels = [str(ex["label"]) for ex in EXERCISES]
    sub_tabs = st.tabs(labels)
    for ex, sub_tab in zip(EXERCISES, sub_tabs):
        with sub_tab:
            render_fn = ex["render"]
            if callable(render_fn):
                render_fn()


def render() -> None:
    render_tab()
