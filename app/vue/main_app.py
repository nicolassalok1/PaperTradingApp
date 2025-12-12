"""
PaperTradingApp Streamlit entrypoint.
Configures the page, builds top-level navigation, and dispatches to page renderers.
"""

from typing import Callable, Dict
from pathlib import Path
import importlib
import pkgutil
import sys

import pandas as pd
import streamlit as st
import altair as alt

alt.themes.enable("dark")

import plotly.io as pio

pio.templates.default = "plotly_dark"

from app.vue.tabs.tab_dashboard_v2 import render_tab as render_dashboard_v2
from app.vue.tabs.tab_trading import render_tab as render_trading
from app.vue.tabs.tab_portfolio_and_risk import render_tab as render_portfolio_and_risk
from app.vue.tabs.tab_hedging_systems import render_tab as render_hedging_systems

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Top‑level grouping and ordering (user‑visible labels)
TAB_GROUPS = {
    "📊 Overview": [
        "📊 Dashboard v2",
    ],
    "📈 Trading": [
        "📈 Options",
        "💹 Trading",
        "📊 Portfolio & Risk",
        "🛡️ Hedging Systems",
    ],
    "🧮 Models": [
        "🧮 Yield Curve",
        "🧮 Calibration",
    ],
}


# Explicit labels for tab modules (override TAB_LABEL when present)
DEFAULT_LABEL_OVERRIDES: Dict[str, str] = {
    "tab_dashboard_v2": "📊 Dashboard v2",
    "tab_options": "📈 Options",
    "tab_trading": "💹 Trading",
    "tab_portfolio_and_risk": "📊 Portfolio & Risk",
    "tab_hedging_systems": "🛡️ Hedging Systems",
    "tab_yieldcurve": "🧮 Yield Curve",
    "tab_calibration": "🧮 Calibration",
    # Internal trading sub‑tabs (kept inside 💹 Trading only)
    "tab_alpaca_spot": "Alpaca Spot",
    "tab_alpaca_orders": "Advanced Orders",
    "tab_alpaca_options_trading": "Alpaca Options",
}


# Labels that should never appear as top‑level tabs
EXCLUDED_LABELS: set[str] = {
    "Alpaca Spot",
    "Advanced Orders",
    "Alpaca Options",
}


def _derive_label(module_name: str) -> str:
    base = module_name.removeprefix("tab_")
    parts = base.split("_")
    return " ".join(p.capitalize() for p in parts if p)


def autodiscover_tabs() -> Dict[str, Callable[[], None]]:
    tabs_dir = Path(__file__).resolve().parent / "tabs"
    tab_map: Dict[str, Callable[[], None]] = {}

    for module_info in pkgutil.iter_modules([str(tabs_dir)]):
        if not module_info.name.startswith("tab_"):
            continue
        module = importlib.import_module(f"app.vue.tabs.{module_info.name}")

        # Prefer explicit overrides, then TAB_LABEL, then a derived name
        label = DEFAULT_LABEL_OVERRIDES.get(module_info.name)
        if not label:
            label = getattr(module, "TAB_LABEL", None)
        if not label:
            label = _derive_label(module_info.name)

        render_fn = getattr(module, "render_tab", None)
        if not callable(render_fn):
            render_fn = getattr(module, "render", None)
        if not callable(render_fn):
            render_fn = getattr(module, "run_dashboard", None)

        if callable(render_fn):
            tab_map[label] = render_fn

    return tab_map


def ordered_tab_labels(all_tabs: Dict[str, Callable[[], None]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    # Respect explicit group ordering first
    for _, labels in TAB_GROUPS.items():
        for lbl in labels:
            if lbl in EXCLUDED_LABELS:
                continue
            if lbl in all_tabs and lbl not in seen:
                ordered.append(lbl)
                seen.add(lbl)

    # Add any remaining tabs (non‑grouped) in alpha order
    for lbl in sorted(all_tabs.keys()):
        if lbl in EXCLUDED_LABELS:
            continue
        if lbl not in seen:
            ordered.append(lbl)
            seen.add(lbl)

    return ordered


def sidebar_menu(all_tabs: Dict[str, Callable[[], None]], tab_labels: list[str]) -> None:
    """Sidebar kept for branding/info; navigation handled by tabs only."""
    st.sidebar.markdown("### PaperTradingApp")
    st.sidebar.info(
        "Navigation via les onglets en haut. "
        "La barre latérale reste disponible pour d'éventuels réglages."
    )


ALL_TABS = autodiscover_tabs()
# Fallback registration if autodiscovery misses key tabs
if "📊 Dashboard v2" not in ALL_TABS:
    ALL_TABS["📊 Dashboard v2"] = render_dashboard_v2
if "💹 Trading" not in ALL_TABS:
    ALL_TABS["💹 Trading"] = render_trading
if "📊 Portfolio & Risk" not in ALL_TABS:
    ALL_TABS["📊 Portfolio & Risk"] = render_portfolio_and_risk
if "🛡️ Hedging Systems" not in ALL_TABS:
    ALL_TABS["🛡️ Hedging Systems"] = render_hedging_systems


def _configure_page() -> None:
    """Global Streamlit page configuration."""
    try:
        st.set_page_config(
            page_title="AI Trading Bot",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
    except Exception:
        # set_page_config must be called once; ignore if already set
        pass


def _inject_global_styles() -> None:
    theme_path = Path(__file__).parent / "styles" / "theme_animated.css"
    if theme_path.exists():
        with open(theme_path, "r", encoding="utf-8") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)


def _patch_streamlit_charts() -> None:
    """Ensure charts are non-scrollable by default (Plotly)."""
    if getattr(st, "_charts_patched", False):
        return

    _orig_plotly_chart = st.plotly_chart

    def _plotly_chart(fig, use_container_width: bool = True, **kwargs):
        cfg = kwargs.pop("config", {}) or {}
        base_cfg = {"scrollZoom": False, "displayModeBar": False}
        merged_cfg = {**base_cfg, **cfg}
        return _orig_plotly_chart(
            fig, use_container_width=use_container_width, config=merged_cfg, **kwargs
        )

    st.plotly_chart = _plotly_chart  # type: ignore[assignment]
    st._charts_patched = True  # type: ignore[attr-defined]


def _arrow_safe_df(df):
    """Ensure DataFrame is Arrow/Streamlit friendly (e.g., UUID -> str)."""
    if not isinstance(df, pd.DataFrame):
        return df
    df_safe = df.copy()
    for col in df_safe.columns:
        if df_safe[col].dtype == object:
            df_safe[col] = df_safe[col].apply(
                lambda x: x
                if isinstance(x, (str, int, float, bool)) or x is None
                else str(x)
            )
    return df_safe


def _patch_streamlit_dataframe() -> None:
    """Wrap st.dataframe to coerce non-Arrow-friendly objects to strings."""
    if getattr(st, "_dataframe_patched", False):
        return

    _orig_df = st.dataframe

    def _dataframe(data, *args, **kwargs):
        data_safe = _arrow_safe_df(data) if isinstance(data, pd.DataFrame) else data
        return _orig_df(data_safe, *args, **kwargs)

    st.dataframe = _dataframe  # type: ignore[assignment]
    st._dataframe_patched = True  # type: ignore[attr-defined]


def main() -> None:
    _configure_page()
    _inject_global_styles()
    _patch_streamlit_charts()
    _patch_streamlit_dataframe()

    tab_labels = ordered_tab_labels(ALL_TABS)
    if not tab_labels:
        st.error("Aucun onglet disponible.")
        return

    sidebar_menu(ALL_TABS, tab_labels)

    def _render_tab(label: str) -> None:
        render_fn = ALL_TABS.get(label)
        if not render_fn:
            st.error("Onglet introuvable.")
            return
        try:
            render_fn()
        except Exception as exc:  # defensive: avoid blank page if a tab crashes
            st.error(f"Onglet '{label}' non rendu : {exc}")
            st.exception(exc)

    tabs = st.tabs(tab_labels)
    for label, tab in zip(tab_labels, tabs):
        with tab:
            _render_tab(label)


if __name__ == "__main__":
    main()

