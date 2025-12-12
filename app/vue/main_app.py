"""
PaperTradingApp Streamlit entrypoint.
Configures the page, builds top-level navigation, and dispatches to page renderers.
"""

from typing import Callable, Dict
from pathlib import Path
import importlib
import pkgutil
import sys

import streamlit as st
import altair as alt
alt.themes.enable("dark")

import plotly.io as pio
pio.templates.default = "plotly_dark"

from app.vue.tabs.tab_portfolio_allocation import render_tab as render_portfolio_allocation
from app.vue.tabs.tab_hedger_v2 import render_tab as render_hedger_v2
from app.vue.tabs.tab_dashboard_v2 import render_tab as render_dashboard_v2
from app.vue.tabs.tab_hedger_rl_live_v2 import render_tab as render_hedger_rl_live_v2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TAB_GROUPS = {
    "📊 Analyse": [
        "📊 Dashboard v2",
        "🧮 Yield Curve",
        "🧱 Risk Management",
        "🧪 Trading Systems / Backtest",
        "🧮 Calibration",
    ],
    "📈 Trading": [
        "🛒 Buy/Sell",
        "💹 Advanced Orders",
        "📐 Portfolio Allocation",
        "🛡️ Hedger v2",
        "🤖 Hedger RL Live v2",
        "🛡️ Hedger",
    ],
    "🧰 Maintenance": [
        "📊 Dashboard",
    ],
}

DEFAULT_LABEL_OVERRIDES: Dict[str, str] = {
    "tab_dashboard": "📊 Dashboard",
    "tab_dashboard_v2": "📊 Dashboard v2",
    "tab_buy_sell": "🛒 Buy/Sell",
    "tab_portfolio": "💼 Portfolio",
    "tab_options": "📈 Options",
    "tab_hedger": "🛡️ Hedger",
    "tab_yieldcurve": "🧮 Yield Curve",
    "tab_backtest": "🧪 Trading Systems / Backtest",
    "tab_calibration": "🧮 Calibration",
    "tab_alpaca_orders": "💹 Advanced Orders",
    "tab_risk_management": "🧱 Risk Management",
    "tab_portfolio_allocation": "📐 Portfolio Allocation",
    "tab_hedger_v2": "🛡️ Hedger v2",
    "tab_hedger_rl_live_v2": "🤖 Hedger RL Live v2",
}

EXCLUDED_LABELS = {
    "🛡️ Hedger",
    "🛒 Buy/Sell",
    "🧪 Trading Systems / Backtest",
    "📊 Dashboard",
    "💼 Portfolio",
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

        label = getattr(module, "TAB_LABEL", None)
        if not label:
            label = DEFAULT_LABEL_OVERRIDES.get(module_info.name)
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
    seen = set()
    for _, labels in TAB_GROUPS.items():
        for lbl in labels:
            if lbl in EXCLUDED_LABELS:
                continue
            if lbl in all_tabs and lbl not in seen:
                ordered.append(lbl)
                seen.add(lbl)
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
    st.sidebar.info("Navigation via les onglets en haut. La barre latérale reste disponible pour d'éventuels réglages.")


ALL_TABS = autodiscover_tabs()
# Fallback registration if autodiscovery misses the new tab
if "📐 Portfolio Allocation" not in ALL_TABS:
    ALL_TABS["📐 Portfolio Allocation"] = render_portfolio_allocation
if "🛡️ Hedger v2" not in ALL_TABS:
    ALL_TABS["🛡️ Hedger v2"] = render_hedger_v2
if "📊 Dashboard v2" not in ALL_TABS:
    ALL_TABS["📊 Dashboard v2"] = render_dashboard_v2
if "🤖 Hedger RL Live v2" not in ALL_TABS:
    ALL_TABS["🤖 Hedger RL Live v2"] = render_hedger_rl_live_v2


def _configure_page() -> None:
    """Global Streamlit page configuration."""
    try:
        st.set_page_config(
            page_title="AI Trading Bot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
    except Exception:
        # set_page_config must be called once; ignore if already set
        pass


def _inject_global_styles() -> None:
    from pathlib import Path
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
        return _orig_plotly_chart(fig, use_container_width=use_container_width, config=merged_cfg, **kwargs)

    st.plotly_chart = _plotly_chart  # type: ignore[assignment]
    st._charts_patched = True  # type: ignore[attr-defined]


def main() -> None:
    _configure_page()
    _inject_global_styles()
    _patch_streamlit_charts()

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
