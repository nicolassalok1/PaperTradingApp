"""
PaperTradingApp Streamlit entrypoint.
Configures the page, builds top-level navigation, and dispatches to page renderers.
"""

from typing import Any, Callable, Dict
from pathlib import Path
import importlib
import pkgutil
import sys
import time

import pandas as pd
import streamlit as st
import altair as alt

alt.themes.enable("dark")

import plotly.io as pio

pio.templates.default = "plotly_dark"

from app.controller import trading_controller as trading_ctrl

from app.vue.tabs.tab_dashboard_v2 import render_tab as render_dashboard_v2
from app.vue.tabs.tab_trading import render_tab as render_trading
from app.vue.tabs.tab_portfolio_and_risk import render_tab as render_portfolio_and_risk
from app.vue.tabs.tab_hedging_systems import render_tab as render_hedging_systems
from app.vue.tabs.tab_bots import render_tab as render_bots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Top-level grouping and ordering (user-visible labels)
TAB_GROUPS = {
    "🏠 Vue d’ensemble": [
        "📊 Dashboard",
        "🧭 Portefeuille & Risque",
    ],
    "💹 Trading": [
        "💹 Trading",
        "🛡️ Hedging Systems",
    ],
    "🧩 Modèles": [
        "🧮 Yield Curve",
        "🧪 Calibration avancée",
        "📈 Options",
    ],
    "🧰 Outils": [
        "🤖 Bots",
    ],
}

# Explicit labels for tab modules (override TAB_LABEL when present)
DEFAULT_LABEL_OVERRIDES: Dict[str, str] = {
    "tab_dashboard_v2": "📊 Dashboard",
    "tab_options": "📈 Options",
    "tab_trading": "💹 Trading",
    "tab_portfolio_and_risk": "🧭 Portefeuille & Risque",
    "tab_hedging_systems": "🛡️ Hedging Systems",
    "tab_yieldcurve": "🧮 Yield Curve",
    "tab_advanced_calibration": "🧪 Calibration avancée",
    "tab_bots": "🤖 Bots",
    # Internal trading sub-tabs (kept inside 💹 Trading only)
    "tab_alpaca_spot": "Alpaca Spot",
    "tab_alpaca_orders": "Advanced Orders",
    "tab_alpaca_options": "Alpaca Options",
}


# Labels that should never appear as top-level tabs
EXCLUDED_LABELS: set[str] = {
    "Alpaca Spot",
    "Advanced Orders",
    "Alpaca Options",
}

_ALPACA_STATUS_STATE_KEY = "alpaca_boot_status"
_ALPACA_STATUS_TS_KEY = "alpaca_boot_status_ts"
_ALPACA_STATUS_TTL_SEC = 60.0


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

    # Intentionally do not expose non-grouped tabs at the top-level.
    return ordered


def sidebar_menu(all_tabs: Dict[str, Callable[[], None]], tab_labels: list[str]) -> None:
    """Sidebar kept for branding/info; navigation handled by tabs only."""
    st.sidebar.markdown("### PaperTradingApp")
    st.sidebar.caption("Interface de paper trading orientée modèles (MVC).")
    st.sidebar.info("Navigation via les onglets en haut. La barre latérale sert d’aide et de réglages.")
    with st.sidebar.expander("Guide rapide", expanded=False):
        st.markdown(
            "- Commence par `📊 Dashboard` pour vérifier l’état du compte.\n"
            "- Utilise `🧮 Yield Curve` si tu veux un `r(T)` cohérent.\n"
            "- Calibre une surface dans `🧪 Calibration avancée`, puis envoie-la vers `📈 Options`.\n"
            "- Tout ce qui envoie des ordres Alpaca est volontairement explicite et peut être dangereux."
        )


ALL_TABS = autodiscover_tabs()
# Fallback registration if autodiscovery misses key tabs
if "📊 Dashboard" not in ALL_TABS:
    ALL_TABS["📊 Dashboard"] = render_dashboard_v2
if "💹 Trading" not in ALL_TABS:
    ALL_TABS["💹 Trading"] = render_trading
if "🧭 Portefeuille & Risque" not in ALL_TABS:
    ALL_TABS["🧭 Portefeuille & Risque"] = render_portfolio_and_risk
if "🛡️ Hedging Systems" not in ALL_TABS:
    ALL_TABS["🛡️ Hedging Systems"] = render_hedging_systems
if "🤖 Bots" not in ALL_TABS:
    ALL_TABS["🤖 Bots"] = render_bots


def _configure_page() -> None:
    """Global Streamlit page configuration."""
    try:
        st.set_page_config(
            page_title="PaperTradingApp",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
    except Exception:
        # set_page_config must be called once; ignore if already set
        pass


def _inject_global_styles() -> None:
    theme_path = Path(__file__).parent / "styles" / "theme_animated.css"
    if theme_path.exists():
        if hasattr(st, "html"):
            try:
                st.html(theme_path)  # type: ignore[arg-type,attr-defined]
                return
            except Exception:
                pass

        with open(theme_path, "r", encoding="utf-8") as f:  # pragma: no cover
            css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)  # pragma: no cover


def _patch_streamlit_charts() -> None:
    """Ensure charts are non-scrollable by default (Plotly)."""
    if getattr(st, "_charts_patched", False):
        return

    _orig_plotly_chart = st.plotly_chart

    def _plotly_chart(fig, width="stretch", **kwargs):
        cfg = kwargs.pop("config", {}) or {}
        base_cfg = {"scrollZoom": False, "displayModeBar": False}
        merged_cfg = {**base_cfg, **cfg}
        return _orig_plotly_chart(
            fig, width=width, config=merged_cfg, **kwargs
        )

    st.plotly_chart = _plotly_chart  # type: ignore[assignment]
    st._charts_patched = True  # type: ignore[attr-defined]


def _arrow_safe_df(df):
    """Ensure DataFrame is Arrow/Streamlit friendly (e.g., UUID -> str)."""
    if not isinstance(df, pd.DataFrame):
        return df
    df_safe = df.copy()
    # Streamlit/Arrow warns when column names are of mixed type and silently coerces
    # them to str ("...not roundtrip correctly"). Normalise here (display-only copy)
    # so the table converts cleanly and quietly.
    if len({type(c) for c in df_safe.columns}) > 1:
        df_safe.columns = [str(c) for c in df_safe.columns]
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


def _coerce_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    try:
        return bool(val)
    except Exception:
        return None


def _format_clock_dt(val: Any) -> str:
    if val is None or val == "":
        return ""
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M")
    except Exception:
        try:
            return str(val)
        except Exception:
            return ""


def _load_alpaca_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "error": None,
        "clock_error": None,
        "market_open": None,
        "next_open": "",
        "next_close": "",
        "trading_enabled": None,
        "blocked_reasons": [],
        "account_status": "",
    }

    try:
        account = trading_ctrl.get_orders_account()
    except Exception as exc:
        status["error"] = str(exc)
        return status

    if not account:
        status["error"] = "Aucune donnee de compte Alpaca retournee."
        return status

    status["account_status"] = str(account.get("status") or "")
    blocked_reasons: list[str] = []
    for field, label in [
        ("trading_blocked", "trading bloque"),
        ("account_blocked", "compte bloque"),
        ("trade_suspended_by_user", "trading suspendu par l'utilisateur"),
    ]:
        if _coerce_bool(account.get(field)):
            blocked_reasons.append(label)

    account_status_norm = status["account_status"].strip().upper()
    if account_status_norm and account_status_norm not in {"ACTIVE"}:
        blocked_reasons.append(f"statut compte = {account_status_norm}")

    status["blocked_reasons"] = blocked_reasons
    status["trading_enabled"] = False if blocked_reasons else True

    try:
        clock = trading_ctrl.get_orders_clock()
    except Exception as exc:
        clock = None
        status["clock_error"] = str(exc)

    status["market_open"] = _coerce_bool((clock or {}).get("is_open"))
    status["next_open"] = _format_clock_dt((clock or {}).get("next_open"))
    status["next_close"] = _format_clock_dt((clock or {}).get("next_close"))
    return status


def _get_alpaca_status_cached() -> dict[str, Any]:
    now = time.time()
    ts = st.session_state.get(_ALPACA_STATUS_TS_KEY)
    cached = st.session_state.get(_ALPACA_STATUS_STATE_KEY)

    try:
        ts_f = float(ts) if ts is not None else None
    except Exception:
        ts_f = None

    if isinstance(cached, dict) and ts_f is not None and (now - ts_f) < _ALPACA_STATUS_TTL_SEC:
        return cached

    status = _load_alpaca_status()
    st.session_state[_ALPACA_STATUS_STATE_KEY] = status
    st.session_state[_ALPACA_STATUS_TS_KEY] = now
    return status


def _render_alpaca_status_banner() -> None:
    status = _get_alpaca_status_cached()
    error = status.get("error")
    if error:
        st.warning(f"Verification Alpaca impossible : {error}")
        return

    market_open = status.get("market_open")
    trading_enabled = status.get("trading_enabled")
    blocked_reasons = status.get("blocked_reasons") or []
    clock_error = status.get("clock_error")

    status_bits = []
    if market_open is True:
        status_bits.append("Marche US ouvert")
    elif market_open is False:
        status_bits.append("Marche US ferme")
    else:
        status_bits.append("Etat du marche inconnu")

    if trading_enabled is True:
        status_bits.append("trading Alpaca autorise")
    elif trading_enabled is False:
        reason = "; ".join(blocked_reasons) if blocked_reasons else "trading Alpaca bloque"
        status_bits.append(reason)
    else:
        status_bits.append("capacite de trading inconnue")

    detail_parts = []
    next_open = status.get("next_open")
    next_close = status.get("next_close")
    if market_open is False and next_open:
        detail_parts.append(f"prochaine ouverture: {next_open}")
    if market_open is True and next_close:
        detail_parts.append(f"prochaine fermeture: {next_close}")
    if clock_error and market_open is None:
        detail_parts.append(f"horloge Alpaca indisponible ({clock_error})")

    body = " ; ".join(status_bits)
    if detail_parts:
        body = f"{body} ({' - '.join(detail_parts)})"

    if trading_enabled is False:
        st.error(body)
    elif market_open is False:
        st.warning(body)
    elif trading_enabled is True and market_open is True:
        st.success(body)
    else:
        st.info(body)


def main() -> None:
    _configure_page()
    _inject_global_styles()
    _patch_streamlit_charts()
    _patch_streamlit_dataframe()

    tab_labels = ordered_tab_labels(ALL_TABS)
    if not tab_labels:
        st.error("Aucun onglet disponible.")
        return

    _render_alpaca_status_banner()

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
