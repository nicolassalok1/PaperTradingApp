"""
🌡️ Vol Implicite — port Streamlit du dashboard Tkinter "Implied Volatility
Trading Dashboard" (ex-Interactive Brokers), branché sur Alpaca.

- Série historique : volatilité réalisée (barres daily Alpaca, fallback Stooq/Yahoo).
- IV courante : ATM ~30 j depuis les snapshots d'options Alpaca.
- Analyse : régimes par percentile roulant, régressions de vol forward,
  signal de mean reversion — mêmes constructions que la version IB.

MVC : la vue ne parle qu'au contrôleur (`iv_dashboard_controller`).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.controller import iv_dashboard_controller as ctrl
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "🌡️ Vol Implicite"

_STATE_KEY = "iv_dashboard_result"

_DURATION_CHOICES = {"1 an": 1.0, "2 ans": 2.0, "3 ans": 3.0, "5 ans": 5.0}

# Dark-theme palette (aligned with the other tabs)
_COL_RED = "#f87171"
_COL_ORANGE = "#fb923c"
_COL_GREEN = "#34d399"
_COL_BLUE = "#60a5fa"
_COL_GOLD = "#fbbf24"
_COL_NEUTRAL = "#e5e7eb"
_COL_MUTED = "#9ca3af"

_REGIME_COLORS = {
    "high": _COL_RED,
    "above": _COL_ORANGE,
    "normal": _COL_NEUTRAL,
    "below": _COL_BLUE,
    "low": _COL_GREEN,
    "unknown": _COL_MUTED,
}
_SIGNAL_COLORS = {
    "down": _COL_BLUE,
    "up": _COL_RED,
    "neutral": _COL_NEUTRAL,
    "unknown": _COL_MUTED,
}


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def _chip(caption: str, label: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:0.25rem;">
          <div style="font-size:0.78rem;color:{_COL_MUTED};margin-bottom:0.35rem;">{caption}</div>
          <span style="display:inline-block;padding:0.30rem 0.7rem;border-radius:999px;
                       border:1px solid {color};color:{color};font-weight:600;
                       font-size:0.9rem;background:rgba(255,255,255,0.03);">
            {label}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _vol_level_color(vol: float) -> str:
    """Legacy thresholds: > 40% high (red), < 15% low (green)."""
    if not np.isfinite(vol):
        return _COL_MUTED
    if vol > 0.40:
        return _COL_RED
    if vol < 0.15:
        return _COL_GREEN
    return _COL_NEUTRAL


def _fmt_pct(x: Any, digits: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(v):
        return "N/A"
    return f"{v * 100:.{digits}f}%"


def _base_layout(fig: go.Figure, *, height: int = 380, ytitle: str = "", xtitle: str = "") -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title=xtitle or None,
        yaxis_title=ytitle or None,
        hovermode="closest",
    )


# --------------------------------------------------------------------------- #
# Form
# --------------------------------------------------------------------------- #
def _render_form() -> None:
    with st.form("iv_dashboard_form"):
        c1, c2, c3, c4 = st.columns([1.4, 1.1, 1.1, 1.0])
        with c1:
            symbol = st.text_input("Symbole", value="SPY", help="Sous-jacent US (actions/ETF).")
        with c2:
            duration = st.selectbox("Durée", list(_DURATION_CHOICES.keys()), index=1)
        with c3:
            rv_window = st.number_input(
                "Fenêtre RV (jours)",
                min_value=5,
                max_value=120,
                value=20,
                step=1,
                help="Fenêtre de la volatilité réalisée (série historique).",
            )
        with c4:
            st.write("")
            submitted = st.form_submit_button("🔍 Analyser", type="primary", width="stretch")

        with st.expander("Paramètres avancés", expanded=False):
            a1, a2, a3 = st.columns(3)
            with a1:
                forward_window = st.number_input(
                    "Fenêtre forward (jours)", min_value=5, max_value=90, value=30, step=1
                )
            with a2:
                percentile_window = st.number_input(
                    "Fenêtre percentile (jours)", min_value=60, max_value=756, value=252, step=1
                )
            with a3:
                include_iv = st.checkbox(
                    "IV courante via options Alpaca",
                    value=True,
                    help=(
                        "Interroge la chaîne d'options Alpaca pour l'IV ATM ~30 j. "
                        "Flux configurable via ALPACA_OPTION_DATA_FEED (indicative/opra)."
                    ),
                )

    if submitted:
        sym = (symbol or "").strip().upper()
        if not sym:
            st.warning("Entre un symbole avant de lancer l'analyse.")
            return
        with st.spinner(f"Analyse de la volatilité de {sym} via Alpaca…"):
            try:
                result = ctrl.get_iv_analysis(
                    sym,
                    years=_DURATION_CHOICES.get(duration, 2.0),
                    rv_window=int(rv_window),
                    forward_window=int(forward_window),
                    percentile_window=int(percentile_window),
                    include_current_iv=bool(include_iv),
                )
                result["generated_at"] = pd.Timestamp.now().strftime("%H:%M:%S")
                st.session_state[_STATE_KEY] = result
            except Exception as exc:  # noqa: BLE001 — surfaced to the user
                st.error(f"Analyse impossible : {exc}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _render_metrics(result: Dict[str, Any]) -> None:
    current_vol = float(result.get("current_vol", float("nan")))
    regime = result.get("regime") or {}
    pct = result.get("current_percentile", float("nan"))
    stats = result.get("vol_stats") or {}

    st.markdown("#### Série de volatilité (réalisée)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _chip(
            f"RV courante ({result.get('rv_window')} j, annualisée)",
            _fmt_pct(current_vol),
            _vol_level_color(current_vol),
        )
    with c2:
        st.metric(
            f"Percentile ({result.get('percentile_window')} j)",
            _fmt_pct(pct, digits=1),
            help="Rang percentile de la vol courante dans sa fenêtre roulante.",
        )
    with c3:
        _chip(
            "Régime courant",
            str(regime.get("label", "N/A")),
            _REGIME_COLORS.get(str(regime.get("key")), _COL_MUTED),
        )
    with c4:
        _chip(
            "Signal mean reversion",
            str(regime.get("signal_label", "N/A")),
            _SIGNAL_COLORS.get(str(regime.get("signal_key")), _COL_MUTED),
        )
    st.caption(
        f"Plage de vol — min : {_fmt_pct(stats.get('min'))} | "
        f"moyenne : {_fmt_pct(stats.get('mean'))} | max : {_fmt_pct(stats.get('max'))}"
    )

    st.markdown("#### Volatilité implicite (options Alpaca)")
    current_iv = result.get("current_iv")
    if current_iv:
        iv_val = float(current_iv.get("iv", float("nan")))
        spread = result.get("iv_minus_rv")
        iv_pct = result.get("iv_vs_series_percentile", float("nan"))
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            _chip(
                f"IV ATM ~{current_iv.get('dte', '?')} j",
                _fmt_pct(iv_val),
                _vol_level_color(iv_val),
            )
        with d2:
            st.metric(
                "Spread IV − RV",
                f"{float(spread) * 100:+.2f} pts" if spread is not None else "N/A",
                help="Prime de risque de volatilité : IV ATM moins vol réalisée courante.",
            )
        with d3:
            st.metric(
                "Percentile IV vs série RV",
                _fmt_pct(iv_pct, digits=1),
                help="Position de l'IV courante dans la distribution récente de la RV.",
            )
        with d4:
            # No regime / mean-reversion chip here: an IV ranked inside the RV
            # distribution sits structurally high (variance risk premium) and says
            # nothing about IV richness vs its own history.
            st.caption(
                "Lecture : ce percentile mesure la prime de risque de vol (IV vs RV), "
                "pas la cherté de l'IV vs son propre historique — aucun signal de "
                "mean reversion n'en est dérivé."
            )
        st.caption(
            f"Échéance {current_iv.get('expiry')} ({current_iv.get('dte')} j) · "
            f"{current_iv.get('n_contracts')} contrats · méthode : {current_iv.get('method')} · "
            f"flux : {current_iv.get('feed')}"
        )
    else:
        st.warning(
            "IV courante indisponible — "
            + str(result.get("iv_error") or "chaîne d'options Alpaca inaccessible.")
        )

    iv_history = result.get("iv_history")
    n_iv_obs = 0 if iv_history is None else int(len(iv_history))
    if n_iv_obs:
        st.caption(
            f"Historique IV local : {n_iv_obs} observation(s) quotidienne(s) "
            "accumulée(s) dans `cache/IVHistory/` (une par jour d'analyse)."
        )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _render_series_chart(result: Dict[str, Any]) -> None:
    series: pd.DataFrame = result["series"]
    vol = series["vol"]

    q25 = float(vol.quantile(0.25))
    q75 = float(vol.quantile(0.75))
    mean = float(vol.mean())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=vol,
            mode="lines",
            name=f"Vol réalisée ({result.get('rv_window')} j)",
            line=dict(color=_COL_BLUE, width=1.6),
            hovertemplate="%{x|%d %b %Y} · %{y:.2%}<extra></extra>",
        )
    )
    for level, name, color in (
        (q75, "75e percentile", _COL_RED),
        (q25, "25e percentile", _COL_GREEN),
        (mean, "Moyenne", _COL_NEUTRAL),
    ):
        fig.add_hline(y=level, line_dash="dash", line_width=1, line_color=color, opacity=0.7)
        fig.add_trace(
            go.Scatter(
                x=[series.index[0]],
                y=[level],
                mode="lines",
                name=name,
                line=dict(color=color, dash="dash", width=1),
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[series.index[-1]],
            y=[float(vol.iloc[-1])],
            mode="markers",
            name="RV courante",
            marker=dict(color=_COL_RED, size=11, symbol="circle"),
            hovertemplate="RV courante · %{y:.2%}<extra></extra>",
        )
    )

    iv_history = result.get("iv_history")
    if iv_history is not None and not iv_history.empty:
        fig.add_trace(
            go.Scatter(
                x=iv_history["date"],
                y=iv_history["iv"],
                mode="lines+markers",
                name="IV ATM (historique local)",
                line=dict(color=_COL_GOLD, width=1),
                marker=dict(color=_COL_GOLD, size=5),
                hovertemplate="%{x|%d %b %Y} · IV %{y:.2%}<extra></extra>",
            )
        )
    current_iv = result.get("current_iv")
    if current_iv:
        fig.add_trace(
            go.Scatter(
                x=[series.index[-1]],
                y=[float(current_iv["iv"])],
                mode="markers",
                name="IV ATM courante",
                marker=dict(color=_COL_GOLD, size=13, symbol="star"),
                hovertemplate="IV ATM · %{y:.2%}<extra></extra>",
            )
        )

    fig.update_layout(title="Série de volatilité et bandes de régime")
    _base_layout(fig, height=400, ytitle="Vol annualisée")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, key="iv_dash_series_chart")


def _render_forward_chart(result: Dict[str, Any]) -> None:
    analysis = result["analysis"]
    df: pd.DataFrame = analysis["df"]
    reg = analysis["reg_forward"]
    fw = analysis["forward_window"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["current_vol"],
            y=df["forward_vol"],
            mode="markers",
            name="Observations",
            marker=dict(color=_COL_BLUE, size=5, opacity=0.55),
            hovertemplate="courante %{x:.2%} · forward %{y:.2%}<extra></extra>",
        )
    )
    x_range = np.linspace(float(df["current_vol"].min()), float(df["current_vol"].max()), 100)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=reg["slope"] * x_range + reg["intercept"],
            mode="lines",
            name=f"Régression (R² = {reg['r2']:.3f})",
            line=dict(color=_COL_RED, width=2),
        )
    )
    lo = float(min(df["current_vol"].min(), df["forward_vol"].min()))
    hi = float(max(df["current_vol"].max(), df["forward_vol"].max()))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="y = x (aucun changement)",
            line=dict(color=_COL_MUTED, width=1, dash="dash"),
        )
    )
    fig.update_layout(
        title=(
            f"Vol forward {fw} j vs vol courante — "
            f"y = {reg['slope']:.3f}x + {reg['intercept']:.3f}"
        )
    )
    _base_layout(fig, height=380, xtitle="Vol courante", ytitle=f"Vol forward moyenne {fw} j")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, key="iv_dash_forward_chart")


def _render_diff_chart(result: Dict[str, Any]) -> None:
    analysis = result["analysis"]
    df: pd.DataFrame = analysis["df"]
    intersection = float(analysis["intersection"])

    high_mask = df["current_vol"] > intersection
    low_mask = ~high_mask

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.loc[high_mask, "current_vol"],
            y=df.loc[high_mask, "vol_diff"],
            mode="markers",
            name=f"Régime vol haute (n={analysis['n_high']})",
            marker=dict(color=_COL_RED, size=5, opacity=0.55),
            hovertemplate="courante %{x:.2%} · diff %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[low_mask, "current_vol"],
            y=df.loc[low_mask, "vol_diff"],
            mode="markers",
            name=f"Régime vol basse (n={analysis['n_low']})",
            marker=dict(color=_COL_BLUE, size=5, opacity=0.55),
            hovertemplate="courante %{x:.2%} · diff %{y:.2%}<extra></extra>",
        )
    )

    for reg_key, mask, color, label in (
        ("reg_high", high_mask, _COL_RED, "haute"),
        ("reg_low", low_mask, _COL_BLUE, "basse"),
    ):
        reg = analysis.get(reg_key)
        if not reg or not int(mask.sum()):
            continue
        x_seg = df.loc[mask, "current_vol"]
        x_rng = np.linspace(float(x_seg.min()), float(x_seg.max()), 100)
        fig.add_trace(
            go.Scatter(
                x=x_rng,
                y=reg["slope"] * x_rng + reg["intercept"],
                mode="lines",
                name=f"Rég. vol {label} (R² = {reg['r2']:.3f})",
                line=dict(color=color, width=2),
            )
        )

    fig.add_hline(y=0.0, line_dash="dash", line_width=1, line_color=_COL_MUTED, opacity=0.8)
    fig.add_vline(
        x=intersection,
        line_dash="dot",
        line_width=1.5,
        line_color=_COL_GREEN,
        annotation_text=f"Split régimes ({intersection:.3f})",
        annotation_font_color=_COL_GREEN,
    )
    fig.update_layout(title="Diff de vol (forward − courante) par régime")
    _base_layout(fig, height=380, xtitle="Vol courante", ytitle="Diff de vol")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, key="iv_dash_diff_chart")


# --------------------------------------------------------------------------- #
# Log / diagnostics
# --------------------------------------------------------------------------- #
def _format_reg(reg: Dict[str, Any] | None, title: str) -> list[str]:
    if not reg:
        return [f"{title}: données insuffisantes pour la régression."]
    return [
        f"{title}:",
        (
            f"  pente {reg['slope']:.4f} · intercept {reg['intercept']:.4f} · "
            f"R² {reg['r2']:.4f} · p-value {reg['p_value']:.4g} · n={reg['n']}"
        ),
    ]


def _render_log(result: Dict[str, Any]) -> None:
    with st.expander("Journal & détails des régressions", expanded=False):
        lines: list[str] = []
        analysis = result.get("analysis")
        if analysis:
            lines += _format_reg(analysis["reg_forward"], "Régression 1 — vol forward ~ vol courante")
            lines.append(f"  intersection avec y=x à vol = {analysis['intersection']:.4f}")
            lines += _format_reg(analysis["reg_diff"], "Régression 2 — diff de vol ~ vol courante")
            lines += _format_reg(
                analysis.get("reg_high"),
                f"Régime VOL HAUTE (vol > {analysis['intersection']:.3f}, n={analysis['n_high']})",
            )
            lines += _format_reg(
                analysis.get("reg_low"),
                f"Régime VOL BASSE (vol ≤ {analysis['intersection']:.3f}, n={analysis['n_low']})",
            )
            for ins in analysis.get("insights", []):
                lines.append(f"INSIGHT : {ins}")
            lines.append("")
        for msg in result.get("log", []):
            lines.append(str(msg))
        st.code("\n".join(lines) or "Aucun message.", language=None)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render_tab() -> None:
    render_page_header(
        "Vol Implicite (IV)",
        "Régimes de volatilité & mean reversion — RV historique + IV ATM via Alpaca.",
        icon="🌡️",
        badge="Alpaca",
    )

    _render_form()

    result = st.session_state.get(_STATE_KEY)
    if not isinstance(result, dict):
        st.info(
            "Entre un symbole (ex. SPY) puis clique sur **🔍 Analyser**. "
            "La série historique est la volatilité réalisée (Alpaca n'expose pas "
            "d'historique d'IV) ; l'IV ATM courante vient de la chaîne d'options Alpaca "
            "et s'accumule jour après jour dans un historique local."
        )
        return

    st.caption(
        f"**{result.get('symbol')}** · source série : {result.get('source')} · "
        f"{len(result.get('series', []))} points · généré à {result.get('generated_at', '—')}"
    )

    _render_metrics(result)
    st.divider()

    _render_series_chart(result)

    if result.get("analysis"):
        col_left, col_right = st.columns(2)
        with col_left:
            _render_forward_chart(result)
        with col_right:
            _render_diff_chart(result)
    else:
        st.info(
            "Analyse forward vol indisponible : "
            + str(result.get("analysis_error") or "données insuffisantes.")
        )

    _render_log(result)


def render() -> None:
    """Keep consistency with other tabs if a generic router is used."""
    render_tab()
