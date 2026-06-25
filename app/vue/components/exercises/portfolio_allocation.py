"""
Streamlit view for the SPX/VIX Portfolio Allocation exercise.

Re-implementation of the neutral reference component (`reference/ui/
PortfolioAllocation.tsx`) using the app's design system: native Streamlit
widgets, the dark theme tokens, and altair charts (the lib used by the Yield
Curve tab). Same FR labels, same 8 metrics, same compliance badges, same two
charts (NAV log + weight path with cap reference lines), same caveat.

The compute is the validated Python engine, reached only through the controller.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app.controller import exercises_controller as ctrl

_PRICES_KEY = "pa_prices"
_LABEL_KEY = "pa_price_label"
_RESULT_KEY = "pa_result"


@st.cache_data(show_spinner=False)
def _bundled_prices() -> pd.DataFrame:
    return ctrl.pa_load_csv(ctrl.pa_bundled_csv_path())


@st.cache_data(show_spinner="Chargement depuis Yahoo (^GSPC, ^VIX)…")
def _yahoo_prices(start: str, end: str | None) -> pd.DataFrame:
    return ctrl.pa_fetch_yahoo(start, end)


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def _money(x: float) -> str:
    return f"${x / 1e6:.1f}mm"


def _stat(col, label: str, value: str, sub: str | None = None, ok: bool | None = None) -> None:
    col.metric(label, value)
    if ok is True:
        col.caption(f":green[✓] {sub or 'conforme'}")
    elif ok is False:
        col.caption(f":red[✗] {sub or 'hors limite'}")
    elif sub:
        col.caption(sub)


def _set_prices(df: pd.DataFrame, label: str) -> None:
    st.session_state[_PRICES_KEY] = df
    st.session_state[_LABEL_KEY] = (
        f"{label} — {len(df):,} lignes ({df.index[0].date()} → {df.index[-1].date()})"
    )
    st.session_state.pop(_RESULT_KEY, None)


def render() -> None:
    st.markdown("### Portfolio Allocation — SPX / VIX")
    st.caption(
        "Portefeuille deux actifs **long SPX** (moteur de rendement) + **long VIX** "
        "(couverture convexe), pondéré en inverse-vol, mis à l'échelle sur une cible de "
        "vol ex-ante de 10 %, puis projeté sur les caps gross / par-instrument / VaR. "
        "Point-in-time, rebalancé quotidiennement."
    )

    # ---- data source ----
    source = st.radio(
        "Source de données",
        ["Fichier CSV", "Yahoo Finance"],
        horizontal=True,
        key="pa_source",
    )

    if source == "Fichier CSV":
        c1, c2 = st.columns([3, 2])
        uploaded = c1.file_uploader("Choisir un CSV (Date,SPX,VIX)", type=["csv"], key="pa_uploader")
        if c2.button("Charger le jeu d'exemple (spx_vix_daily.csv)", width="stretch"):
            try:
                _set_prices(_bundled_prices(), "spx_vix_daily.csv")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Lecture du CSV impossible : {exc}")
        if uploaded is not None:
            try:
                _set_prices(ctrl.pa_load_csv(uploaded), uploaded.name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Lecture du CSV impossible : {exc}")
    else:
        if st.button("Charger depuis Yahoo (^GSPC, ^VIX)", width="stretch"):
            try:
                _set_prices(_yahoo_prices("1990-01-01", None), "Yahoo Finance")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Chargement Yahoo impossible : {exc}")
        st.caption(
            "Le serveur Streamlit récupère ^GSPC / ^VIX directement — aucun backend HTTP, pas de CORS."
        )

    label = st.session_state.get(_LABEL_KEY)
    if label:
        st.caption(label)

    # ---- mandate parameters ----
    st.markdown("##### Paramètres du mandat")
    p1, p2, p3, p4, p5 = st.columns(5)
    lookback = p1.number_input("Lookback (j)", min_value=20, max_value=2000, value=252, step=1, key="pa_lookback")
    vol_t = p2.number_input("Cible vol (%)", min_value=1.0, max_value=50.0, value=10.0, step=0.5, key="pa_vol")
    gross = p3.number_input("Cap gross (%)", min_value=10.0, max_value=500.0, value=150.0, step=5.0, key="pa_gross")
    name = p4.number_input("Cap / instrument (%)", min_value=10.0, max_value=300.0, value=100.0, step=5.0, key="pa_name")
    var_l = p5.number_input("Cap VaR 95% (%)", min_value=0.25, max_value=20.0, value=2.5, step=0.25, key="pa_var")

    prices = st.session_state.get(_PRICES_KEY)
    run = st.button("Lancer le backtest", type="primary", disabled=prices is None, width="stretch")
    if run and prices is not None:
        cfg = ctrl.pa_build_config(
            lookback=int(lookback),
            vol_target=vol_t / 100.0,
            gross_cap=gross / 100.0,
            name_cap=name / 100.0,
            var_limit=var_l / 100.0,
        )
        try:
            metrics, series = ctrl.pa_run_backtest(prices, cfg)
            st.session_state[_RESULT_KEY] = {"metrics": metrics, "series": series, "cfg": cfg}
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backtest échoué : {exc}")

    result = st.session_state.get(_RESULT_KEY)
    if result:
        _render_results(result)


def _render_results(result: dict) -> None:
    m = result["metrics"]
    cfg = result["cfg"]

    vol_ok = abs(m["realisedVol"] - cfg["vol_target"]) <= 0.01 + 1e-9
    var_ok = m["realisedVar95_1d"] <= cfg["var_limit"] + 1e-9
    gross_ok = m["avgGrossExposure"] <= cfg["gross_cap"] + 1e-9
    name_ok = m["maxNameWeight"] <= cfg["name_cap"] + 1e-9

    r1 = st.columns(4)
    _stat(r1[0], "Rendement annualisé (CAGR)", _pct(m["annReturnCagr"]))
    _stat(r1[1], "Vol réalisée", _pct(m["realisedVol"]), sub=f"cible {_pct(cfg['vol_target'], 0)} ±1%", ok=vol_ok)
    _stat(r1[2], "Sharpe (rf=0)", f"{m['sharpeRf0']:.2f}", sub=f"SPX seul : {m['benchmarkSpxOnly']['sharpe']:.2f}")
    _stat(r1[3], "Drawdown max", _pct(m["maxDrawdown"]))

    r2 = st.columns(4)
    _stat(r2[0], "VaR 95% 1j réalisée", _pct(m["realisedVar95_1d"], 2), sub=f"limite {_pct(cfg['var_limit'], 1)}", ok=var_ok)
    _stat(r2[1], "Gross moyen", _pct(m["avgGrossExposure"], 0), sub=f"cap {_pct(cfg['gross_cap'], 0)}", ok=gross_ok)
    _stat(r2[2], "Poids max / instrument", _pct(m["maxNameWeight"], 0), sub=f"cap {_pct(cfg['name_cap'], 0)}", ok=name_ok)
    _stat(r2[3], "Turnover (×/an)", f"{m['annualisedTurnover']:.2f}", sub=f"coût {m['annualisedCostDrag'] * 1e4:.1f} bp/an")

    st.caption(
        f"{m['sampleStart']} → {m['sampleEnd']} · {m['tradingDays']:,} jours · "
        f"NAV finale {_money(m['finalNavUsd'])} · corr(SPX,VIX) {m['corrSpxVixDaily']:.2f} · "
        f"vol VIX {_pct(m['annVolVix'], 0)}"
    )

    df = pd.DataFrame(result["series"])
    df["date"] = pd.to_datetime(df["date"])
    df["navM"] = df["nav"] / 1e6

    # ---- NAV (log) ----
    st.markdown("##### NAV cumulée (échelle log, nette de coûts)")
    nav_chart = (
        alt.Chart(df)
        .mark_line(color="#E63946", strokeWidth=1.6)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("navM:Q", title="NAV ($mm)", scale=alt.Scale(type="log")),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("navM:Q", title="NAV $mm", format=".1f")],
        )
        .properties(height=260)
    )
    st.altair_chart(nav_chart, width="stretch")

    # ---- weights with cap reference lines ----
    st.markdown("##### Trajectoire des poids (fraction de NAV)")
    disp = df.rename(columns={"wSpx": "w SPX", "wVix": "w VIX", "gross": "gross |w|"})
    w_long = disp.melt(
        id_vars="date",
        value_vars=["w SPX", "w VIX", "gross |w|"],
        var_name="serie",
        value_name="poids",
    )
    lines = (
        alt.Chart(w_long)
        .mark_line(strokeWidth=1.3)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("poids:Q", title="Poids"),
            color=alt.Color(
                "serie:N",
                title=None,
                scale=alt.Scale(
                    domain=["w SPX", "w VIX", "gross |w|"],
                    range=["#1f6f54", "#b5651d", "#9aa0a6"],
                ),
            ),
            tooltip=[alt.Tooltip("date:T", title="Date"), "serie:N", alt.Tooltip("poids:Q", format=".3f")],
        )
    )
    rule_name = alt.Chart(pd.DataFrame({"y": [cfg["name_cap"]]})).mark_rule(color="#E63946", strokeDash=[4, 4]).encode(y="y:Q")
    rule_gross = alt.Chart(pd.DataFrame({"y": [cfg["gross_cap"]]})).mark_rule(color="#7d3c98", strokeDash=[4, 4]).encode(y="y:Q")
    rule_zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#666666").encode(y="y:Q")
    st.altair_chart((lines + rule_name + rule_gross + rule_zero).properties(height=240), width="stretch")
    st.caption("Lignes de référence : :red[—] cap / instrument · :violet[—] cap gross.")

    # ---- caveat (must stay visible) ----
    st.warning(
        "**Mise en garde — le VIX spot n'est pas investable.** Le backtest capte le rendement de "
        "l'indice VIX sans friction. En réel (futures VIX / VXX), le roll en contango détruit une "
        "grande part de la jambe VIX : lis le Sharpe affiché comme une **borne haute**, pas une "
        f"espérance. Le benchmark SPX-seul (Sharpe {m['benchmarkSpxOnly']['sharpe']:.2f}) est plus "
        "proche d'un plancher atteignable. La série SPX est l'indice prix (hors dividendes), ce qui "
        "sous-estime la jambe actions d'environ 2 %/an."
    )
