"""
📡 Kalman Filters — estimation du niveau moyen OU par filtre de Kalman.

Adapté du "Kalman Trading System" Quant Guild (kts.py) et du notebook
"What Mean Reversion Should Look Like" (ktsmr.ipynb) :

- calibration AR(1)/OU (φ, μ, σ) sur une fenêtre de barres ;
- filtre de Kalman dont l'ÉTAT est le niveau moyen estimé (fair value),
  avec le levier de bruit « Confiance prix ↔ Confiance OU » (via R) ;
- bandes de réversion centre ± k·σ_stat et signaux long/short/close ;
- backtest out-of-sample comparant centre figé (μ̂) et centre Kalman —
  l'illustration du « piège de la réversion » du notebook ;
- prévision OU (points violets) avec incertitude ;
- suivi live du spot (fragment auto-rafraîchi) et ordres paper Alpaca
  derrière un garde-fou explicite.

Vue MVC : n'importe QUE des contrôleurs (jamais app.model / app.utils).
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.controller import kalman_controller as ctrl
from app.controller import trading_controller as trading_ctrl
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "📡 Kalman Filters"

# GitHub-dark palette (same family as kts.py / le thème de l'app)
_GREEN = "#3fb950"
_RED = "#f85149"
_ORANGE = "#f0883e"
_PURPLE = "#a371f7"
_BLUE = "#58a6ff"
_LIGHT_GREEN = "#7ee787"
_GRAY = "#8b949e"

_LIVE_TAPE_MAX = 300


# ---------------------------------------------------------------------------
# Helpers (pure view-side)
# ---------------------------------------------------------------------------
def _closes_from_bars(bars: list[dict]) -> list[float]:
    return [float(b["c"]) for b in bars]


def _center_series(result: dict, n: int, use_kalman_center: bool) -> np.ndarray:
    """Band center per bar: the level expected BEFORE the bar (prior), or frozen mu-hat."""
    if use_kalman_center:
        return np.asarray(result["x_pred"], dtype=float)
    return np.full(n, float(result["params"]["mu"]), dtype=float)


def _signal_from_price(price: float, center: float, band: float) -> tuple[str, str]:
    """(libellé, couleur) du signal mean-reversion au prix courant."""
    if not np.isfinite(price) or not np.isfinite(center) or band <= 0:
        return "—", _GRAY
    if price > center + band:
        return "SHORT (au-dessus de la bande haute)", _RED
    if price < center - band:
        return "LONG (sous la bande basse)", _GREEN
    return "FLAT (dans les bandes)", _GRAY


def _fmt(x: float | None, nd: int = 4) -> str:
    try:
        xv = float(x)  # type: ignore[arg-type]
        if not np.isfinite(xv):
            return "—"
        return f"{xv:.{nd}f}"
    except Exception:
        return "—"


def _build_main_chart(
    bars: list[dict],
    result: dict,
    *,
    use_kalman_center: bool,
    show_markers: bool,
    live_x: float | None,
) -> go.Figure:
    n = len(bars)
    idx = list(range(n))
    ts = [str(b.get("t", "")) for b in bars]
    x_filt = np.asarray(result["x_filt"], dtype=float)
    mu = float(result["params"]["mu"])
    band = float(result["band_halfwidth"])
    center = _center_series(result, n, use_kalman_center)
    upper, lower = center + band, center - band

    fig = go.Figure()

    # Bandes (dessinées d'abord pour rester sous les prix) — centre = niveau attendu
    # avant la barre (prior) quand le centre Kalman est choisi
    fig.add_trace(
        go.Scatter(x=idx, y=upper, mode="lines", name="Bande haute (+k·σ_stat)",
                   line=dict(color=_RED, width=1, dash="dot"), hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(x=idx, y=lower, mode="lines", name="Bande basse (−k·σ_stat)",
                   line=dict(color=_GREEN, width=1, dash="dot"),
                   fill="tonexty", fillcolor="rgba(88,166,255,0.07)", hoverinfo="skip")
    )

    # Chandeliers OHLC
    fig.add_trace(
        go.Candlestick(
            x=idx,
            open=[b["o"] for b in bars],
            high=[b["h"] for b in bars],
            low=[b["l"] for b in bars],
            close=[b["c"] for b in bars],
            name="OHLC",
            text=ts,
            increasing=dict(line=dict(color=_GREEN), fillcolor=_GREEN),
            decreasing=dict(line=dict(color=_RED), fillcolor=_RED),
            opacity=0.85,
        )
    )

    # Niveau moyen Kalman (historique filtré)
    fig.add_trace(
        go.Scatter(x=idx, y=x_filt, mode="lines", name="Niveau moyen Kalman (x)",
                   line=dict(color=_ORANGE, width=2))
    )

    # μ̂ long terme
    fig.add_trace(
        go.Scatter(x=[0, n - 1], y=[mu, mu], mode="lines", name="μ̂ (fenêtre de calibration)",
                   line=dict(color=_GRAY, width=1, dash="dash"), hoverinfo="skip")
    )

    # Niveau live (état courant du filtre, mis à jour par les ticks)
    if live_x is not None and np.isfinite(live_x):
        fig.add_trace(
            go.Scatter(x=[0, n - 1], y=[live_x, live_x], mode="lines",
                       name="Niveau live (ticks)", line=dict(color=_LIGHT_GREEN, width=2),
                       hoverinfo="skip")
        )

    # Séparation calibration / out-of-sample (ancrage début seulement)
    if result.get("calib_anchor") == "head" and result.get("oos_start"):
        w = int(result["oos_start"])
        if 0 < w < n:
            fig.add_vline(x=w - 0.5, line_width=1, line_dash="dot", line_color=_GRAY)
            fig.add_annotation(x=w - 0.5, y=1.02, yref="paper", showarrow=False,
                               text="calibration ⟶ hors-échantillon", font=dict(size=10, color=_GRAY))

    # Prévision OU + incertitude
    fc = result.get("forecast") or {}
    fc_mean = list(fc.get("mean") or [])
    if fc_mean:
        fx = list(range(n, n + len(fc_mean)))
        fu = list(fc.get("upper") or [])
        fl = list(fc.get("lower") or [])
        if fu and fl:
            fig.add_trace(go.Scatter(x=fx, y=fu, mode="lines", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=fx, y=fl, mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor="rgba(163,113,247,0.15)",
                                     name="Incertitude prévision (±1σ)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fx, y=fc_mean, mode="markers", name="Prévision OU",
                                 marker=dict(color=_PURPLE, size=7)))

    # Marqueurs de trades du backtest sélectionné
    if show_markers:
        variant = "kalman" if use_kalman_center else "static"
        bt = (result.get("backtests") or {}).get(variant) or {}
        trades = bt.get("trades") or []
        off = int(result.get("oos_start") or 0)
        ex_x = [off + int(t["index"]) for t in trades if t.get("action") == "exit"]
        ex_y = [float(t["price"]) for t in trades if t.get("action") == "exit"]
        lg_x = [off + int(t["index"]) for t in trades if t.get("action") == "entry" and t.get("side") == "long"]
        lg_y = [float(t["price"]) for t in trades if t.get("action") == "entry" and t.get("side") == "long"]
        sh_x = [off + int(t["index"]) for t in trades if t.get("action") == "entry" and t.get("side") == "short"]
        sh_y = [float(t["price"]) for t in trades if t.get("action") == "entry" and t.get("side") == "short"]
        if lg_x:
            fig.add_trace(go.Scatter(x=lg_x, y=lg_y, mode="markers", name="Entrée long",
                                     marker=dict(color=_GREEN, size=11, symbol="triangle-up",
                                                 line=dict(color="white", width=1))))
        if sh_x:
            fig.add_trace(go.Scatter(x=sh_x, y=sh_y, mode="markers", name="Entrée short",
                                     marker=dict(color=_RED, size=11, symbol="triangle-down",
                                                 line=dict(color="white", width=1))))
        if ex_x:
            fig.add_trace(go.Scatter(x=ex_x, y=ex_y, mode="markers", name="Sortie (retour au centre)",
                                     marker=dict(color=_GRAY, size=9, symbol="x")))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Barre", rangeslider=dict(visible=False)),
        yaxis=dict(title="Prix"),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, font=dict(size=10)),
        hovermode="x unified",
    )
    return fig


def _build_equity_chart(result: dict) -> go.Figure | None:
    bts = result.get("backtests") or {}
    if not bts:
        return None
    off = int(result.get("oos_start") or 0)
    fig = go.Figure()
    styles = {
        "static": ("Bandes figées (μ̂ de la fenêtre)", _GRAY, "dash"),
        "kalman": ("Bandes Kalman (centre adaptatif)", _ORANGE, "solid"),
    }
    for key, (label, color, dash) in styles.items():
        bt = bts.get(key)
        if not bt:
            continue
        eq = list(bt.get("equity") or [])
        if not eq:
            continue
        fig.add_trace(
            go.Scatter(x=[off + i for i in range(len(eq))], y=eq, mode="lines",
                       name=label, line=dict(color=color, width=2, dash=dash))
        )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Barre (hors-échantillon)"),
        yaxis=dict(title="PnL cumulé (par unité)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
    )
    return fig


def _render_backtest_stats(result: dict) -> None:
    bts = result.get("backtests") or {}
    col_static, col_kalman = st.columns(2)
    labels = [
        ("static", col_static, "Bandes figées — μ̂ de la fenêtre (le « piège »)"),
        ("kalman", col_kalman, "Bandes Kalman — centre adaptatif (le remède)"),
    ]
    for key, col, title in labels:
        bt = bts.get(key) or {}
        stats = bt.get("stats") or {}
        with col:
            st.markdown(f"**{title}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("PnL total (unité)", _fmt(stats.get("total_pnl"), 2))
            c2.metric("Allers-retours", str(int(stats.get("n_round_trips") or 0)))
            hr = stats.get("hit_rate")
            c3.metric("Taux de gain", f"{float(hr) * 100:.0f}%" if hr is not None and np.isfinite(float(hr)) else "—")
            c4, c5, c6 = st.columns(3)
            c4.metric("Max drawdown", _fmt(stats.get("max_drawdown"), 2))
            c5.metric("Entrées", str(int(stats.get("n_entries") or 0)))
            c6.metric("Position ouverte", _fmt(stats.get("open_position"), 0))


# ---------------------------------------------------------------------------
# Onglet
# ---------------------------------------------------------------------------
def render_tab() -> None:
    render_page_header(
        "Kalman Filters — niveau moyen OU",
        "Le filtre estime la « fair value » d'un processus d'Ornstein-Uhlenbeck ; "
        "bandes de réversion, backtest hors-échantillon, prévision OU et suivi live.",
        icon="📡",
        badge="OU · Mean Reversion",
    )

    ss = st.session_state

    # --- Données & calibration -------------------------------------------
    st.markdown("### Données & calibration OU")
    col_t, col_b, col_w, col_a, col_go = st.columns([1.4, 1.1, 1.1, 1.6, 1.3])
    with col_t:
        ticker = st.text_input("Ticker", value=ss.get("kf_ticker", "AAPL"), key="kf_ticker_input").strip().upper()
    with col_b:
        bar_label = st.selectbox("Barres", ctrl.bar_choice_labels(), index=1, key="kf_bar_label")
    with col_w:
        calib_window = int(
            st.number_input("Fenêtre calib (barres)", min_value=5, max_value=2000, value=60, step=5,
                            key="kf_calib_window")
        )
    with col_a:
        anchor_label = st.radio(
            "Ancrage de la calibration",
            ["Début de série (backtest honnête)", "Fin de série (trading live)"],
            key="kf_calib_anchor",
            help=(
                "Début : φ, μ, σ estimés sur les W premières barres, le reste sert de "
                "backtest hors-échantillon (méthode du notebook). Fin : estimés sur les W "
                "dernières barres comme kts.py en live — tout devient in-sample, backtest désactivé."
            ),
        )
    with col_go:
        st.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)
        load_clicked = st.button("⟳ Charger & calibrer OU", type="primary", width="stretch")

    if load_clicked and ticker:
        with st.spinner(f"Chargement des barres {ticker}…"):
            loaded = ctrl.load_history(ticker, bar_label, max_bars=1000)
        if loaded.get("success"):
            ss["kf_data"] = {"ticker": ticker, "bar_label": bar_label, **loaded}
            ss["kf_ticker"] = ticker
            ss.pop("kf_live_state", None)
            ss.pop("kf_live_meta", None)
            ss.pop("kf_live_tape", None)
            ss.pop("kf_orders_lock", None)
            st.toast(loaded.get("message", "OK"), icon="✅")
        else:
            st.warning(loaded.get("message", "Chargement impossible."))

    data = ss.get("kf_data")
    if not data or not data.get("bars"):
        st.info(
            "Charge un historique pour calibrer le modèle OU : le filtre de Kalman estimera "
            "ensuite le niveau moyen (ligne orange), les bandes de réversion et la prévision OU. "
            "Granularités intraday via Yahoo ; « 1 jour » utilise le cache local."
        )
        return

    bars = data["bars"]
    closes = _closes_from_bars(bars)
    if data.get("ticker") != ticker:
        st.caption(f"⚠️ Données affichées : **{data.get('ticker')}** ({data.get('bar_label')}). "
                   f"Clique « Charger & calibrer OU » pour passer à {ticker}.")

    # --- Réglages du filtre ----------------------------------------------
    st.markdown("### Filtre de Kalman : Confiance prix ↔ Confiance modèle OU")
    col_lever, col_k, col_center, col_fc = st.columns([2.2, 1.4, 1.6, 1.0])
    with col_lever:
        noise_lever = st.slider(
            "0 = suivre les prix · 100 = suivre l'OU pur", 0, 100,
            int(ss.get("kf_noise_lever", 50)), key="kf_noise_lever",
            help="Implémenté via le bruit d'observation R. Le filtre part de P0 = R (parité kts.py) : "
                 "le premier gain vaut ≈ φ²/(1+φ²) quel que soit le levier, puis décroît en 1/n ; "
                 "à 100, une fois ce transitoire absorbé, K ≈ 0 et le niveau se fige sur sa moyenne glissante.",
        )
    with col_k:
        band_k = st.slider("k (largeur des bandes, ×σ_stat)", 0.25, 3.0,
                           float(ss.get("kf_band_k", 1.0)), 0.25, key="kf_band_k")
    with col_center:
        center_choice = st.radio(
            "Centre des bandes", ["Niveau Kalman (adaptatif)", "μ̂ figé (fenêtre)"],
            key="kf_center_mode", horizontal=False,
            help="Le notebook trade autour d'un μ̂ figé (et saigne quand il est biaisé) ; "
                 "kts.py centre sur le niveau Kalman qui s'adapte.",
        )
    with col_fc:
        forecast_steps = int(st.number_input("Prévision (barres)", 1, 60, 5, key="kf_forecast_steps"))

    use_kalman_center = center_choice.startswith("Niveau Kalman")
    calib_anchor = "tail" if anchor_label.startswith("Fin") else "head"

    result = ctrl.run_pipeline(
        {
            "closes": closes,
            "calib_window": calib_window,
            "noise_lever": noise_lever,
            "band_k": band_k,
            "forecast_steps": forecast_steps,
            "mu_mode": "window",
            "calib_anchor": calib_anchor,
        }
    )
    if not result.get("success"):
        st.warning(result.get("message", "Calibration impossible."))
        return
    ss["kf_result_state"] = result["state"]  # pour le suivi live

    params = result["params"]
    phi = float(params["phi"])
    sigma_stat = float(result["sigma_stat"])
    band = float(result["band_halfwidth"])

    # --- Paramètres estimés ----------------------------------------------
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("φ (AR1)", _fmt(phi))
    m2.metric("μ̂ (niveau long terme)", _fmt(params["mu"], 2))
    m3.metric("σ_ε (par barre)", _fmt(params["sigma_eps"]))
    m4.metric("Demi-vie (barres)", _fmt(params["half_life_bars"], 1))
    m5.metric("σ_stat", _fmt(sigma_stat))
    m6.metric("Gain de Kalman K", _fmt(result.get("kalman_gain_last")))
    if phi >= 0.98:
        st.warning(
            "φ ≥ 0,98 : réversion très faible — le processus est proche d'une marche aléatoire. "
            "« There is no long-term mean price of a stock » (TL;DW du notebook) : les bandes "
            "autour d'un μ̂ figé n'ont guère de sens ici ; le centre Kalman adaptatif est plus prudent."
        )

    # --- Graphique principal ---------------------------------------------
    live_state = ss.get("kf_live_state")
    live_x = float(live_state["x"]) if live_state else None
    show_markers = bool(result.get("backtests"))
    fig = _build_main_chart(
        bars, result,
        use_kalman_center=use_kalman_center,
        show_markers=show_markers,
        live_x=live_x,
    )
    st.plotly_chart(fig, key="kf_main_chart")
    st.caption(
        f"{data.get('ticker')} · {data.get('bar_label')} · {len(bars)} barres · "
        f"dernière : {bars[-1].get('t', '—')} (UTC) · bandes = centre ± {band_k:.2f}·σ_stat = ±{_fmt(band, 3)}"
        + (" · centre Kalman = niveau attendu avant la barre (prior)" if use_kalman_center else "")
    )

    # --- Signal courant ---------------------------------------------------
    last_close = float(closes[-1])
    center_now = float(result["x_pred"][-1]) if use_kalman_center else float(params["mu"])
    sig_label, sig_color = _signal_from_price(last_close, center_now, band)
    z = (last_close - center_now) / sigma_stat if sigma_stat > 0 else float("nan")
    st.markdown(
        f"**Signal (dernière barre)** : <span style='color:{sig_color};font-weight:700'>{sig_label}</span> "
        f"· z = {_fmt(z, 2)} · prix {_fmt(last_close, 2)} vs centre {_fmt(center_now, 2)}",
        unsafe_allow_html=True,
    )

    # --- Backtest hors-échantillon ---------------------------------------
    st.markdown("### Backtest hors-échantillon — le piège de la réversion")
    if result.get("backtests"):
        w = int(result.get("calib_window") or 0)
        st.caption(
            f"Phase 1 : les {w} premières barres estiment (φ, μ̂, σ). Phase 2 : les "
            f"{int(result.get('n_bars', 0)) - w} barres restantes sont tradées sans avoir été vues. "
            "Règles du notebook : short au-dessus de la bande haute, long sous la bande basse, "
            "sortie au retour du prix sur le centre. PnL mark-to-market par unité."
        )
        _render_backtest_stats(result)
        eq_fig = _build_equity_chart(result)
        if eq_fig is not None:
            st.plotly_chart(eq_fig, key="kf_equity_chart")
        st.caption(
            "💡 Le « piège » (ktsmr.ipynb) : un μ̂ estimé sur une petite fenêtre est structurellement "
            "biaisé — les bandes figées font entrer du mauvais côté de la vraie moyenne et le PnL "
            "saigne. Le centre Kalman (kts.py) absorbe ce biais en réestimant le niveau à chaque barre."
        )
    elif calib_anchor == "tail":
        st.info(
            "Backtest désactivé : la calibration est ancrée sur la fin de série (mode trading live, "
            "comme kts.py) — toutes les barres sont in-sample. Repasse sur « Début de série » pour "
            "un backtest honnête."
        )
    else:
        st.info(
            f"Backtest désactivé : la fenêtre de calibration ({int(result.get('calib_window') or 0)} barres) "
            f"couvre toute la série ({int(result.get('n_bars', 0))} barres) — il ne reste pas au moins "
            "2 barres hors-échantillon. Réduis la fenêtre ou charge plus de barres."
        )

    # --- Suivi live -------------------------------------------------------
    st.markdown("### Suivi live (spot)")
    col_live, col_poll = st.columns([1.2, 1.0])
    with col_live:
        live_on = st.toggle("Activer le suivi live", key="kf_live_on",
                            help="Rafraîchit le spot périodiquement et met à jour l'état du filtre "
                                 "tick par tick (predict + update), comme le stream de kts.py.")
    with col_poll:
        poll_s = st.selectbox("Rafraîchissement (s)", [2, 5, 10, 30], index=1,
                              key="kf_live_poll", disabled=not live_on)

    bar_seconds = int(data.get("bar_seconds") or 0)

    @st.fragment(run_every=(f"{int(poll_s)}s" if live_on else None))
    def _live_panel() -> None:
        if not st.session_state.get("kf_live_on"):
            st.caption("Suivi live désactivé.")
            return
        ref = st.session_state.get("kf_result_state") or {}
        if not ref:
            st.caption("Lance d'abord une calibration.")
            return
        state = st.session_state.get("kf_live_state")
        meta = dict(st.session_state.get("kf_live_meta") or {})
        calib_sig = [ref.get("phi"), ref.get("mu"), ref.get("Q")]
        # Recalibration (fenêtre, ancrage, données) : l'état live repart de l'état calibré.
        # Le levier ne change que R : on le resynchronise sans toucher x/P.
        if not state or meta.get("calib_sig") != calib_sig:
            state = dict(ref)
            meta = {"calib_sig": calib_sig, "last_tick_ts": None, "x_pred": None}
        else:
            state = dict(state)
            state["R"] = ref.get("R", state.get("R"))

        px = ctrl.latest_price(data.get("ticker") or ticker)
        if px is None:
            st.session_state["kf_live_state"] = state
            st.session_state["kf_live_meta"] = meta
            st.warning(
                "Pas de prix live : le suivi exige les clés Alpaca (dernier trade / quote). "
                "Sans elles, aucune clôture Stooq en cache n'est injectée dans le filtre."
            )
            return

        # Un tick du filtre = une barre de la granularité calibrée (φ, Q sont par barre) :
        # le spot se rafraîchit à chaque poll, l'état n'avance qu'au rythme des barres.
        now_ts = time.time()
        last_ts = meta.get("last_tick_ts")
        due = last_ts is None or bar_seconds <= 0 or (now_ts - float(last_ts)) >= bar_seconds
        if due:
            step = ctrl.kalman_step(state, px, forecast_steps=forecast_steps)
            if not step.get("success"):
                st.warning(step.get("message", "Tick impossible."))
                return
            state = step["state"]
            meta["last_tick_ts"] = now_ts
            meta["x_pred"] = float(step["x_pred"])
        st.session_state["kf_live_state"] = state
        st.session_state["kf_live_meta"] = meta
        x_now = float(state["x"])
        x_prior = float(meta.get("x_pred") if meta.get("x_pred") is not None else x_now)

        tape = list(st.session_state.get("kf_live_tape") or [])
        tape.append({"t": datetime.now().isoformat(timespec="seconds"), "price": float(px), "x": x_now})
        st.session_state["kf_live_tape"] = tape[-_LIVE_TAPE_MAX:]

        center_live = x_prior if use_kalman_center else float(params["mu"])
        live_sig, live_color = _signal_from_price(float(px), center_live, band)
        z_live = (float(px) - center_live) / sigma_stat if sigma_stat > 0 else float("nan")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dernier prix", _fmt(px, 2))
        c2.metric("Niveau Kalman x", _fmt(x_now, 2))
        c3.metric("z-score", _fmt(z_live, 2))
        c4.markdown(
            f"<div style='padding-top:1.4em;color:{live_color};font-weight:700'>{live_sig}</div>",
            unsafe_allow_html=True,
        )
        if last_ts is not None and bar_seconds > 0:
            remaining = max(0, int(bar_seconds - (now_ts - float(meta.get("last_tick_ts") or now_ts))))
            st.caption(f"Prochain tick du filtre dans ~{remaining} s (une barre = {bar_seconds} s).")

        tape_now = st.session_state["kf_live_tape"]
        if len(tape_now) >= 2:
            df_tape = pd.DataFrame(tape_now)
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(x=df_tape["t"], y=df_tape["price"], mode="lines",
                                          name="Spot", line=dict(color=_BLUE, width=2)))
            fig_live.add_trace(go.Scatter(x=df_tape["t"], y=df_tape["x"], mode="lines",
                                          name="Niveau Kalman", line=dict(color=_ORANGE, width=2)))
            fig_live.add_trace(go.Scatter(x=[df_tape["t"].iloc[0], df_tape["t"].iloc[-1]],
                                          y=[center_live + band, center_live + band], mode="lines",
                                          name="Bande haute", line=dict(color=_RED, width=1, dash="dot")))
            fig_live.add_trace(go.Scatter(x=[df_tape["t"].iloc[0], df_tape["t"].iloc[-1]],
                                          y=[center_live - band, center_live - band], mode="lines",
                                          name="Bande basse", line=dict(color=_GREEN, width=1, dash="dot")))
            fig_live.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                                   legend=dict(orientation="h", y=1.05, x=0, font=dict(size=10)),
                                   xaxis=dict(title=None, showticklabels=False), yaxis=dict(title="Prix"))
            st.plotly_chart(fig_live, key="kf_live_chart")
        st.caption("Spot = dernier trade Alpaca (ou milieu de quote). Horodatages de la tape en heure locale.")

    _live_panel()

    # --- Trading papier ---------------------------------------------------
    st.markdown("### Trading papier (Alpaca)")
    with st.expander("Passer des ordres market sur le signal", expanded=False):
        st.caption(
            "⚠️ Envoie de vrais ordres **paper** via le compte Alpaca de l'app (endpoint paper "
            "verrouillé côté service). Rien n'est envoyé tant que la case n'est pas cochée."
        )
        orders_enabled = st.checkbox("Activer l'envoi d'ordres paper Alpaca", key="kf_orders_enabled")
        qty = int(st.number_input("Quantité", min_value=1, max_value=100000, value=10, key="kf_order_qty"))
        sym = (data.get("ticker") or ticker or "").upper()

        # Après un envoi, les boutons restent verrouillés tant que l'utilisateur n'a pas
        # relu les positions : la position du broker ne reflète pas un ordre en attente
        # (qty inchangée), un second clic inverserait la position.
        locked = bool(ss.get("kf_orders_lock"))
        if orders_enabled and locked:
            st.success(str(ss.get("kf_last_order_msg") or f"Ordre envoyé sur {sym}."))
            st.info("Positions à relire avant un nouvel envoi (un ordre en attente n'y apparaît pas encore).")
            if st.button("🔓 Relire les positions et déverrouiller", key="kf_btn_unlock"):
                ss.pop("kf_orders_lock", None)
                ss.pop("kf_last_order_msg", None)
                locked = False

        pos_qty = None      # position signée (qty)
        avail_qty = None    # part non engagée dans un ordre ouvert (qty_available)
        if orders_enabled:
            try:
                positions = trading_ctrl.get_spot_positions() or []
                for p in positions:
                    if str(p.get("symbol", "")).upper() == sym:
                        pos_qty = float(p.get("qty") or 0.0)
                        avail_raw = p.get("qty_available")
                        avail_qty = float(avail_raw) if avail_raw not in (None, "") else pos_qty
                        break
                st.caption(
                    f"Position actuelle sur {sym} : **{pos_qty if pos_qty is not None else 0:g}**"
                    + (f" (disponible : {avail_qty:g})" if avail_qty is not None and avail_qty != pos_qty else "")
                )
            except Exception as exc:
                st.warning(f"Positions indisponibles (clés Alpaca absentes ?) : {exc}")

        close_qty = abs(avail_qty) if avail_qty else 0.0
        close_is_fractional = close_qty > 0 and abs(close_qty - round(close_qty)) > 1e-9
        if close_is_fractional:
            st.caption(
                f"Position fractionnaire ({close_qty:g}) : fermeture impossible depuis cet onglet "
                "(les ordres market y partent en GTC, refusés par Alpaca pour les quantités fractionnaires)."
            )
            close_qty = 0.0

        def _send(side: str, q: float, label: str) -> None:
            try:
                res = trading_ctrl.send_spot_market_order(sym, float(q), side)
            except Exception as exc:
                st.error(f"Ordre refusé : {exc}")
                return
            oid = (res or {}).get("id") or (res or {}).get("client_order_id") or ""
            ss["kf_orders_lock"] = True
            ss["kf_last_order_msg"] = f"{label} envoyé ({sym} ×{q:g}). {('Ordre ' + str(oid)) if oid else ''}"
            # Re-rendre tout de suite avec les boutons verrouillés : un second clic
            # arrivant sur la frame courante ne doit pas pouvoir repartir.
            st.rerun()

        can_send = orders_enabled and not locked
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("🟢 Long (buy)", disabled=not can_send, width="stretch", key="kf_btn_long"):
                _send("buy", qty, "Long")
        with b2:
            if st.button("🔴 Short (sell)", disabled=not can_send, width="stretch", key="kf_btn_short"):
                _send("sell", qty, "Short")
        with b3:
            if st.button("⚪ Close position", disabled=(not can_send or close_qty <= 0),
                         width="stretch", key="kf_btn_close"):
                _send("sell" if (avail_qty or 0) > 0 else "buy", close_qty, "Close")

        st.caption(
            f"Signal courant : {sig_label} (z = {_fmt(z, 2)}). L'onglet ne trade jamais seul — "
            "les boutons ci-dessus sont la seule voie d'exécution, comme dans kts.py."
        )


def render() -> None:
    render_tab()
