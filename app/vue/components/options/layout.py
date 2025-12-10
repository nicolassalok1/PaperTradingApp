import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from app.controller import options_controller as oc
from app.vue.components.selector import choose_option_select
from app.vue.components.options_text import render_option_text as _txt
from app.vue.components.options.shared import load_cached_option_history, load_options_meta

compute_crr_heatmaps = oc.compute_crr_heatmaps
heatmap_axis = oc.heatmap_axis
render_heatmap = oc.render_heatmap
build_crr_tree = oc.build_crr_tree
plot_crr_tree = oc.plot_crr_tree


def render_options_history_block() -> None:
    """
    Render:
      - subheader 'Historique 1 an du ticker (prix de cloture)'
      - meta table from load_options_meta()
      - 1y close chart from load_cached_option_history()
    """
    st.subheader("Historique 1 an du ticker (prix de cloture)")
    ticker = st.session_state.get("heston_cboe_ticker") or st.session_state.get("tkr_common") or ""
    ticker = str(ticker or "").strip().upper()
    if not ticker:
        st.info("Charge un ticker via la calibration Heston pour afficher l'historique 1 an.")
        return

    meta = load_options_meta()
    meta_row = {
        "Ticker cache": meta.get("ticker"),
        "S0_ref cache": meta.get("S0_ref"),
        "r cache": meta.get("r"),
        "q cache": meta.get("q"),
    }
    meta_df = pd.DataFrame([meta_row])
    st.dataframe(meta_df, hide_index=True)

    tkr_hist, df_hist = load_cached_option_history()
    if df_hist is None or df_hist.empty or "Close" not in df_hist.columns:
        st.info(
            "Pas d'historique disponible pour ce ticker dans le cache. "
            "Clique sur Refresh dans l'onglet Options pour le telecharger."
        )
        return

    if not isinstance(df_hist.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
        try:
            df_hist.index = pd.to_datetime(df_hist.index)
        except Exception:
            st.info(
                "Pas d'historique disponible pour ce ticker dans le cache. "
                "Clique sur Refresh dans l'onglet Options pour le telecharger."
            )
            return

    start_dt = df_hist.index.min()
    end_dt = df_hist.index.max()
    start_label = start_dt.strftime("%Y-%m-%d") if hasattr(start_dt, "strftime") else str(start_dt)
    end_label = end_dt.strftime("%Y-%m-%d") if hasattr(end_dt, "strftime") else str(end_dt)
    tkr_display = (tkr_hist or ticker).strip().upper()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist["Close"], mode="lines", name="Close"))
    fig.update_layout(
        title=f"{tkr_display} - Close (1 an) [{start_label} → {end_label}]",
        xaxis_title="Date",
        yaxis_title="Prix",
    )
    st.plotly_chart(fig, config={"staticPlot": True, "scrollZoom": False})


def option_panel(title: str, subtitle: str | None = None):
    """
    Start a product panel with unified styling.
    Returns a container that can be used to nest content if needed.
    """
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)
    return st.container()


def params_expander(label: str = "Parametres"):
    """
    Convenience wrapper for parameter sections.
    """
    return st.expander(label)


def compute_button(label: str = "Calculer le prix") -> bool:
    """
    Standardized compute button.
    """
    # Avoid duplicate implicit keys by auto-generating one when none is provided.
    counter = st.session_state.get("_compute_btn_counter", 0)
    st.session_state["_compute_btn_counter"] = counter + 1
    key = f"compute_btn_{counter}"
    return st.button(label, key=key)


def render_crr_payoff_surface(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_char: str = "c",
    n_steps: int = 25,
) -> None:
    """
    Render a CRR payoff surface (Call/Put) heatmap around (S0, K)
    using shared option helpers.
    """
    if compute_crr_heatmaps is None or heatmap_axis is None or render_heatmap is None:
        st.info("CRR heatmaps indisponibles (dépendances optionnelles manquantes).")
        return
    if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        st.warning("Parametres invalides pour la surface CRR.")
        return
    span_S = 0.25 * float(S0)
    span_K = 0.25 * float(K)
    s_values = heatmap_axis(float(S0), span_S, n_points=31)
    k_values = heatmap_axis(float(K), span_K, n_points=31)
    call_matrix, put_matrix = compute_crr_heatmaps(
        s_values, k_values, float(T), float(r), float(sigma), int(n_steps)
    )
    call_fig = render_heatmap(call_matrix, k_values, s_values, title="Payoff CRR - Call")
    put_fig = render_heatmap(put_matrix, k_values, s_values, title="Payoff CRR - Put")
    st.plotly_chart(call_fig, config={"staticPlot": True, "scrollZoom": False})
    st.plotly_chart(put_fig, config={"staticPlot": True, "scrollZoom": False})


# ---------------------------------------------------------------------------
# GPT-style analysis wrappers (UI-level)
# ---------------------------------------------------------------------------


def render_payoff_text(option_label: str, option_tag: str):
    """Display payoff text in the UI."""
    msg = _txt(option_label, option_tag)
    st.info(msg)


def render_crr_tree(option_obj, r: float, sigma: float, n_steps: int):
    """Build and display a CRR tree preview."""
    spot_tree, value_tree = build_crr_tree(option_obj, r=r, sigma=sigma, n_steps=n_steps)
    fig = plot_crr_tree(spot_tree, value_tree)
    st.pyplot(fig, clear_figure=True)


def render_heatmap_diagnostics(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    option_char: str,
):
    """Render CRR heatmap diagnostics for call/put surfaces."""
    if compute_crr_heatmaps is None or heatmap_axis is None or render_heatmap is None:
        st.info("Diagnostics heatmap indisponibles (dépendances optionnelles manquantes).")
        return
    if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        st.warning("Parametres invalides pour les diagnostics heatmap.")
        return
    span_S = 0.25 * float(S0)
    span_K = 0.25 * float(K)
    s_values = heatmap_axis(float(S0), span_S, n_points=31)
    k_values = heatmap_axis(float(K), span_K, n_points=31)
    call_matrix, put_matrix = compute_crr_heatmaps(
        s_values, k_values, float(T), float(r), float(sigma), int(n_steps)
    )
    call_fig = render_heatmap(call_matrix, k_values, s_values, title="Surface CRR Call")
    put_fig = render_heatmap(put_matrix, k_values, s_values, title="Surface CRR Put")
    st.plotly_chart(call_fig, config={"staticPlot": True, "scrollZoom": False})
    st.plotly_chart(put_fig, config={"staticPlot": True, "scrollZoom": False})
