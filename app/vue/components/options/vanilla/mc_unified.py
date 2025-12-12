import streamlit as st
import numpy as np

from app.vue.components.options import controller_bridge as cb


def _style_label(style: str) -> str:
    return {
        "european": "Européenne",
        "bermudan": "Bermudienne",
        "american": "Américaine",
    }.get(style, style.title())


def render_mc_panel(default_style: str = "european"):
    ctx = cb.get_option_context()
    if not cb.ensure_close_history(ctx):
        return
    S0 = float(ctx.get("S0") or 100.0)
    ticker = ctx.get("ticker") or ""
    _k = ctx["_k"]
    key_prefix = f"mc_{default_style}"

    st.subheader("Pricing Monte Carlo unifié (LSMC)")
    cols = st.columns(3)
    with cols[0]:
        style_choice = st.selectbox(
            "Style",
            ["european", "bermudan", "american"],
            index=["european", "bermudan", "american"].index(default_style)
            if default_style in ["european", "bermudan", "american"]
            else 0,
            key=_k(f"{key_prefix}_style"),
            format_func=_style_label,
        )
        option_type = st.selectbox("Type", ["call", "put"], index=0, key=_k(f"{key_prefix}_type"))
        T = st.slider("Maturité (années)", min_value=0.05, max_value=2.0, value=0.5, step=0.05, key=_k(f"{key_prefix}_T"))
    with cols[1]:
        strike = st.slider(
            "Strike",
            min_value=0.5 * S0,
            max_value=1.5 * S0,
            value=S0,
            step=0.5,
            key=_k(f"{key_prefix}_strike"),
        )
        sigma = st.slider("Volatilité", min_value=0.05, max_value=1.0, value=0.25, step=0.01, key=_k(f"{key_prefix}_sigma"))
        n_paths = st.number_input("N paths", min_value=2000, max_value=50000, value=20000, step=2000, key=_k(f"{key_prefix}_npaths"))
    with cols[2]:
        n_steps = st.number_input("N steps (temps)", min_value=25, max_value=1000, value=252, step=25, key=_k(f"{key_prefix}_nsteps"))
        if style_choice == "bermudan":
            ex_ctrl = st.number_input("Nb dates d'exercice", min_value=1, max_value=50, value=8, step=1, key=_k(f"{key_prefix}_berm_ndates"))
        elif style_choice == "american":
            ex_ctrl = st.number_input("Fréquence exercice (par an)", min_value=50, max_value=500, value=252, step=25, key=_k(f"{key_prefix}_am_freq"))
        else:
            ex_ctrl = None

    st.info("Pricing Monte Carlo désactivé pour les Vanilla. Utilise le pricer Black-Scholes.")
