import streamlit as st
import pandas as pd
import numpy as np
from app.vue.components.options.controller_bridge import *


def render_tab_rainbow():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    if not ensure_close_history(ctx):
        return
    # --------------------------------
    hist_tkr = ticker

    """Placeholder Rainbow panel extracted from ancien; logic to be re-wired."""
    # --- Contexte exotiques ---
    common_spot_value = float(st.session_state.get("common_spot_value", 100.0))
    hist_tkr = resolve_common_underlying()
    S0 = float(common_spot_value)
    # -----------------------------

    st.info("Panel Rainbow extrait : logique complète à réintégrer (pricing, dashboard push).")

    # --- Bouton Add-to-Dashboard Clean ---
    if "price" in locals() and st.button("Ajouter au dashboard", key=_k("rainbow_add_clean")):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
