import pandas as pd
import streamlit as st

from app.controller import yieldcurve_controller as yc
from app.vue.components.page_utils import render_page_header


def render():
    snapshot = yc.get_curve_snapshot(risk_free_maturity=1.0, ensure_cache=False)

    maturities = snapshot.get("maturities") or []
    zero_rates = snapshot.get("zero_rates") or []
    discount_factors = snapshot.get("discount_factors") or []
    risk_free_rate = snapshot.get("risk_free_rate")
    risk_free_maturity = snapshot.get("risk_free_maturity")
    source_path = snapshot.get("source_path")

    render_page_header(
        "Yield Curve",
        "Visualisation des taux zc et facteurs d'actualisation (lecture seule)",
        icon="🧭",
        badge="Rates",
    )

    if source_path:
        st.caption(f"Source: {source_path}")
    else:
        st.caption("Source attendue: cache/yield_curve.csv")

    col_rf, col_ref = st.columns([2, 1])
    with col_rf:
        if risk_free_rate is not None:
            st.metric(
                "Risk-free rate (pricing)",
                f"{float(risk_free_rate) * 100:.2f} %",
                f"réf: {float(risk_free_maturity):.2f}y",
            )
        else:
            st.warning("Risk-free rate indisponible.")
    with col_ref:
        st.write("")
        st.write("Taux utilisé par les pricers (zc).")

    def _df_from_series(values: list[float], label: str) -> pd.DataFrame | None:
        if not maturities or not values:
            return None
        try:
            df = pd.DataFrame({"Maturity (years)": maturities, label: values}).set_index(
                "Maturity (years)"
            )
            return df
        except Exception:
            return None

    st.markdown("#### Zero rates (term structure)")
    df_zero = _df_from_series(zero_rates, "Zero rate")
    if df_zero is not None and not df_zero.empty:
        st.line_chart(df_zero, height=260)
        st.dataframe(df_zero.reset_index().rename(columns={"Zero rate": "Zero rate (dec)"}), hide_index=True)
    else:
        st.info("Aucune courbe à afficher. Dépose `cache/yield_curve.csv` pour alimenter la vue.")

    st.markdown("#### Discount factors")
    df_df = _df_from_series(discount_factors, "Discount factor")
    if df_df is not None and not df_df.empty:
        st.area_chart(df_df, height=220)
    else:
        st.info("Pas de discount factors calculables sans courbe.")
