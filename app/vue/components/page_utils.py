import pandas as pd
import streamlit as st
import altair as alt

from app.controller import dashboard_controller as dash_ctrl


def render_page_header(
    title: str, subtitle: str | None = None, *, icon: str = "📈", badge: str | None = None
) -> None:
    """Render a stylized page title card to keep headers consistent."""
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="page-hero__icon">{icon}</div>
            <div class="page-hero__titles">
                <div class="page-hero__title">{title}</div>
                {f'<div class="page-hero__subtitle">{subtitle}</div>' if subtitle else ''}
            </div>
            {f'<div class="page-hero__badge">{badge}</div>' if badge else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_closing_history_chart(ticker: str, key_prefix: str, *, location_label: str = "") -> None:
    """Display a simple line chart for closing prices over 1y."""
    ticker_norm = (ticker or "").strip().upper()
    if not ticker_norm:
        return
    df_hist, cache_path, from_cache = dash_ctrl.load_or_fetch_closing_history(
        ticker_norm, period="1y", interval="1d"
    )
    if df_hist is None or df_hist.empty:
        msg_key = f"_history_msg_{key_prefix}_{ticker_norm}"
        if not st.session_state.get(msg_key):
            st.info(f"Clôtures introuvables pour {ticker_norm} sur 1 an.")
            st.session_state[msg_key] = True
        return

    date_col = df_hist.columns[0]
    price_col = "Close" if "Close" in df_hist.columns else df_hist.columns[1]
    df_plot = df_hist[[date_col, price_col]].rename(columns={price_col: ticker_norm})
    df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[ticker_norm])
    df_plot = df_plot.set_index(date_col)
    if df_plot.empty:
        st.info(f"Clôtures introuvables pour {ticker_norm} sur 1 an.")
        return

    st.caption(f"Clôtures 1 an {ticker_norm} | {'cache' if from_cache else 'téléchargées'}")
    df_plot_reset = df_plot.reset_index().rename(columns={date_col: "Date"})
    chart = (
        alt.Chart(df_plot_reset)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y(f"{ticker_norm}:Q", title="Close"),
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)
    if cache_path:
        st.caption(f"Source: {cache_path.name}")
