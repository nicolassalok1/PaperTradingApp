import pandas as pd
import streamlit as st
import altair as alt

from app.controller import risk_management_controller as ctrl
from app.vue.components.page_utils import render_page_header


def _render_account_snapshot(account: dict) -> None:
    st.markdown("### Account snapshot")
    equity = account.get("equity", 0.0)
    cash = account.get("cash", 0.0)
    pv = account.get("portfolio_value", equity)
    col1, col2, col3 = st.columns(3)
    col1.metric("Equity", f"${equity:,.2f}")
    col2.metric("Cash", f"${cash:,.2f}")
    col3.metric("Portfolio Value", f"${pv:,.2f}")


def _render_exposure(per_position: list[dict], exposure: float, net_exposure: float) -> None:
    st.markdown("### Exposure")
    if not per_position:
        st.info("No positions to compute exposure.")
        return
    df = pd.DataFrame(per_position)
    df["abs_mv"] = df["market_value"].abs()
    pie_df = df[["symbol", "abs_mv"]].rename(columns={"abs_mv": "Exposure"})
    chart = (
        alt.Chart(pie_df)
        .mark_arc()
        .encode(theta="Exposure", color="symbol", tooltip=["symbol", "Exposure"])
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Total exposure: ${exposure:,.2f}")

    st.markdown("### Net exposure (long vs short)")
    long_expo = df[df["market_value"] > 0]["market_value"].sum()
    short_expo = df[df["market_value"] < 0]["market_value"].abs().sum()
    net_df = pd.DataFrame({"Side": ["Long", "Short"], "Exposure": [long_expo, short_expo]})
    st.bar_chart(net_df.set_index("Side"), use_container_width=True)
    st.caption(f"Net exposure: ${net_exposure:,.2f}")


def _render_pnl_and_var(unrealized_pnl_total: float, var_lite: float) -> None:
    st.markdown("### PnL & VaR-lite")
    col1, col2 = st.columns(2)
    col1.metric("Unrealized PnL", f"${unrealized_pnl_total:,.2f}")
    col2.metric("VaR (historical, 95%)", f"${var_lite:,.2f}")


def _render_position_table(per_position: list[dict]) -> None:
    st.markdown("### Per-position risk metrics")
    if not per_position:
        st.info("No positions to display.")
        return
    df = pd.DataFrame(per_position)
    if not df.empty:
        st.dataframe(df, hide_index=True, use_container_width=True)


def _render_alerts(alerts: list[str]) -> None:
    st.markdown("### Alerts")
    if not alerts:
        st.success("No active risk alerts.")
        return
    for alert in alerts:
        st.error(alert)


def _render_pnl_chart(pnl_series: pd.Series | None) -> None:
    if pnl_series is None or pnl_series.empty:
        st.info("PnL series unavailable.")
        return
    st.markdown("### Rolling PnL")
    df = pnl_series.reset_index()
    df.columns = ["Date", "PnL"]
    df["Date"] = pd.to_datetime(df["Date"])
    st.line_chart(df.set_index("Date"), height=260, use_container_width=True)


def render_tab() -> None:
    render_page_header(
        "Risk Management",
        "Exposure, PnL, VaR-lite and alerts",
        icon="??",
        badge="Risk",
    )

    account = ctrl.get_account()
    summary = ctrl.get_risk_summary()

    _render_account_snapshot(account)
    st.divider()
    _render_exposure(summary.get("per_position_metrics", []), summary.get("exposure", 0.0), summary.get("net_exposure", 0.0))
    st.divider()
    _render_pnl_and_var(summary.get("unrealized_pnl_total", 0.0), summary.get("var_lite", 0.0))
    st.divider()
    _render_position_table(summary.get("per_position_metrics", []))
    st.divider()
    _render_alerts(summary.get("alerts", []))
    st.divider()
    _render_pnl_chart(summary.get("pnl_series"))


def render() -> None:
    """Maintain parity with other tabs if a router expects render()."""
    render_tab()
