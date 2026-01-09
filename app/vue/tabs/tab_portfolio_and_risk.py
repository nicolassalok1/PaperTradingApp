import pandas as pd
import streamlit as st
import altair as alt

from app.controller import portfolio_and_risk_controller as ctrl
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "🧭 Portefeuille & Risque"


def _render_account_snapshot(account: dict) -> None:
    st.markdown("### Aperçu du compte")
    equity = account.get("equity", 0.0)
    cash = account.get("cash", 0.0)
    pv = account.get("portfolio_value", equity)
    col1, col2, col3 = st.columns(3)
    col1.metric("Equity", f"${equity:,.2f}")
    col2.metric("Cash", f"${cash:,.2f}")
    col3.metric("Portfolio Value", f"${pv:,.2f}")


def _render_exposure(
    per_position: list[dict], exposure: float, net_exposure: float
) -> None:
    st.markdown("### Exposition")
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
    st.altair_chart(chart, width="stretch")
    st.caption(f"Total exposure: ${exposure:,.2f}")

    st.markdown("### Exposition net (long vs short)")
    long_expo = df[df["market_value"] > 0]["market_value"].sum()
    short_expo = df[df["market_value"] < 0]["market_value"].abs().sum()
    net_df = pd.DataFrame(
        {"Side": ["Long", "Short"], "Exposure": [long_expo, short_expo]}
    )
    st.bar_chart(net_df.set_index("Side"), width="stretch")
    st.caption(f"Net exposure: ${net_exposure:,.2f}")


def _render_pnl_and_var(unrealized_pnl_total: float, var_lite: float) -> None:
    st.markdown("### PnL & VaR-lite")
    col1, col2 = st.columns(2)
    col1.metric("Unrealized PnL", f"${unrealized_pnl_total:,.2f}")
    col2.metric("VaR (historical, 95%)", f"${var_lite:,.2f}")


def _render_position_table(per_position: list[dict]) -> None:
    st.markdown("### Risque par position")
    if not per_position:
        st.info("No positions to display.")
        return
    df = pd.DataFrame(per_position)
    if not df.empty:
        st.dataframe(df, hide_index=True, width="stretch")


def _render_alerts(alerts: list[str]) -> None:
    st.markdown("### Alertes")
    if not alerts:
        st.success("No active risk alerts.")
        return
    for alert in alerts:
        st.error(alert)


def _render_pnl_chart(pnl_series: pd.Series | None) -> None:
    if pnl_series is None or pnl_series.empty:
        st.info("PnL series unavailable.")
        return
    st.markdown("### PnL glissant")
    df = pnl_series.reset_index()
    df.columns = ["Date", "PnL"]
    df["Date"] = pd.to_datetime(df["Date"])
    st.line_chart(df.set_index("Date"), height=260, width="stretch")


def _alloc_method_mapping(label: str) -> str:
    mapping = {
        "EigenPortfolio (PCA-based)": "eigen",
    }
    return mapping.get(label, "eigen")


def _render_allocation_results(result: dict) -> None:
    symbols = result.get("symbols", [])
    weights = result.get("target_weights", [])
    if not symbols or not weights:
        st.warning("No allocation results.")
        return
    df = pd.DataFrame({"Symbol": symbols, "Target Weight": weights})
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("Target weights (sum=1).")
    fig = df.set_index("Symbol").plot.pie(y="Target Weight", autopct="%1.1f%%").figure
    st.pyplot(fig, clear_figure=True)


def _render_orders_table(orders: list[dict]) -> None:
    if not orders:
        st.info("No rebalance orders required.")
        return
    df_orders = pd.DataFrame(orders)
    st.dataframe(df_orders, hide_index=True, width="stretch")
    turnover = df_orders["qty"].abs().sum()
    st.caption(f"Approx turnover (sum abs qty): {turnover:.4f}")


def _render_rebalancing_tools() -> None:
    st.markdown("### Allocation & Rebalancement")

    method_label = st.selectbox(
        "Optimization method",
        options=["EigenPortfolio (PCA-based)"],
        index=0,
        key="por_alloc_method",
    )
    lookback_days = st.number_input(
        "Lookback days", min_value=20, max_value=365, value=60, step=5
    )
    method = _alloc_method_mapping(method_label)

    if st.button("Compute target allocation", type="primary"):
        allocation_result = ctrl.compute_allocation(method, lookback_days)
        _render_allocation_results(allocation_result)

    if st.button("Generate rebalance plan"):
        plan_result = ctrl.generate_rebalance_plan(method, lookback_days)
        orders = plan_result.get("orders", [])
        _render_orders_table(orders)

    st.warning(
        "Executing live orders will send market orders to Alpaca. Use with caution."
    )
    if st.button("Execute rebalance on Alpaca (DANGEROUS)", type="secondary"):
        exec_result = ctrl.execute_rebalance(method, lookback_days)
        executions = exec_result.get("executions", [])
        if executions:
            st.success("Rebalance executed.")
            st.dataframe(
                pd.DataFrame(executions), hide_index=True, width="stretch"
            )
        else:
            st.info("No executions performed.")


def render_tab() -> None:
    render_page_header(
        "Portefeuille & Risque",
        "Allocation, exposition, PnL, VaR-lite et alertes — centralisé en un seul écran.",
        icon="🧭",
        badge="Risk",
    )

    account = ctrl.get_account()
    summary = ctrl.get_risk_summary()

    # Top row: account + alerts vs. PnL
    col_left, col_right = st.columns(2)
    with col_left:
        _render_account_snapshot(account)
        st.divider()
        _render_alerts(summary.get("alerts", []))

    with col_right:
        _render_pnl_and_var(
            summary.get("unrealized_pnl_total", 0.0), summary.get("var_lite", 0.0)
        )
        with st.expander("Rolling PnL", expanded=False):
            _render_pnl_chart(summary.get("pnl_series"))

    st.divider()

    # Middle row: exposure vs. allocation & rebalance tools
    col_expo, col_alloc = st.columns(2)
    with col_expo:
        _render_exposure(
            summary.get("per_position_metrics", []),
            summary.get("exposure", 0.0),
            summary.get("net_exposure", 0.0),
        )

    with col_alloc:
        _render_rebalancing_tools()

    st.divider()

    # Bottom: detailed per-position table kept in an expander
    with st.expander("Per-position risk metrics (details)", expanded=False):
        _render_position_table(summary.get("per_position_metrics", []))


def render() -> None:
    """Compatibility alias if a router expects render()."""
    render_tab()
