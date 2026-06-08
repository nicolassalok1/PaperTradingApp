import pandas as pd
import streamlit as st
import altair as alt

from app.controller import portfolio_and_risk_controller as ctrl
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "🧭 Portefeuille & Risque"


def _render_account_snapshot(account: dict) -> None:
    return


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


def _render_positions_breakdown() -> None:
    st.markdown("### Positions (spot & options)")
    positions = ctrl.get_positions_breakdown()

    equities = positions.get("equities") or []
    options = positions.get("options") or []

    if not equities and not options:
        st.info("Aucune position Alpaca.")
        return

    if equities:
        st.caption("Positions spot")
        st.dataframe(pd.DataFrame(equities), hide_index=True, width="stretch")
    if options:
        st.caption("Positions options")
        st.dataframe(pd.DataFrame(options), hide_index=True, width="stretch")


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
    _ALLOC_STATE_KEY = "por_last_allocation"

    if st.button("Compute target allocation", type="primary"):
        allocation_result = ctrl.compute_allocation(method, lookback_days)
        st.session_state[_ALLOC_STATE_KEY] = {
            "result": allocation_result,
            "method": method,
            "lookback": lookback_days,
        }
        _render_allocation_results(allocation_result)

    if st.button("Generate rebalance plan"):
        alloc_state = st.session_state.get(_ALLOC_STATE_KEY, {})
        allocation_result = alloc_state.get("result")
        if (
            allocation_result is None
            or alloc_state.get("method") != method
            or alloc_state.get("lookback") != lookback_days
        ):
            allocation_result = ctrl.compute_allocation(method, lookback_days)
            st.session_state[_ALLOC_STATE_KEY] = {
                "result": allocation_result,
                "method": method,
                "lookback": lookback_days,
            }
        plan_result = ctrl.generate_rebalance_plan(method, lookback_days)
        target = plan_result.get("target") or allocation_result or {}
        if target and "target_weights" not in target and "weights" in target:
            target = {
                **target,
                "target_weights": target.get("weights", []),
            }
        _render_allocation_results(target)
        # Show current vs target weights to help debug why no orders may be generated
        current = plan_result.get("current") or {}
        curr_symbols = current.get("symbols") or []
        curr_weights = current.get("weights") or []
        curr_map = {s: float(w) for s, w in zip(curr_symbols, curr_weights)}
        rows = []
        for sym, tw in zip(target.get("symbols", []), target.get("target_weights", [])):
            cw = curr_map.get(sym, 0.0)
            rows.append({"Symbol": sym, "Current Weight": cw, "Target Weight": tw, "Δ weight": tw - cw})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        orders = plan_result.get("orders", [])
        _render_orders_table(orders)
        st.session_state[_ALLOC_STATE_KEY] = {
            "result": target,
            "method": method,
            "lookback": lookback_days,
        }

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

    # Positions overview (spot + options) replaces the account/PnL header section
    _render_positions_breakdown()
    st.divider()

    summary = ctrl.get_risk_summary()

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
