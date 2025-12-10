from typing import List

import pandas as pd
import streamlit as st

from app.controller import portfolio_allocation_controller as ctrl
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "📐 Portfolio Allocation"


def _render_snapshot():
    snapshot = ctrl.get_portfolio_snapshot()
    equity = snapshot.get("equity", 0.0)
    cash = snapshot.get("cash", 0.0)
    symbols = snapshot.get("symbols", [])
    weights = snapshot.get("current_weights", [])

    st.markdown("### Portfolio Snapshot (Alpaca)")
    c1, c2 = st.columns(2)
    c1.metric("Equity", f"${equity:,.2f}")
    c2.metric("Cash", f"${cash:,.2f}")

    if symbols and weights:
        df = pd.DataFrame({"Symbol": symbols, "Weight": weights})
        df["Weight"] = df["Weight"].astype(float)
        st.bar_chart(df.set_index("Symbol"), use_container_width=True)
    else:
        st.info("No positions detected on Alpaca.")


def _method_mapping(label: str) -> str:
    mapping = {
        "Markowitz – Minimum Variance": "markowitz_min_var",
        "Markowitz – Maximum Sharpe": "markowitz_max_sharpe",
        "Risk Parity (ERC)": "risk_parity",
        "EigenPortfolio (PCA-based)": "eigen",
    }
    return mapping.get(label, "markowitz_min_var")


def _render_allocation_results(result: dict):
    symbols = result.get("symbols", [])
    weights = result.get("target_weights", [])
    if not symbols or not weights:
        st.warning("No allocation results.")
        return
    df = pd.DataFrame({"Symbol": symbols, "Target Weight": weights})
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption("Target weights (sum=1).")
    fig = df.set_index("Symbol").plot.pie(y="Target Weight", autopct="%1.1f%%").figure
    st.pyplot(fig, clear_figure=True)


def _render_orders_table(orders: List[dict]):
    if not orders:
        st.info("No rebalance orders required.")
        return
    df_orders = pd.DataFrame(orders)
    st.dataframe(df_orders, hide_index=True, use_container_width=True)
    turnover = df_orders["qty"].abs().sum()
    st.caption(f"Approx turnover (sum abs qty): {turnover:.4f}")


def render_tab() -> None:
    render_page_header(
        "📐 Portfolio Allocation",
        "Compute optimal weights from Alpaca positions and rebalance live.",
        icon="📐",
        badge="Alpaca",
    )

    _render_snapshot()
    st.divider()

    method_label = st.selectbox(
        "Optimization method",
        [
            "Markowitz – Minimum Variance",
            "Markowitz – Maximum Sharpe",
            "Risk Parity (ERC)",
            "EigenPortfolio (PCA-based)",
        ],
    )
    lookback_days = st.number_input("Lookback days", min_value=20, max_value=365, value=60, step=5)
    method = _method_mapping(method_label)

    if st.button("Compute target allocation", type="primary"):
        allocation_result = ctrl.compute_allocation(method, lookback_days)
        _render_allocation_results(allocation_result)

    if st.button("Generate rebalance plan"):
        plan_result = ctrl.generate_rebalance_plan(method, lookback_days)
        orders = plan_result.get("orders", [])
        _render_orders_table(orders)

    st.warning("Executing live orders will send market orders to Alpaca. Use with caution.")
    if st.button("Execute rebalance on Alpaca (DANGEROUS)", type="secondary"):
        exec_result = ctrl.execute_rebalance(method, lookback_days)
        executions = exec_result.get("executions", [])
        if executions:
            st.success("Rebalance executed.")
            st.dataframe(pd.DataFrame(executions), hide_index=True, use_container_width=True)
        else:
            st.info("No executions performed.")


def render():
    render_tab()
