import pandas as pd
import streamlit as st

from app.controller import rl_live_v2_controller as ctrl

TAB_LABEL = "🤖 Hedger RL Live v2"


def _render_snapshot():
    st.subheader("Live Snapshot (Alpaca)")
    snap = ctrl.get_live_snapshot()
    account = snap.get("account", {})
    positions = snap.get("positions", [])
    equity = float(account.get("equity", 0.0) or 0.0)
    cash = float(account.get("cash", 0.0) or 0.0)
    pv = float(account.get("portfolio_value", 0.0) or 0.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity", f"${equity:,.2f}")
    c2.metric("Cash", f"${cash:,.2f}")
    c3.metric("Portfolio Value", f"${pv:,.2f}")
    if positions:
        df = pd.DataFrame(positions)
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No positions detected.")


def _render_rl_suggestion():
    st.subheader("RL Hedge Suggestion")
    underlying = st.text_input("Underlying symbol", value="AAPL", key="rl_live_underlying").upper()
    if st.button("Generate RL Hedge Suggestion", key="rl_live_generate"):
        suggestion = ctrl.get_rl_suggestion(underlying)
        st.json(
            {
                "state_vector": suggestion.get("state_vector"),
                "greeks": suggestion.get("greeks"),
                "action": suggestion.get("action"),
            }
        )
        action = suggestion.get("action", {})
        q_vals = action.get("q_values", [])
        if q_vals:
            st.bar_chart(pd.DataFrame({"q": q_vals}), key="rl_live_qvals")
        st.session_state["_last_rl_suggestion_under"] = underlying


def _render_execute():
    st.subheader("Execute Hedge")
    underlying = st.text_input(
        "Underlying for execution",
        value=st.session_state.get("_last_rl_suggestion_under", "AAPL"),
        key="rl_live_exec_underlying",
    ).upper()
    if st.button("Execute Hedge NOW on Alpaca", key="rl_live_exec"):
        result = ctrl.execute_rl_hedge(underlying)
        st.json(result)


def _render_backtester():
    st.subheader("RL Visual Backtester")
    underlying = st.text_input("Underlying for backtest", value="AAPL", key="rl_live_bt_underlying").upper()
    lookback = st.number_input(
        "Lookback days",
        min_value=10,
        max_value=365,
        value=60,
        step=5,
        key="rl_live_bt_lookback",
    )
    if st.button("Run RL Backtest", key="rl_live_bt_run"):
        res = ctrl.run_backtest(underlying, int(lookback))
        pnl_curve = res.get("pnl_curve", [])
        hedge_err = res.get("hedge_error_curve", [])
        pos_curve = res.get("positions_curve", [])
        if pnl_curve:
            df_pnl = pd.DataFrame(pnl_curve)
            st.line_chart(df_pnl.set_index("t"), use_container_width=True, key="rl_live_bt_pnl")
            final_pnl = df_pnl["pnl"].sum()
            max_dd = (df_pnl["pnl"].cumsum().cummax() - df_pnl["pnl"].cumsum()).max()
            st.metric("Final PnL", f"${final_pnl:,.2f}")
            st.metric("Max Drawdown (PnL curve)", f"${max_dd:,.2f}")
        if hedge_err:
            df_err = pd.DataFrame(hedge_err)
            st.line_chart(df_err.set_index("t"), use_container_width=True, key="rl_live_bt_err")
            avg_err = df_err["error"].mean()
            st.metric("Avg Hedge Error", f"{avg_err:.4f}")
        if pos_curve:
            df_pos = pd.DataFrame(pos_curve)
            st.line_chart(df_pos.set_index("t"), use_container_width=True, key="rl_live_bt_pos")


def render_tab():
    st.title("🤖 Hedger RL Live v2 — Greeks + Backtester + Alpaca Execution")
    _render_snapshot()
    st.markdown("---")
    _render_rl_suggestion()
    st.markdown("---")
    _render_execute()
    st.markdown("---")
    _render_backtester()


def render():
    render_tab()
