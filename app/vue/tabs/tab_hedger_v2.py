import pandas as pd
import streamlit as st

from app.controller import hedger_v2_controller as ctrl

TAB_LABEL = "🛡️ Hedger v2"


def _render_account():
    st.markdown("### Account Snapshot (Alpaca)")
    account = ctrl.get_account_snapshot()
    equity = float(account.get("equity", 0.0) or 0.0)
    cash = float(account.get("cash", 0.0) or 0.0)
    pv = float(account.get("portfolio_value", equity) or equity)
    bp = float(account.get("buying_power", 0.0) or 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"${equity:,.2f}")
    c2.metric("Cash", f"${cash:,.2f}")
    c3.metric("Portfolio Value", f"${pv:,.2f}")
    c4.metric("Buying Power", f"${bp:,.2f}")


def _render_positions():
    st.markdown("### Equity Positions")
    equities = ctrl.get_equity_positions()
    if equities:
        st.dataframe(pd.DataFrame(equities), hide_index=True, use_container_width=True)
    else:
        st.info("No US equity positions.")

    st.markdown("### Option Positions")
    options = ctrl.get_option_positions()
    if options:
        st.dataframe(pd.DataFrame(options), hide_index=True, use_container_width=True)
    else:
        st.info("No option positions.")


def _render_manual_trading():
    st.markdown("### Manual Trading (Market Orders)")
    symbol = st.text_input("Symbol", value="AAPL").upper()
    qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
    side = st.radio("Side", options=["buy", "sell"], horizontal=True)
    if st.button("Send manual market order", type="primary"):
        try:
            resp = ctrl.manual_order(symbol, qty, side)
            st.success(f"Order sent: {resp}")
        except Exception as exc:
            st.error(f"Order failed: {exc}")


def _render_dqn_panel():
    st.markdown("### DQN Hedging Panel")
    underlying = st.text_input("Underlying symbol", value="AAPL").upper()

    if st.button("Get DQN hedge suggestion"):
        try:
            suggestion = ctrl.get_dqn_hedge_suggestion(underlying)
            st.info(
                f"Suggestion: side={suggestion.get('side')} | delta_qty={suggestion.get('delta_qty')} | comment={suggestion.get('comment')}"
            )
        except Exception as exc:
            st.error(f"Failed to get suggestion: {exc}")

    if st.button("Execute DQN hedge on Alpaca"):
        try:
            result = ctrl.execute_dqn_hedge(underlying)
            st.success(f"Executed: {result}")
        except Exception as exc:
            st.error(f"Hedge execution failed: {exc}")

    st.caption("Note: DQN is a simple placeholder; future versions can improve policy/training.")


def render_tab() -> None:
    st.title("🛡️ Hedger v2 — Alpaca")
    st.markdown(
        "View Alpaca equities & options, trade manually, and use a basic DQN to propose hedges on the underlying."
    )

    _render_account()
    st.divider()
    _render_positions()
    st.divider()
    _render_manual_trading()
    st.divider()
    _render_dqn_panel()


def render():
    render_tab()
