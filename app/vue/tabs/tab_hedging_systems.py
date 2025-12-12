import pandas as pd
import streamlit as st

from app.controller import hedger_v2_controller as ctrl
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "??? Hedging Systems"


def _render_account() -> None:
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


def _render_positions() -> None:
    st.markdown("### Positions (equities & options)")
    equities = ctrl.get_equity_positions()
    options = ctrl.get_option_positions()

    if not equities and not options:
        st.info("No Alpaca positions.")
        return

    if equities:
        st.caption("Equity positions")
        st.dataframe(pd.DataFrame(equities), hide_index=True, use_container_width=True)

    if options:
        st.caption("Option positions")
        st.dataframe(pd.DataFrame(options), hide_index=True, use_container_width=True)


def _render_dqn_panel() -> None:
    st.markdown("### DQN Hedging Panel")
    underlying = st.text_input("Underlying symbol", value="AAPL").upper()

    if st.button("Get DQN hedge suggestion"):
        try:
            suggestion = ctrl.get_dqn_hedge_suggestion(underlying)
            st.info(
                f"Suggestion: side={suggestion.get('side')} | "
                f"delta_qty={suggestion.get('delta_qty')} | "
                f"comment={suggestion.get('comment')}"
            )
        except Exception as exc:
            st.error(f"Failed to get suggestion: {exc}")

    if st.button("Execute DQN hedge on Alpaca"):
        try:
            result = ctrl.execute_dqn_hedge(underlying)
            st.success(f"Executed: {result}")
        except Exception as exc:
            st.error(f"Hedge execution failed: {exc}")

    st.caption(
        "Note: DQN is a simple placeholder; future versions can improve policy/training."
    )


def render_tab() -> None:
    render_page_header(
        "Hedging Systems - Alpaca",
        "DQN-based Alpaca options hedging (no manual orders).",
        icon="???",
        badge="Hedging",
    )

    # Top row: account snapshot vs DQN hedging panel
    col_left, col_right = st.columns(2)
    with col_left:
        _render_account()

    with col_right:
        _render_dqn_panel()

    st.divider()

    # Positions shown read-only
    _render_positions()


def render() -> None:
    render_tab()
