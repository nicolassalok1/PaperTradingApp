import pandas as pd
import streamlit as st

from app.controller import alpaca_orders_controller as ctrl
from app.vue.components.page_utils import render_page_header


def _to_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _render_account_section() -> None:
    st.markdown("### Account metrics")
    try:
        account = ctrl.get_account()
    except Exception as exc:
        st.error(f"Unable to load account: {exc}")
        return

    if not account:
        st.info("No account data returned.")
        return

    equity = _to_float(account.get("equity"))
    cash = _to_float(account.get("cash"))
    buying_power = _to_float(account.get("buying_power") or account.get("buying_power_usd"))
    col1, col2, col3 = st.columns(3)
    col1.metric("Equity", f"${equity:,.2f}")
    col2.metric("Cash", f"${cash:,.2f}")
    col3.metric("Buying power", f"${buying_power:,.2f}")
    st.caption(
        f"Account: {account.get('id') or account.get('account_number') or 'n/a'} | "
        f"Status: {account.get('status', 'unknown')}"
    )


def _render_positions_section() -> None:
    st.markdown("### Positions")
    try:
        positions = ctrl.get_positions()
    except Exception as exc:
        st.error(f"Unable to load positions: {exc}")
        return

    if not positions:
        st.info("No open positions.")
        return

    df = pd.DataFrame(positions)
    preferred_cols = [
        "symbol",
        "qty",
        "side",
        "avg_entry_price",
        "current_price",
        "market_value",
        "unrealized_pl",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if cols:
        df = df[cols]
    st.dataframe(df, hide_index=True, use_container_width=True)


def _render_orders_section() -> None:
    st.markdown("### Open orders")
    try:
        orders = ctrl.get_open_orders()
    except Exception as exc:
        st.error(f"Unable to load orders: {exc}")
        return

    if not orders:
        st.info("No open orders.")
        return

    df = pd.DataFrame(orders)
    preferred_cols = ["id", "symbol", "side", "qty", "type", "time_in_force", "status", "limit_price", "stop_price"]
    cols = [c for c in preferred_cols if c in df.columns]
    if cols:
        df = df[cols]
    st.dataframe(df, hide_index=True, use_container_width=True)


def _submit_with_feedback(label: str, submit_fn, *args):
    try:
        order = submit_fn(*args)
        order_id = order.get("id") or order.get("client_order_id") or "submitted"
        st.success(f"{label} sent: {order_id}")
        return True
    except Exception as exc:
        st.error(f"{label} failed: {exc}")
        return False


def _render_limit_form() -> None:
    st.markdown("### Limit order")
    with st.form("alpaca_limit_order"):
        symbol = st.text_input("Symbol", placeholder="AAPL").upper()
        qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
        price = st.number_input("Limit price", min_value=0.0, value=100.0, step=0.01, format="%.4f")
        side = st.radio("Side", options=["Buy", "Sell"], horizontal=True)
        submitted = st.form_submit_button("Submit limit order", type="primary")
    if submitted:
        if not symbol:
            st.warning("Enter a symbol.")
        else:
            _submit_with_feedback("Limit order", ctrl.create_limit_order, symbol, qty, price, side.lower())


def _render_stop_loss_form() -> None:
    st.markdown("### Stop-loss order")
    st.caption("Side is inverted to exit the position (long -> sell stop, short -> buy stop).")
    with st.form("alpaca_stop_loss_order"):
        symbol = st.text_input("Symbol (stop-loss)", placeholder="TSLA").upper()
        qty = st.number_input("Quantity (stop-loss)", min_value=0.0, value=1.0, step=1.0)
        stop_price = st.number_input("Stop price", min_value=0.0, value=90.0, step=0.01, format="%.4f")
        side = st.radio("Current position side", options=["Buy", "Sell"], horizontal=True, key="stop_side")
        submitted = st.form_submit_button("Submit stop-loss", type="primary")
    if submitted:
        if not symbol:
            st.warning("Enter a symbol.")
        else:
            _submit_with_feedback(
                "Stop-loss order",
                ctrl.create_stop_loss,
                symbol,
                qty,
                stop_price,
                side.lower(),
            )


def _render_take_profit_form() -> None:
    st.markdown("### Take-profit order")
    st.caption("Side is inverted to exit the position.")
    with st.form("alpaca_take_profit_order"):
        symbol = st.text_input("Symbol (take-profit)", placeholder="MSFT").upper()
        qty = st.number_input("Quantity (take-profit)", min_value=0.0, value=1.0, step=1.0)
        tp_price = st.number_input("Take-profit price", min_value=0.0, value=120.0, step=0.01, format="%.4f")
        side = st.radio("Current position side ", options=["Buy", "Sell"], horizontal=True, key="tp_side")
        submitted = st.form_submit_button("Submit take-profit", type="primary")
    if submitted:
        if not symbol:
            st.warning("Enter a symbol.")
        else:
            _submit_with_feedback(
                "Take-profit order",
                ctrl.create_take_profit,
                symbol,
                qty,
                tp_price,
                side.lower(),
            )


def _render_stop_limit_form() -> None:
    st.markdown("### Stop-limit order")
    with st.form("alpaca_stop_limit_order"):
        symbol = st.text_input("Symbol (stop-limit)", placeholder="AMZN").upper()
        qty = st.number_input("Quantity (stop-limit)", min_value=0.0, value=1.0, step=1.0)
        stop_price = st.number_input("Stop price (trigger)", min_value=0.0, value=95.0, step=0.01, format="%.4f")
        limit_price = st.number_input("Limit price (execution)", min_value=0.0, value=94.0, step=0.01, format="%.4f")
        side = st.radio("Side", options=["Buy", "Sell"], horizontal=True, key="stop_limit_side")
        submitted = st.form_submit_button("Submit stop-limit", type="primary")
    if submitted:
        if not symbol:
            st.warning("Enter a symbol.")
        else:
            _submit_with_feedback(
                "Stop-limit order",
                ctrl.create_stop_limit,
                symbol,
                qty,
                stop_price,
                limit_price,
                side.lower(),
            )


def _render_bracket_form() -> None:
    st.markdown("### Bracket order")
    st.caption("Entry (limit), take-profit (limit), and stop-loss (stop) legs sent together.")
    with st.form("alpaca_bracket_order"):
        symbol = st.text_input("Symbol (bracket)", placeholder="NFLX").upper()
        qty = st.number_input("Quantity (bracket)", min_value=0.0, value=1.0, step=1.0)
        entry_price = st.number_input("Entry price (limit)", min_value=0.0, value=100.0, step=0.01, format="%.4f")
        stop_price = st.number_input("Stop-loss price", min_value=0.0, value=90.0, step=0.01, format="%.4f")
        tp_price = st.number_input("Take-profit price", min_value=0.0, value=120.0, step=0.01, format="%.4f")
        side = st.radio("Entry side", options=["Buy", "Sell"], horizontal=True, key="bracket_side")
        submitted = st.form_submit_button("Submit bracket order", type="primary")
    if submitted:
        if not symbol:
            st.warning("Enter a symbol.")
        else:
            _submit_with_feedback(
                "Bracket order",
                ctrl.create_bracket_order,
                symbol,
                qty,
                entry_price,
                stop_price,
                tp_price,
                side.lower(),
            )


def render_tab() -> None:
    render_page_header(
        "Advanced Orders",
        "Alpaca limit, stop, take-profit, stop-limit, and bracket orders",
        icon="??",
        badge="Alpaca",
    )
    _render_account_section()
    st.divider()
    _render_positions_section()
    st.divider()
    _render_orders_section()
    st.divider()
    _render_limit_form()
    st.divider()
    _render_stop_loss_form()
    st.divider()
    _render_take_profit_form()
    st.divider()
    _render_stop_limit_form()
    st.divider()
    _render_bracket_form()


def render() -> None:
    """Keeps parity with other tabs if a generic router is used."""
    render_tab()
