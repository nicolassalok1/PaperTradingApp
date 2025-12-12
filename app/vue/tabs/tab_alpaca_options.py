import pandas as pd
import streamlit as st

from app.controller import trading_controller as ctrl
from app.vue.components.page_utils import render_page_header


def _to_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _render_account_section() -> None:
    st.markdown("### Account metrics")
    try:
        account = ctrl.get_orders_account()
    except Exception as exc:
        st.error(f"Unable to load account: {exc}")
        return

    if not account:
        st.info("No account data returned.")
        return

    equity = _to_float(account.get("equity"))
    cash = _to_float(account.get("cash"))
    buying_power = _to_float(
        account.get("buying_power")
        or account.get("buying_power_usd")
        or account.get("multiplier")
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Equity", f"${equity:,.2f}")
    col2.metric("Cash", f"${cash:,.2f}")
    col3.metric("Buying power", f"${buying_power:,.2f}")
    st.caption(
        f"Account: {account.get('id') or account.get('account_number') or 'n/a'} | "
        f"Status: {account.get('status', 'unknown')}"
    )


def _render_option_positions_section() -> None:
    st.markdown("### Option positions")
    try:
        positions = ctrl.get_option_positions()
    except Exception as exc:
        st.error(f"Unable to load option positions: {exc}")
        return

    if not positions:
        st.info("No open option positions.")
        return

    df = pd.DataFrame(positions)
    preferred_cols = [
        "symbol",
        "asset_class",
        "qty",
        "side",
        "market_value",
        "unrealized_pl",
        "avg_entry_price",
        "current_price",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if cols:
        df = df[cols]
    st.dataframe(df, hide_index=True, use_container_width=True)


def _render_option_orders_section() -> None:
    st.markdown("### Open option orders")
    try:
        orders = ctrl.get_open_option_orders()
    except Exception as exc:
        st.error(f"Unable to load option orders: {exc}")
        return

    if not orders:
        st.info("No open option orders.")
        return

    df = pd.DataFrame(orders)
    preferred_cols = [
        "id",
        "symbol",
        "asset_class",
        "side",
        "qty",
        "type",
        "time_in_force",
        "status",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if cols:
        df = df[cols]
    st.dataframe(df, hide_index=True, use_container_width=True)


def _render_option_market_order_form() -> None:
    st.markdown("### Market order (options)")
    st.caption(
        "Enter the OPRA option symbol as used in Alpaca "
        "(e.g., AAPL240621C00150000) to buy or sell contracts."
    )
    with st.form("alpaca_option_market_order"):
        option_symbol = st.text_input(
            "Option symbol (OPRA)",
            placeholder="AAPL240621C00150000",
        ).upper()
        qty = st.number_input("Contracts", min_value=1, value=1, step=1)
        side = st.radio("Side", options=["Buy", "Sell"], horizontal=True)
        submitted = st.form_submit_button("Send option market order", type="primary")

    if submitted:
        if not option_symbol:
            st.warning("Please enter an option symbol.")
            return
        try:
            order = ctrl.create_option_market_order(option_symbol, qty, side.lower())
            order_id = order.get("id") or order.get("client_order_id") or "order sent"
            st.success(f"Option order sent: {order_id}")
        except Exception as exc:
            st.error(f"Option order failed: {exc}")


def render_tab() -> None:
    render_page_header(
        "Alpaca Options",
        "Trade options via Alpaca (market orders) and monitor option positions.",
        icon="💹",
        badge="Alpaca",
    )
    _render_account_section()
    st.divider()
    _render_option_positions_section()
    st.divider()
    _render_option_orders_section()
    st.divider()
    _render_option_market_order_form()


def render() -> None:
    """Keeps parity with other tabs if a generic router is used."""
    render_tab()

