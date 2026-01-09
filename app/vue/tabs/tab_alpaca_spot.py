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
    st.markdown("### Métriques du compte")
    try:
        account = ctrl.get_spot_account()
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


def _render_positions_section() -> None:
    st.markdown("### Positions")
    try:
        positions = ctrl.get_spot_positions()
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
        "market_value",
        "cost_basis",
        "avg_entry_price",
        "current_price",
        "unrealized_pl",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if cols:
        df = df[cols]
    st.dataframe(df, hide_index=True, width="stretch")


def _render_orders_section() -> None:
    st.markdown("### Open orders")
    try:
        orders = ctrl.get_spot_open_orders()
    except Exception as exc:
        st.error(f"Unable to load orders: {exc}")
        return

    if not orders:
        st.info("No open orders.")
        return

    df = pd.DataFrame(orders)
    preferred_cols = ["id", "symbol", "side", "qty", "type", "time_in_force", "status"]
    cols = [c for c in preferred_cols if c in df.columns]
    if cols:
        df = df[cols]
    st.dataframe(df, hide_index=True, width="stretch")


def _render_order_form() -> None:
    st.markdown("### Market order")
    with st.form("alpaca_spot_order"):
        symbol = st.text_input("Symbol", placeholder="AAPL or TSLA").upper()
        qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
        side = st.radio("Side", options=["Buy", "Sell"], horizontal=True)
        submitted = st.form_submit_button("Send market order", type="primary")

    if submitted:
        if not symbol:
            st.warning("Please enter a symbol.")
            return
        try:
            order = ctrl.send_spot_market_order(symbol, qty, side.lower())
            order_id = order.get("id") or order.get("client_order_id") or "order sent"
            st.success(f"Order sent: {order_id}")
        except Exception as exc:
            st.error(f"Order failed: {exc}")


def _render_price_history() -> None:
    st.markdown("### Price history")
    symbol = st.text_input("Symbol for history", placeholder="MSFT").upper()
    timeframe = st.selectbox(
        "Timeframe",
        options=["1Day", "1Hour", "30Min", "15Min", "5Min", "1Min"],
        index=0,
    )
    limit = st.slider("Number of bars", min_value=20, max_value=500, value=120, step=10)

    if not symbol:
        st.info("Enter a symbol to load price history.")
        return

    try:
        df_hist = ctrl.get_price_history(symbol, timeframe=timeframe, limit=limit)
    except Exception as exc:
        st.error(f"Unable to load history: {exc}")
        return

    if df_hist is None or df_hist.empty:
        st.info("No price history returned.")
        return

    df_hist = df_hist.copy()
    if "time" in df_hist.columns:
        df_hist["time"] = pd.to_datetime(df_hist["time"], errors="coerce")
        df_hist = df_hist.dropna(subset=["time"])
        df_hist = df_hist.sort_values("time")
        df_hist = df_hist.set_index("time")

    price_col = None
    for candidate in ["close", "Close", "c"]:
        if candidate in df_hist.columns:
            price_col = candidate
            break

    if not price_col:
        st.dataframe(df_hist, hide_index=True, width="stretch")
        return

    st.line_chart(df_hist[[price_col]], height=300, width="stretch")
    st.caption(f"{len(df_hist)} bars loaded for {symbol} ({timeframe}).")


def render_tab() -> None:
    render_page_header(
        "Alpaca Spot",
        "Vue compte + positions + ordres spot (actions/ETF) via Alpaca.",
        icon="⚡",
        badge="Alpaca",
    )
    _render_account_section()
    st.divider()
    _render_positions_section()
    st.divider()
    _render_orders_section()
    st.divider()
    _render_order_form()
    st.divider()
    _render_price_history()


def render() -> None:
    """Keep consistency with other tabs if a generic router is used."""
    render_tab()
