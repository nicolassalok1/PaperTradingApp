import pandas as pd
import streamlit as st

from app.controller import trading_controller as ctrl
from app.model.options.logic import download_options_alpaca, fetch_alpaca_option_tickers
from app.vue.components.page_utils import render_page_header


_CHAIN_STATE_KEY = "alpaca_options_chain_df"
_CHAIN_TICKER_KEY = "alpaca_options_chain_ticker"
_TICKERS_STATE_KEY = "alpaca_options_underlyings"


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
    cols = [c for c in df.columns if c in preferred_cols]
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
    cols = [c for c in df.columns if c in preferred_cols]
    if cols:
        df = df[cols]
    st.dataframe(df, hide_index=True, use_container_width=True)


def _get_chain_from_state() -> pd.DataFrame | None:
    df = st.session_state.get(_CHAIN_STATE_KEY)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def _render_manual_opra_form() -> None:
    with st.expander("Advanced: enter OPRA symbol manually"):
        st.caption(
            "If you already know the exact OPRA symbol as used in Alpaca "
            "(e.g., AAPL240621C00150000), you can enter it directly below."
        )
        with st.form("alpaca_option_market_order_manual"):
            option_symbol = st.text_input(
                "Option symbol (OPRA)",
                placeholder="AAPL240621C00150000",
            ).upper()
            qty = st.number_input("Contracts (manual)", min_value=1, value=1, step=1)
            side = st.radio("Side (manual)", options=["Buy", "Sell"], horizontal=True)
            submitted = st.form_submit_button("Send option market order (manual)", type="secondary")

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


def _render_option_market_order_form() -> None:
    st.markdown("### Trade options via Alpaca chain")
    st.caption(
        "Choose a ticker, option type, time to maturity and strike. "
        "We fetch the options chain from Alpaca and build the OPRA symbol for you."
    )

    # --- Load / cache the available underlying tickers from Alpaca ---
    tickers = st.session_state.get(_TICKERS_STATE_KEY)
    if tickers is None:
        with st.spinner("Loading available tickers from Alpaca..."):
            tickers = fetch_alpaca_option_tickers(limit=200)
        st.session_state[_TICKERS_STATE_KEY] = tickers

    # --- Load / cache the options chain for a given underlying ---
    default_ticker = st.session_state.get(_CHAIN_TICKER_KEY)
    col_ticker, col_button = st.columns([3, 1])
    with col_ticker:
        if tickers:
            if not default_ticker or default_ticker not in tickers:
                default_ticker = tickers[0]
            ticker = st.selectbox(
                "Underlying ticker",
                options=tickers,
                index=tickers.index(default_ticker),
            )
        else:
            default_ticker = default_ticker or "AAPL"
            ticker = st.text_input("Underlying ticker", default_ticker).upper().strip()
    with col_button:
        load_clicked = st.button("Load options chain", use_container_width=True)

    if load_clicked and ticker:
        try:
            with st.spinner(f"Loading options for {ticker} from Alpaca..."):
                df_chain = download_options_alpaca(ticker)
        except Exception as exc:
            st.error(f"Unable to load options from Alpaca: {exc}")
            df_chain = None

        if df_chain is None or df_chain.empty:
            st.warning(f"No options returned for {ticker}.")
        else:
            st.session_state[_CHAIN_TICKER_KEY] = ticker
            st.session_state[_CHAIN_STATE_KEY] = df_chain
            st.success(f"{len(df_chain)} contracts loaded for {ticker}.")

    df_chain = _get_chain_from_state()

    if df_chain is None:
        st.info("Load an options chain above to select a contract, or use manual OPRA entry below.")
        _render_manual_opra_form()
        return

    df = df_chain.copy()
    required_cols = {"opra", "K", "T", "type"}
    if not required_cols.issubset(df.columns):
        st.warning("Options chain is missing required fields to build the selector. Falling back to manual entry.")
        _render_manual_opra_form()
        return

    # Clean and derive days to expiry
    try:
        df["T"] = pd.to_numeric(df["T"], errors="coerce")
        df = df.dropna(subset=["T"])
        df = df[df["T"] > 0].copy()
    except Exception:
        pass

    if df.empty:
        st.info("Options chain is empty after filtering; try reloading or another ticker.")
        _render_manual_opra_form()
        return

    df["days_to_expiry"] = (df["T"] * 365.0).round().astype(int)

    # --- User selects type, maturity, side ---
    type_options = ["Call", "Put"]
    col1, col2, col3 = st.columns(3)
    with col1:
        opt_type_label = st.selectbox("Option type", type_options)
        opt_type = "call" if opt_type_label.lower().startswith("c") else "put"

    df_type = df[df["type"].astype(str).str.lower() == opt_type]
    if df_type.empty:
        st.info(f"No {opt_type_label.lower()}s found in the loaded chain.")
        _render_manual_opra_form()
        return

    maturities = sorted(df_type["days_to_expiry"].unique())
    with col2:
        maturity_days = st.selectbox(
            "Time to maturity (days)",
            options=maturities,
            format_func=lambda d: f"{int(d)} days",
        )

    df_slice = df_type[df_type["days_to_expiry"] == maturity_days].copy()
    if df_slice.empty:
        st.info("No contracts for this maturity; try another selection.")
        _render_manual_opra_form()
        return

    with col3:
        side = st.radio("Side", options=["Buy", "Sell"], horizontal=True)

    df_slice = df_slice.sort_values("K")
    strikes = list(df_slice["K"].unique())

    if not strikes:
        st.info("No strikes available for this selection.")
        _render_manual_opra_form()
        return

    with st.form("alpaca_option_market_order_from_chain"):
        col_k, col_q = st.columns(2)
        with col_k:
            strike = st.selectbox(
                "Strike",
                options=strikes,
                format_func=lambda k: f"{k:g}",
            )
        with col_q:
            qty = st.number_input("Contracts", min_value=1, value=1, step=1)

        submitted = st.form_submit_button("Send option market order", type="primary")

    if submitted:
        chosen = df_slice[df_slice["K"] == strike]
        if chosen.empty:
            st.error("Selected strike not found in current slice.")
            return
        row = chosen.iloc[0]
        opra_symbol = str(row.get("opra") or "").strip().upper()
        if not opra_symbol:
            st.error("No OPRA symbol available for the selected contract.")
            return
        try:
            order = ctrl.create_option_market_order(opra_symbol, qty, side.lower())
            order_id = order.get("id") or order.get("client_order_id") or "order sent"
            st.success(f"Option order sent: {order_id}")
        except Exception as exc:
            st.error(f"Option order failed: {exc}")

    _render_manual_opra_form()


def render_tab() -> None:
    render_page_header(
        "Alpaca Options",
        "Trade options via Alpaca: select ticker, expiry and strike from the live options chain.",
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
