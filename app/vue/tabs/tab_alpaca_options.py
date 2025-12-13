import os
from pathlib import Path
import time
import json

import pandas as pd
import streamlit as st

from app.controller import trading_controller as ctrl
from app.controller import options_controller as opt_ctrl
from app.vue.components.page_utils import render_page_header
from app.vue.components.ui_helpers import render_quickstart


_CHAIN_STATE_KEY = "alpaca_options_chain_df"
_CHAIN_TICKER_KEY = "alpaca_options_chain_ticker"
_TICKERS_STATE_KEY = "alpaca_options_underlyings"
_TICKERS_META_STATE_KEY = "alpaca_options_underlyings_meta"
_CLOCK_STATE_KEY = "alpaca_orders_clock"
_CLOCK_TS_STATE_KEY = "alpaca_orders_clock_ts"

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_repo_relative_path(path: Path) -> Path:
    try:
        if path.is_absolute():
            return path
    except Exception:
        pass
    return _REPO_ROOT / path

_OPTIONABLE_TICKERS_CSV = Path(
    os.getenv("ALPACA_OPTIONABLE_TICKERS_PATH", "data/alpaca_optionable_tickers.csv")
)
_OPTIONABLE_TICKERS_CSV = _resolve_repo_relative_path(_OPTIONABLE_TICKERS_CSV)

_PREFERRED_DEFAULTS: list[str] = ["SPY", "AAPL", "MSFT", "TSLA", "QQQ"]
_CLOCK_TTL_SEC = 60.0


def _get_orders_clock_cached() -> dict | None:
    now = time.time()
    ts = st.session_state.get(_CLOCK_TS_STATE_KEY)
    clock = st.session_state.get(_CLOCK_STATE_KEY)
    try:
        ts_f = float(ts) if ts is not None else None
    except Exception:
        ts_f = None

    if isinstance(clock, dict) and ts_f is not None and (now - ts_f) < _CLOCK_TTL_SEC:
        return clock

    try:
        clock = ctrl.get_orders_clock()
    except Exception:
        clock = None

    st.session_state[_CLOCK_STATE_KEY] = clock
    st.session_state[_CLOCK_TS_STATE_KEY] = now
    return clock if isinstance(clock, dict) else None


def _clock_is_open(clock: dict | None) -> bool | None:
    if not isinstance(clock, dict):
        return None
    is_open = clock.get("is_open")
    if isinstance(is_open, bool):
        return is_open
    if is_open is None:
        return None
    try:
        return bool(is_open)
    except Exception:
        return None


def _format_clock_dt(val) -> str:
    if val is None or val == "":
        return ""
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M")
    except Exception:
        try:
            return str(val)
        except Exception:
            return ""


def _format_alpaca_error(exc: Exception) -> str:
    raw = str(exc or "").strip()
    if not raw:
        return "Option order failed."
    try:
        if raw.startswith("{") and raw.endswith("}"):
            payload = json.loads(raw)
            if isinstance(payload, dict):
                code = payload.get("code")
                msg = payload.get("message") or raw
                return f"Option order failed: {msg}" + (f" (code {code})" if code else "")
    except Exception:
        pass
    return f"Option order failed: {raw}"


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
        clock = _get_orders_clock_cached()
        is_open = _clock_is_open(clock)
        status = "OPEN" if is_open else "CLOSED" if is_open is False else "n/a"
        next_open = _format_clock_dt((clock or {}).get("next_open"))
        if status != "n/a":
            st.caption(f"Market: {status}" + (f" | next open: {next_open}" if next_open and status == "CLOSED" else ""))

        st.caption(
            "If you already know the exact OPRA symbol as used in Alpaca "
            "(e.g., AAPL240621C00150000), you can enter it directly below."
        )
        with st.form("alpaca_option_market_order_manual"):
            option_symbol = st.text_input(
                "Option symbol (OPRA)",
                placeholder="AAPL240621C00150000",
            ).upper()
            order_type = st.radio(
                "Order type (manual)",
                options=["Market", "Limit"],
                horizontal=True,
            )
            limit_price = None
            if order_type == "Limit":
                limit_price = st.number_input(
                    "Limit price (manual)",
                    min_value=0.01,
                    value=0.10,
                    step=0.01,
                )
            qty = st.number_input("Contracts (manual)", min_value=1, value=1, step=1)
            side = st.radio("Side (manual)", options=["Buy", "Sell"], horizontal=True)
            disable_market = order_type == "Market" and is_open is False
            submitted = st.form_submit_button(
                f"Send option {order_type.lower()} order (manual)",
                type="secondary",
                disabled=disable_market,
            )

        if submitted:
            if not option_symbol:
                st.warning("Please enter an option symbol.")
                return
            try:
                if order_type == "Limit":
                    if limit_price is None or float(limit_price) <= 0:
                        st.warning("Please enter a valid limit price.")
                        return
                    order = ctrl.create_option_limit_order(option_symbol, qty, float(limit_price), side.lower())
                else:
                    order = ctrl.create_option_market_order(option_symbol, qty, side.lower())
                order_id = order.get("id") or order.get("client_order_id") or "order sent"
                st.success(f"Option order sent: {order_id}")
            except Exception as exc:
                st.error(_format_alpaca_error(exc))
                if "market hours" in str(exc).lower():
                    st.info("Market orders are only accepted during market hours. Use a limit order or try again when the market is open.")


def _render_option_market_order_form() -> None:
    st.markdown("### Trade options via Alpaca chain")
    st.caption(
        "Choose an underlying (from the precomputed list), option type, time to maturity and strike. "
        "We fetch the options chain from Alpaca and build the OPRA symbol for you."
    )

    # --- Load / cache the optionable underlyings from CSV (built offline) ---
    tickers = st.session_state.get(_TICKERS_STATE_KEY)
    tickers_meta = st.session_state.get(_TICKERS_META_STATE_KEY)

    if tickers is None or tickers_meta is None:
        tickers = []
        tickers_meta = {"n_contracts_by_symbol": {}}
        try:
            if _OPTIONABLE_TICKERS_CSV.exists():
                df_tickers = pd.read_csv(_OPTIONABLE_TICKERS_CSV)
                if not df_tickers.empty:
                    sym_col = "symbol" if "symbol" in df_tickers.columns else df_tickers.columns[0]
                    df_tickers = df_tickers.copy()
                    df_tickers[sym_col] = df_tickers[sym_col].astype(str).str.strip().str.upper()
                    df_tickers = df_tickers[df_tickers[sym_col] != ""].copy()
                    tickers = sorted(set(df_tickers[sym_col].tolist()))

                    if "n_contracts" in df_tickers.columns:
                        df_tickers["n_contracts"] = pd.to_numeric(df_tickers["n_contracts"], errors="coerce")
                        counts = (
                            df_tickers.dropna(subset=["n_contracts"])
                            .groupby(sym_col)["n_contracts"]
                            .max()
                            .astype(int)
                            .to_dict()
                        )
                        tickers_meta["n_contracts_by_symbol"] = counts
        except Exception as exc:
            st.warning(f"Unable to load optionable tickers list: {exc}")
            tickers = []
            tickers_meta = {"n_contracts_by_symbol": {}}

        st.session_state[_TICKERS_STATE_KEY] = tickers
        st.session_state[_TICKERS_META_STATE_KEY] = tickers_meta

    if tickers:
        st.caption(f"{len(tickers):,} optionable underlyings loaded from {_OPTIONABLE_TICKERS_CSV}")
    else:
        if _OPTIONABLE_TICKERS_CSV.exists():
            st.warning(f"Optionable tickers CSV is empty: {_OPTIONABLE_TICKERS_CSV}")
        else:
            st.info(
                "Optionable tickers CSV not found. Generate it with:\n"
                "`python scripts/build_optionable_universe.py`"
            )

    # --- Choose underlying & load the options chain ---
    default_ticker = st.session_state.get(_CHAIN_TICKER_KEY)
    col_ticker, col_reload, col_button = st.columns([3, 1, 1])
    with col_ticker:
        if tickers:
            if not default_ticker or default_ticker not in tickers:
                for sym in _PREFERRED_DEFAULTS:
                    if sym in tickers:
                        default_ticker = sym
                        break
                else:
                    default_ticker = tickers[0]

            counts_by_symbol = (tickers_meta or {}).get("n_contracts_by_symbol") or {}

            def _format_ticker(sym: str) -> str:
                try:
                    n = counts_by_symbol.get(sym)
                except Exception:
                    n = None
                if n is None:
                    return sym
                try:
                    n_int = int(n)
                except Exception:
                    return sym
                suffix = "+" if n_int >= 100 else ""
                return f"{sym} ({n_int}{suffix})"

            ticker = st.selectbox(
                "Underlying ticker",
                options=tickers,
                index=tickers.index(default_ticker),
                format_func=_format_ticker,
            )
        else:
            default_ticker = default_ticker or "AAPL"
            ticker = st.text_input("Underlying ticker", default_ticker).upper().strip()
    with col_reload:
        if st.button("Reload list", type="secondary", use_container_width=True):
            st.session_state.pop(_TICKERS_STATE_KEY, None)
            st.session_state.pop(_TICKERS_META_STATE_KEY, None)
            st.rerun()
    with col_button:
        load_clicked = st.button("Load options chain", use_container_width=True)

    with st.expander("Advanced: fetch settings", expanded=False):
        st.caption("Tip: Alpaca snapshots are paginated; this fetch pulls all pages by default.")
        feed = st.selectbox(
            "Options feed",
            options=["indicative", "opra"],
            index=0,
            help="`indicative` is usually available; `opra` may require OPRA agreement/subscription.",
            key="alpaca_options_feed",
        )
        min_days_to_expiry = st.number_input(
            "Min days to expiry",
            min_value=0,
            max_value=365,
            value=1,
            step=1,
            help="Use 0 to include same-day expiry contracts.",
            key="alpaca_options_min_days_to_expiry",
        )
        cache_to_csv = st.checkbox(
            "Cache chain to CSV",
            value=True,
            help="Writes `cache/AlpacaOptionChains/options_alpaca_{TICKER}.csv` for debugging/reuse.",
            key="alpaca_options_cache_to_csv",
        )

    if load_clicked and ticker:
        try:
            with st.spinner(f"Loading options for {ticker} from Alpaca..."):
                df_chain = opt_ctrl.download_alpaca_options_chain(
                    ticker,
                    feed=str(feed or "indicative"),
                    min_days_to_expiry=int(min_days_to_expiry) if min_days_to_expiry is not None else 1,
                    include_spot=True,
                    cache_to_csv=bool(cache_to_csv),
                )
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

    clock = _get_orders_clock_cached()
    is_open = _clock_is_open(clock)
    market_status = "OPEN" if is_open else "CLOSED" if is_open is False else "n/a"

    spot_val = None
    try:
        if "S0" in df_chain.columns:
            s0 = pd.to_numeric(df_chain["S0"], errors="coerce").dropna()
            if not s0.empty:
                spot_val = float(s0.iloc[0])
    except Exception:
        spot_val = None

    chain_ticker = st.session_state.get(_CHAIN_TICKER_KEY) or str(ticker or "").strip().upper()
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Underlying", chain_ticker or "n/a")
    col_m2.metric("Spot price", f"${spot_val:,.2f}" if spot_val is not None and spot_val == spot_val else "n/a")
    col_m3.metric("Contracts loaded", f"{len(df_chain):,}")
    col_m4.metric("Market", market_status)
    if market_status == "CLOSED":
        next_open = _format_clock_dt((clock or {}).get("next_open"))
        if next_open:
            st.caption(f"Next market open: {next_open}")

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

        order_type = st.radio("Order type", options=["Market", "Limit"], horizontal=True)
        limit_price = None
        if order_type == "Limit":
            limit_price = st.number_input(
                "Limit price",
                min_value=0.01,
                value=0.10,
                step=0.01,
            )

        disable_market = order_type == "Market" and is_open is False
        if disable_market:
            st.info("Market is closed: market orders are disabled. Switch to a limit order or try again during market hours.")

        submitted = st.form_submit_button(
            f"Send option {order_type.lower()} order",
            type="primary",
            disabled=disable_market,
        )

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
            if order_type == "Limit":
                if limit_price is None or float(limit_price) <= 0:
                    st.warning("Please enter a valid limit price.")
                    return
                order = ctrl.create_option_limit_order(opra_symbol, qty, float(limit_price), side.lower())
            else:
                order = ctrl.create_option_market_order(opra_symbol, qty, side.lower())
            order_id = order.get("id") or order.get("client_order_id") or "order sent"
            st.success(f"Option order sent: {order_id}")
        except Exception as exc:
            st.error(_format_alpaca_error(exc))
            if "market hours" in str(exc).lower():
                st.info("Market orders are only accepted during market hours. Use a limit order or try again when the market is open.")

    _render_manual_opra_form()


def render_tab() -> None:
    render_page_header(
        "Alpaca Options",
        "Chaîne d’options live Alpaca: sélection ticker/échéance/strike puis envoi d’ordres.",
        icon="💹",
        badge="Alpaca",
    )
    render_quickstart(
        "Guide rapide",
        [
            "Charge la chaîne d’options, puis filtre par maturité/strike selon ton besoin.",
            "Vérifie le sens (buy/sell) et la quantité avant d’envoyer un ordre.",
        ],
        expanded=False,
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
