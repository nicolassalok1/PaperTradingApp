import pandas as pd
import streamlit as st

from app.controller import dashboard_v2_controller as ctrl

TAB_LABEL = "📊 Dashboard v2"


def _render_top_kpis(summary: dict, drawdown: dict, risk: dict):
    equity = summary.get("equity", 0.0)
    cash = summary.get("cash", 0.0)
    pv = summary.get("portfolio_value", 0.0)
    bp = summary.get("buying_power", 0.0)
    unreal = summary.get("unrealized_pl_total", 0.0)
    realized = summary.get("realized_pl_total", 0.0)
    max_dd = drawdown.get("max_drawdown", 0.0)
    gross = risk.get("gross_exposure", 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"${equity:,.2f}")
    c2.metric("Cash", f"${cash:,.2f}")
    c3.metric("Portfolio Value", f"${pv:,.2f}")
    c4.metric("Buying Power", f"${bp:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Unrealized PnL", f"${unreal:,.2f}", delta=f"{unreal:,.2f}")
    c6.metric("Realized PnL", f"${realized:,.2f}", delta=f"{realized:,.2f}")
    c7.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    c8.metric("Gross Exposure", f"${gross:,.2f}")


def _render_positions(title: str, data):
    st.subheader(title)
    if data:
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
    else:
        st.info("No data to display.")


def _render_charts(lookback_days: int):
    left, right = st.columns(2)
    with left:
        st.subheader("Equity & Portfolio Value Over Time")
        eq_curve = ctrl.get_equity_curve(lookback_days)
        if eq_curve:
            df_eq = pd.DataFrame(eq_curve)
            df_eq["date"] = pd.to_datetime(df_eq["date"])
            st.line_chart(
                df_eq.set_index("date")[["equity", "portfolio_value"]],
                use_container_width=True,
            )
        else:
            st.info("No equity curve data.")
    with right:
        st.subheader("PnL & Volatility")
        pnl_ts = ctrl.get_pnl_timeseries(lookback_days)
        vol_ts = ctrl.get_volatility(lookback_days, window=20)
        if pnl_ts:
            df_pnl = pd.DataFrame(pnl_ts)
            df_pnl["date"] = pd.to_datetime(df_pnl["date"])
            st.line_chart(
                df_pnl.set_index("date")[["pnl", "cum_pnl"]], use_container_width=True
            )
        if vol_ts:
            df_vol = pd.DataFrame(vol_ts)
            df_vol["date"] = pd.to_datetime(df_vol["date"])
            st.line_chart(df_vol.set_index("date"), use_container_width=True)


def _render_exposure():
    st.subheader("Exposure by Symbol")
    data = ctrl.get_exposure_by_symbol()
    if data:
        df = pd.DataFrame(data)
        st.bar_chart(
            df.set_index("symbol")["market_value"], use_container_width=True
        )
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No exposure data.")

    sectors = ctrl.get_exposure_by_sector()
    if sectors:
        st.subheader("Exposure by Sector")
        df_sec = pd.DataFrame(sectors)
        st.bar_chart(
            df_sec.set_index("sector")["market_value"], use_container_width=True
        )


def _render_pnl_attribution():
    st.subheader("PnL Attribution by Symbol")
    pnl_attr = ctrl.get_pnl_attribution()
    by_symbol = pnl_attr.get("by_symbol", [])
    if by_symbol:
        df_attr = pd.DataFrame(by_symbol)
        st.dataframe(df_attr, hide_index=True, use_container_width=True)
        if "unrealized_pnl" in df_attr.columns and "symbol" in df_attr.columns:
            st.bar_chart(
                df_attr.set_index("symbol")["unrealized_pnl"],
                use_container_width=True,
            )
    else:
        st.info("No PnL attribution available.")
    total_unreal = pnl_attr.get("total_unrealized", 0.0)
    total_real = pnl_attr.get("total_realized", 0.0)
    c1, c2 = st.columns(2)
    c1.metric(
        "Total Unrealized PnL",
        f"${total_unreal:,.2f}",
        delta=f"{total_unreal:,.2f}",
    )
    c2.metric(
        "Total Realized PnL",
        f"${total_real:,.2f}",
        delta=f"{total_real:,.2f}",
    )


def _render_trade_history():
    st.subheader("Trade History (Alpaca)")
    days_back = st.number_input(
        "Days back", min_value=1, max_value=365, value=30, step=1
    )
    limit = st.number_input(
        "Max orders", min_value=10, max_value=500, value=200, step=10
    )
    symbol_filter = st.text_input("Filter symbol (optional)", value="").strip().upper()
    trades = ctrl.get_trade_history(limit=int(limit), days_back=int(days_back))
    if symbol_filter:
        trades = [
            t for t in trades if (t.get("symbol") or "").upper() == symbol_filter
        ]
    if trades:
        st.dataframe(pd.DataFrame(trades), hide_index=True, use_container_width=True)
    else:
        st.info("No trades found for the period.")


def _render_risk_panel():
    st.subheader("Live Risk Snapshot")
    risk = ctrl.get_live_risk_snapshot()
    gross = risk.get("gross_exposure", 0.0)
    net = risk.get("net_exposure", 0.0)
    largest = risk.get("largest_position_pct", 0.0)
    var_lite = risk.get("var_lite", None)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Exposure", f"${gross:,.2f}")
    c2.metric("Net Exposure", f"${net:,.2f}")
    c3.metric("Largest Position %", f"{largest*100:.2f}%")
    c4.metric("VaR-lite (proxy)", f"${var_lite:,.2f}" if var_lite is not None else "n/a")
    if largest > 0.4:
        st.warning("Largest position exceeds 40% of gross exposure.")
    if var_lite is not None and var_lite < -0.05 * gross:
        st.warning("VaR-lite indicates elevated downside risk.")


def render_tab() -> None:
    st.title("📊 Dashboard v2 - Alpaca Overview")

    summary = ctrl.get_account_summary()
    drawdown = ctrl.get_drawdowns()
    risk = ctrl.get_live_risk_snapshot()
    _render_top_kpis(summary, drawdown, risk)
    st.markdown("---")

    st.subheader("Charts & Analytics")
    lookback_days = st.slider(
        "Lookback (days)", min_value=30, max_value=365, value=90, step=15
    )
    _render_charts(lookback_days)
    st.markdown("---")

    _render_exposure()
    st.markdown("---")

    _render_positions("Spot Positions (Alpaca)", ctrl.get_spot_positions())
    st.markdown("---")
    _render_positions("Option Positions (Alpaca)", ctrl.get_option_positions())
    st.markdown("---")

    _render_pnl_attribution()
    st.markdown("---")

    _render_trade_history()
    st.markdown("---")

    _render_risk_panel()


def render():
    render_tab()

