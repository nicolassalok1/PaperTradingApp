import pandas as pd
import streamlit as st
import altair as alt

from app.controller import dashboard_v2_controller as ctrl
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "📊 Dashboard"


def _render_account_overview(summary: dict, drawdown: dict, risk: dict) -> None:
    """Top-level KPIs giving an immediate sense of the Alpaca account."""
    equity = float(summary.get("equity", 0.0) or 0.0)
    cash = float(summary.get("cash", 0.0) or 0.0)
    pv = float(summary.get("portfolio_value", equity) or equity)
    bp = float(summary.get("buying_power", 0.0) or 0.0)
    unreal = float(summary.get("unrealized_pl_total", 0.0) or 0.0)
    realized = float(summary.get("realized_pl_total", 0.0) or 0.0)
    max_dd = float(drawdown.get("max_drawdown", 0.0) or 0.0)
    gross = float(risk.get("gross_exposure", 0.0) or 0.0)
    net = float(risk.get("net_exposure", 0.0) or 0.0)
    largest = float(risk.get("largest_position_pct", 0.0) or 0.0)

    # Compact bar view to compare key dollar metrics at a glance
    chart_rows = [
        ("Equity", equity),
        ("Cash", cash),
        ("Portfolio Value", pv),
        ("Buying Power", bp),
        ("Gross Exposure", gross),
        ("Net Exposure", net),
    ]
    df_chart = pd.DataFrame(chart_rows, columns=["Metric", "Value"])
    if not df_chart.empty:
        st.markdown("#### Vue synthétique (barres)")
        base = (
            alt.Chart(df_chart)
            .encode(
                y=alt.Y("Metric:N", sort="-x", title=None),
                x=alt.X("Value:Q", title="USD"),
                color=alt.condition(
                    alt.datum.Value >= 0,
                    alt.value("#34d399"),  # green for positive
                    alt.value("#f87171"),  # red for negative
                ),
                tooltip=[
                    alt.Tooltip("Metric:N", title="Stat"),
                    alt.Tooltip("Value:Q", title="Valeur", format=",.2f"),
                ],
            )
        )
        bars = base.mark_bar(cornerRadius=6)
        labels = base.mark_text(
            align="left", baseline="middle", dx=6, color="#e5e7eb"
        ).encode(text=alt.Text("Value:Q", format=",.2f"))
        st.altair_chart((bars + labels).properties(height=240), use_container_width=True)


def _render_performance_section(lookback_days: int) -> None:
    """Compact performance view: equity vs PnL."""
    col_eq, col_pnl = st.columns(2)

    with col_eq:
        st.markdown("#### Courbe d’equity")
        eq_curve = ctrl.get_equity_curve(lookback_days)
        if eq_curve:
            df_eq = pd.DataFrame(eq_curve)
            if not df_eq.empty and "date" in df_eq.columns:
                df_eq["date"] = pd.to_datetime(df_eq["date"])
                df_eq = df_eq.set_index("date")
                st.line_chart(
                    df_eq[["equity", "portfolio_value"]],
                    width="stretch",
                )
            else:
                st.info("Equity curve unavailable.")
        else:
            st.info("No equity data for the selected window.")

    with col_pnl:
        st.markdown("#### PnL dans le temps")
        pnl_ts = ctrl.get_pnl_timeseries(lookback_days)
        if pnl_ts:
            df_pnl = pd.DataFrame(pnl_ts)
            if not df_pnl.empty and "date" in df_pnl.columns:
                df_pnl["date"] = pd.to_datetime(df_pnl["date"])
                df_pnl = df_pnl.set_index("date")
                cols = [c for c in ["pnl", "cum_pnl"] if c in df_pnl.columns]
                if cols:
                    st.line_chart(df_pnl[cols], width="stretch")
                else:
                    st.info("PnL columns not available.")
            else:
                st.info("PnL timeseries unavailable.")
        else:
            st.info("No PnL data for the selected window.")


def _render_positions_and_exposure() -> None:
    """Side-by-side positions snapshot and exposure by symbol."""
    col_pos, col_expo = st.columns(2)

    with col_pos:
        st.markdown("#### Positions ouvertes")
        spot = ctrl.get_spot_positions()
        options = ctrl.get_option_positions()

        if not spot and not options:
            st.info("No open Alpaca positions.")
        else:
            if spot:
                with st.expander("Actions", expanded=True):
                    st.dataframe(
                        pd.DataFrame(spot),
                        hide_index=True,
                        width="stretch",
                    )
            if options:
                with st.expander("Options", expanded=True):
                    st.dataframe(
                        pd.DataFrame(options),
                        hide_index=True,
                        width="stretch",
                    )

    with col_expo:
        st.markdown("#### Exposition par symbole")
        exposure = ctrl.get_exposure_by_symbol()
        if exposure:
            df_expo = pd.DataFrame(exposure)
            if not df_expo.empty and "symbol" in df_expo.columns:
                df_expo = df_expo.set_index("symbol")
                if "market_value" in df_expo.columns:
                    st.bar_chart(
                        df_expo["market_value"],
                        width="stretch",
                    )
                st.dataframe(
                    df_expo.reset_index(),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("Exposure data malformed.")
        else:
            st.info("No exposure data available.")


def _render_pnl_attribution_expander() -> None:
    """Optional detail view: PnL breakdown by symbol."""
    with st.expander("PnL attribution (détails)", expanded=False):
        pnl_attr = ctrl.get_pnl_attribution()
        by_symbol = pnl_attr.get("by_symbol", [])
        if not by_symbol:
            st.info("No PnL attribution available.")
            return

        df_attr = pd.DataFrame(by_symbol)
        if df_attr.empty:
            st.info("PnL attribution is empty.")
            return

        st.dataframe(df_attr, hide_index=True, width="stretch")
        if "symbol" in df_attr.columns and "unrealized_pnl" in df_attr.columns:
            st.bar_chart(
                df_attr.set_index("symbol")["unrealized_pnl"],
                width="stretch",
            )

        total_unreal = float(pnl_attr.get("total_unrealized", 0.0) or 0.0)
        total_real = float(pnl_attr.get("total_realized", 0.0) or 0.0)
        c1, c2 = st.columns(2)
        c1.metric("Total Unrealized PnL", f"${total_unreal:,.2f}")
        c2.metric("Total Realized PnL", f"${total_real:,.2f}")


def _render_trade_history_expander() -> None:
    """Compact trade history browser kept out of the main flow."""
    with st.expander("Historique des ordres (Alpaca)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            days_back = st.number_input(
                "Days back", min_value=1, max_value=365, value=30, step=1
            )
        with col2:
            limit = st.number_input(
                "Max orders", min_value=10, max_value=500, value=200, step=10
            )
        with col3:
            symbol_filter = st.text_input(
                "Filter by symbol (optional)", value=""
            ).strip().upper()

        trades = ctrl.get_trade_history(limit=int(limit), days_back=int(days_back))
        if symbol_filter:
            trades = [
                t for t in trades if (t.get("symbol") or "").upper() == symbol_filter
            ]

        if trades:
            st.dataframe(
                pd.DataFrame(trades),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No trades for the selected period.")


def render_tab() -> None:
    render_page_header(
        "Dashboard (Alpaca)",
        "Vue d’ensemble du compte: cash, PnL, exposition, positions et historique.",
        icon="📊",
        badge="Overview",
    )

    summary = ctrl.get_account_summary()
    drawdown = ctrl.get_drawdowns()
    risk = ctrl.get_live_risk_snapshot()

    _render_account_overview(summary, drawdown, risk)
    st.divider()

    st.markdown("### Positions & exposition")
    _render_positions_and_exposure()

    st.divider()
    _render_pnl_attribution_expander()
    _render_trade_history_expander()


def render() -> None:
    render_tab()
