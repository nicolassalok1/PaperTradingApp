import time

import pandas as pd
import streamlit as st

from app.controller import buy_sell_controller as ctrl

floor_4 = ctrl.floor_4
from app.vue.components.page_utils import render_closing_history_chart, render_page_header


def render():
    load_portfolio = ctrl.load_portfolio
    save_portfolio = ctrl.save_portfolio

    def _dashboard_price(symbol: str, fallback: float = 0.0) -> float:
        # Unified spot API: use app.model.market_data (same as closes) to avoid deltas vs dashboard prices.
        return ctrl.get_market_price(symbol, fallback)

    def _trade_spot(symbol: str, fallback: float = 0.0) -> float:
        return ctrl.trade_spot_with_fallback(symbol, fallback)

    def _compute_spot_totals(portfolio: dict) -> tuple[float, float, float]:
        return ctrl.compute_spot_totals_with_price(portfolio)

    def _buy_asset(symbol: str, quantity: float, price: float, source: str = "manual"):
        return ctrl.buy_asset(
            symbol,
            quantity,
            price,
            source=source,
        )

    def _sell_asset(symbol: str, quantity: float, price: float, source: str = "manual"):
        return ctrl.sell_asset(
            symbol,
            quantity,
            price,
            source=source,
        )

    render_page_header(
        "Buy / Sell",
        "Execution manuelle, calcul des coûts et log des ordres",
        icon="💱",
        badge="Spot",
    )

    with st.expander("📘 Comprendre Buy/Sell"):
        st.markdown(
            """
            ### 💰 Ce que vous faites dans Buy/Sell

            Cet onglet est votre **poste d’exécution manuelle** : c’est ici que vous décidez
            consciemment d’entrer, renforcer, réduire ou retourner une position, en contrôlant précisément prix et quantité.

            Le bloc *Buy / Cover Asset* permet soit d’acheter pour être ou rester **Long**, soit d’acheter pour **couvrir un short**.
            Vous choisissez la *Direction* (Long/Short), le symbole, la quantité et le prix d’exécution.

            Le bloc *Sell / Short Asset* sert à gérer les positions existantes : vendre une partie d’un long, le clôturer entièrement,
            ou vendre au-delà de votre quantité actuelle pour devenir **net short** sur un actif.

            À droite, vous voyez à chaque fois la position en place (quantité, prix moyen, sens long/short) et le P&L estimé du trade
            avant de cliquer, ce qui vous aide à visualiser l’impact concret de l’ordre sur votre portefeuille.

            Utilisez cet onglet pour **intervenir manuellement** malgré vos systèmes automatiques : prendre des profits, couper une perte,
            inverser une position ou initier un short tactique, tout en gardant en tête le P&L et le risque global de votre compte.
            """
        )

    # BUY Section (stacked layout)
    st.markdown("### 📈 Buy / Cover Asset")
    buy_side = st.radio(
        "Direction",
        options=["Long", "Short"],
        index=0,
        horizontal=True,
        key="buy_side",
    )
    buy_symbol = st.text_input("Symbol to Buy", placeholder="e.g., AAPL", key="buy_symbol").upper()
    fetch_buy_history = st.button(
        "🔄 Charger les clôtures 1Y pour ce ticker", key="btn_fetch_buy_history"
    )
    if fetch_buy_history:
        st.session_state["_buy_history_prefetched_symbol"] = ""  # reset to force refetch
        st.session_state.pop("_history_msg_buy_history_" + buy_symbol, None)
        st.session_state["_buy_history_fetch_triggered"] = True
    if buy_symbol and st.session_state.get("_buy_history_fetch_triggered"):
        try:
            with st.spinner(f"Préchargement des clôtures 1Y pour {buy_symbol}..."):
                df_prefetch_buy, _, _ = ctrl.fetch_closing_history(
                    buy_symbol, period="1y", interval="1d"
                )
            if df_prefetch_buy is not None and not df_prefetch_buy.empty:
                st.session_state["_buy_history_prefetched_symbol"] = buy_symbol
        except Exception:
            pass
        render_closing_history_chart(buy_symbol, "buy_history", location_label="Buy/Sell")
    elif buy_symbol:
        st.caption("Clique sur le bouton ci-dessus pour charger les clôtures 1Y.")
    buy_quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="buy_qty")

    if buy_symbol:
        spot_price = _trade_spot(buy_symbol, 0.0)
        trade_price = floor_4(spot_price)
        if trade_price > 0:
            st.info(f"Current price: ${trade_price:.4f}")
            total_cost = buy_quantity * trade_price
            st.metric("Total Cost", f"${total_cost:.4f}")

            if st.button("✅ Execute Order", type="primary", key="exec_buy"):
                order_executed = False
                if buy_side == "Long":
                    result = _buy_asset(buy_symbol, buy_quantity, trade_price)
                    st.success(f"Bought {buy_quantity} units of {buy_symbol} @ ${trade_price:.4f}")
                    if result:
                        side = result.get("side", "long").upper()
                        st.info(
                            f"New position: {result.get('quantity', 0)} units @ price ${trade_price:.4f} "
                            f"({side})"
                        )
                    else:
                        st.info("Position fully closed.")
                    order_executed = True
                else:
                    if _sell_asset(buy_symbol, buy_quantity, trade_price):
                        st.success(
                            f"Shorted {buy_quantity} units of {buy_symbol} @ ${trade_price:.4f}"
                        )
                        portfolio_after = load_portfolio()
                        new_pos = portfolio_after.get(buy_symbol)
                        if new_pos:
                            side = new_pos.get("side", "short").upper()
                            st.info(
                                f"New position: {new_pos.get('quantity', 0)} units @ price ${trade_price:.4f} "
                                f"({side})"
                            )
                        order_executed = True
                    else:
                        st.error("Failed to execute order")
                if order_executed:
                    ctrl.clear_closing_history_cache(buy_symbol)
                    time.sleep(1)
                    st.rerun()
        else:
            st.error(f"Could not fetch price for {buy_symbol}")

    st.divider()

    # SELL / SHORT Section (stacked layout)
    st.markdown("### 📉 Sell / Short Asset")
    my_portfolio = load_portfolio()

    if my_portfolio:
        # Lazy-load 1y closing history for the first ticker to warm the cache (silent).
        sell_options = sorted(my_portfolio.keys())
        default_sell_symbol = sell_options[0] if sell_options else None
        prefetched_symbol = st.session_state.get("_sell_history_prefetched_symbol")
        need_prefetch = default_sell_symbol and prefetched_symbol != default_sell_symbol
        if default_sell_symbol and need_prefetch:
            status_box = st.empty()
            try:
                with st.spinner(f"Préchargement des clôtures 1Y pour {default_sell_symbol}..."):
                    df_prefetch, _, _ = ctrl.fetch_closing_history(
                        default_sell_symbol, period="1y", interval="1d"
                    )
                if df_prefetch is not None and not df_prefetch.empty:
                    status_box.success(f"Clôtures 1Y chargées pour {default_sell_symbol}.")
                    st.session_state["_sell_history_prefetched_symbol"] = default_sell_symbol
                else:
                    status_box.info("Clôtures introuvables sur 1 an pour ce ticker.")
            except Exception:
                status_box.info(
                    "Clôtures non chargées automatiquement. Sélectionne un ticker pour réessayer."
                )

        sell_symbol = st.selectbox(
            "Symbol to Sell/Short",
            options=sell_options,
            key="sell_symbol",
        )

        render_closing_history_chart(sell_symbol, "sell_history", location_label="Buy/Sell")

        if sell_symbol:
            position = my_portfolio[sell_symbol]
            current_qty = float(position.get("quantity", 0.0) or 0.0)
            entry_price = float(
                position.get("s_0_price", position.get("current_price", 0.0)) or 0.0
            )
            side = position.get("side", "long")

            sell_quantity = st.number_input(
                "Quantity to Sell (you can sell more than you hold to go net short)",
                min_value=1,
                value=1,
                step=1,
                key="sell_qty",
            )

            action_options = ["Sell/Short more"]
            if side == "short":
                action_options.append("Buy to cover")
            if len(action_options) > 1:
                action = st.radio(
                    "Action",
                    options=action_options,
                    index=0,
                    horizontal=True,
                    key="sell_action",
                )
            else:
                action = action_options[0]

            market_price = _trade_spot(sell_symbol, 0.0)
            trade_price = floor_4(market_price)
            if trade_price > 0:
                st.info(f"Current market price used: ${trade_price:.4f}")

                if action == "Buy to cover":
                    cash_flow = -sell_quantity * trade_price  # cash out
                    pnl = (avg_price - trade_price) * sell_quantity
                    notional = avg_price * sell_quantity
                    pnl_pct = (pnl / notional * 100) if notional > 0 else 0.0
                    st.metric("Cash Outlay", f"${-cash_flow:.2f}")
                    st.metric("P&L (per this cover)", f"${pnl:.2f}", delta=f"{pnl_pct:.2f}%")

                    if st.button("✅ Buy to cover", type="primary", key="exec_cover"):
                        result = _buy_asset(sell_symbol, sell_quantity, trade_price)
                        st.success(
                            f"Bought {sell_quantity} units of {sell_symbol} @ market ${trade_price:.4f} to cover short"
                        )
                        if result:
                            side_new = result.get("side", "long").upper()
                            st.info(
                                f"New position: {result.get('quantity', 0)} units @ price ${trade_price:.4f} "
                                f"({side_new})"
                            )
                        else:
                            st.info("Position fully closed.")
                        ctrl.clear_closing_history_cache(sell_symbol)
                        time.sleep(1)
                        st.rerun()
                else:
                    total_proceeds = sell_quantity * trade_price
                    if side == "long":
                        pnl = (trade_price - entry_price) * sell_quantity
                    else:
                        pnl = (entry_price - trade_price) * sell_quantity
                    notional = abs(entry_price * sell_quantity)
                    pnl_pct = (pnl / notional * 100) if notional > 0 else 0.0

                    st.metric("Total Proceeds", f"${total_proceeds:.2f}")
                    if st.button("✅ Execute Sell / Short", type="primary", key="exec_sell"):
                        if _sell_asset(sell_symbol, sell_quantity, trade_price):
                            st.success(
                                f"Sold {sell_quantity} units of {sell_symbol} @ market ${trade_price:.4f}"
                            )
                            ctrl.clear_closing_history_cache(sell_symbol)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to execute sell order")
            else:
                st.error(f"Could not fetch price for {sell_symbol}")
    else:
        st.info("No assets in portfolio to sell or short")

    st.markdown("---")
    st.subheader("🧾 Trading Orders Log")
    trades_log = ctrl.load_trades_log()
    if trades_log:
        # Show most recent first, cap to last 50 rows for readability
        sorted_trades = sorted(trades_log, key=lambda e: e.get("timestamp", ""), reverse=True)[:50]
        log_rows = []
        for entry in sorted_trades:
            meta = entry.get("meta", {})
            log_rows.append(
                {
                    "Timestamp": entry.get("timestamp"),
                    "Symbol": entry.get("symbol"),
                    "Side": entry.get("side", entry.get("action")),
                    "Qty": entry.get("quantity"),
                    "Spot Price": entry.get("price"),
                    "Source": entry.get("source") or "unknown",
                }
            )
        st.dataframe(pd.DataFrame(log_rows), hide_index=True)
        st.caption("Derniers ordres exécutés (manuels et automatiques).")
    else:
        st.info("Aucun ordre exécuté pour le moment.")
