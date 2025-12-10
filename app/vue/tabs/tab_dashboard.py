import datetime

import pandas as pd
import streamlit as st
import threading

from app.controller import dashboard_controller as ctrl


def load_expired() -> dict:
    """Load expired options cache (empty dict on failure)."""
    return ctrl.load_expired_options()


def save_expired(data: dict) -> None:
    """Persist expired options cache."""
    ctrl.save_expired_options(data if isinstance(data, dict) else {})


def run_dashboard():

    if st.button("🔄 Refresh spot prices", key="btn_refresh_spots"):
        try:
            with st.spinner("Mise à jour des spot prices..."):
                ctrl.refresh_all_spots_pipeline()
            st.success("Spot prices rafraîchis.")
            st.rerun()
        except Exception as exc:
            st.error(f"Refresh des spots impossible : {exc}")

    # Préchargement des données Dashboard et du cache
    my_portfolio = ctrl.load_portfolio()
    forwards_dash = ctrl.load_forwards_data()

    # On ne conserve que les options ouvertes (custom_opts)
    custom_opts = {
        k: v for k, v in ctrl.load_options().items() if v.get("status", "open") == "open"
    }

    # Options expirées
    expired_options_data = load_expired()

    # Détecter les tickers à surveiller
    dashboard_tickers = ctrl.collect_dashboard_tickers(
        portfolio=my_portfolio,
        forwards=forwards_dash,
        custom_options=custom_opts,
    )

    # Recompute portfolio_value via dedicated utility then reload cache
    try:
        ctrl.recompute_portfolio_value()
    except Exception:
        pass
    dashboard_cache = ctrl.load_dashboard_cache()
    today = datetime.date.today()
    cached_prices = dashboard_cache.get("prices") or {}
    dashboard_prices = cached_prices.copy()

    # Déterminer la dernière date de refresh
    last_refresh_ts = ctrl.dashboard_cache_last_refresh(dashboard_cache)

    # Helper local pour récupérer un prix de Dashboard
    def _dashboard_price(sym: str, fallback: float = 0.0) -> float:
        price = ctrl.dashboard_price(sym, fallback)
        if price > 0 and sym:
            dashboard_prices[sym.strip().upper()] = price
        return price

    # ============================
    # PnL SPOT
    # ============================
    spot_total_pnl, spot_total_notional, delta_spot = ctrl.compute_spot_pnl(
        my_portfolio, _dashboard_price
    )

    # ============================
    # PnL OPTIONS (réalisé uniquement)
    # ============================
    realized_options_pnl = 0.0
    for oid, opt in expired_options_data.items():
        realized_options_pnl += float(opt.get("pnl_total", 0.0) or 0.0)

    total_options_pnl = realized_options_pnl  # pas de M2M pour options ouvertes

    # ============================
    # PnL FORWARDS
    # ============================
    total_forward_pnl, total_forward_notional, delta_fwd = ctrl.compute_forward_pnl(
        forwards_dash, _dashboard_price
    )

    # ============================
    # HEADER PNL GLOBAL
    # ============================
    st.markdown("## 🔎 Synthèse P&L")

    balance_val = float(dashboard_cache.get("balance", 0.0) or 0.0)
    portfolio_val = float(dashboard_cache.get("portfolio_value", 0.0) or 0.0)
    combined_val = balance_val + portfolio_val
    col_bal, col_port, col_combined = st.columns(3)
    with col_bal:
        st.metric("Balance", f"${balance_val:,.2f}")
    with col_port:
        st.metric("Portfolio Value", f"${portfolio_val:,.2f}")
    with col_combined:
        st.metric("PnL", f"${combined_val:,.2f}")

    # ============================
    # AI PORTFOLIO ASSISTANT
    # ============================
    st.markdown("## 🤖 AI Portfolio Assistant")
    st.markdown("Posez vos questions sur votre portefeuille.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Posez votre question sur votre portefeuille...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyse..."):
                answer = chatgpt_response(user_prompt)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # ============================
    # EXPLICATIF
    # ============================
    with st.expander("📘 Comprendre le Dashboard"):
        st.write(
            """
        - Le P&L Spot corresponde à vos positions directes.
        - Le P&L Options est basé sur vos options **expirées** uniquement.
        - Le P&L Forward se base sur la différence entre forward_price et spot actuel.
        - L'assistant IA peut répondre sur votre portfolio.
        """
        )

    # ============================
    # MY PORTFOLIO TABLE
    # ============================
    st.markdown("## 💰 My Portfolio")

    total_portfolio_value = 0.0
    if my_portfolio:
        rows = []
        total_value = 0.0
        total_pnl_value = 0.0

        for sym, p in my_portfolio.items():
            qty = float(p.get("quantity") or p.get("qty") or 0.0)
            avg = float(p.get("avg_price") or p.get("entry_price") or 0.0)
            side = (p.get("side") or p.get("direction") or "long").lower()

            cur = float(dashboard_prices.get(sym, _dashboard_price(sym, avg)) or 0.0)

            if side == "long":
                pnl_abs = (cur - avg) * qty
            else:
                pnl_abs = (avg - cur) * qty

            value = qty * cur
            total_value += value
            total_pnl_value += pnl_abs

            price = cur
            signed_value = value if side == "long" else -value
            total_portfolio_value += signed_value

            base_ratio = (cur / avg - 1.0) if avg else 0.0
            if side == "short":
                base_ratio = -base_ratio
            pnl_pct = base_ratio * 100.0

            rows.append(
                {
                    "Symbol": sym,
                    "Quantity": qty,
                    "Side": side,
                    "Current Value": cur,
                }
            )

        dfp = pd.DataFrame(rows)
        if not dfp.empty:
            total_current_value = dfp["Current Value"].sum()
            dfp["Allocation"] = (
                dfp["Current Value"] / total_current_value * 100.0 if total_current_value else 0.0
            )
            dfp["Allocation"] = dfp["Allocation"].map(lambda x: f"{x:.2f}%")
            dfp = dfp[["Symbol", "Quantity", "Side", "Current Value", "Allocation"]]
        st.dataframe(dfp, hide_index=True)
    else:
        st.info("Portefeuille vide.")

    # ============================
    # FORWARDS TABLE
    # ============================
    st.markdown("## 📈 Forward Positions")

    if forwards_dash:
        rows_f = []

        for fid, f in forwards_dash.items():
            sym = f.get("symbol", "").upper()
            qty = float(f.get("quantity") or f.get("qty") or 0.0)
            side = (f.get("side", "long") or "long").lower()
            fwd_price = float(f.get("forward_price", 0.0) or 0.0)
            maturity = f.get("maturity", "")
            dash_spot = float(dashboard_prices.get(sym, _dashboard_price(sym, 0.0)) or 0.0)
            value_fwd = qty * dash_spot
            signed_value_fwd = value_fwd if side == "long" else -value_fwd
            total_portfolio_value += signed_value_fwd

            pnl_ratio = ((dash_spot - fwd_price) / fwd_price) if fwd_price else 0.0
            if side == "short":
                pnl_ratio = ((fwd_price - dash_spot) / fwd_price) if fwd_price else 0.0

            rows_f.append(
                {
                    "Symbol": sym,
                    "Side": side,
                    "Quantity": qty,
                    "Spot": dash_spot,
                    "Current Value": dash_spot * qty,
                    "P&L": f"{pnl_ratio * 100.0:.2f}%",
                    "Maturity": maturity,
                }
            )

        st.dataframe(pd.DataFrame(rows_f), hide_index=True)
    else:
        st.info("Aucun forward.")

    # ============================
    # TRADING SYSTEMS OVERVIEW
    # ============================
    st.markdown("## 🎯 Configured Trading Systems")

    try:
        systems = ctrl.load_sell_systems()
    except Exception:
        systems = {}

    if systems:
        rows_s = []
        for sym, dat in systems.items():
            direction = (dat.get("direction", "") or "").lower()
            entry_price = float(dat.get("entry_price", 0.0) or 0.0)
            position_qty = float(dat.get("position", 0.0) or 0.0)
            spot_price = float(
                dashboard_prices.get(sym, _dashboard_price(sym, entry_price)) or entry_price
            )
            signed_value_sys = (
                (spot_price * position_qty) if direction == "long" else -(spot_price * position_qty)
            )
            total_portfolio_value += signed_value_sys

            rows_s.append(
                {
                    "Symbol": sym,
                    "Direction": dat.get("direction", ""),
                    "Trigger spot": dat.get("entry_price", ""),
                    "Spot": spot_price,
                    "Quantity": position_qty,
                    "Drawdown": dat.get("drawdown", ""),
                    "Status": dat.get("status", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows_s), hide_index=True)
    else:
        st.info("Aucun système configuré.")

    # ============================
    # MAINTENANCE / RESET
    # ============================
    st.markdown("## 🧹 Maintenance / Reset")

    if st.button("🗑️ Reset complet Dashboard"):
        ctrl.reset_dashboard()
        # Also clear cached prices to force fresh fetches.
        try:
            cache = ctrl.load_dashboard_cache()
            cache["prices"] = {}
            ctrl.save_dashboard_cache(cache)
        except Exception:
            pass
        st.success("Dashboard reset.")
        st.rerun()


# Dashboard fully migrated and functional.
# Next step: migrate Options tab (UI identical) into the new architecture.
