import time

import pandas as pd
import streamlit as st

from app.controller import portfolio_controller as ctrl
from app.controller import dashboard_controller as dash_ctrl
from app.controller import buy_sell_controller as trade_ctrl
from app.vue.components.page_utils import render_page_header


def render():
    load_portfolio = ctrl.load_portfolio
    save_portfolio = ctrl.save_portfolio

    def _buy_asset(symbol: str, qty: float, price: float):
        return trade_ctrl.buy_asset(symbol, qty, price, source="rebalance")

    def _sell_asset(symbol: str, qty: float, price: float):
        return trade_ctrl.sell_asset(symbol, qty, price, source="rebalance")

    render_page_header(
        "Portfolio Allocation",
        "Eigen-portfolio : poids cibles et ordres proposés",
        icon="🧭",
        badge="Risque",
    )

    try:
        st.markdown("### 📐 Allocation & ordres eigen-portfolio")
        portfolio_dict = load_portfolio()
        if not portfolio_dict:
            st.info(
                "Portfolio vide. Ajoute quelques actifs dans l'onglet Buy/Sell avant de calculer une allocation."
            )
            st.caption("ℹ️ Aucun actif spot détecté pour l'instant.")
        else:
            pf_df = ctrl.portfolio_to_df(portfolio_dict)
            tickers = sorted(pf_df["ticker"].unique()) if not pf_df.empty else []

            include_option_underlyings = st.checkbox(
                "Inclure aussi les sous-jacents des options ouvertes (lazy load)",
                value=False,
                key="alloc_include_options_underlyings",
            )
            option_underlyings: list[str] = []
            if include_option_underlyings:
                try:
                    book_opts = ctrl.load_options()
                    option_underlyings = sorted(
                        {
                            (opt.get("underlying") or "").strip().upper()
                            for opt in book_opts.values()
                            if str(opt.get("status", "open")).lower() == "open"
                        }
                    )
                except Exception:
                    option_underlyings = []

            all_tickers = sorted({t for t in tickers + option_underlyings if t})
            if option_underlyings:
                st.caption(
                    f"Tickers détectés : {', '.join(tickers) if tickers else 'aucun'}"
                    f" | Sous-jacents options : {', '.join(option_underlyings)}"
                )
            else:
                st.caption(f"Tickers détectés : {', '.join(tickers) if tickers else 'aucun'}")

            if pf_df.empty or not all_tickers:
                st.info("Aucune position exploitable. Ajoute des positions (quantités non nulles).")
            else:
                try:
                    price_panel, price_sources = ctrl.load_price_panel(
                        all_tickers,
                        loader=dash_ctrl.load_or_fetch_closing_history,
                        period="1y",
                        interval="1d",
                    )
                except Exception as exc:
                    st.error(f"Chargement des prix impossible : {exc}")
                    price_panel, price_sources = None, {}

                st.caption(
                    f"État chargement prix - rows: {0 if price_panel is None else len(price_panel)} | "
                    f"cols: {0 if price_panel is None else price_panel.shape[1] if hasattr(price_panel, 'shape') else 0}"
                )
                if price_panel is None or price_panel.empty:
                    st.info("Chargement des clôtures 1 an en cours/échoué (lazy).")
                else:
                    try:
                        orders_tmp, eigen_w_signed, _ = ctrl.compute_eigen(pf_df, price_panel)
                        latest_prices_perf = price_panel.ffill().iloc[-1]
                        pf_perf = pf_df.copy()
                        pf_perf["spot"] = pf_perf["ticker"].map(latest_prices_perf)
                        pf_perf = pf_perf.dropna(subset=["spot"])
                        pf_perf["side_sign"] = pf_perf["side"].apply(
                            lambda s: 1.0 if str(s).lower() == "long" else -1.0
                        )
                        pf_perf["current_value"] = (
                            pf_perf["quantity"] * pf_perf["spot"] * pf_perf["side_sign"]
                        )
                        gross_perf = pf_perf["current_value"].abs().sum()
                        if gross_perf > 0:
                            w_current = pf_perf.set_index("ticker")["current_value"] / gross_perf
                            rets = price_panel.pct_change().dropna(how="any")
                            w_current = w_current.reindex(rets.columns).fillna(0.0)
                            w_eigen = eigen_w_signed.reindex(rets.columns).fillna(0.0)
                            port_ret_current = (rets * w_current).sum(axis=1)
                            port_ret_eigen = (rets * w_eigen).sum(axis=1)
                            curves = pd.DataFrame(
                                {
                                    "Current": (1 + port_ret_current).cumprod() - 1,
                                    "Eigen": (1 + port_ret_eigen).cumprod() - 1,
                                }
                            )
                            st.markdown("### Courbes de performance (portfolio vs eigen)")
                            st.line_chart(curves)
                        else:
                            st.info("Valeur brute du portfolio nulle : courbes non tracées.")
                    except Exception as exc:
                        st.warning(f"Impossible de calculer les courbes de performance : {exc}")

                    cached_list = [t for t, meta in price_sources.items() if meta.get("from_cache")]
                    dl_list = [t for t, meta in price_sources.items() if not meta.get("from_cache")]
                    if cached_list or dl_list:
                        st.caption(
                            f"Prix chargés via cache: {', '.join(cached_list) or '-'} vs téléchargés: {', '.join(dl_list) or '-'}"
                        )

                    try:
                        orders_df, eigen_weights, latest_prices = ctrl.compute_eigen(
                            pf_df, price_panel
                        )
                    except Exception as exc:
                        st.error(f"Impossible de générer la trading order list : {exc}")
                        st.exception(exc)
                        orders_df = pd.DataFrame()
                        eigen_weights = pd.Series(dtype=float)
                        latest_prices = pd.Series(dtype=float)

                    if not eigen_weights.empty:
                        weights_df = (
                            eigen_weights.rename("weight")
                            .reset_index()
                            .rename(columns={"index": "ticker"})
                        )
                        st.markdown("### 📐 Poids eigen (normalisés)")
                        st.dataframe(weights_df, hide_index=True)

                    if not orders_df.empty:
                        st.markdown("### 📋 Trading order list (eigen-portfolio)")
                        st.dataframe(orders_df, hide_index=True)
                        if st.button(
                            "↔ Appliquer au portfolio & recalculer P&L",
                            type="primary",
                            key="btn_apply_eigen_orders",
                        ):
                            executed, skipped = ctrl.apply_orders(orders_df, latest_prices)
                            if executed > 0:
                                st.success(
                                    f"{executed} ordres appliqués via buy/sell. Rafraîchis le Dashboard pour voir le P&L."
                                )
                            else:
                                st.info(
                                    "Aucun ordre appliqué (quantités nulles ou prix indisponibles)."
                                )
                            if skipped:
                                st.caption(f"Sauts: {', '.join(skipped)}")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.info(
                            "Aucun ordre à appliquer (poids eigen neutres ou données insuffisantes)."
                        )
    except Exception as exc:
        st.error(f"Onglet Portfolio Allocation non rendu : {exc}")
        st.exception(exc)
