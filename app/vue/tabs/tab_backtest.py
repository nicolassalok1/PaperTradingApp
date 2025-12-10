import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

from app.controller import backtest_controller as ctrl
from app.vue.components.page_utils import render_closing_history_chart, render_page_header

floor_2 = ctrl.floor_2
floor_4 = ctrl.floor_4


def render():
    alt = None
    plt = None

    def _buy_asset(symbol: str, qty: float, price: float, *, source: str, meta: dict | None = None):
        return ctrl.buy_asset_with_balance(
            symbol,
            qty,
            price,
            source=source,
            meta=meta,
        )

    def _sell_asset(
        symbol: str, qty: float, price: float, *, source: str, meta: dict | None = None
    ):
        return ctrl.sell_asset_with_balance(
            symbol,
            qty,
            price,
            source=source,
            meta=meta,
        )

    render_page_header(
        "Trading Systems / Backtest",
        "Déclencheurs automatiques multi-niveaux et exécution lazy",
        icon="🤖",
        badge="Auto",
    )

    st.markdown(
        """
        <style>
        .ts-highlight {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.65rem;
            margin: 0.2rem 0 0.8rem 0;
        }
        .ts-pill {
            border: 1px solid #234066;
            background: linear-gradient(140deg, rgba(18, 38, 61, 0.8) 0%, rgba(9, 21, 34, 0.9) 100%);
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
        }
        .ts-pill .pill-label { color: #9fb5d7; font-size: 0.85rem; }
        .ts-pill .pill-value { color: #e8f1ff; font-weight: 800; font-size: 1.1rem; line-height: 1.2; }
        .ts-pill .pill-sub { color: #7da6d1; font-size: 0.82rem; }
        .ts-pill.primary { border-color: #2f7de6; background: linear-gradient(135deg, rgba(47, 125, 230, 0.18) 0%, rgba(17, 45, 76, 0.7) 100%); }
        .ts-pill.accent { border-color: #3a9b76; background: linear-gradient(135deg, rgba(58, 155, 118, 0.14) 0%, rgba(11, 46, 39, 0.7) 100%); }

        .ts-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
            gap: 1rem;
            align-items: stretch;
        }
        @media (max-width: 1100px) { .ts-grid { grid-template-columns: 1fr; } }
        .ts-card {
            border: 1px solid #234066;
            background: linear-gradient(150deg, rgba(12, 25, 39, 0.9) 0%, rgba(6, 12, 22, 0.9) 100%);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.4);
        }
        .ts-card h4 { margin: 0 0 0.35rem 0; color: #e8f1ff; }
        .ts-card .ts-sub { color: #9fb5d7; margin: 0 0 0.6rem 0; }
        .ts-input-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.65rem; margin: 0.35rem 0 0.55rem 0; }
        .ts-actions { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.6rem; margin-top: 0.4rem; }
        .ts-card .stButton>button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid #2f5d8c;
            background: linear-gradient(135deg, #1b3a5a 0%, #102742 100%);
            color: #e8f1ff;
            font-weight: 700;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
        }
        .ts-card .stButton>button:hover { border-color: #3a6dad; background: linear-gradient(135deg, #24507c 0%, #173452 100%); }
        .ts-ghost-btn .stButton>button {
            background: linear-gradient(135deg, #102236 0%, #0b1a2b 100%);
            border: 1px solid #2a4b72;
            color: #c9dbff;
            box-shadow: none;
        }
        .ts-ghost-btn .stButton>button:hover { border-color: #3a6dad; color: #ffffff; }
        .ts-chart-card .chart-title { color: #e8f1ff; font-weight: 800; margin-bottom: 0.2rem; }
        .ts-chart-card .chart-caption { color: #9fb5d7; margin-bottom: 0.55rem; }

        [data-testid="stExpander"] {
            border: 1px solid #234066;
            background: rgba(10, 21, 34, 0.7);
            border-radius: 14px;
        }
        [data-testid="stExpander"] summary { font-weight: 700; color: #e8f1ff; }
        .ts-manage-note { color: #9fb5d7; margin: 0.15rem 0 0.6rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Comprendre Trading Systems"):
        st.markdown(
            """
            ### 💡 Ce que vous faites dans Trading Systems

            Ici, vous ne passez pas d'ordres immédiats : vous **concevez des systèmes automatiques** (long ou short)
            qui interviendront pour vous à différents niveaux de prix prédéfinis.

            Le champ *Symbol* sert à choisir l'actif que vous voulez suivre
            de façon structurée (indice, action, ETF, crypto, etc.).

            Le champ *Direction* vous permet de choisir si le système doit exploiter une **hausse** (Long) ou une **baisse** (Short) :
            - **Long** : le robot cherche à accumuler ou renforcer sur l'actif
            - **Short** : le robot cherche à construire ou renforcer une position vendeuse

            *Number of Levels* définit combien de paliers d'intervention vous voulez.
            Chaque niveau correspond à un prix où le système déclenchera automatiquement un ordre (dans le sens choisi).

            *Drawdown %* contrôle l'écart et la direction des niveaux :
            - **Valeur négative** : niveaux en dessous du prix d'entrée (buy the dip / rachat de short)
            - **Valeur positive** : niveaux au-dessus du prix d'entrée (short plus haut / pyramider sur une tendance haussière)

            Plus le pourcentage est faible, plus les niveaux sont serrés; plus il est élevé, plus les niveaux sont espacés.

            Utilisez cet onglet pour **planifier à l'avance** comment vous voulez que vos positions se construisent ou se réduisent,
            sans avoir à rester devant les écrans à chaque mouvement de marché.
            """
        )

    st.markdown("### 🤖 Trading Systems Studio")

    equities = ctrl.load_equities()
    equities, _ = ctrl.auto_execute_trading_levels(
        equities,
        load_ts_exec_log=ctrl.load_ts_exec_log,
        save_ts_exec_log=ctrl.save_ts_exec_log,
        floor_fn=floor_2,
        buy_fn=lambda sym, qty, price: _buy_asset(
            sym, qty, price, source="trading_system", meta={"trigger": "trading_system"}
        ),
        sell_fn=lambda sym, qty, price: _sell_asset(
            sym, qty, price, source="trading_system", meta={"trigger": "trading_system"}
        ),
    )
    current_count = len(equities)
    remaining_slots = max(0, 10 - current_count)

    st.markdown(
        f"""
        <div class="ts-highlight">
            <div class="ts-pill primary">
                <div class="pill-label">Systems live</div>
                <div class="pill-value">{current_count}/10</div>
                <div class="pill-sub">Limite soft à 10 robots</div>
            </div>
            <div class="ts-pill">
                <div class="pill-label">Slots disponibles</div>
                <div class="pill-value">{remaining_slots}</div>
                <div class="pill-sub">1 niveau par système</div>
            </div>
            <div class="ts-pill accent">
                <div class="pill-label">Lazy execution</div>
                <div class="pill-value">Auto cache</div>
                <div class="pill-sub">Données 1 an</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if current_count >= 10:
        st.error(
            "Maximum limit reached! You cannot add more than 10 equities. Please remove one first."
        )
    else:
        levels = 1  # Fixed to a single level
        with st.container():
            st.markdown('<div class="ts-grid">', unsafe_allow_html=True)
            col_form, col_chart = st.columns([1.05, 1])

            with col_form:
                st.markdown('<div class="ts-card">', unsafe_allow_html=True)
                st.markdown("#### Configure un système")
                st.markdown(
                    '<div class="ts-sub">Sélectionne le ticker, la direction et l&#39;espacement. Chaque ajout crée un niveau automatique unique.</div>',
                    unsafe_allow_html=True,
                )

                col_symbol, col_fetch = st.columns([2.2, 1])
                with col_symbol:
                    symbol = st.text_input("Symbol", placeholder="e.g., AAPL").upper()
                with col_fetch:
                    st.markdown('<div class="ts-ghost-btn">', unsafe_allow_html=True)
                    if st.button(
                        "📂 Fetch closing prices",
                        key="btn_fetch_ts_prices",
                        use_container_width=True,
                    ):
                        if symbol:
                            try:
                                df_hist, _, from_cache = ctrl.load_or_fetch_closing_history(
                                    symbol, period="1y", interval="1d"
                                )
                                if df_hist is not None and not df_hist.empty:
                                    st.session_state["ts_history_ready"] = True
                                    st.session_state["ts_history_symbol"] = symbol
                                    st.session_state["ts_history_from_cache"] = from_cache
                                    # Fetch spot in the same flow
                                    price_data = ctrl.fetch_spot_price(symbol) or 0.0
                                    spot_val = float(price_data or 0.0)
                                    if spot_val > 0:
                                        st.session_state["ts_spot_ready"] = True
                                        st.session_state["ts_spot_symbol"] = symbol
                                        st.session_state["ts_spot_value"] = spot_val
                                        st.success(
                                            f"Clôtures 1 an {'cache' if from_cache else 'téléchargées'} + spot {symbol} ~ {spot_val}"
                                        )
                                    else:
                                        st.session_state["ts_spot_ready"] = False
                                        st.error(f"Spot introuvable pour {symbol}.")
                                else:
                                    st.error(f"Aucune clôture disponible pour {symbol}.")
                            except Exception as exc:
                                st.error(
                                    f"Impossible de récupérer les clôtures pour {symbol}: {exc}"
                                )
                        else:
                            st.info("Renseigne un symbole avant de charger les clôtures.")
                    st.markdown("</div>", unsafe_allow_html=True)

                dir_col, qty_col, dd_col = st.columns([1.2, 1, 1])
                with dir_col:
                    direction = st.radio(
                        "Direction",
                        options=["Long", "Short"],
                        index=0,
                        horizontal=True,
                        key="add_equity_direction",
                    )
                with qty_col:
                    qty_per_trade = st.number_input("Qty per trade", min_value=1, value=1, step=1)
                with dd_col:
                    drawdown = st.number_input(
                        "Drawdown %", min_value=-50.0, max_value=50.0, value=5.0, step=0.1
                    )

                st.markdown('<div class="ts-actions">', unsafe_allow_html=True)
                add_clicked = False
                # Defer can_add computation until after the chart section
                st.session_state["_pending_add_click"] = st.button(
                    "Add system",
                    type="primary",
                    key="btn_add_equity",
                    help="Ajouter le système sélectionné",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_chart:
                st.markdown('<div class="ts-card ts-chart-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="chart-title">1y closing prices</div>', unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="chart-caption">Prévisualise le ticker avant de l&#39;activer. Si vide, charge un symbole pour afficher la courbe.</div>',
                    unsafe_allow_html=True,
                )
                ready = st.session_state.get("ts_history_ready", False)
                ready_sym = st.session_state.get("ts_history_symbol") or ""
                if ready and symbol and symbol.upper() == ready_sym.upper():
                    render_closing_history_chart(
                        symbol, "trading_systems_history", location_label="Trading Systems"
                    )
                    st.caption("Les clôtures sont mises en cache pour des chargements rapides.")
                else:
                    st.info("Clique sur Fetch closing prices après avoir saisi un ticker.")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        spot_file = ctrl.CACHE_CSV_DIR / f"spot_{symbol}.csv" if symbol else None
        can_add = (
            st.session_state.get("ts_history_ready")
            and st.session_state.get("ts_spot_ready")
            and symbol
            and spot_file is not None
            and spot_file.exists()
        )

        add_clicked = bool(st.session_state.pop("_pending_add_click", False))
        if add_clicked and can_add:
            success, info, equities = ctrl.add_trading_system(
                equities,
                symbol=symbol,
                direction=direction,
                qty_per_trade=qty_per_trade,
                drawdown_pct=drawdown,
                levels=levels,
                get_data_fn=ctrl.get_data,
                save_equities_fn=ctrl.save_equities,
                clear_cache_fn=ctrl.clear_closing_history_cache,
            )
            if success:
                entry_price = info.get("entry_price", 0.0)
                direction_label = str(info.get("direction", direction)).capitalize()
                sym_label = info.get("symbol", symbol)
                st.success(f"✅ Added {sym_label} ({direction_label}) at ${entry_price:.2f}")
                st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.error(info.get("error", "Unable to add equity"))
        elif add_clicked and not can_add:
            st.error(
                "Clôtures et spot requis avant d'ajouter un système (clic sur Fetch closing prices)."
            )

    st.markdown("---")
    st.markdown("### 🛠️ Manage Your Trading Systems")
    st.markdown(
        """
        <div class="ts-manage-note">
        Pilotez vos systèmes automatiques déjà configurés : c'est votre salle de contrôle pour voir comment vos robots long/short sont positionnés.
        Le toggle <em>Active</em> active/désactive un système. Le tableau <em>Price Levels</em> montre tous les niveaux d'intervention.
        </div>
        """,
        unsafe_allow_html=True,
    )

    equities_list = ctrl.load_equities()

    if equities_list:
        for symbol, data in equities_list.items():
            direction = data.get("direction", "long")
            with st.expander(
                f"{symbol} ({direction.upper()}) - Status: {data['status']}", expanded=False
            ):
                col1, col2, col3, col4, col5, col6 = st.columns(6)

                with col1:
                    st.markdown(f"**Direction:** {direction.capitalize()}")

                with col2:
                    st.markdown(f"**Position:** {data['position']}")

                with col3:
                    st.markdown(f"**Entry Price:** ${data['entry_price']:.2f}")

                with col4:
                    st.markdown(f"**Drawdown:** {data['drawdown']*100:.1f}%")

                with col5:
                    qty_disp = int(data.get("quantity", 1) or 1)
                    st.markdown(f"**Qty per trade:** {qty_disp}")

                with col6:
                    current_status = data["status"]
                    new_status = st.toggle(
                        "Active",
                        value=current_status == "On",
                        key=f"toggle_manage_{symbol}",
                    )

                    if (new_status and current_status == "Off") or (
                        not new_status and current_status == "On"
                    ):
                        equities_list = ctrl.set_trading_system_status(
                            equities_list,
                            symbol,
                            new_status,
                            save_equities_fn=ctrl.save_equities,
                        )
                        st.rerun()

                st.markdown("**Price Levels:**")
                levels_df = pd.DataFrame(
                    [
                        {
                            "Price": f"${v:.2f}",
                            "Qty (per trigger)": int(data.get("quantity", 1) or 1),
                        }
                        for _, v in data["levels"].items()
                    ]
                )
                st.dataframe(levels_df, hide_index=True)

                try:
                    hist = yf.Ticker(symbol).history(period="1mo", interval="1d")
                except Exception:
                    hist = pd.DataFrame()
                if not hist.empty and {"Close"}.issubset(hist.columns):
                    week_hist = hist.tail(7)
                    price_df = week_hist.reset_index()[["Date", "Close"]].rename(
                        columns={"Close": "Price"}
                    )
                    lvl_items = sorted(data["levels"].items(), key=lambda kv: kv[0])
                    level_rows = []
                    palette = [
                        "#e53935",
                        "#8e24aa",
                        "#3949ab",
                        "#1e88e5",
                        "#00897b",
                        "#43a047",
                        "#fdd835",
                        "#fb8c00",
                        "#6d4c41",
                    ]
                    for idx, (lvl_key, lvl_price) in enumerate(lvl_items):
                        level_rows.append(
                            {
                                "Level": f"Level {lvl_key}",
                                "Price": float(lvl_price),
                                "color": palette[idx % len(palette)],
                            }
                        )
                    price_min = price_df["Price"].min() if not price_df.empty else 0.0
                    price_max = price_df["Price"].max() if not price_df.empty else 0.0
                    pad_min = price_min * 0.9
                    pad_max = price_max * 1.1 if price_max > 0 else price_max

                    if alt is not None and level_rows:
                        levels_df_plot = pd.DataFrame(level_rows)
                        chart_price = (
                            alt.Chart(price_df)
                            .mark_line(color="#90caf9")
                            .encode(
                                x=alt.X("Date:T", title="Date"),
                                y=alt.Y(
                                    "Price:Q",
                                    title="Price",
                                    scale=alt.Scale(domain=[pad_min, pad_max]),
                                ),
                                tooltip=["Date:T", "Price:Q"],
                            )
                        )
                        chart_lvls = (
                            alt.Chart(levels_df_plot)
                            .mark_rule(strokeDash=[6, 4], strokeWidth=1.6)
                            .encode(
                                y=alt.Y("Price:Q", scale=alt.Scale(domain=[pad_min, pad_max])),
                                color=alt.Color(
                                    "Level:N",
                                    scale=alt.Scale(
                                        range=(
                                            [r["color"] for r in level_rows] if level_rows else None
                                        )
                                    ),
                                ),
                                tooltip=["Level:N", alt.Tooltip("Price:Q", format=".2f")],
                            )
                        )
                        st.altair_chart(
                            (chart_price + chart_lvls).properties(height=240, width="container")
                        )
                    elif plt is not None:
                        fig_ts, ax_ts = plt.subplots(figsize=(7, 3))
                        ax_ts.plot(
                            price_df["Date"],
                            price_df["Price"],
                            label="Close",
                            color="#90caf9",
                            linewidth=1.8,
                        )
                        for idx, row in enumerate(level_rows):
                            ax_ts.axhline(
                                y=row["Price"],
                                linestyle=(0, (6, 4)),
                                color=row["color"],
                                linewidth=1.2,
                                label=row["Level"] if idx < len(level_rows) else None,
                            )
                        ax_ts.set_ylim(pad_min, pad_max)
                        ax_ts.set_title(f"{symbol} - historique et niveaux auto")
                        ax_ts.set_xlabel("Date")
                        ax_ts.set_ylabel("Price")
                        ax_ts.grid(alpha=0.3, linestyle="--")
                        if level_rows:
                            ax_ts.legend(loc="best", fontsize=8)
                        st.pyplot(fig_ts, clear_figure=True)
                else:
                    st.info("Historique indisponible pour afficher les niveaux automatisés.")

                if st.button(f"🗑️ Remove {symbol}", key=f"remove_manage_{symbol}"):
                    equities_list = ctrl.remove_trading_system(
                        equities_list,
                        symbol,
                        save_equities_fn=ctrl.save_equities,
                    )
                    st.success(f"Removed {symbol}")
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("No trading systems configured yet. Add an equity above to get started.")
