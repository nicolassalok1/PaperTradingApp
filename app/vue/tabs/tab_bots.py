from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from app.controller import bots_controller
from app.vue.components.page_utils import render_page_header


TAB_LABEL = "🤖 Bots"
IN_PROGRESS_NOTE = (
    "Cette section est en cours d'implementation. Le contenu est masque le temps de finaliser les features."
)

def _as_df(rows) -> pd.DataFrame:
    try:
        df = pd.DataFrame(rows or [])
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _render_assistant() -> None:
    st.subheader("Assistant")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        refresh = st.button("Refresh snapshot", width="stretch")
    with col_b:
        st.caption("OpenAI est optionnel; l’onglet fonctionne sans clé.")

    if refresh or "bots_snapshot" not in st.session_state:
        st.session_state["bots_snapshot"] = bots_controller.get_snapshot()

    snapshot = st.session_state.get("bots_snapshot") or {}
    warnings = snapshot.get("warnings") or []
    if warnings:
        st.warning("\n".join(str(w) for w in warnings))

    account = snapshot.get("account") or {}
    st.markdown("**Account**")
    st.json(account)

    st.markdown("**Positions (spot)**")
    st.dataframe(_as_df(snapshot.get("spot_positions")), width="stretch")

    st.markdown("**Positions (options)**")
    st.dataframe(_as_df(snapshot.get("option_positions")), width="stretch")

    st.markdown("**Open orders**")
    st.dataframe(_as_df(snapshot.get("open_orders")), width="stretch")

    st.divider()
    st.markdown("**Copilot**")
    q = st.text_area("Question", key="bots_assistant_question", height=80)
    if st.button("Ask", type="primary"):
        answer = bots_controller.ask(q, snapshot=snapshot)
        st.session_state["bots_assistant_answer"] = answer
    if st.session_state.get("bots_assistant_answer"):
        st.markdown(st.session_state["bots_assistant_answer"])


def _render_execution() -> None:
    st.subheader("Exécution (Grid/DCA)")

    configs = bots_controller.list_grid_configs()
    symbols = sorted(configs.keys())

    left, right = st.columns([1, 1])
    with left:
        selected = st.selectbox("Bot", options=["<new>"] + symbols, index=0)
    with right:
        st.caption("Par défaut: `dry-run` (aucun ordre soumis).")

    selected_cfg = configs.get(selected) if selected and selected != "<new>" else None
    defaults = asdict(selected_cfg) if selected_cfg is not None else {}

    with st.form("grid_bot_config_form", clear_on_submit=False):
        symbol = st.text_input("Symbol", value=str(defaults.get("symbol") or ""))
        enabled = st.checkbox("Enabled", value=bool(defaults.get("enabled", False)))
        dry_run = st.checkbox("Dry run (no submit)", value=bool(defaults.get("dry_run", True)))

        c1, c2, c3 = st.columns(3)
        with c1:
            qty = st.number_input(
                "Qty",
                min_value=0.0,
                value=float(defaults.get("qty", 1.0) or 1.0),
                step=1.0,
            )
        with c2:
            n_levels = st.number_input(
                "Levels",
                min_value=1,
                max_value=50,
                value=int(defaults.get("n_levels", 5) or 5),
                step=1,
            )
        with c3:
            step_pct = st.number_input(
                "Step (%)",
                min_value=0.01,
                max_value=50.0,
                value=float(defaults.get("step_pct", 0.05) or 0.05) * 100.0,
                step=0.25,
            )

        save = st.form_submit_button("Save config", type="primary")
        if save:
            cfg = bots_controller.save_grid_config(
                symbol=symbol,
                enabled=enabled,
                qty=float(qty),
                n_levels=int(n_levels),
                step_pct=float(step_pct) / 100.0,
                dry_run=dry_run,
            )
            st.success(f"Saved: {cfg.symbol}")

    st.markdown("**Saved configs**")
    st.dataframe(_as_df([asdict(c) for c in configs.values()]), width="stretch")

    if selected_cfg is None:
        st.info("Sélectionne un bot existant pour le lancer.")
        return

    st.divider()
    st.markdown("**Run once**")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        allow_submit = st.checkbox("Allow submit (unsafe)", value=False)
    with col2:
        allow_live = st.checkbox("Allow live trading", value=False)
    with col3:
        if st.button("Run", width="stretch"):
            st.session_state["bots_grid_last_report"] = bots_controller.run_grid_once(
                selected_cfg,
                allow_submit=allow_submit,
                allow_live=allow_live,
            )

    report = st.session_state.get("bots_grid_last_report")
    if not report:
        return

    for msg in report.get("warnings") or []:
        st.warning(msg)
    for msg in report.get("errors") or []:
        st.error(msg)

    st.markdown("**Plan**")
    plan = {
        "symbol": report.get("symbol"),
        "reference_price": report.get("reference_price"),
        "desired_prices": report.get("desired_prices"),
        "existing_open_limit_buy_prices": report.get("existing_open_limit_buy_prices"),
        "to_submit_prices": report.get("to_submit_prices"),
    }
    st.json(plan)

    if report.get("submitted_orders"):
        st.markdown("**Submitted orders**")
        st.dataframe(_as_df(report.get("submitted_orders")), width="stretch")
    if report.get("simulated_orders"):
        st.markdown("**Simulated orders**")
        st.dataframe(_as_df(report.get("simulated_orders")), width="stretch")


def _render_volatility() -> None:
    st.subheader("Volatilité")

    straddle_tab, crush_tab, regime_tab, meanrev_tab, markov_tab = st.tabs(
        ["Straddle", "IV Crush", "Realized Vol Regime", "Mean Reversion", "Markov"]
    )

    with straddle_tab:
        st.markdown("Prix + greeks Black-Scholes (proxy) via les moteurs déjà présents.")
        sym = st.text_input("Ticker (optional)", value="SPY", key="bots_straddle_ticker")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Load spot", width="stretch"):
                spot = bots_controller.get_spot_price(sym)
                if spot:
                    st.session_state["bots_straddle_spot"] = float(spot)
        with c2:
            st.caption("Source spot: Stooq (fallback Yahoo) via `fetch_spot_price`.")

        spot_val = float(st.session_state.get("bots_straddle_spot") or 0.0) or 0.0
        S0 = st.number_input("Spot", min_value=0.0, value=float(spot_val), step=0.1)
        K = st.number_input("Strike", min_value=0.0, value=float(S0), step=0.1)
        days = st.number_input("Days to expiry", min_value=0, value=30, step=1)
        iv_pct = st.number_input("IV (%)", min_value=0.01, value=25.0, step=0.5)
        r_pct = st.number_input("r (%)", value=0.0, step=0.25)
        q_pct = st.number_input("q (%)", value=0.0, step=0.25)

        if st.button("Compute straddle", type="primary"):
            snap = bots_controller.straddle_snapshot(
                spot=float(S0),
                strike=float(K),
                days_to_expiry=int(days),
                iv=float(iv_pct) / 100.0,
                r=float(r_pct) / 100.0,
                q=float(q_pct) / 100.0,
            )
            st.session_state["bots_straddle_snapshot"] = snap

        snap = st.session_state.get("bots_straddle_snapshot")
        if snap:
            st.json(snap)

    with crush_tab:
        st.markdown("Comparaison pré/post (spot + IV) sur un straddle au même strike.")
        pre_spot = st.number_input("Pre spot", min_value=0.0, value=100.0, step=0.5)
        post_spot = st.number_input("Post spot", min_value=0.0, value=102.0, step=0.5)
        strike = st.number_input("Strike", min_value=0.0, value=float(pre_spot), step=0.5)
        days = st.number_input("Days to expiry", min_value=0, value=30, step=1, key="bots_crush_days")
        pre_iv = st.number_input("Pre IV (%)", min_value=0.01, value=40.0, step=0.5)
        post_iv = st.number_input("Post IV (%)", min_value=0.01, value=25.0, step=0.5)
        qty = st.number_input("Qty", min_value=0.0, value=1.0, step=1.0)

        if st.button("Compute IV crush", type="primary"):
            out = bots_controller.straddle_iv_crush(
                pre_spot=float(pre_spot),
                post_spot=float(post_spot),
                strike=float(strike),
                days_to_expiry=int(days),
                pre_iv=float(pre_iv) / 100.0,
                post_iv=float(post_iv) / 100.0,
                qty=float(qty),
            )
            st.session_state["bots_iv_crush"] = out

        out = st.session_state.get("bots_iv_crush")
        if out:
            st.json(out)

    with regime_tab:
        sym = st.text_input("Symbol", value="SPY", key="bots_regime_symbol")
        period = st.selectbox("History", options=["6mo", "1y", "2y", "5y"], index=2)
        window = st.number_input("Window (days)", min_value=5, max_value=252, value=20, step=1)

        if st.button("Compute regime", type="primary"):
            res = bots_controller.realized_vol_regime(
                sym,
                period=str(period),
                window=int(window),
                annualization=252,
            )
            st.session_state["bots_vol_regime"] = res

        res = st.session_state.get("bots_vol_regime")
        if not res:
            st.info("Clique sur **Compute regime** pour générer le résultat.")
        elif res.get("error"):
            st.error(res["error"])
        else:
            st.metric("Current vol", value=f"{res.get('current_vol', 0.0):.2%}")
            st.metric("Percentile", value=f"{res.get('percentile', 0.0):.1%}")
            st.metric("Regime", value=str(res.get("regime")))

            df = _as_df(res.get("series"))
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date")
            if not df.empty and "vol" in df.columns:
                st.line_chart(df["vol"], height=220)
            st.dataframe(df.tail(30), width="stretch")

    with meanrev_tab:
        st.markdown("Analyse de mean-reversion sur la volatilité réalisée (OHLC).")
        sym = st.text_input("Symbol", value="SPY", key="bots_meanrev_symbol")
        period = st.selectbox(
            "History",
            options=["6mo", "1y", "2y", "5y"],
            index=2,
            key="bots_meanrev_period",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            vol_window = st.number_input(
                "Vol window (days)",
                min_value=5,
                max_value=252,
                value=20,
                step=1,
                key="bots_meanrev_vol_window",
            )
        with c2:
            forward_window = st.number_input(
                "Forward avg (days)",
                min_value=5,
                max_value=90,
                value=30,
                step=1,
                key="bots_meanrev_forward_window",
            )

        if st.button("Compute mean reversion", type="primary"):
            res = bots_controller.realized_vol_mean_reversion(
                sym,
                period=str(period),
                vol_window=int(vol_window),
                forward_window=int(forward_window),
                annualization=252,
            )
            st.session_state["bots_meanrev"] = res

        res = st.session_state.get("bots_meanrev")
        if not res:
            st.info("Clique sur **Compute mean reversion** pour générer le résultat.")
        elif res.get("error"):
            st.error(res["error"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Current vol", value=f"{res.get('current_vol', 0.0):.2%}")
            c2.metric("Percentile", value=f"{res.get('percentile', 0.0):.1%}")
            c3.metric("Hint", value=str(res.get("mean_reversion_hint") or "N/A"))

            st.caption(f"Split vol (intersection y=x): {float(res.get('split_x') or 0.0):.2%}")
            st.json(
                {
                    "reg_forward": res.get("reg_forward"),
                    "reg_vol_diff_all": res.get("reg_vol_diff_all"),
                    "reg_vol_diff_high": res.get("reg_vol_diff_high"),
                    "reg_vol_diff_low": res.get("reg_vol_diff_low"),
                }
            )

            df = _as_df(res.get("series"))
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])

            if df.empty:
                st.info("Pas assez de données pour afficher les graphiques.")
            else:
                st.markdown("**Scatter: current vs forward**")
                try:
                    st.scatter_chart(df, x="current_vol", y="forward_vol", height=240)
                except Exception:
                    st.dataframe(
                        df[["current_vol", "forward_vol"]].tail(200),
                        width="stretch",
                    )

                st.markdown("**Scatter: current vs (forward-current)**")
                try:
                    st.scatter_chart(df, x="current_vol", y="vol_diff", height=240)
                except Exception:
                    st.dataframe(df[["current_vol", "vol_diff"]].tail(200), width="stretch")

                if "date" in df.columns:
                    df_ts = df.set_index("date")
                    cols = [c for c in ["current_vol", "forward_vol"] if c in df_ts.columns]
                    if cols:
                        st.markdown("**Time series**")
                        st.line_chart(df_ts[cols], height=220)

                st.dataframe(df.tail(30), width="stretch")

    with markov_tab:
        st.markdown("Matrice de transition Markov sur régimes de volatilité réalisée (quantiles).")
        sym = st.text_input("Symbol", value="SPY", key="bots_markov_symbol")
        period = st.selectbox(
            "History",
            options=["6mo", "1y", "2y", "5y"],
            index=2,
            key="bots_markov_period",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            window = st.number_input(
                "Vol window (days)",
                min_value=5,
                max_value=252,
                value=20,
                step=1,
                key="bots_markov_window",
            )
        with c2:
            n_states = st.number_input(
                "States",
                min_value=2,
                max_value=6,
                value=3,
                step=1,
                key="bots_markov_states",
            )

        if st.button("Compute Markov matrix", type="primary"):
            res = bots_controller.markov_vol_transition(
                sym,
                period=str(period),
                window=int(window),
                annualization=252,
                n_states=int(n_states),
            )
            st.session_state["bots_markov"] = res

        res = st.session_state.get("bots_markov")
        if not res:
            st.info("Clique sur **Compute Markov matrix** pour générer le résultat.")
        elif res.get("error"):
            st.error(res["error"])
        else:
            st.caption(
                f"Current state: {res.get('current_state')} | Next probs: {res.get('next_state_probs')}"
            )
            st.json({"cuts": res.get("cuts"), "labels": res.get("labels")})

            labels = res.get("labels") or []
            mat = res.get("transition_matrix") or []
            if labels and mat:
                df_m = pd.DataFrame(mat, index=labels, columns=labels)
                st.markdown("**Transition matrix (rows sum to 1)**")
                st.dataframe(df_m, width="stretch")

            df_s = _as_df(res.get("series"))
            if not df_s.empty and "date" in df_s.columns:
                df_s["date"] = pd.to_datetime(df_s["date"], errors="coerce")
            if not df_s.empty:
                st.dataframe(df_s.tail(40), width="stretch")


def _render_placeholder(section_name: str) -> None:
    st.info(f"{section_name} est en cours d'implementation. {IN_PROGRESS_NOTE}")


def render_tab() -> None:
    render_page_header(
        "Bots & Assistant",
        IN_PROGRESS_NOTE,
        icon="🤖",
        badge="Tools",
    )

    assistant_tab, execution_tab, vol_tab = st.tabs(["Assistant", "Exécution", "Volatilité"])

    with assistant_tab:
        _render_placeholder("Assistant")
        # _render_assistant()  # TODO: re-activer quand la section sera prete

    with execution_tab:
        _render_placeholder("Exécution")
        # _render_execution()  # TODO: re-activer quand la section sera prete

    with vol_tab:
        _render_placeholder("Volatilité")
        # _render_volatility()  # TODO: re-activer quand la section sera prete


def render() -> None:
    render_tab()
