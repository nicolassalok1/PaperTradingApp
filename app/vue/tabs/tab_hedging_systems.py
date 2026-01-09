import pandas as pd
import streamlit as st

from app.controller import hedger_v2_controller as ctrl
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "🛡️ Hedging Systems"


def _render_account() -> None:
    st.markdown("### Aperçu du compte (Alpaca)")
    account = ctrl.get_account_snapshot()
    equity = float(account.get("equity", 0.0) or 0.0)
    cash = float(account.get("cash", 0.0) or 0.0)
    pv = float(account.get("portfolio_value", equity) or equity)
    bp = float(account.get("buying_power", 0.0) or 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"${equity:,.2f}")
    c2.metric("Cash", f"${cash:,.2f}")
    c3.metric("Portfolio Value", f"${pv:,.2f}")
    c4.metric("Buying Power", f"${bp:,.2f}")


def _render_positions() -> list[dict]:
    st.markdown("### Positions (actions & options)")
    equities = ctrl.get_equity_positions()
    options = ctrl.get_option_positions()

    if not equities and not options:
        st.info("No Alpaca positions.")
        return []

    if equities:
        st.caption("Equity positions")
        st.dataframe(pd.DataFrame(equities), hide_index=True, width="stretch")

    if options:
        st.caption("Option positions")
        st.dataframe(pd.DataFrame(options), hide_index=True, width="stretch")

    return options or []


def _render_dqn_panel(option_positions: list[dict]) -> None:
    st.markdown("### Panneau de couverture (DQN)")

    with st.expander("Modèle DQN (statut & entraînement)", expanded=False):
        status = ctrl.get_dqn_model_status()
        weights_ok = bool(status.get("weights_exists"))
        if weights_ok:
            st.success("Modèle DQN trouvé (cache).")
        else:
            st.warning("Aucun modèle DQN en cache. Entraîne-le une première fois.")

        meta = status.get("meta") or {}
        trained_utc = meta.get("trained_utc") or meta.get("loaded_utc") or "n/a"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Version", str(status.get("version", "n/a")))
        c2.metric("Trained/Loaded (UTC)", str(trained_utc))
        c3.metric("Eval |abs(delta)|", f"{(meta.get('eval_avg_abs_total_delta') or 0.0):.3f}")
        c4.metric("Train avg loss", f"{(meta.get('train_avg_loss') or 0.0):.6f}")

        train_steps = st.slider(
            "Train steps (synthetic env)",
            min_value=500,
            max_value=25_000,
            value=int((meta.get("train_steps") or 7_500)),
            step=250,
        )
        seed = st.number_input("Seed", min_value=0, value=int(meta.get("config", {}).get("seed", 42)), step=1)
        force_retrain = st.checkbox("Force retrain (overwrite cache)", value=False)
        if st.button("Train / update DQN", type="primary"):
            with st.spinner("Training DQN (numpy)..."):
                meta_out = ctrl.train_dqn_model(
                    train_steps=int(train_steps),
                    seed=int(seed),
                    force_retrain=bool(force_retrain),
                )
            st.success("DQN entraîné et sauvegardé.")
            with st.expander("Détails (meta)", expanded=False):
                st.json(meta_out)

    if option_positions:
        choices = [
            f"{pos.get('symbol', '')} | {pos.get('side', '')} {pos.get('qty', '')}"
            for pos in option_positions
        ]
        selected_label = st.selectbox("Option à couvrir", choices)
        try:
            idx = choices.index(selected_label)
        except ValueError:
            idx = 0
        option_symbol = str(option_positions[idx].get("symbol", "")).upper()
        mode = "option"
    else:
        st.info("Aucune position option détectée. Entre un underlying pour tester le hedger.")
        option_symbol = ""
        underlying_symbol = st.text_input("Underlying symbol", placeholder="AAPL").upper()
        mode = "underlying"

    b1, b2 = st.columns(2)

    with b1:
        if st.button("Get DQN hedge suggestion"):
            try:
                if mode == "option":
                    suggestion = ctrl.get_dqn_hedge_suggestion_for_option(option_symbol)
                    ref = option_symbol
                else:
                    suggestion = ctrl.get_dqn_hedge_suggestion(underlying_symbol)
                    ref = underlying_symbol

                st.info(
                    f"Suggestion for {ref}: side={suggestion.get('side')} | "
                    f"delta_qty={suggestion.get('delta_qty')} | "
                    f"comment={suggestion.get('comment')}"
                )
                with st.expander("Détails (Q-values / state)", expanded=False):
                    st.write({"q_values": suggestion.get("q_values"), "state": suggestion.get("state")})
            except Exception as exc:
                st.error(f"Failed to get suggestion: {exc}")

    with b2:
        if st.button("Execute DQN hedge on Alpaca"):
            try:
                if mode == "option":
                    result = ctrl.execute_dqn_hedge_for_option(option_symbol)
                    ref = option_symbol
                else:
                    result = ctrl.execute_dqn_hedge(underlying_symbol)
                    ref = underlying_symbol
                st.success(f"Executed hedge for {ref}: {result}")
            except Exception as exc:
                st.error(f"Hedge execution failed: {exc}")

    st.caption("DQN v1 (numpy): replay buffer + target network, entraîné sur un environnement synthétique.")


def render_tab() -> None:
    render_page_header(
        "Hedging Systems (Alpaca)",
        "Couverture options: suggestions DQN (prototype) et exécution via Alpaca.",
        icon="🛡️",
        badge="Hedging",
    )

    # Account snapshot
    _render_account()
    st.divider()

    # Positions shown read-only
    option_positions = _render_positions()

    st.divider()
    _render_dqn_panel(option_positions)


def render() -> None:
    render_tab()
