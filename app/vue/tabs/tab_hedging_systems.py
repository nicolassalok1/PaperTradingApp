import pandas as pd
import streamlit as st

from app.controller import hedger_v2_controller as ctrl
from app.vue.components.page_utils import render_page_header
from app.vue.components.ui_helpers import render_quickstart

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
        return

    if equities:
        st.caption("Equity positions")
        st.dataframe(pd.DataFrame(equities), hide_index=True, use_container_width=True)

    if options:
        st.caption("Option positions")
        st.dataframe(pd.DataFrame(options), hide_index=True, use_container_width=True)

    return options


def _render_dqn_panel(option_positions: list[dict]) -> None:
    st.markdown("### Panneau de couverture (DQN)")

    choices = [
        f"{pos.get('symbol', '')} | {pos.get('side', '')} {pos.get('qty', '')}"
        for pos in option_positions
    ]
    selected_label = st.selectbox("Option to hedge", choices)
    try:
        idx = choices.index(selected_label)
    except ValueError:
        idx = 0
    option_symbol = str(option_positions[idx].get("symbol", "")).upper()

    if st.button("Get DQN hedge suggestion"):
        try:
            suggestion = ctrl.get_dqn_hedge_suggestion_for_option(option_symbol)
            st.info(
                f"Suggestion for {option_symbol}: side={suggestion.get('side')} | "
                f"delta_qty={suggestion.get('delta_qty')} | "
                f"comment={suggestion.get('comment')}"
            )
        except Exception as exc:
            st.error(f"Failed to get suggestion: {exc}")

    if st.button("Execute DQN hedge on Alpaca"):
        try:
            result = ctrl.execute_dqn_hedge_for_option(option_symbol)
            st.success(f"Executed hedge for {option_symbol}: {result}")
        except Exception as exc:
            st.error(f"Hedge execution failed: {exc}")

    st.caption(
        "Note: DQN is a simple placeholder; future versions can improve policy/training."
    )


def render_tab() -> None:
    render_page_header(
        "Hedging Systems (Alpaca)",
        "Couverture options: suggestions DQN (prototype) et exécution via Alpaca.",
        icon="🛡️",
        badge="Hedging",
    )
    render_quickstart(
        "Guide rapide",
        [
            "Ce module est expérimental: considère les suggestions comme une aide, pas une vérité.",
            "Vérifie toujours la position sélectionnée avant d’exécuter une couverture.",
            "Exécuter enverra des ordres sur Alpaca: fais-le seulement en connaissance de cause.",
        ],
        expanded=False,
    )

    # Account snapshot
    _render_account()
    st.divider()

    # Positions shown read-only
    option_positions = _render_positions()

    # DQN hedging panel shown only if there are option positions to hedge
    if option_positions:
        st.divider()
        _render_dqn_panel(option_positions)


def render() -> None:
    render_tab()
