import pandas as pd
import streamlit as st

from app.controller import hedging_controller as ctrl
from app.vue.components.page_utils import render_page_header


TAB_LABEL = "?? Options Hedging"


def _to_orders_df(orders) -> pd.DataFrame:
    rows = []
    for o in orders or []:
        if hasattr(o, "to_dict"):
            data = o.to_dict()
        elif isinstance(o, dict):
            data = o
        else:
            continue
        rows.append(
            {
                "Symbol": data.get("symbol"),
                "Asset Type": data.get("asset_type"),
                "Side": data.get("side"),
                "Quantity": data.get("quantity"),
                "Order Type": data.get("order_type"),
                "Estimated Price": data.get("estimated_price"),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Symbol", "Asset Type", "Side", "Quantity", "Order Type", "Estimated Price"]
    )


def render_tab() -> None:
    render_page_header(
        "Options Hedging (DQN)",
        "Compute DQN-based hedging orders and optionally execute them via Alpaca.",
        icon="??",
        badge="Options",
    )

    option_specs = ctrl.load_option_specs()
    if not option_specs:
        st.info("No options found in options_portfolio.json.")
        return

    st.markdown("### Hedging context")
    labels = [
        f"{o.id} | {o.symbol} {o.option_type.upper()} K={o.strike}"
        for o in option_specs
    ]
    choice = st.selectbox("Option to hedge", options=labels, index=0)
    option = option_specs[labels.index(choice)]

    hedge_lot = st.number_input(
        "Hedge lot (underlying quantity per action)",
        min_value=0.1,
        max_value=10_000.0,
        value=1.0,
        step=0.1,
        key="hedging_hedge_lot",
    )

    st.markdown("### Trading mode")
    mode_label = st.radio(
        "Execution mode",
        options=["Paper (default, safe)", "Live (explicit opt-in)"],
        index=0,
        horizontal=True,
        key="hedging_mode",
    )
    live_mode = mode_label.startswith("Live")
    if live_mode:
        live_confirm = st.checkbox(
            "I understand this will send live orders to Alpaca.",
            key="hedging_live_confirm",
        )
    else:
        live_confirm = False

    st.markdown("### Step 1 – Run Hedging")
    if st.button("Run Hedging", type="primary", key="btn_run_hedging"):
        try:
            # Agent state is optional; if a trained agent has been stored in
            # session_state by another workflow, it can be passed through here.
            agent_key = f"dqn_agent_state_{option.id}"
            agent_state = st.session_state.get(agent_key)
            orders = ctrl.compute_hedging_orders(
                option,
                hedge_lot=float(hedge_lot),
                agent_state=agent_state,
            )
            st.session_state["hedging_orders"] = [
                o.to_dict() if hasattr(o, "to_dict") else dict(o) for o in orders
            ]
            st.session_state["hedging_executed"] = False
            if not orders:
                st.info("DQN recommends to hold (no hedge order).")
        except Exception as exc:
            st.error(f"Failed to compute hedging orders: {exc}")

    orders_data = st.session_state.get("hedging_orders", [])
    if orders_data:
        st.markdown("### Proposed hedging orders (not executed)")
        df_orders = _to_orders_df(orders_data)
        st.dataframe(df_orders, hide_index=True, use_container_width=True)
        mode = "live" if live_mode else "paper"
        st.caption(f"Execution mode selected: **{mode.upper()}**. Orders are not sent until confirmation.")
    else:
        st.info("Run hedging to generate proposed orders.")
        return

    st.markdown("### Step 2 – Confirm Execution")
    already_executed = st.session_state.get("hedging_executed", False)
    if already_executed:
        st.success("Hedging orders have been executed in this session.")
        return

    if live_mode and not live_confirm:
        st.warning("Enable the confirmation checkbox above to allow live execution.")
        disabled = True
    else:
        disabled = False

    if st.button("Confirm Execution", type="primary", disabled=disabled, key="btn_confirm_hedging"):
        try:
            mode = "live" if live_mode else "paper"
            results = ctrl.execute_orders(orders_data, mode=mode)
            st.session_state["hedging_executed"] = True
            st.session_state["hedging_execution_result"] = results
            st.success(f"Executed {len(results)} order(s) in {mode.upper()} mode.")
        except Exception as exc:
            st.error(f"Failed to execute hedging orders: {exc}")


def render() -> None:
    """Compatibility alias for generic routers."""
    render_tab()

