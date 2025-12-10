import pandas as pd
import streamlit as st

from app.controller import hedger_controller as ctrl
from app.vue.components.page_utils import render_page_header


def render():
    ok, err = ctrl.check_heston_support()
    if not ok:
        st.error(
            "Le module Heston (ou PyTorch) est manquant. Installez la dépendance pour utiliser l'onglet Hedger."
            + (f" Détail: {err}" if err else "")
        )
        return

    render_page_header(
        "DQN Hedger",
        "Calibration Heston légère et simulation de hedge par DQN",
        icon="🧠",
        badge="Options",
    )

    option_specs = ctrl.load_option_specs()
    if not option_specs:
        st.info("Aucune option dans options_portfolio.json.")
        return

    opt_labels = [f"{o.id} | {o.symbol} {o.option_type.upper()} K={o.strike}" for o in option_specs]
    choice = st.selectbox("Option (depuis options_portfolio.json)", options=opt_labels, index=0)
    opt = option_specs[opt_labels.index(choice)]
    hp_effective = opt.heston_params or {}
    _local_hp_key = f"hedger_heston_params_{opt.id}"

    st.markdown("### Option sélectionnée")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "ID": opt.id,
                    "Underlying": opt.symbol,
                    "Type": opt.option_type.upper(),
                    "Side": "Long" if opt.side == 1 else "Short",
                    "Strike": opt.strike,
                    "Maturity (y)": opt.maturity_years,
                    "Qty": opt.quantity,
                    "S0": opt.S0,
                }
            ]
        ),
        hide_index=True,
    )

    st.markdown("### Hyperparamètres DQN / Hedge")
    episodes = st.number_input(
        "Episodes d'entraînement", min_value=5, max_value=300, value=50, step=5, key="hedger_ep"
    )
    steps_path = st.number_input(
        "Pas par trajectoire", min_value=20, max_value=300, value=120, step=10, key="hedger_steps"
    )
    hedge_lot = st.number_input(
        "Quantité de hedge par action",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key="hedger_lot",
    )

    # Calibration placeholder (UI can invoke controller.calibrate_heston_params in future).
    st.session_state.setdefault(_local_hp_key, hp_effective)

    if st.button("⚙️ Entraîner un DQN léger", key=f"train_dqn_{opt.id}"):
        progress_dqn = st.progress(0, text="Entrainement DQN en cours...")
        result = ctrl.train_agent(
            opt, steps=int(steps_path), episodes=int(episodes), hedge_lot=float(hedge_lot)
        )
        progress_dqn.progress(1.0, text="Entrainement terminé")
        st.success("Entrainement terminé (léger).")
        st.session_state[f"dqn_agent_state_{opt.id}"] = result.get("agent_state")

    st.markdown("### 🎯 Simulation online (Heston path)")
    run_sim = st.button("Lancer une simulation de hedge", key=f"run_sim_{opt.id}")
    if run_sim:
        if f"dqn_agent_state_{opt.id}" not in st.session_state:
            st.warning("Entraîne d'abord le DQN.")
        else:
            progress_sim = st.progress(0, text="Simulation en cours...")
            logs = ctrl.run_simulation(
                opt,
                steps=int(steps_path),
                hedge_lot=float(hedge_lot),
                agent_state=st.session_state[f"dqn_agent_state_{opt.id}"],
            )
            total_steps = max(1, len(logs))
            for idx, _ in enumerate(logs):
                progress_sim.progress(
                    min(1.0, (idx + 1) / total_steps), text=f"Etape {idx + 1}/{total_steps}"
                )
            st.success("Simulation terminée.")
            if logs:
                df_logs = pd.DataFrame(logs)
                st.dataframe(df_logs, hide_index=True)
