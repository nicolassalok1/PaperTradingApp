import json
import pandas as pd
import streamlit as st

from app.controller.calibration_controller import CalibrationController

TAB_LABEL = "🧮 Calibration"


def render_tab():
    ctrl = CalibrationController()
    st.title("Calibration (scaffold)")
    st.caption("Architecture prête pour les futurs modèles de calibration.")

    st.subheader("Source de données de marché")
    source_options = ["file_upload", "api_placeholder"]
    source_choice = st.selectbox(
        "Source de surface",
        options=source_options,
        format_func=lambda v: "Fichier CSV" if v == "file_upload" else "API placeholder",
    )

    uploaded_df = None
    uploaded_file = st.file_uploader("Charger une surface IV (CSV optionnel)", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.dataframe(uploaded_df.head(), hide_index=True, use_container_width=True)
        except Exception as exc:
            st.warning(f"Impossible de lire le CSV: {exc}")

    st.subheader("Choix du modèle")
    models = ctrl.get_models()
    model_choice = st.selectbox(
        "Modèle",
        options=models,
        index=0 if models else 0,
        format_func=lambda v: str(v).upper(),
    )

    st.subheader("Contraintes (JSON)")
    constraints_raw = st.text_area("Contraintes", value="{}", height=120)
    constraints = None
    if constraints_raw:
        try:
            constraints = json.loads(constraints_raw)
        except Exception as exc:
            st.warning(f"JSON invalide: {exc}")

    ticker = st.text_input("Ticker (optionnel)", value="")

    if st.button("Lancer la calibration (placeholder)", type="primary"):
        payload = {
            "model": model_choice,
            "source": source_choice,
            "ticker": ticker.strip() or None,
            "constraints": constraints if isinstance(constraints, dict) else None,
            "surface_path": None,
        }
        result = ctrl.submit(payload)
        if result.get("success"):
            st.success(result.get("message", "OK"))
        else:
            st.info(result.get("message", "Calibration non implémentée."))
        details = result.get("details") or {}
        if uploaded_df is not None:
            details = {**details, "uploaded_rows": len(uploaded_df)}
        if details:
            st.json(details)


# Backward-compatible alias
render = render_tab
