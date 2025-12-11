import numpy as np
import streamlit as st
from app.controller import options_controller as oc
from app.vue.components.options.controller_bridge import *

# Bridge controller functions used locally
fetch_iv_surface = oc.fetch_iv_surface
interpolate_surface = oc.interpolate_surface
load_iv_from_csv = oc.load_iv_from_csv
calibrate_heston_stub = oc.calibrate_heston_stub
price_heston_fft = oc.price_heston_fft
price_heston_call = oc.price_heston_call
heston_delta = oc.heston_delta
heston_vega = oc.heston_vega
plot_surface_3d = oc.plot_surface_3d
plot_heatmap = oc.heston_plot_heatmap

def _safe_calibrate_heston(strikes, maturities, iv_matrix):
    """
    TensorFlow-free fallback calibration using simple IV statistics.
    """
    try:
        iv_flat = iv_matrix[np.isfinite(iv_matrix)]
        iv_med = float(np.nanmedian(iv_flat)) if iv_flat.size else 0.5
        if not np.isfinite(iv_med) or iv_med <= 0:
            iv_med = 0.5
    except Exception:
        iv_med = 0.5

    sigma = max(0.01, iv_med)
    v0 = sigma**2
    return {
        "kappa": 1.5,
        "theta": v0,
        "sigma": sigma,
        "rho": -0.4,
        "v0": v0,
    }


def render_tab_heston():
    # --- Contexte global Options ---
    ctx = get_option_context()
    S0 = ctx["S0"]
    ticker = ctx["ticker"]
    close_series = ctx["close_series"]
    _k = ctx["_k"]
    # --------------------------------
    hist_tkr = ticker

    st.markdown("## 🧮 Modèle de Heston — Interface Professionnelle")
    st.caption("Calibration NN | Pricing CF | FFT (stub) | IV Surface 3D")
    st.markdown("---")

    common_spot_value = float(st.session_state.get("common_spot_value", 100.0))
    mc_label_to_model = {
        "Black–Scholes (MC)": "bs",
        "rHeston (MC)": "rheston",
        "rBergomi (MC)": "rbergomi",
        "SABR (MC)": "sabr",
        "Volterra (MC)": "volterra",
    }
    mc_choice = st.selectbox(
        "Modèle de pricing Monte Carlo",
        options=list(mc_label_to_model.keys()),
        index=0,
        key=f"mc_model_{_k('heston')}",
        help="Black–Scholes (MC) implémenté, les autres seront ajoutés progressivement.",
    )
    mc_model = mc_label_to_model.get(mc_choice, list(mc_label_to_model.values())[0])

    tab_calib, tab_pricing, tab_surface = st.tabs(["Calibration NN", "Pricing", "IV Surface"])

    with tab_calib:
        st.subheader("Calibration NN du modele de Heston")

        ticker = st.text_input(
            "Ticker (IV via yfinance)",
            st.session_state.get("heston_cboe_ticker", ""),
            placeholder="ex: SPY",
        )
        use_csv = st.checkbox("Charger depuis un CSV ?", value=False)
        fetch_iv = st.button("Charger la surface IV", key="btn_fetch_iv_surface")

        df_iv = None
        ticker_norm = (ticker or "").strip().upper()
        if use_csv:
            iv_file = st.file_uploader("Televerser un CSV d'IV", type="csv")
            if iv_file:
                df_iv = load_iv_from_csv(iv_file)
        else:
            cached_tkr = st.session_state.get("_iv_surface_ticker")
            if fetch_iv and ticker_norm:
                df_iv = fetch_iv_surface(ticker_norm)
                st.session_state["heston_cboe_ticker"] = ticker_norm
                if df_iv is not None:
                    st.session_state["_iv_surface_cache"] = df_iv
                    st.session_state["_iv_surface_ticker"] = ticker_norm
            elif cached_tkr and cached_tkr == ticker_norm:
                df_iv = st.session_state.get("_iv_surface_cache")

        has_iv_data = df_iv is not None and len(df_iv) > 0
        if has_iv_data:
            maturities, strikes, iv_matrix = interpolate_surface(df_iv)
            if iv_matrix is None:
                st.warning("Interpolation echouee.")
            else:
                st.info("Surface IV chargee et interpolee (maturites x strikes).")
                if st.button("Calibrer via NN"):
                    try:
                        # TensorFlow-free calibration stub.
                        params = calibrate_heston_stub(strikes, maturities, iv_matrix)
                        st.success("Calibration terminée (approx heuristique, sans TF).")
                        st.json(params)
                        st.session_state["heston_params"] = params
                    except Exception as exc:
                        st.error(f"Calibration NN indisponible: {exc}")
        elif use_csv or fetch_iv:
            st.warning("Aucune surface IV recuperee.")
        elif ticker_norm:
            st.info('Clique sur "Charger la surface IV" pour récupérer la surface.')
        else:
            st.info("Renseigne un ticker puis clique sur le bouton pour charger la surface IV.")

    with tab_pricing:
        st.subheader("Pricing europeen Heston")

        K = st.number_input("Strike", min_value=1.0, value=100.0)
        T = st.number_input("Maturite (annees)", min_value=0.01, value=0.25)
        r = st.number_input("Taux sans risque r", value=0.01)
        q = st.number_input("Dividende q", value=0.00)
        sigma_mc = float(st.session_state.get("common_sigma_value", 0.2))

        params = st.session_state.get(
            "heston_params",
            {
                "kappa": 1.5,
                "theta": 0.04,
                "sigma": 0.6,
                "rho": -0.5,
                "v0": 0.04,
            },
        )

        if st.button("Pricer Heston (CF)"):
            price = price_heston_call(common_spot_value, K, T, r, q, **params)
            delta = heston_delta(common_spot_value, K, T, r, q, params)
            vega = heston_vega(common_spot_value, K, T, r, q, params)

            st.success(f"Prix Heston = {price:.4f}")
            st.metric("Delta", f"{delta:.4f}")
            st.metric("Vega", f"{vega:.4f}")
            ticker_mc = (
                (ticker or st.session_state.get("heston_cboe_ticker") or st.session_state.get("tkr_common") or "")
                .strip()
                .upper()
            )
            if ticker_mc:
                try:
                    price_mc_val = compute_price_mc(
                        {"ticker": ticker_mc, "K": K, "T": T, "sigma": sigma_mc},
                        mc_model=mc_model,
                    )
                    st.metric("Prix (Monte Carlo)", f"{price_mc_val:.4f}")
                except Exception as exc:
                    st.warning(f"Pricing Monte Carlo indisponible: {exc}")
            else:
                st.info("Renseigne un ticker pour activer le pricing Monte Carlo.")

        if st.button("Pricer via FFT (approx)"):
            st.info("Version FFT skeleton - retourne un stub.")
            fft_price = price_heston_fft(common_spot_value, K, T, r, q, params)
            st.write(f"FFT approx: {fft_price}")

    with tab_surface:
        st.subheader("Surface IV (3D / Heatmap)")

        ticker_surf = st.text_input(
            "Ticker surface (yfinance)",
            st.session_state.get("heston_cboe_ticker", ""),
            placeholder="ex: SPY",
            key="ticker_surf_input",
        )
        fetch_surf = st.button("Charger la surface IV pour affichage", key="btn_fetch_iv_surface_plot")
        df_surf = None
        ticker_surf_norm = (ticker_surf or "").strip().upper()
        cached_tkr = st.session_state.get("_iv_surface_ticker")
        if fetch_surf and ticker_surf_norm:
            df_surf = fetch_iv_surface(ticker_surf_norm)
            st.session_state["heston_cboe_ticker"] = ticker_surf_norm
            if df_surf is not None:
                st.session_state["_iv_surface_cache"] = df_surf
                st.session_state["_iv_surface_ticker"] = ticker_surf_norm
        elif cached_tkr and cached_tkr == ticker_surf_norm:
            df_surf = st.session_state.get("_iv_surface_cache")

        has_surface = df_surf is not None and len(df_surf) > 0
        if has_surface:
            maturities, strikes, iv_matrix = interpolate_surface(df_surf)

            if iv_matrix is None:
                st.warning("Interpolation impossible.")
            else:
                st.plotly_chart(
                    plot_surface_3d(maturities, strikes, iv_matrix),
                    config={"staticPlot": True, "scrollZoom": False},
                )
                st.plotly_chart(
                    plot_heatmap(maturities, strikes, iv_matrix),
                    config={"staticPlot": True, "scrollZoom": False},
                )
        elif fetch_surf:
            st.warning("Aucune IV recuperee pour ce ticker.")
        elif ticker_surf_norm:
            st.info('Clique sur "Charger la surface IV pour affichage" pour lancer le fetch.')
        else:
            st.info("Renseigne un ticker puis déclenche le chargement de la surface IV.")

    # --- Bouton Add-to-Dashboard Clean ---
    if "price" in locals() and st.button("Ajouter au dashboard", key="heston_add_dashboard_btn"):
        payload = {
            "underlying": ticker,
            "S0": S0,
            "price": float(price),
        }
        oid = add_option_to_dashboard_clean(payload)
        log_action("add_option", {"id": oid, "payload": payload})
        st.success(f"Option ajoutée au dashboard (id={oid})")
    # -------------------------------------
