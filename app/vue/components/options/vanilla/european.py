import streamlit as st

from app.vue.components.options.layout import (
    compute_button,
    option_panel,
    params_expander,
    render_crr_payoff_surface,
    render_crr_tree,
    render_heatmap_diagnostics,
    render_payoff_text,
)
from app.vue.components.options.pricing import price_heston_european_call


class _EuropeanOption:
    """Lightweight option object for CRR tree display."""

    def __init__(self, s0: float, K: float, T: float, call: bool = True):
        self.s0 = s0
        self.K = K
        self.T = T
        self.call = call

    def payoff(self, s):
        return (s - self.K) if self.call else (self.K - s)


def render():
    option_panel("Option europeenne (Heston)")
    base = "eu_heston_"
    with params_expander():
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.number_input(
                "Sous-jacent (S0)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}s0",
            )
            K = st.number_input(
                "Strike (K)",
                value=100.0,
                min_value=0.01,
                step=1.0,
                key=f"{base}k",
            )
            T = st.number_input(
                "Maturite T (annees)",
                value=1.0,
                min_value=0.01,
                step=0.05,
                format="%.2f",
                key=f"{base}t",
            )
        with col2:
            r = st.number_input(
                "Taux sans risque (r)",
                value=0.02,
                step=0.005,
                format="%.4f",
                key=f"{base}r",
            )
            q = st.number_input(
                "Dividend yield (q)",
                value=0.00,
                step=0.005,
                format="%.4f",
                key=f"{base}q",
            )
            option_type = st.selectbox("Type d'option", ["Call", "Put"], index=0, key=f"{base}type")
            sigma_proxy = st.number_input(
                "Volatilite (sigma)",
                value=0.2,
                min_value=0.0001,
                step=0.01,
                format="%.4f",
                key=f"{base}sigma_proxy",
            )
        with col3:
            kappa = st.number_input("kappa", value=2.0, step=0.1, key=f"{base}kappa")
            theta = st.number_input(
                "theta", value=0.04, step=0.01, format="%.4f", key=f"{base}theta"
            )
            sigma = st.number_input(
                "eta (vol vol)", value=0.5, step=0.05, format="%.4f", key=f"{base}eta"
            )
            rho = st.number_input("rho", value=-0.7, step=0.05, format="%.4f", key=f"{base}rho")
            v0 = st.number_input(
                "v0 (var initiale)", value=0.04, step=0.01, format="%.4f", key=f"{base}v0"
            )

    if compute_button():
        try:
            price = price_heston_european_call(
                S0=S0,
                K=K,
                r=r,
                q=q,
                T=T,
                kappa=kappa,
                theta=theta,
                sigma=sigma,
                rho=rho,
                v0=v0,
                option_type="call" if option_type.lower().startswith("c") else "put",
            )
            st.success(f"Prix Heston {option_type} ~ {price:.4f}")
        except Exception as exc:
            st.error(f"Echec du pricing Heston : {exc}")

    st.markdown("---")
    st.subheader("Analyse complete (GPT-style)")
    with st.expander("Afficher l'analyse complete", expanded=False):
        render_payoff_text(option_type, "payoff")
        render_heatmap_diagnostics(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma_proxy,
            n_steps=25,
            option_char="c" if option_type.lower().startswith("c") else "p",
        )
        try:
            opt_obj = _EuropeanOption(S0, K, T, call=option_type.lower().startswith("c"))
            render_crr_tree(opt_obj, r=r, sigma=sigma_proxy, n_steps=10)
        except Exception:
            st.info("Arbre CRR non disponible pour ces parametres.")

    st.markdown("---")
    st.subheader("Surface de payoff (CRR)")
    with st.expander("Previsualisation de la surface de payoff", expanded=False):
        render_crr_payoff_surface(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma_proxy,
            option_char="c" if option_type.lower().startswith("c") else "p",
        )
