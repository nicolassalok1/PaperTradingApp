import importlib.util
from pathlib import Path

import streamlit as st


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Impossible de charger le module {name} depuis {path}")
    spec.loader.exec_module(module)
    return module


BASE_DIR = Path(__file__).parent
SCRIPTS_APP_PATH = BASE_DIR / "scripts" / "streamlit_app.py"
SOURCES_APP_PATH = BASE_DIR / "sources" / "streamlit_app3.py"

st.set_page_config(page_title="Volatility + RL Dashboards", layout="wide")
st.title("Volatility + RL Dashboards")
st.caption("Deux applications Streamlit réunies dans un seul projet, séparées par onglets.")

try:
    scripts_app = _load_module("volatility_scripts_app", SCRIPTS_APP_PATH)
except Exception as exc:
    scripts_app = None
    st.error(f"Impossible de charger l’app scripts : {exc}")

try:
    sources_app = _load_module("rl_finance_sources_app", SOURCES_APP_PATH)
except Exception as exc:
    sources_app = None
    st.error(f"Impossible de charger l’app sources : {exc}")


tabs_main = st.tabs(["Volatility Tools", "RL & Finance Lab"])

with tabs_main[0]:
    st.subheader("Volatility Tools")
    if scripts_app is None:
        st.warning("Module scripts introuvable.")
    else:
        vol_tabs = st.tabs(
            [
                "Volatility Crush Analyzer",
                "IV Dashboard",
                "Earnings IV Crush",
            ]
        )
        scripts_app.render_vol_crush_tab(vol_tabs[0])
        scripts_app.render_iv_dashboard_tab(vol_tabs[1])
        scripts_app.render_earnings_dashboard_tab(vol_tabs[2])

with tabs_main[1]:
    st.subheader("RL & Quant Finance Lab")
    if sources_app is None:
        st.warning("Module sources introuvable.")
    else:
        rl_tabs = st.tabs(
            [
                "Agents RL",
                "Pricing & Hedging",
                "Signals",
                "Strategies",
                "NLP Sentiment",
            ]
        )

        with rl_tabs[0]:
            sub_tabs = st.tabs(["Trading DQL", "Hedging DQL", "Allocation"])
            with sub_tabs[0]:
                sources_app.render_rl_trading()
            with sub_tabs[1]:
                sources_app.render_rl_hedging()
            with sub_tabs[2]:
                sources_app.render_rl_allocation()

        with rl_tabs[1]:
            sub_tabs = st.tabs(["Pricing", "Delta Hedging"])
            with sub_tabs[0]:
                sources_app.render_pricing_derivatives()
            with sub_tabs[1]:
                sources_app.render_hedging_derivatives()

        with rl_tabs[2]:
            sub_tabs = st.tabs(
                [
                    "Stock Prediction",
                    "BTC PCA",
                    "Eigen Portfolio",
                    "Yield Curve Build",
                    "Yield Curve Predict",
                ]
            )
            with sub_tabs[0]:
                sources_app.render_signals_stock_prediction()
            with sub_tabs[1]:
                sources_app.render_signals_btc_pca()
            with sub_tabs[2]:
                sources_app.render_signals_eigen_portfolio()
            with sub_tabs[3]:
                sources_app.render_signals_yield_curve_construction()
            with sub_tabs[4]:
                sources_app.render_signals_yield_curve_prediction()

        with rl_tabs[3]:
            sub_tabs = st.tabs(["BTC MA", "Crypto Allocation", "RL SP500"])
            with sub_tabs[0]:
                sources_app.render_strategies_btc_ma()
            with sub_tabs[1]:
                sources_app.render_strategies_crypto_allocation()
            with sub_tabs[2]:
                sources_app.render_strategies_rl_sp500()

        with rl_tabs[4]:
            sources_app.render_strategies_nlp_sentiment()
