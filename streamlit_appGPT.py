"""
Streamlit-based trading and options pricing dashboard.

Responsibilities:
- Render the multi-tab UI (dashboard, trading systems, forwards, options pricing).
- Persist portfolios/systems/options in local JSON files under ./database/GPTab/jsons.
- Call external services: Alpaca (positions/orders/prices), yfinance (price history),
  OpenAI ChatGPT, CBOE, and various pricing libraries (torch, tensorflow, scipy, etc.).

Side effects & assumptions:
- Reads environment variables (e.g., OPENAI_API_KEY) and local JSON files.
- Performs network I/O to market/pricing APIs; failures are surfaced in the UI.
- Writes to disk in the working directory; callers must ensure filesystem access.
"""

import os
import sys
from pathlib import Path


def _coerce_env_value(val):
    # Ensure any Path/bytes/other objects become plain strings for HTTP clients.
    try:
        import os as _os
        if hasattr(_os, "fspath"):
            return val if isinstance(val, str) else _os.fspath(val)
    except Exception:
        pass
    return val if isinstance(val, str) else str(val)


def _coerce_all_env_to_str():
    for k, v in list(os.environ.items()):
        os.environ[k] = _coerce_env_value(v)


# Pre-coerce everything before importing libs that may read env vars
_coerce_all_env_to_str()

# Désactive torch.compile / torch._dynamo pour éviter les imports cassés sur certaines versions
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import streamlit as st
from openai import OpenAI
import json
import alpaca_trade_api as tradeapi
import time
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import datetime
import math
import yfinance as yf
import re
import inspect
import time
import threading
import shutil

# Configuration
APP_DIR = Path(__file__).resolve().parent
DATASETS_DIR = APP_DIR / "database" / "GPTab"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_CSV_DIR = DATASETS_DIR / "cache_CSV"
CACHE_CSV_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR = DATASETS_DIR / "jsons"
OLD_JSON_DIR = APP_DIR / "database" / "jsons"
try:
    if OLD_JSON_DIR.exists() and not JSON_DIR.exists():
        shutil.copytree(OLD_JSON_DIR, JSON_DIR, dirs_exist_ok=True)
        # Supprime l'ancien répertoire après migration pour éviter la duplication
        shutil.rmtree(OLD_JSON_DIR, ignore_errors=True)
except Exception:
    pass
JSON_DIR.mkdir(parents=True, exist_ok=True)
# Scripts/pricing now sous scripts/scriptsGPT
SCRIPTS_DIR = APP_DIR / "scripts" / "scriptsGPT"
PRICING_DIR = SCRIPTS_DIR / "pricing_scripts"
NOTEBOOKS_SCRIPTS_DIR = APP_DIR / "notebooks" / "scripts"
PRICING_SCRIPT_DIR = DATASETS_DIR / "pricing_script"
CACHE_OPTIONS_HISTORY_FILE = CACHE_CSV_DIR / "options_last_history.csv"
CACHE_OPTIONS_CALLS_FILE = CACHE_CSV_DIR / "options_last_calls.csv"
CACHE_OPTIONS_PUTS_FILE = CACHE_CSV_DIR / "options_last_puts.csv"
CACHE_OPTIONS_META_FILE = JSON_DIR / "options_last_meta.json"
LEGACY_OPTIONS_META_FILE = DATASETS_DIR / "options_last_meta.json"
HESTON_PARAMS_FILE = JSON_DIR / "heston_params.json"
LEGACY_HESTON_PARAM_FILES = [
    DATASETS_DIR / "paraml_heston.json",
    DATASETS_DIR / "param_heston.json",
    DATASETS_DIR / "params_heston.json",
    DATASETS_DIR / "heston_params.json",
    APP_DIR / "database" / "jsons" / "paraml_heston.json",
    APP_DIR / "database" / "jsons" / "heston_params.json",
]
BASKET_CLOSING_FILE = CACHE_CSV_DIR / "basket_closing_prices.csv"
CLOSING_CACHE_FILE = CACHE_CSV_DIR / "closing_cache.csv"
TRAIN_FILE = CACHE_CSV_DIR / "train.csv"
TEST_FILE = CACHE_CSV_DIR / "test.csv"

# Migration : déplacer les CSV depuis database/GPTab vers database/GPTab/cache_CSV
def _migrate_csv_caches() -> None:
    legacy_files = [
        "options_last_history.csv",
        "options_last_calls.csv",
        "options_last_puts.csv",
        "basket_closing_prices.csv",
        "closing_cache.csv",
        "closing_prices.csv",
        "train.csv",
        "test.csv",
    ]
    for fname in legacy_files:
        old_path = DATASETS_DIR / fname
        new_path = CACHE_CSV_DIR / fname
        try:
            if old_path.exists() and not new_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.replace(new_path)
        except Exception:
            pass


def _migrate_heston_params_cache() -> None:
    """Ensure legacy Heston params caches are copied to the new JSON path."""
    if HESTON_PARAMS_FILE.exists():
        return
    for legacy_path in LEGACY_HESTON_PARAM_FILES:
        try:
            if legacy_path.exists():
                HESTON_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_path, HESTON_PARAMS_FILE)
                return
        except Exception:
            continue


_migrate_csv_caches()
_migrate_heston_params_cache()
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PRICING_DIR))
sys.path.insert(0, str(PRICING_SCRIPT_DIR))
sys.path.insert(0, str(NOTEBOOKS_SCRIPTS_DIR))
from rates_utils import get_r, get_q
from pricing import (
    bs_price_call,
    bs_price_put,
    payoff_strangle,
    price_butterfly_bs,
    price_call_spread_bs,
    price_condor_bs,
    price_iron_butterfly_bs,
    price_iron_condor_bs,
    price_put_spread_bs,
    price_straddle_bs,
    price_strangle_bs,
    view_asian_arith,
    view_asian_geom,
    view_asset_or_nothing,
    view_barrier,
    view_butterfly,
    view_call_spread,
    view_calendar_spread,
    view_chooser,
    view_condor,
    view_diagonal_spread,
    view_digital,
    view_forward_start,
    view_iron_butterfly,
    view_iron_condor,
    view_lookback,
    view_lookback_fixed,
    view_put_spread,
    view_quanto,
    view_cliquet,
    view_rainbow,
    view_straddle,
    view_strangle,
    price_heston_carr_madan,
)
DATA_FILE = JSON_DIR / "equities.json"
PORTFOLIO_FILE = JSON_DIR / "portfolio.json"
SELL_SYSTEMS_FILE = JSON_DIR / "sell_systems.json"
OPTIONS_BOOK_FILE = JSON_DIR / "options_portfolio.json"
OPTIONS_BOOK_FILE_LEGACY = JSON_DIR / "options_book.json"
OPTIONS_PORTFOLIO_FILE = OPTIONS_BOOK_FILE  # legacy name kept for compatibility
EXPIRED_OPTIONS_FILE = OPTIONS_BOOK_FILE    # legacy name kept for compatibility
CUSTOM_OPTIONS_FILE = JSON_DIR / "custom_options.json"
FORWARDS_FILE = JSON_DIR / "forwards.json"
DASHBOARD_CACHE_FILE = JSON_DIR / "Dashboard_cache.json"
LEGACY_EXPIRED_FILE = JSON_DIR / "expired_options.json"
load_dotenv()


def _coerce_env_to_str(name: str) -> str | None:
    """Ensure env vars are plain strings (avoid Path/bytes breaking .startswith)."""
    val = os.getenv(name)
    if val is None:
        return None
    if not isinstance(val, str):
        val = str(val)
        os.environ[name] = val
    return val


# Coerce frequently used env vars (uppercase + lowercase) that feed into HTTP clients
_ENV_KEYS = [
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL",
]
_ENV_KEYS += [k.lower() for k in _ENV_KEYS]
for _env_key in _ENV_KEYS:
    _coerce_env_to_str(_env_key)

# Safety net after load_dotenv
_coerce_all_env_to_str()


def _patch_inspect_pathlike_to_str() -> None:
    """Force inspect.getsourcefile à retourner une str même si un Path est rencontré (torch peut casser sinon)."""
    try:
        original = inspect.getsourcefile

        def _safe_getsourcefile(obj):
            res = original(obj)
            return os.fspath(res) if isinstance(res, (Path, os.PathLike)) else res

        inspect.getsourcefile = _safe_getsourcefile  # type: ignore[assignment]
    except Exception:
        pass


_patch_inspect_pathlike_to_str()

def run_app_options():
    """
    Render the options pricing tab (legacy monolith) inside the main Streamlit app.

    This function wires pricing models (Longstaff, Heston, lookback, etc.), visualization,
    and the ability to push priced structures into the unified dashboard JSON store.
    Side effects: imports heavy numerical libs, sets TensorFlow flags, reads/writes JSON
    files in ./database, and invokes networked data sources (yfinance, requests).
    """
    # TensorFlow et certains modules inspectent sys.argv ; force tout en str pour éviter les Path
    import sys as _sys

    _sys.argv = [str(a) for a in _sys.argv]
    import io
    import math
    import os
    import subprocess
    import sys
    from pathlib import Path
    import time
    import re
    import base64
    import datetime

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # désactive torch._compile/._dynamo pour éviter les imports cassés sur optimizer
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # évite l’initialisation torch._dynamo sur torch.optim

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import requests
    import streamlit as st
    import tensorflow as tf
    import yfinance as yf
    from rates_utils import get_r as get_r_interp, get_q as get_q_yf
    import torch
    import yfinance as yf
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy import stats
    from scipy.interpolate import griddata
    from scipy.linalg import lu_factor, lu_solve
    from scipy.stats import norm
    from typing import Callable

    from Longstaff.option import Option
    from Longstaff.pricing import (
        black_scholes_merton,
        crr_pricing,
        monte_carlo_simulation,
    )
    from Longstaff.process import GeometricBrownianMotion, HestonProcess
    from Lookback.european_call import european_call_option
    from Lookback.lookback_call import lookback_call_option
    from Heston.heston_torch import HestonParams, carr_madan_call_torch

    torch.set_default_dtype(torch.float64)
    def _pick_heston_device():
        """Sélectionne le device pour la calibration NN : CUDA si dispo, sinon CPU."""
        try:
            if torch.cuda.is_available():
                return torch.device("cuda")
        except Exception:
            pass
        return torch.device("cpu")

    HES_DEVICE = _pick_heston_device()
    try:
        HES_DEVICE_LABEL = (
            f"cuda ({torch.cuda.get_device_name(0)})"
            if HES_DEVICE.type == "cuda"
            else HES_DEVICE.type
        )
    except Exception:
        HES_DEVICE_LABEL = str(HES_DEVICE)
    MIN_IV_MATURITY = 0.1


    PLOTLY_CONFIG = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["sendDataToCloud"],
    }

    # Orange theme for "Ajouter au dashboard" buttons
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #ff8c00 !important;
            color: #fff !important;
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def render_add_to_dashboard_button(
        product_label: str,
        option_char: str,
        price_value: float | None,
        strike: float | None,
        maturity: float | None,
        key_prefix: str,
        *,
        strike2: float | None = None,
        spot: float | None = None,
        legs: list[dict] | None = None,
        misc: dict | None = None,
        expanded: bool = True,
    ) -> None:
        """Prompt to push a priced structure into the dashboard JSON."""
        if "add_option_to_dashboard" not in globals():
            st.info("Ajout au dashboard indisponible (fonction manquante).")
            return

        with st.expander(f"📥 Ajouter au dashboard ({product_label})", expanded=expanded):
            display_price = f"${price_value:.6f}" if price_value is not None else "-"
            st.metric("Prix calculé", display_price)
            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l’entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((maturity or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=f"{key_prefix}_qty")
            side = st.selectbox("Sens", ["long", "short"], index=0, key=f"{key_prefix}_side")
            strike_val = float(strike if strike is not None else st.session_state.get("common_strike", 0.0))
            strike2_val = float(strike2) if strike2 is not None else None
            st.caption(
                f"K (strike commun): {strike_val:.4f}"
                + (f" | K2: {strike2_val:.4f}" if strike2_val is not None else "")
            )
            if maturity is not None:
                st.caption(f"T (maturité commune, années): {float(maturity):.4f}")

            if st.button("Ajouter au dashboard", key=f"{key_prefix}_add"):
                base_misc = {
                    "structure": product_label,
                    "legs": legs,
                    "strike2": strike2_val,
                    "spot_at_pricing": float(spot or 0.0),
                }
                if isinstance(misc, dict):
                    base_misc.update(misc)
                misc_payload = base_misc
                price_val = float(price_value) if price_value is not None else 0.0
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": "call" if option_char.lower() == "c" else "put",
                    "product_type": product_label,
                    "type": product_label,
                    "strike": float(strike_val),
                    "strike2": float(strike2_val) if strike2_val is not None else None,
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price_val,
                    "side": side,
                    "S0": float(spot or 0.0),
                    "maturity_years": maturity,
                    "legs": legs,
                    "T_0": today.isoformat(),
                    "price": price_val,
                    "misc": misc_payload,
                }
                try:
                    # Log the JSON write attempt (UI + server log)
                    st.caption(
                        f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}"
                    )
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"{product_label} ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(
                        f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}"
                    )


    def simulate_gbm_paths(S0, r, q, sigma, T, M, N_paths, seed=42):
        """
        Simulate GBM paths under the risk-neutral measure:
            dS_t = (r - q) S_t dt + sigma S_t dW_t
        Returns:
            S : array of shape (M+1, N_paths)
            dt: time step
        """
        dt = T / M
        rng = np.random.default_rng(seed)
        S = np.empty((M + 1, N_paths))
        S[0, :] = S0
        Z = rng.normal(size=(M, N_paths))
        drift = (r - q - 0.5 * sigma**2) * dt
        vol_term = sigma * np.sqrt(dt)
        for t in range(1, M + 1):
            S[t, :] = S[t - 1, :] * np.exp(drift + vol_term * Z[t - 1, :])
        return S, dt


    def price_bermudan_lsmc(
        S0,
        K,
        T,
        r,
        q,
        sigma,
        cpflag="p",
        M=50,
        N_paths=100_000,
        degree=3,
        n_ex_dates=6,
        seed: int = 42,
    ):
        """
        Longstaff–Schwartz Monte Carlo pricing for a Bermudan option
        under risk-neutral GBM (Black–Scholes dynamics).
        """
        S, dt = simulate_gbm_paths(S0, r, q, sigma, T, M, N_paths, seed=seed)
        disc = np.exp(-r * dt)

        if cpflag == "c":
            Y = np.maximum(S - K, 0.0)
        elif cpflag == "p":
            Y = np.maximum(K - S, 0.0)
        else:
            raise ValueError("cpflag must be 'c' or 'p'")

        C = Y[-1, :].copy()
        ex_indices = np.linspace(1, M - 1, max(1, n_ex_dates - 1), dtype=int)
        ex_set = set(ex_indices.tolist())

        for j in range(M - 1, 0, -1):
            C *= disc
            if j in ex_set:
                S_j = S[j, :]
                Y_j = Y[j, :]
                itm = Y_j > 0.0
                if np.any(itm):
                    X = np.vstack([S_j[itm] ** k for k in range(degree + 1)]).T
                    y_reg = C[itm]
                    beta, *_ = np.linalg.lstsq(X, y_reg, rcond=None)
                    X_all = np.vstack([S_j**k for k in range(degree + 1)]).T
                    C_hat = X_all @ beta
                    exercise = (Y_j > C_hat) & itm
                    C[exercise] = Y_j[exercise]

        C *= disc
        price = np.mean(C)
        return float(price)


    # ---------------------------------------------------------------------------
    #  Bermudan / European / American + Barrier (Crank–Nicolson)
    # ---------------------------------------------------------------------------

    class CrankNicolsonBS:
        """
        Solveur Crank–Nicolson pour la PDE de Black–Scholes en log(S).

        Typeflag:
            'Eu'  : option européenne
            'Am'  : option américaine (exercice possible à chaque date de grille)
            'Bmd' : option bermudéenne (exercice possible à certaines dates)
        cpflag:
            'c' : call
            'p' : put
        """

        def __init__(
            self,
            Typeflag: str,
            cpflag: str,
            S0: float,
            K: float,
            T: float,
            vol: float,
            r: float,
            d: float,
            n_spatial: int = 500,
            n_time: int = 600,
            exercise_step: int | None = None,
            n_exercise_dates: int | None = None,
            **_,
        ) -> None:
            self.Typeflag = Typeflag
            self.cpflag = cpflag
            self.S0 = float(S0)
            self.K = float(K)
            self.T = float(T)
            self.vol = float(vol)
            self.r = float(r)
            self.d = float(d)

            self.n_spatial = max(50, int(n_spatial))
            self.n_time = max(50, int(n_time))

            # Deux modes possibles pour la Bermudane :
            # - exercise_step       : exercice tous les 'exercise_step' pas
            # - n_exercise_dates    : nb de dates d'exercice (incluant T)
            # Si les deux sont donnés -> erreur, c'est ambigu.
            if exercise_step is not None and n_exercise_dates is not None:
                raise ValueError(
                    "Spécifie soit exercise_step, soit n_exercise_dates, pas les deux."
                )

            self.exercise_step = int(exercise_step) if exercise_step is not None else None
            self.n_exercise_dates = (
                int(n_exercise_dates) if n_exercise_dates is not None else None
            )

        # -------------------- utils --------------------

        def _resolve_params(
            self,
            Typeflag: str | None,
            cpflag: str | None,
            S0: float | None,
            K: float | None,
            T: float | None,
            vol: float | None,
            r: float | None,
            d: float | None,
        ):
            """Résout les paramètres effectifs sans casser les valeurs 0 éventuelles."""

            Typeflag = self.Typeflag if Typeflag is None else Typeflag
            cpflag = self.cpflag if cpflag is None else cpflag
            S0 = self.S0 if S0 is None else float(S0)
            K = self.K if K is None else float(K)
            T = self.T if T is None else float(T)
            vol = self.vol if vol is None else float(vol)
            r = self.r if r is None else float(r)
            d = self.d if d is None else float(d)
            return Typeflag, cpflag, S0, K, T, vol, r, d

        # -------------------- solveur CN --------------------

        def CN_option_info(
            self,
            Typeflag: str | None = None,
            cpflag: str | None = None,
            S0: float | None = None,
            K: float | None = None,
            T: float | None = None,
            vol: float | None = None,
            r: float | None = None,
            d: float | None = None,
        ) -> tuple[float, float, float, float]:
            """
            Résout la PDE et retourne (Price, Delta, Gamma, Theta).
            """

            Typeflag, cpflag, S0, K, T, vol, r, d = self._resolve_params(
                Typeflag, cpflag, S0, K, T, vol, r, d
            )

            Typeflag = Typeflag.strip()
            cpflag = cpflag.strip()
            if Typeflag not in {"Eu", "Am", "Bmd"}:
                raise ValueError("Typeflag doit être 'Eu', 'Am' ou 'Bmd'.")
            if cpflag not in {"c", "p"}:
                raise ValueError("cpflag doit être 'c' ou 'p'.")

            # Cas trivial T=0
            if T <= 0.0 or self.n_time <= 0:
                payoff0 = max(S0 - K, 0.0) if cpflag == "c" else max(K - S0, 0.0)
                return float(payoff0), 0.0, 0.0, 0.0

            if Typeflag == "Bmd":
                M_lsmc = max(1, min(self.n_time, 50))
                N_paths = 50_000
                n_ex_dates = self.n_exercise_dates or 6
                seed_base = 12345

                def _lsmc_price(s0_val: float, t_val: float) -> float:
                    return price_bermudan_lsmc(
                        S0=s0_val,
                        K=K,
                        T=max(t_val, 1e-6),
                        r=r,
                        q=d,
                        sigma=vol,
                        cpflag=cpflag,
                        M=M_lsmc,
                        N_paths=N_paths,
                        degree=3,
                        n_ex_dates=n_ex_dates,
                        seed=seed_base,
                    )

                price_bmd = _lsmc_price(S0, T)

                bump_s = max(1e-4, 0.01 * S0)
                price_up = _lsmc_price(S0 + bump_s, T)
                price_down = _lsmc_price(max(S0 - bump_s, 1e-6), T)
                delta = (price_up - price_down) / (2.0 * bump_s)
                gamma = (price_up - 2.0 * price_bmd + price_down) / (bump_s**2)

                theta = 0.0
                theta_h = min(max(1.0 / 365.0, 0.01 * T), max(T / 2.0, 1e-6))
                if T > theta_h:
                    price_short = _lsmc_price(S0, T - theta_h)
                    theta = (price_short - price_bmd) / theta_h

                return float(price_bmd), float(delta), float(gamma), float(theta)

            # ----- Grille en log(S) -----
            mu = r - d - 0.5 * vol * vol
            x_max = vol * np.sqrt(max(T, 1e-8)) * 5.0
            n_points = self.n_spatial
            dx = 2.0 * x_max / n_points

            X = np.linspace(-x_max, x_max, n_points + 1)
            max_log = np.log(np.finfo(float).max / max(S0, 1e-12))
            X_clipped = np.clip(X, -max_log, max_log)
            s_grid = S0 * np.exp(X_clipped)

            n_index = np.arange(0, n_points + 1)

            n_time = self.n_time
            dt = T / n_time

            a = 0.25 * dt * ((vol**2) * (n_index**2) - mu * n_index)
            b = -0.5 * dt * ((vol**2) * (n_index**2) + r)
            c = 0.25 * dt * ((vol**2) * (n_index**2) + mu * n_index)

            main_diag_A = 1.0 - b - 2.0 * a
            upper_A = a + c
            lower_A = a - c

            main_diag_B = 1.0 + b + 2.0 * a
            upper_B = -a - c
            lower_B = -a + c

            A = np.zeros((n_points + 1, n_points + 1))
            B = np.zeros((n_points + 1, n_points + 1))

            np.fill_diagonal(A, main_diag_A)
            np.fill_diagonal(A[1:], lower_A[:-1])
            np.fill_diagonal(A[:, 1:], upper_A[:-1])
            A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)

            np.fill_diagonal(B, main_diag_B)
            np.fill_diagonal(B[1:], lower_B[:-1])
            np.fill_diagonal(B[:, 1:], upper_B[:-1])
            B = np.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)

            lu_factor_A = lu_factor(A)

            # Payoff terminal
            if cpflag == "c":
                values = np.maximum(s_grid - K, 0.0)
            else:
                values = np.maximum(K - s_grid, 0.0)

            payoff = values.copy()
            values_prev_time = values.copy()

            S_max = s_grid[-1]
            S_min = s_grid[0]  # pas utilisé mais dispo si besoin

            # ----- Boucle backward -----
            for time_index in range(n_time):
                # Sauvegarde pour theta (un seul pas suffit)
                if time_index == n_time - 1:
                    values_prev_time = values.copy()

                rhs = B.dot(values)
                values = lu_solve(lu_factor_A, rhs)

                t_now = T - (time_index + 1) * dt
                tau = T - t_now  # temps restant à maturité

                # Conditions aux bords
                if cpflag == "c":
                    values[0] = 0.0
                    values[-1] = S_max - K * np.exp(-r * tau)
                else:
                    values[0] = K * np.exp(-r * tau)
                    values[-1] = 0.0

                # Gestion du style
                if Typeflag == "Am":
                    values = np.maximum(values, payoff)
                elif Typeflag == "Eu":
                    pass

            # ----- Grecs par différences finies -----
            middle_index = n_points // 2
            price = values[middle_index]

            s_plus = S0 * np.exp(dx)
            s_minus = S0 * np.exp(-dx)

            v_plus = values[middle_index + 1]
            v_0 = values[middle_index]
            v_minus = values[middle_index - 1]

            delta = (v_plus - v_minus) / (s_plus - s_minus)

            dVdS_plus = (v_plus - v_0) / (s_plus - S0)
            dVdS_minus = (v_0 - v_minus) / (S0 - s_minus)
            gamma = (dVdS_plus - dVdS_minus) / ((s_plus - s_minus) / 2.0)

            theta = -(values[middle_index] - values_prev_time[middle_index]) / dt

            return float(price), float(delta), float(gamma), float(theta)


    def CN_Barrier_option(Typeflag, cpflag, S0, K, Hu, Hd, T, vol, r, d):
        """
        Pricing d'une option barrière par Crank–Nicolson.
        """

        mu = r - d - 0.5 * vol * vol
        x_max = vol * np.sqrt(T) * 5
        n_points = 500
        dx = 2 * x_max / n_points
        X = np.linspace(-x_max, x_max, n_points + 1)
        n_index = np.arange(0, n_points + 1)

        n_time = 600
        dt = T / n_time

        a = 0.25 * dt * ((vol**2) * (n_index**2) - mu * n_index)
        b = -0.5 * dt * ((vol**2) * (n_index**2) + r)
        c = 0.25 * dt * ((vol**2) * (n_index**2) + mu * n_index)

        main_diag_A = 1 - b - 2 * a
        upper_A = a + c
        lower_A = a - c

        main_diag_B = 1 + b + 2 * a
        upper_B = -a - c
        lower_B = -a + c

        A = np.zeros((n_points + 1, n_points + 1))
        B = np.zeros((n_points + 1, n_points + 1))

        np.fill_diagonal(A, main_diag_A)
        np.fill_diagonal(A[1:], lower_A[:-1])
        np.fill_diagonal(A[:, 1:], upper_A[:-1])

        np.fill_diagonal(B, main_diag_B)
        np.fill_diagonal(B[1:], lower_B[:-1])
        np.fill_diagonal(B[:, 1:], upper_B[:-1])

        Ainv = np.linalg.inv(A)

        s_grid = S0 * np.exp(X)
        if cpflag == "c":
            values = np.clip(s_grid - K, 0, 1e10)
        elif cpflag == "p":
            values = np.clip(K - s_grid, 0, 1e10)
        else:
            raise ValueError("cpflag doit être 'c' ou 'p'.")

        typeflag = Typeflag.upper()
        if typeflag in {"UNO", "UO"}:
            values = np.where(s_grid < Hu, values, 0.0)
        elif typeflag == "DNO":
            values = np.where((s_grid > Hd) & (s_grid < Hu), values, 0.0)
        elif typeflag in {"DO"}:
            values = np.where(s_grid > Hd, values, 0.0)
        else:
            raise ValueError("Typeflag doit être 'UNO', 'UO', 'DO' ou 'DNO'.")

        values_prev_time = values.copy()

        for time_index in range(n_time):
            if time_index == n_time - 1:
                values_prev_time = values.copy()

            values = B.dot(values)
            values = Ainv.dot(values)

            s_grid = S0 * np.exp(X)
            if typeflag in {"UNO", "UO"}:
                values = np.where(s_grid < Hu, values, 0.0)
            elif typeflag == "DNO":
                values = np.where((s_grid > Hd) & (s_grid < Hu), values, 0.0)
            elif typeflag == "DO":
                values = np.where(s_grid > Hd, values, 0.0)

        middle_index = n_points // 2
        price = values[middle_index]

        s_plus = S0 * np.exp(dx)
        s_minus = S0 * np.exp(-dx)

        delta = (values[middle_index + 1] - values[middle_index - 1]) / (s_plus - s_minus)

        d_value_d_s_plus = (values[middle_index + 1] - values[middle_index]) / (s_plus - S0)
        d_value_d_s_minus = (values[middle_index] - values[middle_index - 1]) / (S0 - s_minus)
        gamma = (d_value_d_s_plus - d_value_d_s_minus) / ((s_plus - s_minus) / 2.0)

        theta = -(values[middle_index] - values_prev_time[middle_index]) / dt

        return float(price), float(delta), float(gamma), float(theta)


    # ---------------------------------------------------------------------------
    #  Helper Longstaff–Schwartz qui retourne le prix (version locale)
    # ---------------------------------------------------------------------------


    def longstaff_schwartz_price(option: Option, process, n_paths: int, n_steps: int) -> float:
        """
        Implémentation locale de l'algorithme LS, basée sur Longstaff/pricing.py,
        mais qui renvoie le prix comme float.
        """
        from numpy.polynomial import Polynomial

        simulated_paths = process.simulate(s0=option.s0, v0=option.v0, T=option.T, n=n_paths, m=n_steps)
        payoffs = option.payoff(s=simulated_paths)

        continuation_values = np.zeros_like(payoffs)
        continuation_values[-1] = payoffs[-1]

        dt = option.T / n_steps
        discount = np.exp(-process.mu * dt)

        for time_index in range(n_steps - 1, 0, -1):
            polynomial = Polynomial.fit(simulated_paths[time_index], discount * continuation_values[time_index + 1], 5)
            continuation = polynomial(simulated_paths[time_index])
            continuation_values[time_index] = np.where(
                payoffs[time_index] > continuation,
                payoffs[time_index],
                discount * continuation_values[time_index + 1],
            )

        price = discount * np.mean(continuation_values[1])
        return float(price)


    # ---------------------------------------------------------------------------
    #  Outils pour les heatmaps européennes
    # ---------------------------------------------------------------------------

    HEATMAP_GRID_SIZE = 11


    def _heatmap_axis(center: float, span: float, n_points: int = HEATMAP_GRID_SIZE) -> np.ndarray:
        lower = max(0.01, center - span)
        upper = max(lower, center + span)
        if np.isclose(lower, upper) or n_points == 1:
            return np.array([lower])
        return np.linspace(lower, upper, n_points)


    def _render_heatmap(
        matrix: np.ndarray,
        x_values: np.ndarray,
        y_values: np.ndarray,
        title: str,
        xlabel: str = "Spot",
        ylabel: str = "Strike",
        wrap_in_expander: bool = True,
    ) -> None:
        def _draw():
            fig, ax = plt.subplots()
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
                cmap="viridis",
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)

        if wrap_in_expander:
            with st.expander(f"Afficher la heatmap : {title}", expanded=False):
                _draw()
        else:
            _draw()


    def _render_call_put_heatmaps(
        label: str, call_matrix: np.ndarray, put_matrix: np.ndarray, x_values: np.ndarray, y_values: np.ndarray
    ) -> None:
        with st.expander(f"Afficher les heatmaps : {label}", expanded=False):
            st.write(f"Heatmap Call ({label})")
            _render_heatmap(call_matrix, x_values, y_values, f"Call ({label})", wrap_in_expander=False)
            st.write(f"Heatmap Put ({label})")
            _render_heatmap(put_matrix, x_values, y_values, f"Put ({label})", wrap_in_expander=False)


    def _compute_bsm_heatmaps(
        s_values: np.ndarray, k_values: np.ndarray, maturity: float, rate: float, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        call_matrix = np.zeros((len(k_values), len(s_values)))
        put_matrix = np.zeros_like(call_matrix)
        for i, strike in enumerate(k_values):
            for j, spot in enumerate(s_values):
                option_call = Option(s0=spot, T=maturity, K=strike, call=True)
                option_put = Option(s0=spot, T=maturity, K=strike, call=False)
                call_matrix[i, j] = black_scholes_merton(r=rate, sigma=sigma, option=option_call)
                put_matrix[i, j] = black_scholes_merton(r=rate, sigma=sigma, option=option_put)
        return call_matrix, put_matrix


    def _compute_mc_heatmaps(
        s_values: np.ndarray,
        k_values: np.ndarray,
        maturity: float,
        mu: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        process = GeometricBrownianMotion(mu=mu, sigma=sigma)
        discount = np.exp(-mu * maturity)
        call_matrix = np.zeros((len(k_values), len(s_values)))
        put_matrix = np.zeros_like(call_matrix)

        for j, spot in enumerate(s_values):
            simulated_paths = process.simulate(s0=spot, T=maturity, n=n_paths, m=n_steps, v0=None)
            terminal_prices = simulated_paths[-1]
            for i, strike in enumerate(k_values):
                call_matrix[i, j] = np.mean(np.maximum(terminal_prices - strike, 0)) * discount
                put_matrix[i, j] = np.mean(np.maximum(strike - terminal_prices, 0)) * discount

        return call_matrix, put_matrix


    def _vanilla_price_with_dividend(
        option_type: str,
        S0: float,
        K: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> float:
        if T <= 0 or sigma <= 0 or K <= 0 or S0 <= 0:
            intrinsic = max(S0 - K, 0.0) if option_type.lower() in {"call", "c"} else max(K - S0, 0.0)
            return float(intrinsic)
        sqrt_T = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - dividend + 0.5 * sigma * sigma) * T) / sqrt_T
        d2 = d1 - sqrt_T
        if option_type.lower() in {"call", "c"}:
            price = S0 * np.exp(-dividend * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-dividend * T) * norm.cdf(-d1)
        return float(max(price, 0.0))


    def _digital_cash_or_nothing_price(
        option_type: str,
        S0: float,
        K: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
        payout: float = 1.0,
    ) -> float:
        if T <= 0 or sigma <= 0 or payout <= 0:
            return 0.0
        sqrt_T = sigma * np.sqrt(T)
        d2 = (np.log(S0 / K) + (r - dividend - 0.5 * sigma * sigma) * T) / sqrt_T
        disc = payout * np.exp(-r * T)
        if option_type.lower() in {"call", "c"}:
            return float(disc * norm.cdf(d2))
        return float(disc * norm.cdf(-d2))


    def _asset_or_nothing_price(
        option_type: str,
        S0: float,
        K: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> float:
        if T <= 0 or sigma <= 0 or S0 <= 0:
            return 0.0
        sqrt_T = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - dividend + 0.5 * sigma * sigma) * T) / sqrt_T
        disc = S0 * np.exp(-dividend * T)
        if option_type.lower() in {"call", "c"}:
            return float(disc * norm.cdf(d1))
        return float(disc * norm.cdf(-d1))


    def _chooser_option_price(
        S0: float,
        K: float,
        T: float,
        t_choice: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> float:
        # Formula: call(T,K) + put(t_choice, K*exp(- (r - dividend)*(T - t_choice)))
        call_price = _vanilla_price_with_dividend("call", S0, K, T, r, dividend, sigma)
        tau = max(0.0, T - t_choice)
        if tau <= 0:
            return call_price
        K_adj = K * np.exp(-r * (T - t_choice))
        sqrt_tau = sigma * np.sqrt(tau)
        d1 = (np.log(S0 / K_adj) + (r - dividend + 0.5 * sigma * sigma) * tau) / sqrt_tau
        d2 = d1 - sqrt_tau
        put_piece = K_adj * np.exp(-r * t_choice) * norm.cdf(-d2) - S0 * np.exp(-dividend * t_choice) * norm.cdf(-d1)
        return float(max(call_price + put_piece, 0.0))


    def _forward_start_price_mc(
        S0: float,
        r: float,
        q: float,
        sigma: float,
        T_start: float,
        T_end: float,
        k: float,
        n_paths: int = 5000,
        n_steps: int = 200,
        option_type: str = "call",
    ) -> float:
        if T_end <= T_start or sigma <= 0 or n_paths <= 0 or n_steps <= 0:
            return 0.0
        dt = T_end / n_steps
        drift = (r - q - 0.5 * sigma * sigma) * dt
        diff = sigma * np.sqrt(dt)
        disc = np.exp(-r * T_end)
        payoffs = []
        step_choice = int(T_start / dt)
        for _ in range(n_paths):
            s = S0
            s_start = None
            for step in range(n_steps):
                z = np.random.normal()
                s *= np.exp(drift + diff * z)
                if s_start is None and step >= step_choice:
                    s_start = s
            s_start = s if s_start is None else s_start
            strike_dyn = k * s_start
            if option_type.lower() in {"call", "c"}:
                payoff = max(s - strike_dyn, 0.0)
            else:
                payoff = max(strike_dyn - s, 0.0)
            payoffs.append(payoff)
        return float(disc * np.mean(payoffs))


    def _binary_barrier_mc(
        option_type: str,
        barrier_type: str,
        direction: str,
        S0: float,
        K: float,
        barrier: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
        payout: float,
        n_paths: int,
        n_steps: int,
    ) -> float:
        if payout <= 0 or barrier <= 0 or n_paths <= 0 or n_steps <= 0:
            return 0.0
        dt = T / n_steps
        drift = (r - dividend - 0.5 * sigma * sigma) * dt
        diff = sigma * np.sqrt(dt)
        disc = np.exp(-r * T)
        hits = []
        for _ in range(n_paths):
            s = S0
            touched = False
            for _ in range(n_steps):
                z = np.random.normal()
                s *= np.exp(drift + diff * z)
                if (barrier_type == "up" and s >= barrier) or (barrier_type == "down" and s <= barrier):
                    touched = True
                    break
            if direction == "out":
                pay = 0.0 if touched else payout
            else:  # in
                pay = payout if touched else 0.0
            # Optional vanilla style digital with strike K at maturity if not handling knock condition
            if pay == 0.0:
                if option_type.lower() in {"call", "c"}:
                    pay = payout if s >= K else 0.0
                else:
                    pay = payout if s <= K else 0.0
            hits.append(pay)
        return float(disc * np.mean(hits))


    def _cliquet_mc(
        S0: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        n_periods: int,
        cap: float,
        floor: float,
        n_paths: int = 2000,
        seed: int | None = None,
    ) -> float:
        if n_periods <= 0 or n_paths <= 0 or T <= 0:
            return 0.0
        rng = np.random.default_rng(seed)
        dt = T / n_periods
        drift = (r - q - 0.5 * sigma * sigma) * dt
        diff = sigma * np.sqrt(dt)
        disc = np.exp(-r * T)
        payoffs = []
        for _ in range(n_paths):
            s = S0
            coupons = []
            for _ in range(n_periods):
                z = rng.normal()
                s_next = s * np.exp(drift + diff * z)
                ret = (s_next / s) - 1.0
                coupons.append(np.clip(ret, floor, cap))
                s = s_next
            payoffs.append(sum(coupons))
        return float(disc * np.mean(payoffs))


    def _quanto_vanilla_price(
        option_type: str,
        S0: float,
        K: float,
        T: float,
        r_dom: float,
        q_for: float,
        sigma_asset: float,
        sigma_fx: float,
        rho: float,
    ) -> float:
        # Simple quanto adjustment on dividend: q* = q_for + rho*sigma_S*sigma_FX
        q_adj = q_for + rho * sigma_asset * sigma_fx
        return _vanilla_price_with_dividend(option_type, S0, K, T, r_dom, q_adj, sigma_asset)


    def _rainbow_two_asset_mc(
        payoff_on: str,
        S0_a: float,
        S0_b: float,
        sigma_a: float,
        sigma_b: float,
        rho: float,
        K: float,
        T: float,
        r: float,
        q_a: float,
        q_b: float,
        n_paths: int = 5000,
        n_steps: int = 200,
        option_type: str = "call",
    ) -> float:
        if n_paths <= 0 or n_steps <= 0 or T <= 0:
            return 0.0
        dt = T / n_steps
        disc = np.exp(-r * T)
        payoff_list = []
        for _ in range(n_paths):
            s_a, s_b = S0_a, S0_b
            for _ in range(n_steps):
                z1 = np.random.normal()
                z2 = np.random.normal()
                z_b = rho * z1 + np.sqrt(max(0.0, 1 - rho**2)) * z2
                s_a *= np.exp((r - q_a - 0.5 * sigma_a * sigma_a) * dt + sigma_a * np.sqrt(dt) * z1)
                s_b *= np.exp((r - q_b - 0.5 * sigma_b * sigma_b) * dt + sigma_b * np.sqrt(dt) * z_b)
            if payoff_on == "max":
                s_star = max(s_a, s_b)
            else:
                s_star = min(s_a, s_b)
            if option_type.lower() in {"call", "c"}:
                payoff = max(s_star - K, 0.0)
            else:
                payoff = max(K - s_star, 0.0)
            payoff_list.append(payoff)
        return float(disc * np.mean(payoff_list))

    def _barrier_closed_form_price(
        option_type: str,
        barrier_type: str,
        S0: float,
        K: float,
        barrier: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> float:
        if barrier <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("Paramètres invalides pour la formule fermée barrière.")
        if barrier_type == "up" and S0 >= barrier:
            return 0.0
        if barrier_type == "down" and S0 <= barrier:
            return 0.0

        option_flag = option_type.lower()
        phi = 1.0 if option_flag in {"call", "c"} else -1.0
        eta = 1.0 if barrier_type == "down" else -1.0
        mu = (r - dividend - 0.5 * sigma * sigma) / (sigma * sigma)
        sigma_sqrt_T = sigma * np.sqrt(T)
        if sigma_sqrt_T == 0:
            return 0.0
        x1 = (np.log(S0 / K) / sigma_sqrt_T) + (1.0 + mu) * sigma_sqrt_T
        y1 = (np.log((barrier * barrier) / (S0 * K)) / sigma_sqrt_T) + (1.0 + mu) * sigma_sqrt_T
        power1 = (barrier / S0) ** (2.0 * (mu + eta))
        power2 = (barrier / S0) ** (2.0 * mu)
        term1 = phi * S0 * np.exp(-dividend * T) * (norm.cdf(phi * x1) - power1 * norm.cdf(eta * y1))
        term2 = phi * K * np.exp(-r * T) * (norm.cdf(phi * x1 - phi * sigma_sqrt_T) - power2 * norm.cdf(eta * y1 - eta * sigma_sqrt_T))
        price = term1 - term2
        return max(float(price), 0.0)


    def _knock_in_closed_form_price(
        option_type: str,
        barrier_type: str,
        S0: float,
        K: float,
        barrier: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> float:
        vanilla = _vanilla_price_with_dividend(
            option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma
        )
        barrier_out_price = _barrier_closed_form_price(
            option_type=option_type,
            barrier_type=barrier_type,
            S0=S0,
            K=K,
            barrier=barrier,
            T=T,
            r=r,
            dividend=dividend,
            sigma=sigma,
        )
        return max(vanilla - barrier_out_price, 0.0)


    def _barrier_monte_carlo_price(
        option_type: str,
        barrier_type: str,
        S0: float,
        K: float,
        barrier: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
        knock_in: bool = False,
    ) -> float:
        if barrier <= 0 or n_paths <= 0 or n_steps <= 0:
            raise ValueError("Paramètres invalides pour le Monte Carlo barrière.")
        option_type_lower = option_type.lower()
        if barrier_type == "up" and S0 >= barrier:
            if knock_in:
                return _vanilla_price_with_dividend(option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma)
            return 0.0
        if barrier_type == "down" and S0 <= barrier:
            if knock_in:
                return _vanilla_price_with_dividend(option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma)
            return 0.0
        dt = T / n_steps
        drift = (r - dividend - 0.5 * sigma * sigma) * dt
        diffusion = sigma * np.sqrt(dt)
        discount = np.exp(-r * T)
        payoffs = []
        for _ in range(n_paths):
            s = S0
            barrier_hit = False
            for _ in range(n_steps):
                z = np.random.normal()
                s *= np.exp(drift + diffusion * z)
                if barrier_type == "up" and s >= barrier:
                    barrier_hit = True
                    if not knock_in:
                        break
                elif barrier_type == "down" and s <= barrier:
                    barrier_hit = True
                    if not knock_in:
                        break
            if knock_in and not barrier_hit:
                payoffs.append(0.0)
                continue
            if not knock_in and barrier_hit:
                payoffs.append(0.0)
                continue
            if option_type_lower in {"call", "c"}:
                payoff = max(s - K, 0.0)
            else:
                payoff = max(K - s, 0.0)
            payoffs.append(payoff)
        return discount * (float(np.mean(payoffs)) if payoffs else 0.0)


    def _render_barrier_stock_paths(
        S0: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
        barrier: float,
        barrier_type: str,
        n_steps: int,
        title_suffix: str,
    ):
        """Display a few GBM trajectories with the active barrier level overlaid."""
        n_steps = max(5, int(n_steps))
        dt = T / n_steps if n_steps > 0 else T
        times = np.linspace(0.0, T, n_steps + 1)
        n_paths_plot = 5
        paths = np.empty((n_paths_plot, n_steps + 1))

        drift = (r - dividend - 0.5 * sigma * sigma) * dt
        vol_step = sigma * np.sqrt(dt)

        for i in range(n_paths_plot):
            shocks = np.random.normal(size=n_steps)
            log_path = np.empty(n_steps + 1)
            log_path[0] = np.log(S0)
            log_path[1:] = log_path[0] + np.cumsum(drift + vol_step * shocks)
            paths[i] = np.exp(log_path)

        fig, ax = plt.subplots(figsize=(7, 3))
        for i in range(n_paths_plot):
            ax.plot(times, paths[i], alpha=0.65, linewidth=1.4)

        is_up = barrier_type == "up"
        color = "crimson" if is_up else "steelblue"
        label = "Barrière haute" if is_up else "Barrière basse"
        ax.axhline(barrier, color=color, linestyle="--", linewidth=2.0, label=f"{label} = {barrier:.2f}")
        ax.set_xlabel("Temps (années)")
        ax.set_ylabel("Sous-jacent simulé")
        ax.set_title(f"Trajectoires simulées + barrière ({title_suffix})")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


    def _compute_barrier_heatmap_matrix(
        option_type: str,
        barrier_type: str,
        strike_values: np.ndarray,
        offset_values: np.ndarray,
        S0: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.zeros((len(strike_values), len(offset_values)))
        ratio_axis = np.zeros(len(offset_values))

        for j, offset in enumerate(offset_values):
            if barrier_type == "up":
                ratio = 1.1 + offset
            else:
                ratio = max(0.01, 0.9 - offset)
            ratio_axis[j] = ratio

            for i, strike in enumerate(strike_values):
                barrier = max(strike * ratio, 0.01)
                try:
                    price = _barrier_closed_form_price(
                        option_type=option_type,
                        barrier_type=barrier_type,
                        S0=S0,
                        K=float(strike),
                        barrier=float(barrier),
                        T=T,
                        r=r,
                        dividend=dividend,
                        sigma=sigma,
                    )
                except ValueError:
                    price = 0.0
                matrix[i, j] = price

        if np.any(np.diff(ratio_axis) < 0):
            order = np.argsort(ratio_axis)
            ratio_axis = ratio_axis[order]
            matrix = matrix[:, order]

        return matrix, ratio_axis


    def _compute_up_and_out_strike_heatmap(
        option_type: str,
        barrier: float,
        strike_values: np.ndarray,
        maturity_values: np.ndarray,
        spot: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> np.ndarray:
        """
        Construit une matrice de prix up-and-out selon (T, K) pour un spot fixe.
        """
        matrix = np.zeros((len(maturity_values), len(strike_values)))
        for i, maturity in enumerate(maturity_values):
            for j, strike in enumerate(strike_values):
                if strike <= 0.0:
                    matrix[i, j] = 0.0
                    continue
                try:
                    price = _barrier_closed_form_price(
                        option_type=option_type,
                        barrier_type="up",
                        S0=float(spot),
                        K=float(strike),
                        barrier=float(barrier),
                        T=float(maturity),
                        r=r,
                        dividend=dividend,
                        sigma=sigma,
                    )
                except ValueError:
                    price = 0.0
                matrix[i, j] = price
        return matrix


    def _compute_up_and_in_strike_heatmap(
        option_type: str,
        barrier: float,
        strike_values: np.ndarray,
        maturity_values: np.ndarray,
        spot: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> np.ndarray:
        matrix = np.zeros((len(maturity_values), len(strike_values)))
        for i, maturity in enumerate(maturity_values):
            for j, strike in enumerate(strike_values):
                if strike <= 0.0:
                    matrix[i, j] = 0.0
                    continue
                vanilla = _vanilla_price_with_dividend(option_type, spot, float(strike), float(maturity), r, dividend, sigma)
                try:
                    barrier_out = _barrier_closed_form_price(
                        option_type=option_type,
                        barrier_type="up",
                        S0=float(spot),
                        K=float(strike),
                        barrier=float(barrier),
                        T=float(maturity),
                        r=r,
                        dividend=dividend,
                        sigma=sigma,
                    )
                except ValueError:
                    matrix[i, j] = 0.0
                    continue
                matrix[i, j] = max(vanilla - barrier_out, 0.0)
        return matrix


    def _compute_lookback_exact_heatmap(
        s_values: np.ndarray,
        t_values: np.ndarray,
        t_current: float,
        rate: float,
        sigma: float,
    ) -> np.ndarray:
        matrix = np.zeros((len(t_values), len(s_values)))
        for i, maturity in enumerate(t_values):
            for j, spot in enumerate(s_values):
                lookback_opt = lookback_call_option(
                    T=float(maturity), t=float(t_current), S0=float(spot), r=float(rate), sigma=float(sigma)
                )
                matrix[i, j] = lookback_opt.price_exact()
        return matrix


    def _compute_lookback_mc_heatmap(
        s_values: np.ndarray,
        t_values: np.ndarray,
        t_current: float,
        rate: float,
        sigma: float,
        n_iters: int,
    ) -> np.ndarray:
        matrix = np.zeros((len(t_values), len(s_values)))
        for i, maturity in enumerate(t_values):
            for j, spot in enumerate(s_values):
                lookback_opt = lookback_call_option(
                    T=float(maturity), t=float(t_current), S0=float(spot), r=float(rate), sigma=float(sigma)
                )
                matrix[i, j] = lookback_opt.price_monte_carlo(n_iters)
        return matrix


    def _compute_down_in_heatmap(
        option_type: str,
        strike_values: np.ndarray,
        offset_values: np.ndarray,
        S0: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix_out, ratio_axis = _compute_barrier_heatmap_matrix(
            option_type=option_type,
            barrier_type="down",
            strike_values=strike_values,
            offset_values=offset_values,
            S0=S0,
            T=T,
            r=r,
            dividend=dividend,
            sigma=sigma,
        )
        matrix_in = np.zeros_like(matrix_out)
        for i, strike in enumerate(strike_values):
            vanilla = _vanilla_price_with_dividend(
                option_type=option_type, S0=S0, K=float(strike), T=T, r=r, dividend=dividend, sigma=sigma
            )
            matrix_in[i, :] = np.maximum(vanilla - matrix_out[i, :], 0.0)
        return matrix_in, ratio_axis


    def _compute_american_ls_heatmaps(
        s_values: np.ndarray,
        k_values: np.ndarray,
        maturity: float,
        process,
        n_paths: int,
        n_steps: int,
        v0=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        call_matrix = np.zeros((len(k_values), len(s_values)))
        put_matrix = np.zeros_like(call_matrix)
        for i, strike in enumerate(k_values):
            for j, spot in enumerate(s_values):
                option_call = Option(s0=spot, T=maturity, K=strike, v0=v0, call=True)
                option_put = Option(s0=spot, T=maturity, K=strike, v0=v0, call=False)
                call_matrix[i, j] = longstaff_schwartz_price(option_call, process, n_paths, n_steps)
                put_matrix[i, j] = longstaff_schwartz_price(option_put, process, n_paths, n_steps)
        return call_matrix, put_matrix


    def _compute_american_crr_heatmaps(
        s_values: np.ndarray,
        k_values: np.ndarray,
        maturity: float,
        rate: float,
        sigma: float,
        n_tree: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        call_matrix = np.zeros((len(k_values), len(s_values)))
        put_matrix = np.zeros_like(call_matrix)
        for i, strike in enumerate(k_values):
            for j, spot in enumerate(s_values):
                option_call = Option(s0=spot, T=maturity, K=strike, call=True)
                option_put = Option(s0=spot, T=maturity, K=strike, call=False)
                call_matrix[i, j] = crr_pricing(r=rate, sigma=sigma, option=option_call, n=n_tree)
                put_matrix[i, j] = crr_pricing(r=rate, sigma=sigma, option=option_put, n=n_tree)
        return call_matrix, put_matrix


    def _build_crr_tree(option: Option, r: float, sigma: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        if n_steps <= 0:
            raise ValueError("n_steps doit être supérieur à 0.")
        dt = option.T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        a = np.exp(r * dt)
        p = (a - d) / (u - d)
        q = 1 - p

        spot_tree = np.full((n_steps + 1, n_steps + 1), np.nan)
        value_tree = np.full_like(spot_tree, np.nan)

        for level in range(n_steps + 1):
            for up_moves in range(level + 1):
                spot_tree[level, up_moves] = option.s0 * (u**up_moves) * (d ** (level - up_moves))

        payoff_last = option.payoff(spot_tree[n_steps, : n_steps + 1])
        value_tree[n_steps, : n_steps + 1] = payoff_last
        discount = np.exp(-r * dt)

        for level in range(n_steps - 1, -1, -1):
            for up_moves in range(level + 1):
                continuation = discount * (
                    p * value_tree[level + 1, up_moves + 1] + q * value_tree[level + 1, up_moves]
                )
                exercise = option.payoff(np.array([spot_tree[level, up_moves]]))[0]
                value_tree[level, up_moves] = max(exercise, continuation)

        return spot_tree, value_tree


    def _format_tree_matrix(matrix: np.ndarray, precision: int = 4) -> np.ndarray:
        fmt = f"{{:.{precision}f}}"
        formatted = []
        for row in matrix:
            formatted.append([fmt.format(value) if not np.isnan(value) else "" for value in row])
        return np.array(formatted)


    def _plot_crr_tree(spots: np.ndarray, values: np.ndarray) -> plt.Figure:
        n_levels = spots.shape[0]
        fig_width = min(12, 4 + n_levels * 0.25)
        fig_height = min(10, 3 + n_levels * 0.25)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_axis_off()

        def _node_coords(level: int, index: int) -> tuple[float, float]:
            x = index - level / 2
            y = n_levels - 1 - level
            return x, y

        for level in range(n_levels - 1):
            for index in range(level + 1):
                if np.isnan(spots[level, index]):
                    continue
                x_curr, y_curr = _node_coords(level, index)
                x_down, y_down = _node_coords(level + 1, index)
                x_up, y_up = _node_coords(level + 1, index + 1)
                ax.plot([x_curr, x_down], [y_curr, y_down], color="lightgray", linewidth=0.8)
                ax.plot([x_curr, x_up], [y_curr, y_up], color="lightgray", linewidth=0.8)

        x_coords = []
        y_coords = []
        color_values = []
        spots_list = []
        option_list = []

        for level in range(n_levels):
            for index in range(level + 1):
                value = spots[level, index]
                option_value = values[level, index]
                if np.isnan(value) or np.isnan(option_value):
                    continue
                x, y = _node_coords(level, index)
                x_coords.append(x)
                y_coords.append(y)
                color_values.append(option_value)
                spots_list.append(value)
                option_list.append(option_value)

        scatter = ax.scatter(
            x_coords,
            y_coords,
            c=color_values,
            cmap="viridis",
            s=120,
            edgecolors="black",
            linewidths=0.5,
        )
        display_labels = n_levels - 1 <= 10
        if display_labels:
            for x, y, spot_value, option_value in zip(x_coords, y_coords, spots_list, option_list):
                ax.text(x, y + 0.25, f"S={spot_value:.2f}", ha="center", va="bottom", fontsize=7)
                ax.text(x, y - 0.25, f"V={option_value:.2f}", ha="center", va="top", fontsize=7)

        ax.set_ylim(-0.5, n_levels - 0.5)
        ax.set_xlim(min(x_coords, default=-1) - 1, max(x_coords, default=1) + 1)
        ax.set_title("Arbre CRR (couleur = valeur de l'option)")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Valeur de l'option")
        fig.tight_layout()
        return fig


    # ---------------------------------------------------------------------------
    #  Modules Basket & Asian – helpers
    # ---------------------------------------------------------------------------


    @st.cache_data(show_spinner=False)
    def get_option_expiries(ticker: str):
        tk = yf.Ticker(ticker)
        return tk.options or []


    @st.cache_data(show_spinner=False)
    def get_option_surface_from_yf(ticker: str, expiry: str):
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)

        frames = []
        for frame in [chain.calls, chain.puts]:
            tmp = frame[["strike", "impliedVolatility"]].rename(columns={"strike": "K", "impliedVolatility": "iv"})
            tmp["T"] = 0.0
            frames.append(tmp)
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["K", "iv"])
        return df


    @st.cache_data(show_spinner=False)
    def get_spot_and_hist_vol(ticker: str, period: str = "6mo", interval: str = "1d"):
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError("Aucune donnée téléchargée.")
        close = data["Close"]
        spot = float(close.iloc[-1])
        log_returns = np.log(close / close.shift(1)).dropna()
        sigma = float(log_returns.std() * np.sqrt(252))
        hist_df = data.reset_index()
        hist_df["Date"] = pd.to_datetime(hist_df["Date"])
        return spot, sigma, hist_df


    def fetch_closing_prices(tickers, period="1mo", interval="1d"):
        if isinstance(tickers, str):
            tickers = [tickers]
        for var in ["YF_IMPERSONATE", "YF_SCRAPER_IMPERSONATE"]:
            try:
                os.environ.pop(var, None)
            except Exception:
                pass
        try:
            yf.set_config(proxy=None)
        except Exception:
            pass

        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            raise RuntimeError(f"Aucune donnée récupérée pour {tickers} sur {period}.")

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
        else:
            if "Adj Close" in data.columns:
                prices = data[["Adj Close"]].copy()
            elif "Close" in data.columns:
                prices = data[["Close"]].copy()
            else:
                raise RuntimeError("Colonnes de prix introuvables dans les données yfinance.")
            prices.columns = tickers

        prices = prices.reset_index()
        return prices


    def compute_corr_from_prices(prices_df: pd.DataFrame):
        price_cols = [c for c in prices_df.columns if c.lower() != "date"]
        returns = np.log(prices_df[price_cols] / prices_df[price_cols].shift(1)).dropna(how="any")
        if returns.empty:
            raise RuntimeError("Pas assez de données pour calculer la corrélation.")
        return returns.corr()


    def load_closing_prices_with_tickers(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
        if not path.exists():
            return None, []
        try:
            df = pd.read_csv(path)
        except Exception:
            return None, []
        ticker_cols: list[str] = []
        for col in df.columns:
            col_str = str(col).strip()
            if not col_str or col_str.lower() == "date":
                continue
            ticker_cols.append(col_str)
        return df, ticker_cols

    def load_close_series_for_ticker(ticker: str, fallback_value: float | None = None) -> pd.Series:
        """
        Charge une série de clôtures pour un ticker depuis cache_CSV/closing_cache.csv.
        Si absent ou invalide, télécharge via yfinance et met à jour le cache.
        """
        ticker_norm = (ticker or "SPY").strip().upper()
        # 1) Cache local
        try:
            if CLOSING_CACHE_FILE.exists():
                df_cache = pd.read_csv(CLOSING_CACHE_FILE, parse_dates=True)
                date_col = next((c for c in df_cache.columns if str(c).lower() in {"date", "datetime", "index"}), None)
                price_cols = [c for c in df_cache.columns if str(c).lower() not in {"date", "datetime", "index"}]
                price_col = ticker_norm if ticker_norm in df_cache.columns else (price_cols[0] if len(price_cols) == 1 else None)
                if price_col:
                    dates = pd.to_datetime(df_cache[date_col]) if date_col else pd.RangeIndex(len(df_cache))
                    series = pd.Series(df_cache[price_col].values, index=dates, name="Close")
                    if not series.empty:
                        return series
        except Exception:
            pass
        # 2) Téléchargement (un seul ticker) et mise à jour cache
        try:
            df_dl = fetch_closing_prices([ticker_norm], period="1y", interval="1d")
            if df_dl is not None and not df_dl.empty and ticker_norm in df_dl.columns:
                CLOSING_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                df_dl.to_csv(CLOSING_CACHE_FILE, index=False)
                dates = pd.to_datetime(df_dl.iloc[:, 0])
                return pd.Series(df_dl[ticker_norm].values, index=dates, name="Close")
        except Exception:
            pass
        # 3) Fallback simple
        if fallback_value is not None:
            return pd.Series([fallback_value], index=pd.Index([datetime.date.today()]), name="Close")
        return pd.Series(dtype=float)


    class BasketOption:
        def __init__(self, weights, prices, volatility, corr, strike, maturity, rate):
            self.weights = weights
            self.vol = volatility
            self.strike = strike
            self.mat = maturity
            self.rate = rate
            self.corr = corr
            self.prices = prices

        def get_mc(self, m_paths: int = 10000):
            b_ts = stats.multivariate_normal(np.zeros(len(self.weights)), cov=self.corr).rvs(size=m_paths)
            s_ts = self.prices * np.exp((self.rate - 0.5 * self.vol**2) * self.mat + self.vol * b_ts)
            if len(self.weights) > 1:
                payoffs = (np.sum(self.weights * s_ts, axis=1) - self.strike).clip(0)
            else:
                payoffs = np.maximum(s_ts - self.strike, np.zeros(m_paths))
            return float(np.exp(-self.rate * self.mat) * np.mean(payoffs))

        def get_bs_price(self):
            d1 = (np.log(self.prices / self.strike) + (self.rate + 0.5 * self.vol**2) * self.mat) / (
                self.vol * np.sqrt(self.mat)
            )
            d2 = d1 - self.vol * np.sqrt(self.mat)
            bs_price = stats.norm.cdf(d1) * self.prices - stats.norm.cdf(d2) * self.strike * np.exp(-self.rate * self.mat)
            return float(bs_price)


    class DataGen:
        def __init__(self, n_assets: int, n_samples: int):
            if n_samples <= 0:
                raise ValueError("n_samples needs to be positive")
            if n_assets <= 0:
                raise ValueError("n_assets needs to be positive")
            self.n_assets = n_assets
            self.n_samples = n_samples

        def generate(self, corr, strike_price: float, base_price: float, method: str = "bs"):
            mats = np.random.uniform(0.2, 1.1, size=self.n_samples)
            vols = np.random.uniform(0.01, 1.0, size=self.n_samples)
            rates = np.random.uniform(0.02, 0.1, size=self.n_samples)

            strikes = np.random.randn(self.n_samples) + strike_price
            prices = np.random.randn(self.n_samples) + base_price

            if self.n_assets > 1:
                weights = np.random.rand(self.n_samples * self.n_assets).reshape((self.n_samples, self.n_assets))
                weights /= np.sum(weights, axis=1)[:, np.newaxis]
            else:
                weights = np.ones((self.n_samples, self.n_assets))

            labels = []
            for i in range(self.n_samples):
                basket = BasketOption(
                    weights[i],
                    prices[i],
                    vols[i],
                    corr,
                    strikes[i],
                    mats[i],
                    rates[i],
                )
                if method == "bs":
                    labels.append(basket.get_bs_price())
                else:
                    labels.append(basket.get_mc())

            data = pd.DataFrame(
                {
                    "S/K": prices / strikes,
                    "Maturity": mats,
                    "Volatility": vols,
                    "Rate": rates,
                    "Labels": labels,
                    "Prices": prices,
                    "Strikes": strikes,
                }
            )
            for i in range(self.n_assets):
                data[f"Weight_{i}"] = weights[:, i]
            return data


    def simulate_dataset_notebook(n_assets: int, n_samples: int, method: str, corr: np.ndarray, base_price: float, base_strike: float):
        generator = DataGen(n_assets=n_assets, n_samples=n_samples)
        return generator.generate(corr=corr, strike_price=base_strike, base_price=base_price, method=method)


    @st.cache_data(show_spinner=False)
    def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(file_bytes))


    def split_data_nn(data: pd.DataFrame, split_ratio: float = 0.7):
        feature_cols = ["S/K", "Maturity", "Volatility", "Rate"]
        target_col = "Labels"
        train = data.iloc[: int(split_ratio * len(data)), :]
        test = data.iloc[int(split_ratio * len(data)) :, :]
        x_train, y_train = train[feature_cols], train[target_col]
        x_test, y_test = test[feature_cols], test[target_col]
        return x_train, y_train, x_test, y_test


    def build_model_nn(input_dim: int) -> tf.keras.Model:
        inp = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(32, activation="relu")(inp)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        out = tf.keras.layers.Dense(1, activation="relu")(x)
        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=["mean_squared_error"],
        )
        return model


    def price_basket_nn(model: tf.keras.Model, S: float, K: float, maturity: float, volatility: float, rate: float) -> float:
        S_over_K = S / K
        x = np.array([[S_over_K, maturity, volatility, rate]], dtype=float)
        return float(model.predict(x, verbose=0)[0, 0])


    def plot_heatmap_nn(
        model: tf.keras.Model,
        data: pd.DataFrame,
        spot_ref: float | None = None,
        strike_ref: float | None = None,
        maturity_fixed: float = 1.0,
    ):
        df = data.copy()
        if "Prices" not in df.columns and spot_ref is not None:
            df["Prices"] = spot_ref
        if "Strikes" not in df.columns and strike_ref is not None:
            df["Strikes"] = strike_ref

        if not {"Prices", "Strikes"}.issubset(df.columns):
            raise ValueError("Colonnes Prices et Strikes requises pour reproduire la heatmap du notebook.")

        s_min, s_max = df["Prices"].quantile([0.01, 0.99])
        k_min, k_max = df["Strikes"].quantile([0.01, 0.99])
        n_S, n_K = 50, 50
        s_vals = np.linspace(s_min, s_max, n_S)
        k_vals = np.linspace(k_min, k_max, n_K)

        K_grid, S_grid = np.meshgrid(k_vals, s_vals)
        s_over_k_grid = S_grid / K_grid

        sigma_ref = float(df["Volatility"].median())
        rate_ref = float(df["Rate"].median())

        X = np.stack(
            [
                s_over_k_grid.ravel(),
                np.full(s_over_k_grid.size, maturity_fixed),
                np.full(s_over_k_grid.size, sigma_ref),
                np.full(s_over_k_grid.size, rate_ref),
            ],
            axis=1,
        )
        prices_grid = model.predict(X, verbose=0).reshape(n_S, n_K)

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            prices_grid,
            origin="lower",
            extent=[k_vals.min(), k_vals.max(), s_vals.min(), s_vals.max()],
            aspect="auto",
            cmap="viridis",
        )
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Spot S")
        ax.set_title("Heatmap du prix NN en fonction de S et K (T=1 an)")
        fig.colorbar(im, ax=ax, label="Prix NN")
        plt.tight_layout()
        return fig


    def build_grid(
        df: pd.DataFrame,
        spot: float,
        n_k: int = 200,
        n_t: int = 200,
        k_min: float | None = None,
        k_max: float | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
        k_span: float | None = None,
        margin_frac: float = 0.02,
    ):
        if k_min is None or k_max is None:
            if k_span is not None:
                k_min = spot - k_span
                k_max = spot + k_span
            else:
                data_k_min = float(df["K"].min())
                data_k_max = float(df["K"].max())
                delta_k = data_k_max - data_k_min
                pad = delta_k * margin_frac
                k_min = data_k_min - pad
                k_max = data_k_max + pad

        if t_min is None:
            t_min = float(df["T"].min())
        if t_max is None:
            t_max = float(df["T"].max())

        if k_min >= k_max:
            raise ValueError("k_min doit être inférieur à k_max.")
        if t_min >= t_max:
            raise ValueError("t_min doit être inférieur à t_max.")

        k_vals = np.linspace(k_min, k_max, n_k)
        t_vals = np.linspace(t_min, t_max, n_t)

        df = df[(df["K"] >= k_min) & (df["K"] <= k_max)].copy()
        df = df[(df["T"] >= t_min) & (df["T"] <= t_max)]

        if df.empty:
            raise ValueError("Aucun point n'appartient au domaine défini par la grille.")

        df["K_idx"] = np.searchsorted(k_vals, df["K"], side="left").clip(0, n_k - 1)
        df["T_idx"] = np.searchsorted(t_vals, df["T"], side="left").clip(0, n_t - 1)

        grouped = df.groupby(["T_idx", "K_idx"])["iv"].mean().reset_index()

        iv_grid = np.full((n_t, n_k), np.nan, dtype=float)
        for _, row in grouped.iterrows():
            iv_grid[int(row["T_idx"]), int(row["K_idx"])] = row["iv"]

        k_grid, t_grid = np.meshgrid(k_vals, t_vals)
        return k_grid, t_grid, iv_grid


    def make_iv_surface_figure(k_grid, t_grid, iv_grid, title_suffix=""):
        fig = plt.figure(figsize=(12, 5))

        ax3d = fig.add_subplot(1, 2, 1, projection="3d")

        iv_flat = iv_grid[~np.isnan(iv_grid)]
        if iv_flat.size == 0:
            raise ValueError("La grille iv_grid ne contient aucune valeur non-NaN.")
        iv_mean = iv_flat.mean()
        iv_grid_filled = np.where(np.isnan(iv_grid), iv_mean, iv_grid)

        surf = ax3d.plot_surface(
            k_grid,
            t_grid,
            iv_grid_filled,
            rstride=1,
            cstride=1,
            linewidth=0.2,
            antialiased=True,
            cmap="viridis",
        )

        ax3d.set_xlabel("Strike K")
        ax3d.set_ylabel("Maturité T (années)")
        ax3d.set_zlabel("Implied vol")
        ax3d.set_title(f"Surface 3D de volatilité implicite{title_suffix}")

        fig.colorbar(surf, shrink=0.5, aspect=10, ax=ax3d, label="iv")

        ax2d = fig.add_subplot(1, 2, 2)
        im = ax2d.imshow(
            iv_grid_filled,
            extent=[k_grid.min(), k_grid.max(), t_grid.min(), t_grid.max()],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        ax2d.set_xlabel("Strike K")
        ax2d.set_ylabel("Maturité T (années)")
        ax2d.set_title(f"Heatmap IV{title_suffix}")
        fig.colorbar(im, ax=ax2d, label="iv")

        plt.tight_layout()
        return fig


    def btm_asian(strike_type, option_type, spot, strike, rate, sigma, maturity, steps):
        delta_t = maturity / steps
        up = np.exp(sigma * np.sqrt(delta_t))
        down = 1.0 / up
        prob = (np.exp(rate * delta_t) - down) / (up - down)

        spot_paths = [spot]
        avg_paths = [spot]
        strike_paths = [strike]

        for _ in range(steps):
            spot_paths = [s * up for s in spot_paths] + [s * down for s in spot_paths]
            avg_paths = avg_paths + avg_paths
            strike_paths = strike_paths + strike_paths
            for index in range(len(avg_paths)):
                avg_paths[index] = avg_paths[index] + spot_paths[index]

        avg_paths = np.array(avg_paths) / (steps + 1)
        spot_paths = np.array(spot_paths)
        strike_paths = np.array(strike_paths)

        if strike_type == "fixed":
            if option_type == "C":
                payoff = np.maximum(avg_paths - strike_paths, 0.0)
            else:
                payoff = np.maximum(strike_paths - avg_paths, 0.0)
        else:
            if option_type == "C":
                payoff = np.maximum(spot_paths - avg_paths, 0.0)
            else:
                payoff = np.maximum(avg_paths - spot_paths, 0.0)

        option_price = payoff.copy()
        for _ in range(steps):
            length = len(option_price) // 2
            option_price = prob * option_price[:length] + (1 - prob) * option_price[length:]

        return float(option_price[0])


    def hw_btm_asian(strike_type, option_type, spot, strike, rate, sigma, maturity, steps, m_points):
        n_steps = steps
        delta_t = maturity / n_steps
        up = np.exp(sigma * np.sqrt(delta_t))
        down = 1.0 / up
        prob = (np.exp(rate * delta_t) - down) / (up - down)

        avg_grid = []
        strike_vec = np.array([strike] * m_points)

        for j_index in range(n_steps + 1):
            path_up_then_down = np.array(
                [spot * up**j * down**0 for j in range(n_steps - j_index)]
                + [spot * up**(n_steps - j_index) * down**j for j in range(j_index + 1)]
            )
            avg_max = path_up_then_down.mean()

            path_down_then_up = np.array(
                [spot * down**j * up**0 for j in range(j_index + 1)]
                + [spot * down**j_index * up**(j + 1) for j in range(n_steps - j_index)]
            )
            avg_min = path_down_then_up.mean()

            diff = avg_max - avg_min
            avg_vals = [avg_max - diff * k_index / (m_points - 1) for k_index in range(m_points)]
            avg_grid.append(avg_vals)

        avg_grid = np.round(avg_grid, 4)

        payoff = []
        for j_index in range(n_steps + 1):
            avg_vals = np.array(avg_grid[j_index])
            spot_vals = np.array([spot * up**(n_steps - j_index) * down**j_index] * m_points)

            if strike_type == "fixed":
                if option_type == "C":
                    pay = np.maximum(avg_vals - strike_vec, 0.0)
                else:
                    pay = np.maximum(strike_vec - avg_vals, 0.0)
            else:
                if option_type == "C":
                    pay = np.maximum(spot_vals - avg_vals, 0.0)
                else:
                    pay = np.maximum(avg_vals - spot_vals, 0.0)

            payoff.append(pay)

        payoff = np.round(np.array(payoff), 4)

        for n_index in range(n_steps - 1, -1, -1):
            avg_backward = []
            payoff_backward = []

            for j_index in range(n_index + 1):
                path_up_then_down = np.array(
                    [spot * up**j * down**0 for j in range(n_index - j_index)]
                    + [spot * up**(n_index - j_index) * down**j for j in range(j_index + 1)]
                )
                avg_max = path_up_then_down.mean()

                path_down_then_up = np.array(
                    [spot * down**j * up**0 for j in range(j_index + 1)]
                    + [spot * down**j_index * up**(j + 1) for j in range(n_index - j_index)]
                )
                avg_min = path_down_then_up.mean()

                diff = avg_max - avg_min
                avg_vals = np.array([avg_max - diff * k_index / (m_points - 1) for k_index in range(m_points)])
                avg_backward.append(avg_vals)

            avg_backward = np.round(np.array(avg_backward), 4)

            payoff_new = []
            for j_index in range(n_index + 1):
                avg_vals = avg_backward[j_index]
                pay_vals = np.zeros_like(avg_vals)

                avg_up = np.array(avg_grid[j_index])
                avg_down = np.array(avg_grid[j_index + 1])
                pay_up = payoff[j_index]
                pay_down = payoff[j_index + 1]

                for k_index, avg_k in enumerate(avg_vals):
                    if avg_k <= avg_up[0]:
                        fu = pay_up[0]
                    elif avg_k >= avg_up[-1]:
                        fu = pay_up[-1]
                    else:
                        idx = np.searchsorted(avg_up, avg_k) - 1
                        x0, x1 = avg_up[idx], avg_up[idx + 1]
                        y0, y1 = pay_up[idx], pay_up[idx + 1]
                        fu = y0 + (y1 - y0) * (avg_k - x0) / (x1 - x0)

                    if avg_k <= avg_down[0]:
                        fd = pay_down[0]
                    elif avg_k >= avg_down[-1]:
                        fd = pay_down[-1]
                    else:
                        idx = np.searchsorted(avg_down, avg_k) - 1
                        x0, x1 = avg_down[idx], avg_down[idx + 1]
                        y0, y1 = pay_down[idx], pay_down[idx + 1]
                        fd = y0 + (y1 - y0) * (avg_k - x0) / (x1 - x0)

                    pay_vals[k_index] = (prob * fu + (1 - prob) * fd) * np.exp(-rate * delta_t)

                payoff_backward.append(pay_vals)

            avg_grid = avg_backward
            payoff = np.round(np.array(payoff_backward), 4)

        option_price = payoff[0].mean()
        return float(option_price)


    def bs_option_price(time, spot, strike, maturity, rate, sigma, option_kind):
        tau = maturity - time
        if tau <= 0:
            if option_kind == "call":
                return max(spot - strike, 0.0)
            return max(strike - spot, 0.0)

        d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)

        if option_kind == "call":
            price = spot * norm.cdf(d1) - strike * np.exp(-rate * tau) * norm.cdf(d2)
        else:
            price = strike * np.exp(-rate * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        return float(price)


    def asian_geometric_closed_form(spot, strike, rate, sigma, maturity, n_obs, option_type):
        if n_obs < 1:
            return 0.0
        dt = maturity / n_obs
        nu = rate - 0.5 * sigma**2
        sigma_g_sq = (sigma**2) * (n_obs + 1) * (2 * n_obs + 1) / (6 * n_obs**2)
        sigma_g = np.sqrt(sigma_g_sq)
        mu_g = (nu * (n_obs + 1) / (2 * n_obs) + 0.5 * sigma_g_sq) * maturity
        d1 = (np.log(spot / strike) + mu_g + 0.5 * sigma_g_sq * maturity) / (sigma_g * np.sqrt(maturity))
        d2 = d1 - sigma_g * np.sqrt(maturity)
        df = np.exp(-rate * maturity)
        if option_type == "call":
            return float(df * (spot * np.exp(mu_g) * norm.cdf(d1) - strike * norm.cdf(d2)))
        else:
            return float(df * (strike * norm.cdf(-d2) - spot * np.exp(mu_g) * norm.cdf(-d1)))


    def asian_mc_control_variate(
        spot,
        strike,
        rate,
        sigma,
        maturity,
        n_obs,
        n_paths,
        option_type,
        antithetic=True,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)
        dt = maturity / n_obs
        drift = (rate - 0.5 * sigma**2) * dt
        vol_step = sigma * np.sqrt(dt)

        if antithetic:
            n_base = max(1, n_paths // 2)
            z_base = np.random.randn(n_obs, n_base)
            z = np.concatenate([z_base, -z_base], axis=1)
            n_eff = z.shape[1]
        else:
            z = np.random.randn(n_obs, n_paths)
            n_eff = n_paths

        log_s = np.log(spot) + np.cumsum(drift + vol_step * z, axis=0)
        s_paths = np.exp(log_s)

        arith_mean = s_paths.mean(axis=0)
        geom_mean = np.exp(np.log(s_paths).mean(axis=0))
        if option_type == "call":
            arith_payoff = np.maximum(arith_mean - strike, 0.0)
            geom_payoff = np.maximum(geom_mean - strike, 0.0)
        else:
            arith_payoff = np.maximum(strike - arith_mean, 0.0)
            geom_payoff = np.maximum(strike - geom_mean, 0.0)
        closed_geom = asian_geometric_closed_form(spot, strike, rate, sigma, maturity, n_obs, option_type)
        cov = np.cov(arith_payoff, geom_payoff)[0, 1]
        var_geom = np.var(geom_payoff)
        c = cov / var_geom if var_geom > 0 else 0.0
        control_estimator = arith_payoff - c * (geom_payoff - closed_geom)
        disc = np.exp(-rate * maturity)
        disc_payoff = disc * control_estimator
        price = np.mean(disc_payoff)
        stderr = np.std(disc_payoff, ddof=1) / np.sqrt(n_eff)
        return float(price), float(stderr), float(c)


    def compute_asian_price(
        strike_type: str,
        option_type: str,
        model: str,
        spot: float,
        strike: float,
        rate: float,
        sigma: float,
        maturity: float,
        steps: int,
        m_points: int | None,
    ):
        if model == "BTM naïf":
            return btm_asian(
                strike_type=strike_type,
                option_type=option_type,
                spot=spot,
                strike=strike,
                rate=rate,
                sigma=sigma,
                maturity=maturity,
                steps=int(steps),
            )
        m_points_val = int(m_points) if m_points is not None else 10
        return hw_btm_asian(
            strike_type=strike_type,
            option_type=option_type,
            spot=spot,
            strike=strike,
            rate=rate,
            sigma=sigma,
            maturity=maturity,
            steps=int(steps),
            m_points=m_points_val,
        )


    def ui_basket_surface(spot_common, maturity_common, rate_common, strike_common, key_prefix: str = "basket"):
        st.header("Basket – Pricing NN + corrélation (3 actifs)")
        render_unlock_sidebar_button("tab_basket", "🔓 Réactiver T (onglet Basket)")

        min_assets, max_assets = 2, 10
        closing_path = BASKET_CLOSING_FILE
        prices_df_cached, csv_tickers = load_closing_prices_with_tickers(closing_path)

        def _normalize_tickers(candidates: list[str]) -> list[str]:
            cleaned = [str(tk).strip().upper() for tk in candidates if str(tk).strip()]
            trimmed = cleaned[:max_assets]
            if len(trimmed) < min_assets:
                trimmed += ["SPY"] * (min_assets - len(trimmed))
            return trimmed

        def _k(suffix: str) -> str:
            return f"{key_prefix}_{suffix}"

        if "basket_tickers" not in st.session_state:
            default_list = csv_tickers if csv_tickers else ["AAPL", "SPY", "MSFT"]
            st.session_state["basket_tickers"] = _normalize_tickers(default_list)

        with st.container():
            st.subheader("Sélection des assets (2 à 10)")
            btn_col_add, btn_col_remove = st.columns(2)
            with btn_col_add:
                if st.button(
                    "Ajouter un asset",
                    key=_k("btn_add_asset"),
                    disabled=len(st.session_state["basket_tickers"]) >= max_assets,
                ):
                    st.session_state["basket_tickers"].append(
                        f"TICKER{len(st.session_state['basket_tickers']) + 1}"
                    )
            with btn_col_remove:
                removable = list(st.session_state["basket_tickers"])
                remove_choice = st.selectbox(
                    "Sélectionner un asset à retirer",
                    removable if len(removable) > min_assets else ["(aucun)"],
                    index=0 if len(removable) > min_assets else 0,
                    key=_k("select_remove_asset"),
                )
                if st.button(
                    "Retirer l'asset sélectionné",
                    key=_k("btn_remove_asset"),
                    disabled=len(st.session_state["basket_tickers"]) <= min_assets or remove_choice == "(aucun)",
                ):
                    try:
                        st.session_state["basket_tickers"].remove(remove_choice)
                    except ValueError:
                        pass

            tickers = []
            for i, default_tk in enumerate(st.session_state["basket_tickers"]):
                if i % 3 == 0:
                    cols = st.columns(3)
                col = cols[i % 3]
                with col:
                    tick = st.text_input(f"Ticker {i + 1}", value=default_tk, key=_k(f"corr_tk_dynamic_{i}"))
                    tickers.append(tick.strip().upper() or default_tk)
            tickers = tickers[:max_assets]
            if len(tickers) < min_assets:
                tickers += ["SPY"] * (min_assets - len(tickers))
            st.session_state["basket_tickers"] = tickers
        tickers = st.session_state["basket_tickers"]

        period = st.selectbox("Période yfinance", ["1mo", "3mo", "6mo", "1y"], index=0, key=_k("corr_period"))
        interval = st.selectbox("Intervalle", ["1d", "1h"], index=0, key=_k("corr_interval"))

        st.caption(
            "Le calcul de corrélation utilise les prix de clôture présents dans database/GPTab/cache_CSV/basket_closing_prices.csv "
            "(régénéré via yfinance). En cas d'échec, une matrice de corrélation inventée sera utilisée."
        )
        regen_csv = st.button("Mettre à jour la Matrice de Corrélation", key=_k("btn_regen_closing"))
        try:
            if regen_csv or not closing_path.exists():
                prices_df_cached = fetch_closing_prices(tickers, period=period, interval=interval)
                closing_path.parent.mkdir(parents=True, exist_ok=True)
                prices_df_cached.to_csv(closing_path, index=False)
                csv_tickers = [c for c in prices_df_cached.columns if str(c).lower() != "date"]
                st.info(f"database/GPTab/cache_CSV/basket_closing_prices.csv généré via yfinance ({len(prices_df_cached)} lignes)")
                if csv_tickers:
                    st.session_state["basket_tickers"] = _normalize_tickers(csv_tickers)
                    tickers = st.session_state["basket_tickers"]
        except Exception as exc:
            st.warning(f"Impossible de récupérer les prix de clôture : {exc}")

        corr_df = None
        try:
            if prices_df_cached is None:
                prices_df_cached, _ = load_closing_prices_with_tickers(closing_path)
            if prices_df_cached is None:
                raise FileNotFoundError("Impossible de charger database/GPTab/cache_CSV/basket_closing_prices.csv.")
            corr_df = compute_corr_from_prices(prices_df_cached)
            st.success(f"Corrélation calculée à partir de {closing_path.name}")
            st.dataframe(corr_df)
        except Exception as exc:
            st.warning(f"Impossible de calculer la corrélation depuis database/GPTab/cache_CSV/basket_closing_prices.csv : {exc}")
            corr_df = pd.DataFrame(
                [
                    [1.0, 0.6, 0.4],
                    [0.6, 1.0, 0.7],
                    [0.4, 0.7, 1.0],
                ],
                columns=tickers,
                index=tickers,
            )
            st.info("Utilisation d'une matrice de corrélation inventée pour la suite des calculs.")
            st.dataframe(corr_df)

        st.subheader("Dataset Basket pour NN")
        st.caption("Dataset généré automatiquement via DataGen (comme dans le notebook).")
        n_samples = st.slider("Taille du dataset simulé", 1000, 20000, 10000, 1000, key=_k("basket_n_samples"))
        method = st.selectbox("Méthode de pricing pour les labels", ["bs", "mc"], index=0, key=_k("basket_method"))

        df = simulate_dataset_notebook(
            n_assets=len(tickers),
            n_samples=int(n_samples),
            method=method,
            corr=corr_df.values,
            base_price=float(spot_common),
            base_strike=float(strike_common),
        )

        st.write("Aperçu :", df.head())
        st.write("Shape :", df.shape)

        split_ratio = st.slider("Train ratio", 0.5, 0.9, 0.7, 0.05, key=_k("basket_split_ratio"))
        epochs = st.slider("Epochs d'entraînement", 5, 200, 20, 5, key=_k("basket_epochs"))

        x_train, y_train, x_test, y_test = split_data_nn(df, split_ratio=split_ratio)
        Path("data").mkdir(parents=True, exist_ok=True)
        if st.session_state.get("carr_madan_calibrated", False):
            pd.concat([x_train, y_train], axis=1).to_csv(TRAIN_FILE, index=False)
            pd.concat([x_test, y_test], axis=1).to_csv(TEST_FILE, index=False)
            st.info("train.csv et test.csv générés dans cache_CSV (suite calibration Carr–Madan).")
        else:
            st.info("Calibre d'abord le NN Carr–Madan pour générer train.csv et test.csv.")

        st.write(f"Train size: {x_train.shape[0]} | Test size: {x_test.shape[0]}")

        train_button = st.button("Entraîner le modèle NN", key=_k("btn_train_nn"))
        if not train_button:
            st.info("Clique sur 'Entraîner le modèle NN' pour lancer l'apprentissage.")
            return

        tf.keras.backend.clear_session()
        model = build_model_nn(input_dim=x_train.shape[1])
        train_logs: list[str] = []
        log_box = st.empty()

        class StreamlitLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                msg = (
                    f"Epoch {epoch + 1}/{epochs} - loss: {logs.get('loss', float('nan')):.4f} - "
                    f"mse: {logs.get('mean_squared_error', float('nan')):.4f}"
                )
                if "val_loss" in logs or "val_mean_squared_error" in logs:
                    msg += (
                        f" - val_loss: {logs.get('val_loss', float('nan')):.4f} - "
                        f"val_mse: {logs.get('val_mean_squared_error', float('nan')):.4f}"
                    )
                train_logs.append(msg)
                log_box.text("\n".join(train_logs))

        with st.spinner("Entraînement du NN en cours…"):
            history = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                validation_data=(x_test, y_test),
                verbose=0,
                callbacks=[StreamlitLogger()],
            )
        st.success("Entraînement terminé.")

        st.subheader("Courbe MSE NN")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(history.history["mean_squared_error"], label="train")
        ax.plot(history.history["val_mean_squared_error"], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

        st.subheader("Heatmap prix NN (S vs K)")
        try:
            with st.spinner("Calcul de la heatmap…"):
                heatmap_fig = plot_heatmap_nn(
                    model=model,
                    data=df,
                    spot_ref=float(spot_common),
                    strike_ref=float(strike_common),
                    maturity_fixed=1.0,
                )
            st.pyplot(heatmap_fig)
        except Exception as exc:
            st.warning(f"Impossible d'afficher la heatmap : {exc}")

        st.subheader("Surface IV (Strike, Maturité)")
        try:
            with st.spinner("Calcul de la surface IV…"):
                iv_df = df.copy()
                if "Strikes" in iv_df.columns:
                    iv_df["K"] = iv_df["Strikes"]
                else:
                    iv_df["K"] = spot_common / iv_df["S/K"].replace(0.0, np.nan)
                iv_df = iv_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["K", "Maturity", "Volatility"])

                if iv_df.empty:
                    raise ValueError("Pas de données IV exploitables (S/K nuls ou manquants).")

                spot_ref_for_grid = float(iv_df["Prices"].mean()) if "Prices" in iv_df.columns else float(spot_common)

                grid_k, grid_t, grid_iv = build_grid(
                    df=iv_df.rename(columns={"Maturity": "T", "Volatility": "iv"}),
                    spot=spot_ref_for_grid,
                )
                iv_fig = make_iv_surface_figure(grid_k, grid_t, grid_iv, title_suffix=" (dataset NN)")
            st.pyplot(iv_fig)
        except Exception as exc:
            st.warning(f"Impossible d'afficher la surface IV : {exc}")


    def ui_asian_options(
        spot_default,
        sigma_common,
        maturity_common,
        strike_common,
        rate_common,
        key_prefix: str = "asian",
        option_char: str = "c",
    ):
        # Prefix keys to avoid clashes when the module is rendered in multiple tabs.
        def _k(suffix: str) -> str:
            return f"{key_prefix}_{suffix}"

        st.header("Options asiatiques (module Asian)")
        render_unlock_sidebar_button("tab_asian", "🔓 Réactiver T (onglet Asian)")
        render_general_definition_explainer(
            "🌏 Comprendre les options asiatiques",
            (
                "- **Spécificité du payoff** : pour une option asiatique arithmétique, le payoff dépend de la moyenne des prix du sous‑jacent observés à différentes dates entre `0` et `T`, plutôt que du seul `S_T`.\n"
                "- **Effet de lissage** : cette moyenne réduit l’impact des pics de volatilité ponctuels et donne un profil de risque plus \"lissé\" que pour une option européenne standard.\n"
                "- **Conséquences sur le prix** : à paramètres identiques, une option asiatique est généralement moins chère que son équivalent européen car elle réagit moins aux extrêmes de la trajectoire.\n"
                "- **Usage pratique** : ces produits sont fréquemment utilisés dans l’énergie, les matières premières ou les produits structurés pour lisser l’exposition à des prix très volatils.\n"
                "- **Objectif du module** : illustrer le pricing d’options asiatiques par simulation Monte Carlo, avec des variates antithétiques et un contrôle par une option de référence."
            ),
        )
        render_method_explainer(
            "🧮 Méthode Monte Carlo + control variate",
            (
                "- **Étape 1 – Paramétrage de la grille** : pour chaque couple `(K, T)` de la grille choisie, on fixe un nombre d’observations `n_obs` le long de `[0, T]` et un nombre de trajectoires Monte Carlo `n_paths_surface`.\n"
                "- **Étape 2 – Simulation des trajectoires de `S_t`** : pour un spot initial donné, on simule sous la mesure neutre au risque `n_paths_surface` trajectoires du sous‑jacent en découpant `[0, T]` en `n_obs` pas. À chaque pas, on applique le schéma d’Euler du GBM.\n"
                "- **Étape 3 – Utilisation des variates antithétiques** : pour chaque suite de chocs gaussiens utilisée pour générer une trajectoire, on génère une trajectoire \"miroir\" avec les chocs opposés. On obtient ainsi des paires de trajectoires fortement corrélées qui réduisent la variance de l’estimateur.\n"
                "- **Étape 4 – Calcul de la moyenne arithmétique** : sur chaque trajectoire, on calcule la moyenne arithmétique des `S_t` observés aux dates de la grille. Cette moyenne est ensuite utilisée pour déterminer le payoff asiatique (call ou put) à l’échéance.\n"
                "- **Étape 5 – Construction d’une variable de contrôle** : en parallèle, on calcule pour chaque trajectoire le payoff d’une option de référence (par exemple une option européenne ou une option asiatique géométrique) dont on connaît une formule de prix fermée.\n"
                "- **Étape 6 – Correction par control variate** : on corrige l’estimation brute du payoff asiatique en soustrayant la composante due à la variable de contrôle, puis en réajoutant l’espérance théorique de cette variable. Cela réduit significativement la variance de l’estimateur final.\n"
                "- **Étape 7 – Actualisation et moyenne** : les payoffs corrigés sont actualisés au taux `rate_common` jusqu’à la date présente et moyennés sur toutes les trajectoires.\n"
                "- **Étape 8 – Remplissage des surfaces** : on répète ce processus pour chaque point `(K, T)` de la grille, ce qui remplit deux matrices de prix (call et put) utilisées pour tracer les surfaces de prix asiatiques."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – module Asian",
            (
                "- **\"S0 (spot)\"** (via les paramètres communs) : niveau de départ des trajectoires asiatiques.\n"
                "- **\"K (strike)\"** : strike de référence utilisé pour centrer la plage de strikes.\n"
                "- **\"T (maturité, années)\"** : maturité de référence utilisée pour initialiser la plage de maturités.\n"
                "- **\"Taux sans risque r\"** : intervient dans l’actualisation et le drift neutre au risque.\n"
                "- **\"Volatilité σ\"** : volatilité utilisée pour simuler les trajectoires du sous‑jacent.\n"
                "- **\"K min\" / \"K max\"** : bornes de la plage de strikes sur l’axe horizontal des surfaces.\n"
                "- **\"T min (années)\" / \"T max (années)\"** : bornes de la plage de maturités sur l’axe vertical.\n"
                "- **\"Résolution en K\"** et **\"Résolution en T\"** : nombres de points de grille en strike et en maturité.\n"
                "- **\"Nombre de trajectoires Monte Carlo\"** : nombre de trajectoires utilisées pour estimer chaque point de la surface."
            ),
        )
        if spot_default is None:
            st.warning("Aucun téléchargement yfinance : utilisez le spot commun.")
            spot_default = 57830.0
        if sigma_common is None:
            sigma_common = 0.05

        col1, col2 = st.columns(2)
        with col1:
            spot_common = st.session_state.get("common_spot", spot_default)
            strike_common_local = st.session_state.get("common_strike", strike_common)
            st.info(f"Spot commun S0 = {spot_common:.4f}")
            st.info(f"Strike commun K = {strike_common_local:.4f}")
            st.info(f"Taux sans risque commun r = {rate_common:.4f}")
        with col2:
            sigma = sigma_common
            st.info(f"Volatilité commune σ = {sigma:.4f}")
            st.info("Pricing asiatique via Monte Carlo + control variate (méthode notebook).")

        with st.expander("📈 Prix asiatique arithmétique (MC + control variate)", expanded=False):
            progress = st.progress(0)
            try:
                n_obs_price = max(2, int(50 * float(maturity_common)))
                price_asian_call, _, _ = asian_mc_control_variate(
                    spot=float(spot_common),
                    strike=float(strike_common_local),
                    rate=float(rate_common),
                    sigma=float(sigma),
                    maturity=float(maturity_common),
                    n_obs=int(n_obs_price),
                    n_paths=20_000,
                    option_type="call",
                    antithetic=True,
                    seed=None,
                )
                progress.progress(100)
                st.success(f"Prix call asiatique arithmétique (MC + control variate) = {price_asian_call:.6f}")
                render_add_to_dashboard_button(
                    product_label="Asian arithmétique",
                    option_char=option_char,
                    price_value=price_asian_call,
                    strike=strike_common_local,
                    maturity=maturity_common,
                    key_prefix=_k("save_asian_arith"),
                    spot=spot_common,
                    misc={
                        "method": "MC control variate",
                        "n_obs": int(n_obs_price),
                        "n_paths": 20000,
                        "sigma": float(sigma),
                        "r": float(rate_common),
                        "q": float(d_common),
                    },
                )
            except Exception as exc:
                st.error(f"Erreur lors du pricing asiatique : {exc}")
            finally:
                progress.empty()
        st.caption(
            f"Paramètres utilisés pour le prix asiatique : "
            f"S0={spot_common:.4f}, K={strike_common_local:.4f}, "
            f"T={maturity_common:.4f}, r={rate_common:.4f}, σ={sigma:.4f}"
        )

        st.subheader("Heatmaps prix asiatiques (K vs T)")
        k_center = st.session_state.get("common_strike", strike_common)
        k_span = float(st.session_state.get("heatmap_span_value", 25.0))
        k_min = max(0.01, k_center - k_span)
        k_max = k_center + k_span
        col_k, col_t = st.columns(2)
        with col_k:
            st.caption(f"Domaine K commun (span): [{k_min:.2f}, {k_max:.2f}]")
        with col_t:
            t_center = st.session_state.get("common_maturity", maturity_common)
            t_span = float(st.session_state.get("heatmap_maturity_span_value", max(0.05, t_center * 0.5)))
            t_min = max(0.01, t_center - t_span)
            t_max = t_center + t_span
            st.caption(f"Domaine T commun (span): [{t_min:.2f}, {t_max:.2f}]")

        n_k = st.slider("Résolution en K", 10, 40, 20, 2, key=_k("n_k"))
        n_t = st.slider("Résolution en T", 10, 40, 20, 2, key=_k("n_t"))
        n_paths_surface = st.slider("Nombre de trajectoires Monte Carlo", 5_000, 50_000, 20_000, 5_000, key=_k("n_paths"))

        is_call_tab = option_char.lower() == "c"
        heatmap_label = "Call" if is_call_tab else "Put"
        with st.expander(f"Afficher la heatmap {heatmap_label}", expanded=False):
            k_vals = np.linspace(k_min, k_max, n_k)
            t_vals = np.linspace(t_min, t_max, n_t)
            prices = np.zeros((n_t, n_k), dtype=float)

            with st.spinner("Calcul de la surface de prix (MC asiatique)…"):
                progress_surface = st.progress(0)
                total_iters = len(t_vals) * len(k_vals)
                done = 0
                for i_t, t_val in enumerate(t_vals):
                    n_obs = max(2, int(50 * t_val))
                    for i_k, k_val in enumerate(k_vals):
                        price_val, _, _ = asian_mc_control_variate(
                            spot=float(spot_common),
                            strike=float(k_val),
                            rate=float(rate_common),
                            sigma=float(sigma),
                            maturity=float(t_val),
                            n_obs=int(n_obs),
                            n_paths=int(n_paths_surface),
                            option_type="call" if is_call_tab else "put",
                            antithetic=True,
                            seed=None,
                        )
                        prices[i_t, i_k] = price_val
                        done += 1
                        if total_iters > 0:
                            progress_surface.progress(int((done / total_iters) * 100))
                progress_surface.empty()

            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(
                prices,
                origin="lower",
                extent=[k_vals.min(), k_vals.max(), t_vals.min(), t_vals.max()],
                aspect="auto",
                cmap="viridis",
            )
            ax.set_xlabel("Strike K")
            ax.set_ylabel("Maturité T (années)")
            ax.set_title(f"{heatmap_label} asiatique arithmétique (MC + control variate)")
            fig.colorbar(im, ax=ax, label="Prix")
            fig.tight_layout()
            st.pyplot(fig)


    # ---------------------------------------------------------------------------
    #  Module Heston – pipeline complet
    # ---------------------------------------------------------------------------


    def heston_mc_pricer(
        S0: float,
        K: float,
        T: float,
        r: float,
        v0: float,
        theta: float,
        kappa: float,
        sigma_v: float,
        rho: float,
        n_paths: int = 50_000,
        n_steps: int = 100,
        option_type: str = "call",
    ) -> float:
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        S = np.full(n_paths, S0)
        v = np.full(n_paths, v0)
        for _ in range(n_steps):
            z1 = np.random.randn(n_paths)
            z2 = np.random.randn(n_paths)
            z_s = z1
            z_v = rho * z1 + math.sqrt(1 - rho**2) * z2
            v_pos = np.maximum(v, 0)
            S = S * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z_s)
            v = v + kappa * (theta - v_pos) * dt + sigma_v * np.sqrt(v_pos) * sqrt_dt * z_v
            v = np.maximum(v, 0)
        payoff = np.maximum(S - K, 0) if option_type == "call" else np.maximum(K - S, 0)
        return float(math.exp(-r * T) * np.mean(payoff))


    def download_options_cboe(symbol: str, option_type: str) -> tuple[pd.DataFrame, float, float, float]:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol.upper()}.json"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", {})
        options = data.get("options", [])
        spot = float(data.get("current_price") or data.get("close") or np.nan)
        risk_free = float(data.get("risk_free_rate") or 0.02)
        dividend_yield = float(data.get("dividend_yield") or 0.0)
        now = pd.Timestamp.utcnow().tz_localize(None)
        pattern = re.compile(rf"^{symbol.upper()}(?P<expiry>\d{{6}})(?P<cp>[CP])(?P<strike>\d+)$")

        rows: list[dict] = []
        for opt in options:
            match = pattern.match(opt.get("option", ""))
            if not match:
                continue
            cp = match.group("cp")
            if (option_type == "call" and cp != "C") or (option_type == "put" and cp != "P"):
                continue
            expiry_dt = pd.to_datetime(match.group("expiry"), format="%y%m%d")
            T = (expiry_dt - now).total_seconds() / (365.0 * 24 * 3600)
            if T <= 0:
                continue
            T = round(T, 2)
            if T <= MIN_IV_MATURITY:
                continue
            strike = int(match.group("strike")) / 1000.0
            bid = float(opt.get("bid") or 0.0)
            ask = float(opt.get("ask") or 0.0)
            last = float(opt.get("last_trade_price") or 0.0)
            if bid > 0 and ask > 0:
                mid = 0.5 * (bid + ask)
            elif last > 0:
                mid = last
            else:
                mid = np.nan
            if np.isnan(mid) or mid <= 0:
                continue
            iv_val = opt.get("iv", np.nan)
            iv_val = float(iv_val) if iv_val not in (None, "") else np.nan
            rows.append(
                {
                    "S0": spot,
                    "K": strike,
                    "T": T,
                    ("C_mkt" if option_type == "call" else "P_mkt"): round(mid, 2),
                    "iv_market": iv_val,
                }
            )

        df = pd.DataFrame(rows)
        df = df[df["T"] > MIN_IV_MATURITY]
        return df, spot, risk_free, dividend_yield


    @st.cache_data(show_spinner=False)
    def load_cboe_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
        calls_df, spot_calls, rf_calls, div_calls = download_options_cboe(symbol, "call")
        puts_df, spot_puts, rf_puts, div_puts = download_options_cboe(symbol, "put")
        S0_ref = float(np.nanmean([spot_calls, spot_puts]))
        risk_free = float(np.nanmean([rf_calls, rf_puts]))
        dividend_yield = float(np.nanmean([div_calls, div_puts]))
        return calls_df, puts_df, S0_ref, risk_free, dividend_yield


    def prices_from_unconstrained(u: torch.Tensor, S0_t: torch.Tensor, K_t: torch.Tensor, T_t: torch.Tensor, r: float, q: float):
        params = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
        prices = []
        for S0_i, K_i, T_i in zip(S0_t, K_t, T_t):
            price_i = carr_madan_call_torch(S0_i, r, q, T_i, params, K_i)
            prices.append(price_i)
        return torch.stack(prices)


    def heston_nn_loss(
        u: torch.Tensor,
        S0_t: torch.Tensor,
        K_t: torch.Tensor,
        T_t: torch.Tensor,
        C_mkt_t: torch.Tensor,
        r: float,
        q: float,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_prices = prices_from_unconstrained(u, S0_t, K_t, T_t, r, q)
        diff = model_prices - C_mkt_t
        if weights is not None:
            return 0.5 * (weights * diff**2).mean()
        return 0.5 * (diff**2).mean()


    def calibrate_heston_nn(
        df: pd.DataFrame,
        r: float,
        q: float,
        max_iters: int,
        lr: float,
        spot_override: float | None = None,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> HestonParams:
        if df.empty:
            raise ValueError("DataFrame vide.")
        df_clean = df.dropna(subset=["S0", "K", "T", "C_mkt"])
        df_clean = df_clean[(df_clean["T"] > MIN_IV_MATURITY) & (df_clean["C_mkt"] > 0.05)]
        df_clean = df_clean[df_clean.get("iv_market", 0) > 0]
        if df_clean.empty:
            raise ValueError("Pas de points pour la calibration")

        S0_ref = spot_override if spot_override is not None else float(df_clean["S0"].median())
        moneyness = df_clean["K"].values / S0_ref

        S0_t = torch.tensor(df_clean["S0"].values, dtype=torch.float64, device=HES_DEVICE)
        K_t = torch.tensor(df_clean["K"].values, dtype=torch.float64, device=HES_DEVICE)
        T_t = torch.tensor(df_clean["T"].values, dtype=torch.float64, device=HES_DEVICE)
        C_mkt_t = torch.tensor(df_clean["C_mkt"].values, dtype=torch.float64, device=HES_DEVICE)

        weights_np = 1.0 / (np.abs(moneyness - 1.0) + 1e-3)
        weights_np = np.clip(weights_np / weights_np.mean(), 0.5, 5.0)
        weights_t = torch.tensor(weights_np, dtype=torch.float64, device=HES_DEVICE)

        u = torch.zeros(5, dtype=torch.float64, device=HES_DEVICE, requires_grad=True)
        m = torch.zeros_like(u)
        v = torch.zeros_like(u)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        for iteration in range(max_iters):
            if u.grad is not None:
                u.grad.zero_()
            loss_val = heston_nn_loss(u, S0_t, K_t, T_t, C_mkt_t, r, q, weights=weights_t)
            loss_val.backward()
            with torch.no_grad():
                grad = u.grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1 ** (iteration + 1))
                v_hat = v / (1 - beta2 ** (iteration + 1))
                u -= lr * m_hat / (torch.sqrt(v_hat) + eps)
            if progress_callback:
                progress_callback(iteration + 1, max_iters, float(loss_val.detach().cpu()))

        return HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])


    def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


    def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(K - S, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


    def implied_vol_option(price: float, S: float, K: float, T: float, r: float, option_type: str = "call", tol: float = 1e-6, max_iter: int = 100) -> float:
        if T < MIN_IV_MATURITY:
            return np.nan
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        if price <= intrinsic:
            return np.nan
        sigma = 0.3
        for _ in range(max_iter):
            price_est = bs_call(S, K, T, r, sigma) if option_type == "call" else bs_put(S, K, T, r, sigma)
            diff = price_est - price
            if abs(diff) < tol:
                return sigma
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T)
            if vega < 1e-10:
                return np.nan
            sigma = sigma - diff / vega
            if sigma <= 0:
                return np.nan
        return np.nan


    def build_market_surface(
        df: pd.DataFrame,
        price_col: str,
        option_type: str,
        kk_grid: np.ndarray,
        tt_grid: np.ndarray,
        rf_rate: float,
    ) -> np.ndarray | None:
        df = df.dropna(subset=[price_col]).copy()
        df = df[(df["T"] >= MIN_IV_MATURITY) & (df[price_col] > 0)]
        if len(df) < 5:
            return None
        df["iv_calc"] = df.apply(
            lambda row: implied_vol_option(
                row[price_col], row["S0"], row["K"], row["T"], rf_rate, option_type
            ),
            axis=1,
        )
        df = df.dropna(subset=["iv_calc"])
        if df.empty:
            return None
        pts = df[["K", "T"]].to_numpy()
        vals = df["iv_calc"].to_numpy()
        surf = griddata(pts, vals, (kk_grid, tt_grid), method="linear")
        if surf is None or np.all(np.isnan(surf)):
            surf = griddata(pts, vals, (kk_grid, tt_grid), method="nearest")
        else:
            mask = np.isnan(surf)
            if mask.any():
                surf[mask] = griddata(pts, vals, (kk_grid[mask], tt_grid[mask]), method="nearest")
        return surf


    def build_market_price_grid(
        df: pd.DataFrame,
        price_col: str,
        kk_grid: np.ndarray,
        tt_grid: np.ndarray,
    ) -> np.ndarray | None:
        df = df.dropna(subset=[price_col]).copy()
        df = df[(df["T"] >= MIN_IV_MATURITY) & (df[price_col] > 0)]
        if len(df) < 5:
            return None
        pts = df[["K", "T"]].to_numpy()
        vals = df[price_col].to_numpy()
        grid = griddata(pts, vals, (kk_grid, tt_grid), method="linear")
        if grid is None or np.all(np.isnan(grid)):
            grid = griddata(pts, vals, (kk_grid, tt_grid), method="nearest")
        else:
            mask = np.isnan(grid)
            if mask.any():
                grid[mask] = griddata(pts, vals, (kk_grid[mask], tt_grid[mask]), method="nearest")
        return grid

    def _get_cached_iv_for(K_target: float, T_target: float, option_type: str = "call") -> float | None:
        """
        Récupère une vol implicite depuis les données CBOE en cache (session ou CSV),
        en cherchant la maturité la plus proche de T_target puis le K le plus proche.
        """
        opt_type = option_type.lower()
        calls_df = st.session_state.get("heston_calls_df")
        puts_df = st.session_state.get("heston_puts_df")
        if opt_type == "put":
            primary_df, secondary_df = puts_df, calls_df
        else:
            primary_df, secondary_df = calls_df, puts_df

        dfs: list[pd.DataFrame] = []
        for df in (primary_df, secondary_df):
            if df is not None and hasattr(df, "empty") and not df.empty:
                dfs.append(df)

        if not dfs:
            cache_path = CACHE_OPTIONS_PUTS_FILE if opt_type == "put" else CACHE_OPTIONS_CALLS_FILE
            try:
                df_cached = pd.read_csv(cache_path)
                if not df_cached.empty:
                    dfs.append(df_cached)
            except Exception:
                pass

        def _extract_iv(row: pd.Series) -> float | None:
            for col in ("iv_market", "iv", "impliedVolatility", "implied_vol"):
                if col in row and pd.notna(row[col]) and float(row[col]) > 0:
                    return float(row[col])
            price_col = "P_mkt" if opt_type == "put" else "C_mkt"
            if price_col in row and "S0" in row and "T" in row and "K" in row:
                try:
                    price_val = float(row[price_col])
                    if price_val <= 0:
                        return None
                    r_val = float(st.session_state.get("common_rate", 0.02))
                    return implied_vol_option(
                        price=price_val,
                        S=float(row["S0"]),
                        K=float(row["K"]),
                        T=float(row["T"]),
                        r=r_val,
                        option_type="put" if opt_type == "put" else "call",
                    )
                except Exception:
                    return None
            return None

        for df in dfs:
            if df is None or df.empty or "K" not in df.columns or "T" not in df.columns:
                continue
            df_valid = df.dropna(subset=["K", "T"]).copy()
            df_valid = df_valid[(df_valid["K"] > 0) & (df_valid["T"] > 0)]
            if df_valid.empty:
                continue
            df_valid["t_diff"] = (df_valid["T"] - T_target).abs()
            df_valid["k_diff"] = (df_valid["K"] - K_target).abs()
            best_row = df_valid.sort_values(["t_diff", "k_diff"]).iloc[0]
            iv_val = _extract_iv(best_row)
            if iv_val is not None and np.isfinite(iv_val) and iv_val > 0:
                return float(iv_val)
        return None

    def _pick_default_T_near_one(k_ref: float) -> float:
        """Choisit une maturité par défaut (proche de 1 an) en fonction du strike de référence."""
        dfs: list[pd.DataFrame] = []
        for df in (
            st.session_state.get("heston_calls_df"),
            st.session_state.get("heston_puts_df"),
        ):
            if df is not None and hasattr(df, "empty") and not df.empty:
                dfs.append(df)
        if not dfs:
            try:
                df_cached = pd.read_csv(CACHE_OPTIONS_CALLS_FILE)
                if not df_cached.empty:
                    dfs.append(df_cached)
            except Exception:
                pass
        for df in dfs:
            df_valid = df.dropna(subset=["K", "T"])
            df_valid = df_valid[df_valid["T"] > 0]
            if df_valid.empty:
                continue
            df_valid = df_valid.assign(
                t_diff=(df_valid["T"] - 1.0).abs(),
                k_diff=(df_valid["K"] - k_ref).abs(),
            )
            best = df_valid.sort_values(["t_diff", "k_diff"]).iloc[0]
            return float(best["T"])
        try:
            return float(st.session_state.get("T_common", 1.0) or 1.0)
        except Exception:
            return 1.0


    def render_section_explainer(title: str, body: str) -> None:
        """No-op (explications cachées)."""
        return


    def render_general_definition_explainer(title: str, body: str) -> None:
        """No-op (explications cachées)."""
        return


    def render_method_explainer(title: str, body: str) -> None:
        """No-op (explications cachées)."""
        return


    def render_inputs_explainer(title: str, body: str) -> None:
        """No-op (explications cachées)."""
        return


    def render_unlock_sidebar_button(context_key: str, label: str) -> None:
        """Affiche un bouton permettant de réactiver l'input T lorsque Heston a verrouillé la barre latérale."""
        if st.session_state.get("heston_tab_locked"):
            if st.button(label, key=f"unlock_sidebar_{context_key}"):
                st.session_state["heston_tab_locked"] = False
                st.rerun()


    ASIAN_LATEX_DERIVATION = r"""
    **Modèle sous la mesure risque-neutre**

    Sous la mesure risque-neutre $\mathbb{Q}$, le sous-jacent suit
    \[
    dS_t = (r-q)\,S_t\,dt + \sigma S_t\,dW_t, \qquad S_0>0,
    \]
    où $r$ est le taux sans risque, $q$ le dividende continu et $W_t$ un mouvement brownien standard.
    La solution explicite s'écrit
    \[
    S_t = S_0 \exp\Big[(r-q-\tfrac12\sigma^2)t + \sigma W_t\Big].
    \]

    **Option asiatique géométrique**

    On définit la moyenne géométrique continue
    \[
    G_T = \exp\!\left(\frac{1}{T}\int_0^T \ln S_t\,dt\right),
    \]
    et le payoff d'un call géométrique $(G_T-K)^+$.
    En partant de
    \[
    \ln S_t = \ln S_0 + (r-q-\tfrac12\sigma^2)t + \sigma W_t,
    \]
    on montre que
    \[
    \ln G_T = \frac{1}{T}\int_0^T \ln S_t\,dt
    = \ln S_0 + (r-q-\tfrac12\sigma^2)\frac{T}{2}
      + \sigma Y,
    \]
    avec
    \[
    Y = \frac{1}{T}\int_0^T W_t\,dt \sim \mathcal{N}\!\Big(0,\tfrac{T}{3}\Big).
    \]
    Ainsi, $\ln G_T$ est gaussien de moyenne $\mu_G$ et variance $v_G$ :
    \[
    \mu_G = \ln S_0 + (r-q-\tfrac12\sigma^2)\frac{T}{2},\qquad
    v_G = \sigma^2\frac{T}{3},
    \]
    ce qui implique que $G_T$ est lognormal. On introduit une volatilité effective
    \[
    \tilde{\sigma} = \frac{\sigma}{\sqrt{3}},
    \]
    et un niveau initial ajusté $\tilde{S}_0$ (obtenu à partir de la moyenne de $\ln G_T$) de sorte que le pricing du call géométrique s'écrive sous une forme de type Black--Scholes :
    \[
    C_0^{\mathrm{geom}} = \tilde{S}_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2),
    \]
    avec
    \[
    d_1 = \frac{\ln(\tilde{S}_0/K) + (r-q + \tfrac12 \tilde{\sigma}^2)T}{\tilde{\sigma}\sqrt{T}},
    \qquad
    d_2 = d_1 - \tilde{\sigma}\sqrt{T}.
    \]

    **Option asiatique arithmétique et PDE associée**

    La moyenne arithmétique continue est
    \[
    A_T = \frac{1}{T}\int_0^T S_t\,dt,
    \]
    et le payoff du call arithmétique est $(A_T-K)^+$.
    Comme ce payoff dépend du chemin complet, on introduit le processus d'intégrale
    \[
    I_t = \int_0^t S_u\,du,
    \]
    de sorte que $A_T = I_T/T$.
    Le couple $(S_t,I_t)$ est markovien et suit
    \[
    dS_t = (r-q)S_t\,dt + \sigma S_t\,dW_t, \qquad dI_t = S_t\,dt.
    \]

    On définit la fonction de valeur
    \[
    V(t,s,i) = \mathbb{E}^{\mathbb{Q}}\!\left[e^{-r(T-t)}\Big(\tfrac{I_T}{T} - K\Big)^+ \,\big|\, S_t=s,\,I_t=i\right],
    \]
    avec condition terminale
    \[
    V(T,s,i) = \Big(\tfrac{i}{T} - K\Big)^+.
    \]
    Le générateur infinitésimal du couple $(S_t,I_t)$ est
    \[
    \mathcal{L}V = (r-q)s\,V_s + \tfrac12\sigma^2 s^2\,V_{ss} + s\,V_i,
    \]
    et, par le théorème de Feynman--Kac, $V$ vérifie la PDE de valorisation
    \[
    \frac{\partial V}{\partial t}
     + (r-q)s \frac{\partial V}{\partial s}
     + \tfrac12 \sigma^2 s^2 \frac{\partial^2 V}{\partial s^2}
     + s \frac{\partial V}{\partial i}
     - r V = 0,
    \]
    sur $[0,T)\times (0,\infty)\times (0,\infty)$, avec la condition terminale ci‑dessus.
    Cette PDE n'admet pas de solution fermée simple et doit être résolue numériquement (schémas aux différences finies, méthodes spectrales, approches Monte Carlo avancées).
    """


    def render_math_derivation(title: str, body_md: str) -> None:
        """Affiche un menu déroulant contenant la dérivation mathématique, rendue avec LaTeX."""
        with st.expander(title):
            st.markdown(body_md)


    def render_pdf_derivation(title: str, pdf_path: str, download_name: str | None = None) -> None:
        """
        Affiche, dans un menu déroulant, un PDF (par exemple une dérivation LaTeX compilée).
        Le PDF est encodé en base64 et inclus dans une balise <iframe>.
        """
        from pathlib import Path as _Path

        with st.expander(title):
            path = _Path(pdf_path)
            if not path.exists():
                st.info(
                    f"Le fichier PDF '{pdf_path}' n'a pas été trouvé. "
                    "Placez le PDF compilé à cet emplacement pour l'afficher ici."
                )
                return

            with path.open("rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")

            pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="700"
        type="application/pdf"
    ></iframe>
    """
            st.markdown(pdf_display, unsafe_allow_html=True)



    def ui_heston_full_pipeline(auto_run: bool = False):


        _, cached_hist = load_cached_option_history()
        meta_options = load_options_meta()
        meta_tkr = (meta_options.get("ticker") or "").strip().upper()
        default_tkr = meta_tkr or st.session_state.get("tkr_common") or get_last_cached_option_ticker() or "SPY"
        # Ticker input with refresh button beside it
        ticker = st.text_input(
            "Ticker (sous-jacent)",
            value=default_tkr,
            key="heston_cboe_ticker",
            help="Code du sous-jacent coté au CBOE utilisé pour la calibration Heston.",
        ).strip().upper()
        # Forcer l'usage du ticker cache Options si présent (ex: BTC)
        if meta_tkr:
            ticker = meta_tkr
        # Si le ticker saisi change par rapport à la session, reset immédiat des params Heston
        prev_tkr = st.session_state.get("_last_heston_fetch_ticker")
        if prev_tkr and prev_tkr != ticker:
            for k in ["heston_kappa_common", "heston_theta_common", "heston_eta_common", "heston_rho_common", "heston_v0_common"]:
                st.session_state.pop(k, None)
            st.session_state["carr_madan_calibrated"] = False
            try:
                HESTON_PARAMS_FILE.write_text("{}", encoding="utf-8")
            except Exception:
                pass
        fetch_btn = st.button("🔄 Refresh", type="primary", key="heston_cboe_fetch")
        st.session_state["tkr_common"] = ticker
        st.session_state["common_underlying"] = ticker
        rf_rate = float(st.session_state.get("common_rate", 0.02))
        div_yield = float(st.session_state.get("common_dividend", 0.0))
        if fetch_btn:
            prev_tkr = st.session_state.get("_last_heston_fetch_ticker")
            if prev_tkr != ticker:
                # Reset Heston params only when ticker changes
                for k in ["heston_kappa_common", "heston_theta_common", "heston_eta_common", "heston_rho_common", "heston_v0_common"]:
                    st.session_state.pop(k, None)
                st.session_state["carr_madan_calibrated"] = False
                try:
                    HESTON_PARAMS_FILE.write_text("{}", encoding="utf-8")
                except Exception:
                    pass
            st.session_state["_last_heston_fetch_ticker"] = ticker

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            pass  # réservé pour d’autres réglages si nécessaire
        with col_cfg2:
            span_mc = float(st.session_state.get("heatmap_span_value", 20.0))
            n_maturities = 40


        state = st.session_state
        if "heston_calls_df" not in state:
            state.heston_calls_df = None
            state.heston_puts_df = None
            state.heston_S0_ref = None
            state.heston_calib_T_target = None
        calls_df = state.heston_calls_df
        puts_df = state.heston_puts_df
        S0_ref = state.heston_S0_ref
        calib_T_target = state.heston_calib_T_target

        # Ne jamais auto-télécharger : l'utilisateur déclenche le fetch explicitement
        auto_fetch = False
        fetch_btn = fetch_btn or auto_fetch
        st.divider()

        # Pré-remplit le spot commun depuis options_last_meta.json si le ticker correspond
        meta_cache = load_options_meta()
        meta_ticker = (meta_cache.get("ticker") or "").strip().upper()
        meta_spot = meta_cache.get("S0_ref")
        meta_r = meta_cache.get("r")
        meta_q = meta_cache.get("q")
        if meta_ticker == ticker and meta_spot is not None:
            try:
                spot_from_meta = float(meta_spot)
                st.session_state["S0_common"] = spot_from_meta
                st.session_state["common_spot"] = spot_from_meta
                state.heston_S0_ref = spot_from_meta
                S0_ref = spot_from_meta
            except Exception:
                pass
            try:
                if meta_r is not None:
                    st.session_state["common_rate"] = float(meta_r)
                if meta_q is not None:
                    st.session_state["common_dividend"] = float(meta_q)
            except Exception:
                pass
        elif not fetch_btn:
            st.warning(
                "⚠️ Aucun cache options_last_meta.json pour ce ticker. "
                "Saisis un ticker valide puis clique sur « Refresh » pour initialiser le spot commun."
            )
            return

        # Tente de charger d'éventuels paramètres Heston persistés pour ce ticker
        cached_heston_params = load_heston_params_from_json(ticker)
        if cached_heston_params and not fetch_btn:
            required_keys = {"kappa", "theta", "sigma", "rho", "v0"}
            if not required_keys.issubset(cached_heston_params.keys()):
                st.info("Cache Heston incomplet pour ce ticker – calibre le NN pour activer l'onglet.")
            else:
                st.info("Paramètres Heston trouvés dans le cache JSON – réutilisation sans recalibration.")
                try:
                    st.session_state["heston_kappa_common"] = float(cached_heston_params["kappa"])
                    st.session_state["heston_theta_common"] = float(cached_heston_params["theta"])
                    st.session_state["heston_eta_common"] = float(cached_heston_params["sigma"])
                    st.session_state["heston_rho_common"] = float(cached_heston_params["rho"])
                    st.session_state["heston_v0_common"] = float(cached_heston_params["v0"])
                    st.session_state["carr_madan_calibrated"] = True
                    # Recharge les appels/puts éventuellement en cache pour ce ticker
                    if calls_df is None or puts_df is None or S0_ref is None:
                        cached_calls, cached_puts, cached_S0, cached_r, cached_q = load_cached_option_chain(ticker)
                        if cached_calls is not None and cached_puts is not None and cached_S0:
                            state.heston_calls_df = cached_calls
                            state.heston_puts_df = cached_puts
                            state.heston_S0_ref = cached_S0
                            calls_df = cached_calls
                            puts_df = cached_puts
                            S0_ref = cached_S0
                            if cached_r is not None:
                                st.session_state["common_rate"] = float(cached_r)
                            if cached_q is not None:
                                st.session_state["common_dividend"] = float(cached_q)
                            st.session_state["heston_cboe_loaded_once"] = True
                            st.info("Chaînes CBOE rechargées depuis le cache pour le ticker calibré.")
                except Exception:
                    # En cas de problème de parsing, on redemandera une calibration NN.
                    st.info("Lecture des paramètres Heston en cache impossible – calibre le NN pour activer l'onglet.")

        if fetch_btn:
            try:
                ticker_fetch = (st.session_state.get("heston_cboe_ticker") or ticker or "SPY").strip().upper()
                ticker = ticker_fetch
                # Ne pas réécrire la clé du widget texte pour éviter les erreurs Streamlit
                st.session_state["tkr_common"] = ticker_fetch
                st.session_state["common_underlying"] = ticker_fetch
                try:
                    load_cboe_data.clear()
                except Exception:
                    pass
                calls_df, puts_df, S0_ref, rf_rate, div_yield = load_cboe_data(ticker_fetch)
                if ((calls_df is None or calls_df.empty) and (puts_df is None or puts_df.empty)):
                    st.error("🔴 Aucun résultat pour ce ticker (CBOE). Vérifie le code et réessaie.")
                    st.session_state["heston_cboe_loaded_once"] = False
                    return
                try:
                    t_for_rate = float(np.median(calls_df["T"])) if calls_df is not None and not calls_df.empty else 1.0
                    rf_rate = float(get_r(t_for_rate) or rf_rate or 0.02)
                except Exception:
                    rf_rate = float(rf_rate or 0.02)
                try:
                    div_yield = float(get_q(ticker) or div_yield or 0.0)
                except Exception:
                    div_yield = float(div_yield or 0.0)
                state.heston_calls_df = calls_df
                state.heston_puts_df = puts_df
                state.heston_S0_ref = S0_ref
                st.session_state["common_rate"] = float(rf_rate)
                st.session_state["common_dividend"] = float(div_yield)
                st.info(f"📡 Données CBOE chargées pour {ticker_fetch} (cache)")
                st.success(f"{len(calls_df)} calls, {len(puts_df)} puts | S0 ≈ {S0_ref:.2f}")
                save_cached_option_chain(ticker_fetch, calls_df, puts_df, S0_ref, rf_rate, div_yield)
                maturity_list = sorted(calls_df["T"].round(2).unique().tolist())
                st.session_state["cboe_T_options"] = maturity_list
                st.session_state["sidebar_maturity_options"] = maturity_list
                span_sync = float(st.session_state.get("heatmap_span_value", 20.0))
                if maturity_list:
                    rnd_T = float(np.random.choice(maturity_list))
                else:
                    rnd_T = float(round(calls_df["T"].iloc[0], 2))
                eligible_calls = calls_df[
                    (calls_df["T"].round(2) == rnd_T)
                    & calls_df["K"].between(S0_ref - span_sync, S0_ref + span_sync)
                ]
                if eligible_calls.empty:
                    eligible_calls = calls_df[
                        calls_df["K"].between(S0_ref - span_sync, S0_ref + span_sync)
                    ]
                if eligible_calls.empty:
                    eligible_calls = calls_df
                chosen_row = eligible_calls.sample(1).iloc[0]
                chosen_K = float(chosen_row["K"])
                chosen_T = float(round(chosen_row["T"], 2))
                sigma_pick = float(chosen_row.get("iv_market") or np.nan)
                if not np.isfinite(sigma_pick):
                    sigma_pick = implied_vol_option(
                        float(chosen_row.get("C_mkt", np.nan)),
                        float(chosen_row.get("S0", S0_ref)),
                        chosen_K,
                        float(chosen_row["T"]),
                        rf_rate,
                        "call",
                    )
                if not np.isfinite(sigma_pick):
                    sigma_pick = float(st.session_state.get("sigma_common", 0.2))
                prefills = {
                    "S0_common": float(S0_ref),
                    "K_common": chosen_K,
                    "sigma_common": float(np.clip(sigma_pick, 0.01, 5.0)),
                }
                st.session_state["heston_sidebar_prefill"] = prefills
                st.session_state["heston_sidebar_placeholders"] = {
                    "S0_common": f"{prefills['S0_common']:.2f}",
                    "K_common": f"{prefills['K_common']:.2f}",
                    "sigma_common": f"{prefills['sigma_common']:.4f}",
                }
                # Met à jour options_last_meta.json immédiatement
                try:
                    CACHE_OPTIONS_META_FILE.parent.mkdir(parents=True, exist_ok=True)
                    with CACHE_OPTIONS_META_FILE.open("w") as f:
                        json.dump(
                            {
                                "ticker": ticker_fetch,
                                "S0_ref": float(S0_ref),
                                "r": float(rf_rate),
                                "q": float(div_yield),
                            },
                            f,
                        )
                except Exception:
                    pass
                st.session_state["heston_cboe_loaded_once"] = True
                # Refresh cached 1y history for this ticker
                fetch_option_history_to_cache(ticker_fetch)
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Erreur lors du téléchargement des données CBOE : {exc}")

        calls_df = state.heston_calls_df
        puts_df = state.heston_puts_df
        S0_ref = state.heston_S0_ref
        calib_T_target = state.heston_calib_T_target

        calib_band_range: tuple[float, float] | None = None
        calib_T_band = 0.4
        max_iters = 1000
        learning_rate = 0.005

        # Si aucune donnée CBOE, on tente de s'appuyer sur le cache (chaines + historique)
        cache_used = False
        if calls_df is None or puts_df is None or S0_ref is None:
            cached_calls, cached_puts, cached_S0, cached_r, cached_q = load_cached_option_chain(ticker)
            cached_tkr_hist, cached_hist_df = load_cached_option_history()
            if cached_calls is not None and cached_puts is not None and cached_S0:
                calls_df = cached_calls
                puts_df = cached_puts
                S0_ref = cached_S0
                state.heston_calls_df = calls_df
                state.heston_puts_df = puts_df
                state.heston_S0_ref = S0_ref
                if cached_r is not None:
                    st.session_state["common_rate"] = float(cached_r)
                if cached_q is not None:
                    st.session_state["common_dividend"] = float(cached_q)
                cache_used = True
            elif cached_hist_df is not None and not cached_hist_df.empty:
                S0_ref = float(cached_hist_df["Close"].iloc[-1])
                state.heston_S0_ref = S0_ref
                calls_df = pd.DataFrame(columns=["T", "K", "C_mkt", "iv_market"])
                puts_df = pd.DataFrame(columns=["T", "K", "P_mkt", "iv_market"])
                cache_used = True
                st.warning("⚠️ Données CBOE absentes. Utilisation du cache 1 an des clôtures. Clique sur Refresh si besoin.")

        if cache_used:
            cache_age_msg = ""
            try:
                mtime = CACHE_OPTIONS_META_FILE.stat().st_mtime if CACHE_OPTIONS_META_FILE.exists() else None
                if mtime:
                    age_hours = (datetime.datetime.now() - datetime.datetime.fromtimestamp(mtime)).total_seconds() / 3600
                    cache_age_msg = f"Âge du cache options : ~{age_hours:.1f} h"
            except Exception:
                cache_age_msg = ""

            st.info(f"📦 Cache options utilisé. {cache_age_msg or 'Âge du cache inconnu.'} (Refresh pour actualiser).")

        # Si aucun cache ni données disponibles, on bloque l'affichage
        if (calls_df is None and puts_df is None) and S0_ref is None:
            st.warning("⚠️ Charge d'abord les données du ticker (bouton « Récupérer les données du ticker ») pour activer l'onglet Options.")
            return

        if cache_used:
            st.session_state["heston_cboe_loaded_once"] = True
            if calls_df is not None and not calls_df.empty and "T" in calls_df.columns:
                maturity_list = sorted(calls_df["T"].round(2).unique().tolist())
                st.session_state["cboe_T_options"] = maturity_list
                st.session_state["sidebar_maturity_options"] = maturity_list

    # ---------------------------------------------------------------------------
    #  Application Streamlit unifiée
    # ---------------------------------------------------------------------------


    sidebar_prefill = st.session_state.pop("heston_sidebar_prefill", None)
    if sidebar_prefill:
        for key, value in sidebar_prefill.items():
            st.session_state[key] = value


    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        div[data-testid="stStatusWidget"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initial defaults for shared parameters
    placeholder_vals = st.session_state.get("heston_sidebar_placeholders", {})
    heston_tab_locked = st.session_state.get("heston_tab_locked", False)

    default_values = {
        "S0_common": 100.0,
        "K_common": 100.0,
        "T_common": 1.0,
        "sigma_common": 0.2,
        "r_common": 0.05,
        "d_common": 0.0,
        "heatmap_span": 25.0,
        "heston_kappa_common": 2.0,
        "heston_theta_common": 0.04,
        "heston_eta_common": 0.5,
        "heston_rho_common": -0.7,
        "heston_v0_common": 0.04,
    }
    for k, v in default_values.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("heston_cboe_loaded_once", False)

    ui_heston_full_pipeline()

    # Stop rendering le reste de l'onglet tant que le ticker n'est pas fetché
    if not st.session_state.get("heston_cboe_loaded_once", False):
        return

    # Paramètres communs BSM/Heston (définis une seule fois)
    def _safe_float(val, default):
        try:
            return float(default if val is None else val)
        except Exception:
            return float(default)

    S0_common = _safe_float(st.session_state.get("heston_S0_ref"), _safe_float(st.session_state.get("S0_common"), 100.0))
    T_common = _safe_float(st.session_state.get("heston_calib_T_target"), _safe_float(st.session_state.get("T_common"), 1.0))
    st.session_state["T_common"] = T_common
    K_common = _safe_float(st.session_state.get("K_common"), S0_common)
    sigma_common = _safe_float(st.session_state.get("sigma_common"), 0.2)
    r_common = max(_safe_float(st.session_state.get("common_rate"), 0.0), 1e-6)
    d_common = _safe_float(st.session_state.get("common_dividend"), 0.0)
    heatmap_span = _safe_float(st.session_state.get("heatmap_span_value"), 25.0)

    heatmap_spot_values = _heatmap_axis(S0_common, heatmap_span)
    heatmap_strike_values = _heatmap_axis(K_common, heatmap_span)
    heatmap_maturity_span = float(max(0.01, T_common * 0.5))
    heatmap_maturity_values = _heatmap_axis(T_common, heatmap_maturity_span)

    common_spot_value = float(S0_common)
    common_maturity_value = float(T_common)
    common_strike_value = float(K_common)
    common_rate_value = float(r_common)
    common_sigma_value = float(sigma_common)

    st.session_state["common_spot"] = common_spot_value
    st.session_state["common_strike"] = common_strike_value
    st.session_state["common_maturity"] = common_maturity_value
    st.session_state["common_sigma"] = common_sigma_value
    st.session_state["common_rate"] = common_rate_value
    st.session_state["common_dividend"] = float(d_common)
    st.session_state["heatmap_span_value"] = float(heatmap_span)
    st.session_state["heatmap_maturity_span_value"] = float(heatmap_maturity_span)

    st.markdown("---")
    st.subheader("Historique 1 an du ticker (prix de clôture)")
    hist_fig = None
    tkr_hist = st.session_state.get("heston_cboe_ticker", st.session_state.get("tkr_common", "")).strip().upper()
    if not tkr_hist:
        st.info("Charge un ticker via la calibration Heston pour afficher l'historique 1 an.")
    else:
        meta_cache = load_options_meta()
        meta_table = pd.DataFrame.from_dict(
            {
                "Ticker cache": [meta_cache.get("ticker", "")],
                "S0_ref cache": [meta_cache.get("S0_ref")],
                "r cache": [meta_cache.get("r")],
                "q cache": [meta_cache.get("q")],
            }
        )
        st.dataframe(meta_table, use_container_width=True, hide_index=True)
        _, hist_df = load_cached_option_history()

        if hist_df is not None and not hist_df.empty and "Close" in hist_df.columns:
            hist_fig = go.Figure()
            hist_fig.add_trace(
                go.Scatter(
                    x=hist_df.index,
                    y=hist_df["Close"],
                    mode="lines",
                    name="Close",
                )
            )
            # Ensure datetime index for proper ticks
            idx_dt = pd.to_datetime(hist_df.index)
            start_dt = idx_dt.min()
            end_dt = idx_dt.max()
            start_label = start_dt.strftime("%Y-%m-%d") if hasattr(start_dt, "strftime") else str(start_dt)
            end_label = end_dt.strftime("%Y-%m-%d") if hasattr(end_dt, "strftime") else str(end_dt)

            hist_fig.update_layout(
                title=f"{tkr_hist} - Close (1 an)",
                xaxis_title="Date",
                yaxis_title="Prix",
            )
            st.plotly_chart(hist_fig, width="stretch")
        else:
            st.info("Pas d'historique disponible pour ce ticker dans le cache. Clique sur Refresh (Heston) pour le télécharger.")

    heatmap_spot_values = _heatmap_axis(S0_common, heatmap_span)
    heatmap_strike_values = _heatmap_axis(K_common, heatmap_span)
    heatmap_maturity_span = float(max(0.01, T_common * 0.5))
    heatmap_maturity_values = _heatmap_axis(T_common, heatmap_maturity_span)
    common_spot_value = float(S0_common)
    common_maturity_value = float(T_common)
    common_strike_value = float(K_common)
    common_rate_value = float(r_common)
    common_sigma_value = float(sigma_common)
    st.session_state["common_spot"] = common_spot_value
    st.session_state["common_strike"] = common_strike_value
    st.session_state["common_maturity"] = common_maturity_value
    st.session_state["common_sigma"] = common_sigma_value
    st.session_state["common_rate"] = common_rate_value
    st.session_state["common_dividend"] = float(d_common)
    st.session_state["heatmap_span_value"] = float(heatmap_span)
    st.session_state["heatmap_maturity_span_value"] = float(heatmap_maturity_span)

    def render_option_tabs_for_type(option_label: str, option_char: str):
        def _choose_option_select(key_suffix: str, default_char: str) -> tuple[str, str]:
            """Selectbox Call/Put locale sans réécrire d'autres clés de session."""
            default_label = "Call" if default_char.lower() == "c" else "Put"
            saved = st.session_state.get(key_suffix, default_label)
            idx = 0 if saved == "Call" else 1 if saved == "Put" else 0
            choice = st.selectbox(
                "Type d'option",
                ["Call", "Put"],
                index=idx,
                key=key_suffix,
            )
            return choice, ("c" if choice == "Call" else "p")

        # Quick payoff helper for dropdown explanations.
        def _payoff_plot(x_vals, y_vals, title, strike_lines=None):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Payoff"))
            if strike_lines:
                for sl in strike_lines:
                    fig.add_shape(
                        type="line",
                        x0=sl,
                        x1=sl,
                        y0=min(y_vals),
                        y1=max(y_vals),
                        line=dict(color="red", dash="dash"),
                    )
            fig.update_layout(title=title, xaxis_title="Spot à maturité", yaxis_title="Payoff")
            return fig

        def _graph_desc(key_suffix: str) -> str:
            """Retourne une description textuelle (10 lignes) du payoff/produit."""
            desc_map = {
                "american_payoff": """Une option américaine peut être exercée à tout moment entre l’émission et l’échéance.
Le payoff terminal reste celui d’un call ou d’un put vanilla, mais l’acheteur peut l’activer plus tôt.
Sa valeur est toujours au moins égale à celle d’une européenne, car l’option d’exercice supplémentaire a un prix.
Un call américain sur un sous-jacent sans dividende ne s’exerce généralement pas avant T, mais un put peut être anticipé.
Le dividende pousse parfois à lever un call juste avant le détachement, et le taux rend l’exercice du put attractif.
La sensibilité au spot (delta) est souvent plus élevée près du strike, surtout pour les puts ITM.
La convexité (gamma) se concentre autour du strike et peut être accentuée par l’option d’exercice.
Le theta peut devenir moins négatif, voire positif pour certains puts ITM proches de l’échéance.
La volatilité implicite joue sur la valeur de l’option de flexibilité d’exercice, donc sur la prime.
Le profil final est celui d’un payoff vanilla au strike K, mais la décision d’exercice modifie la valeur actuelle.""",
                "bermuda_payoff": """Une option bermudéenne n’est exerçable que sur un ensemble de dates discrètes prédéfinies.
Elle se situe entre une européenne (exercice à T) et une américaine (exercice continu), offrant une flexibilité intermédiaire.
Si aucune date d’exercice anticipé n’est utilisée, le payoff terminal reste celui d’une vanilla au strike K.
Plus il y a de dates d’exercice, plus la valeur se rapproche de l’américaine, car la flexibilité augmente.
Le calendrier des dates influence le choix optimal d’exercice et donc le prix.
Le delta et le theta se situent entre ceux d’une européenne et d’une américaine.
La vega est proche de celle d’une européenne, mais l’impact de la vol sur l’exercice reste présent.
Les dividendes et le niveau des taux guident la décision d’exercer, surtout pour les puts.
Le produit convient aux investisseurs voulant un compromis entre flexibilité et coût.
Le payoff final reste celui d’un call ou d’un put vanilla si l’option n’est pas exercée avant l’échéance.""",
                "asian_graph": """Une option asiatique arithmétique paie sur la moyenne des prix observés, plutôt que sur le spot final.
Ce moyennage réduit l’impact des pics de volatilité et rend souvent l’option moins chère qu’une européenne.
Le nombre d’observations et leur calendrier déterminent la sensibilité au chemin et le niveau de prime.
Le delta est plus lissé qu’une vanilla, car chaque observation influence partiellement le payoff.
Le gamma est plus modéré et la vega généralement plus faible, car l’effet de la vol est dilué dans la moyenne.
Le theta dépend de la cadence des fixings et de la portion déjà observée.
Le payoff final reste de type call ou put, mais sur la moyenne plutôt que sur le spot terminal.
Les options asiatiques sont prisées pour couvrir un prix moyen d’achat ou de vente d’une marchandise.
Elles limitent le risque de mauvais timing tout en conservant une protection ou un levier.
Le graphe de payoff s’apparente à une vanilla mais décalée par l’effet de la moyenne.""",
                "asian_geo_graph": """Une option asiatique géométrique paie sur la moyenne géométrique des prix observés.
Cette moyenne est toujours inférieure ou égale à la moyenne arithmétique, ce qui rend la prime plus basse.
Une formule fermée existe souvent, ce qui facilite le calcul et la couverture.
Le produit est moins sensible aux extrêmes de prix que l’asiatique arithmétique.
Le delta reste lissé et le gamma modéré, car chaque observation pèse de manière multiplicative.
La vega est plus faible que pour l’asiatique arithmétique, l’effet de la vol étant atténué.
Le theta dépend du nombre d’observations et de la partie déjà réalisée.
La géométrique sert fréquemment de contrôle de variance pour des simulations Monte Carlo.
Le payoff final est de type call ou put sur cette moyenne géométrique.
Le graphe de payoff suit une forme vanilla, décalée par la moyenne géométrique plutôt que par le spot final.""",
                "lookback_graph": """Une lookback floating détermine son strike a posteriori en prenant l’extrême du spot pendant la vie de l’option.
Pour un call, le strike peut être le minimum observé ; pour un put, le maximum, ce qui protège contre un mauvais timing.
Le payoff final compare le spot terminal à cet extrême, offrant une convexité forte et une protection renforcée.
Le produit est fortement path-dependent, car chaque observation peut modifier l’extrême retenu.
Le delta dépend de la distance entre le spot et l’extrême ; il peut réagir vivement lors de nouveaux extrêmes.
La vega et le theta diffèrent d’une vanilla car le temps supplémentaire augmente la probabilité d’atteindre des extrêmes.
La maturité accroît la chance d’observer un nouveau max ou min, ce qui influence le prix.
La valorisation se fait souvent par Monte Carlo ou par formules fermées spécialisées.
Le risque de gap est important, car un saut peut fixer un extrême décisif.
Le payoff final reste comparable à une vanilla mais avec un strike déterminé par le parcours du sous-jacent.""",
                "lookback_fixed_graph": """Une lookback fixed fixe son strike dès l’origine, mais son payoff dépend du max ou du min atteint pendant la vie.
Le spot final est comparé à l’extrême historique, ce qui récompense une trajectoire favorable.
Le produit reste path-dependent, même si le strike ne bouge pas après l’émission.
Le delta reflète la distance entre le spot et l’extrême atteint, et peut évoluer brusquement lors d’un nouveau record.
La vega intègre l’incertitude sur l’atteinte de nouveaux extrêmes, ce qui rend le prix sensible à la vol.
Le theta dépend de la probabilité d’étendre l’extrême pendant le temps restant.
Les méthodes de valorisation vont des approches semi-fermées aux Monte Carlo selon les hypothèses.
Le payoff final ressemble à un call ou put, mais il intègre la meilleure ou la pire performance passée.
Le produit convient pour capturer une amélioration de prix sans renoncer au strike initial.
Le graphe de payoff indique une vanilla évaluée avec l’information d’un extrême déjà observé ou attendu.""",
                "forward_start_graph": """Une forward-start choisit son strike à une date future T_start, souvent comme k fois le spot observé ce jour-là.
Avant T_start, l’option reste en attente et vaut une promesse d’option avec strike inconnu.
Après T_start, elle devient une vanilla classique, mais le strike est déjà lié au marché de cette date.
Le produit réduit le risque de mauvais timing sur le strike, utile pour des émissions ou des primes différées.
Le delta est faible avant T_start, car le strike n’est pas encore fixé, puis il augmente après la fixation.
La vega est répartie entre la période avant et après T_start, avec une sensibilité distincte à la vol.
Le theta dépend du délai restant jusqu’à la fixation du strike, puis de la maturité résiduelle.
Le prix dépend du ratio k et de la distribution anticipée du spot à T_start.
La couverture nécessite de suivre le spot pré-fixation pour ajuster l’exposition.
Le payoff final est un call ou un put vanilla évalué avec un strike défini plus tard, mais payé à l’échéance finale.""",
                "cliquet_graph": """Une option cliquet cumule des coupons périodiques capés et floorés, souvent avec réinitialisation du strike.
Chaque période mesure un rendement qui est limité par un cap et un floor, puis ajouté au coupon cumulé.
Le produit est fortement path-dependent, car la séquence de rendements successifs détermine le total payé.
La vega et la theta sont réparties sur toutes les périodes, plutôt que concentrées sur une seule échéance.
Le delta évolue à chaque période selon la distance au cap et au floor et le niveau de spot.
La valorisation se fait souvent par Monte Carlo, car les dépendances entre périodes sont complexes.
Le produit est utilisé pour distribuer un rendement régulier avec des limites de performance.
Les caps/floors protègent contre des chocs excessifs tout en offrant une participation à la hausse ou à la baisse.
Le risque de gap est atténué par les plafonds, mais la corrélation des rendements successifs reste importante.
Le payoff final correspond à la somme des coupons ajustés, payés à l’échéance ou à chaque période selon la structure.""",
                "calendar_graph": """Un calendar spread combine une option courte à maturité proche et une option longue à maturité plus lointaine, au même strike.
Le pari principal est que la valeur temps de la jambe longue dépasse celle de la jambe courte pendant la vie du trade.
Le theta est souvent positif au départ, car on encaisse la prime de l’échéance courte plus vite.
La vega est généralement positive, car la jambe longue réagit plus à une hausse de volatilité.
Le payoff est non linéaire autour du strike et dépend du comportement après l’expiration de la jambe courte.
Le gamma est concentré près de l’échéance proche, mais l’option longue conserve de la convexité au-delà.
Le spread peut être débouclé ou roulé après la première échéance, selon le mouvement du spot et de la vol.
Le produit est sensible au skew et à la term structure de volatilité entre les deux maturités.
Le risque est borné par les primes nettes, mais un mouvement rapide peut coûter la jambe courte.
Le payoff final reflète la valeur résiduelle de la jambe longue moins la jambe courte arrivée à échéance.""",
                "diagonal_graph": """Un diagonal spread mélange une différence de maturité et de strike entre les deux jambes.
Il combine les idées d’un vertical spread (strikes différents) et d’un calendar spread (maturités différentes).
La position permet d’ajuster une vue directionnelle tout en jouant sur la volatilité et le passage du temps.
Le theta dépend des deux échéances : la jambe courte s’érode plus vite, la jambe longue porte plus de valeur temps.
La vega est mixte, car les deux maturités réagissent différemment à un choc de volatilité.
Le profil peut viser une zone de spot spécifique où le spread délivre sa valeur maximale.
Le skew entre strikes et la term structure entre maturités influencent directement le prix.
La gestion active est souvent nécessaire après expiration de la jambe courte ou en cas de mouvement fort.
Le risque est borné par la combinaison de primes, mais le profil reste asymétrique.
Le payoff final est un différentiel de deux vanillas de strikes et de maturités distincts.""",
                "digital_graph": """Une option digitale cash-or-nothing paie un montant fixe si la condition sur le spot est satisfaite à l’échéance.
Pour un call, le paiement est déclenché si S_T dépasse K ; pour un put, s’il passe en dessous.
Le payoff ressemble à un saut de niveau, sans dépendre de l’ampleur du mouvement au-delà du strike.
La vega est concentrée autour du strike, car la probabilité de franchissement dépend fortement de la volatilité.
Le theta peut être abrupt près de l’échéance, la probabilité de franchissement évoluant vite.
Le gamma est également pic autour de K, car une petite variation de spot change la probabilité de paiement.
Le produit sert souvent de brique de base pour des structurés à coupon élevé.
Le risque de gap est important, car un saut peut déclencher ou annuler le paiement d’un coup.
La couverture nécessite de gérer une exposition sensible au spot autour du strike.
Le payoff final est binaire : soit le montant fixe est versé, soit rien n’est payé.""",
                "asset_on_graph": """Une option asset-or-nothing verse le sous-jacent lui-même (ou rien) si la condition est remplie.
Pour un call, l’actif est livré si S_T dépasse K ; pour un put, si S_T passe en dessous.
Le payoff est proportionnel au spot dans le scénario favorable, sinon nul, ce qui combine linéarité et condition binaire.
La sensibilité au spot reste forte, car le paiement dépend directement du niveau final de l’actif.
Le gamma et la vega sont concentrés autour du strike, mais l’ampleur du paiement varie avec S_T.
Le theta se comporte différemment d’une digitale cash car le montant dépend du spot final.
Le produit peut être couvert via des combinaisons de vanillas et de positions sur le sous-jacent.
Le risque de gap crée une discontinuité de valeur s’il survient près du strike.
La valorisation reste proche d’une digitale, mais avec un paiement multiplicatif par S_T.
Le payoff final est soit la remise de l’actif, soit zéro, selon la condition sur le spot.""",
                "chooser_graph": """Une option chooser offre à son détenteur le droit de choisir, à une date t_choice, entre un call et un put.
Avant t_choice, la prime reflète cette flexibilité supplémentaire, supérieure à celle d’une simple vanilla.
Après t_choice, le contrat se transforme en l’option choisie, avec le strike et la maturité restants.
Le produit couvre une incertitude directionnelle jusqu’à une date clé, tout en gardant une protection.
Le delta et la vega sont mixtes avant t_choice, car le profil intègre les deux scénarios potentiels.
Le theta dépend du temps restant avant la décision, puis du temps restant après la conversion.
La valorisation s’appuie souvent sur une combinaison analytique de call et de put avec un facteur de décision.
Le risque de gap avant t_choice peut influencer la préférence pour le call ou le put.
Le produit convient quand l’investisseur veut reporter un choix directionnel tout en verrouillant un coût.
Le payoff final est celui du call ou du put choisi, exercé à l’échéance finale selon le type sélectionné.""",
                "quanto_graph": """Une option quanto paie sur un sous-jacent étranger mais règle dans une autre devise à un taux de change fixé.
Le produit neutralise le risque de change tout en conservant l’exposition au sous-jacent de départ.
Le payoff reste celui d’un call ou d’un put vanilla, mais converti à un taux garanti ou ajusté.
La corrélation entre l’actif et le FX peut être prise en compte dans le drift effectif.
Le delta s’exprime sur le sous-jacent étranger, tandis que l’exposition FX est neutralisée.
La vega porte sur la volatilité de l’actif, l’effet de la vol FX étant limité par la construction.
Le theta reflète la valeur temps dans la devise de règlement, avec le taux de change figé.
La couverture se fait via le sous-jacent étranger et éventuellement des dérivés FX pour ajuster le drift.
Le produit est recherché quand l’investisseur veut éviter le risque de conversion tout en gardant la vue sur l’actif.
Le payoff final est une vanilla en devise locale, calculée sur la performance de l’actif étranger.""",
                "rainbow_graph": """Une option rainbow multi-actifs paie sur l’extrême d’un panier, le plus souvent le max ou le min.
Un call sur le max profite du meilleur performeur ; un put sur le min protège contre le pire actif.
La corrélation entre actifs est déterminante : faible corrélation augmente l’attrait du call sur max et du put sur min.
La vega est répartie entre les actifs, chaque volatilité contribuant à la probabilité d’être l’extrême.
Le delta se déplace vers l’actif le plus susceptible de finir max ou min, évoluant avec le marché.
Le produit ne somme pas les actifs : il choisit un extrême, ce qui rend le payoff non additif.
Les poids peuvent être asymétriques pour refléter des préférences sur certains actifs.
La couverture nécessite de suivre plusieurs sous-jacents et leur corrélation.
Le risque est lié aux sauts d’un actif qui peut soudain devenir l’extrême.
Le payoff final correspond à la valeur de l’actif extrême, comparée au strike pour un call ou un put.""",
                "barrier_graph": """Une option barrière adapte un payoff vanilla en le conditionnant à un niveau franchi ou non.
Une barrière knock-in active l’option si elle est touchée, tandis qu’une knock-out l’éteint.
La direction (up ou down) et le type (call ou put) définissent la zone de surveillance.
Plus la barrière est proche, plus la prime baisse pour une knock-out ou augmente pour une knock-in.
Le produit est path-dependent : la chronologie des niveaux atteints compte, pas seulement le final.
La vega et le theta sont influencés par la proximité de la barrière et par le temps restant.
Le risque de gap est crucial, car un saut peut déclencher ou annuler l’option instantanément.
Les paramètres clés sont le niveau de barrière, son sens, la nature in/out et le style call/put.
Le payoff final est celui d’une vanilla si la condition de barrière est respectée, sinon il est nul ou s’active.
Le produit est prisé pour réduire le coût ou pour cibler un scénario de range précis.""",
                "binary_barrier_graph": """Une barrière digitale paie un montant fixe selon le franchissement d’un niveau et la position finale du spot.
Elle combine la logique d’une barrière (up/down, in/out) et d’un payoff binaire.
La probabilité de paiement dépend du chemin et de la vol, avec un gamma concentré autour de la barrière et du strike.
Une knock-out annule le paiement si la barrière est touchée, alors qu’une knock-in l’active.
Le payout fixe simplifie le montant mais rend le risque de saut très important.
Les paramètres essentiels sont la barrière, sa direction, son type in/out, le montant fixe et l’éventuel strike.
La vega et le theta dépendent de la proximité de la barrière et du temps restant.
La couverture est délicate car la valeur saute quand la barrière est franchie.
Le produit est courant dans les structurés à rendement élevé où la probabilité de survie est rémunérée.
Le payoff final est une fonction en escalier conditionnée par le scénario de barrière et le niveau final.""",
                "basket_graph": """Une option basket paie sur une combinaison pondérée de plusieurs actifs, souvent une moyenne.
La corrélation module la diversification : plus elle est faible, plus la volatilité du panier est réduite.
Le payoff final est de type call ou put sur la valeur agrégée du panier.
La vega dépend des vols individuelles et de la corrélation, chaque composante contribuant au risque global.
Le delta est réparti selon les pondérations et peut se déplacer si un actif domine la variation du panier.
Le produit est path-indépendant si le payoff ne dépend que de la valeur terminale du panier.
Il est utilisé pour exprimer une vue macro ou sectorielle tout en lissant le risque idiosyncratique.
Le prix est sensible à la structure de corrélation : une hausse de corrélation augmente la vol du panier.
La couverture nécessite des rééquilibrages sur chaque actif composant.
Le graphe de payoff ressemble à une vanilla appliquée à la valeur agrégée du panier.""",
                "european_graph": """Une option européenne n’est exerçable qu’à l’échéance, avec un payoff vanilla au strike K.
La prime dépend des paramètres classiques S0, K, T, r, dividende et volatilité implicite.
L’absence d’exercice anticipé simplifie la valorisation et la couverture.
Un call sur un actif sans dividende ne s’exerce jamais avant T, car la valeur temps est conservée.
Le delta, le gamma, le theta et la vega suivent les profils standards du modèle Black-Scholes-Merton.
La surface de prix se lit en fonction de S et K autour des valeurs communes choisies.
Le risque pour l’acheteur est limité à la prime payée, le gain potentiellement illimité pour un call.
La put-call parity relie les prix call et put via le terme forward S0·e^{(r-q)T}.
La volatilité implicite mesurée sur ces options sert de référence pour d’autres produits.
Le payoff final est linéaire au-dessus ou en dessous du strike selon qu’il s’agit d’un call ou d’un put.""",
                "straddle_graph": """Un straddle combine l’achat simultané d’un call et d’un put au même strike K.
Le profil est symétrique et gagne en cas de forte hausse ou de forte baisse du sous-jacent.
Le coût initial est élevé, car on paie deux primes, ce qui crée une zone de perte autour de K.
Le delta initial est proche de zéro mais évolue rapidement dès que le spot s’éloigne du strike.
Le gamma est élevé près du strike, offrant une forte convexité autour de K.
La vega est positive, car une hausse de volatilité augmente la valeur des deux jambes.
Le theta est négatif et peut être important, car deux options perdent de la valeur temps.
Le break-even se situe à K ± (somme des primes), définissant la zone de profit.
Le produit sert à parier sur la magnitude d’un mouvement sans biais directionnel.
Le payoff final forme un V double, somme des payoffs call et put au même strike.""",
                "strangle_graph": """Un strangle achète un put OTM et un call OTM à deux strikes différents, plus éloignés que le spot.
Le coût est inférieur à celui d’un straddle, mais il faut un mouvement plus grand pour être gagnant.
La zone de perte est plus large autour du spot actuel, car les strikes sont décalés.
Le delta initial est proche de zéro mais évolue lorsque le spot s’approche d’un des strikes.
Le gamma est moins concentré qu’un straddle mais reste sensible autour des deux strikes.
La vega est positive, l’option profitant d’une hausse de volatilité qui augmente la probabilité de franchir un strike.
Le theta est négatif mais généralement plus faible qu’un straddle en valeur absolue.
Les break-even se situent autour de chaque strike plus la prime correspondante.
Le produit convient pour parier sur un mouvement important à moindre coût qu’un straddle.
Le payoff final présente deux régions de profit au-delà des strikes, avec une zone plate de perte entre eux.""",
                "call_spread_graph": """Un bull call spread achète un call à strike bas K1 et vend un call à strike plus haut K2.
La jambe courte finance partiellement la jambe longue, réduisant le coût par rapport à un call nu.
Le gain est plafonné au-dessus de K2, car la jambe vendue limite la participation à la hausse.
Le delta est positif mais inférieur à celui d’un call nu, et le gamma est modéré.
La vega est réduite, car une partie du risque de volatilité est vendue via la jambe courte.
Le theta peut être moins négatif, la prime encaissée compensant la perte de valeur temps.
La perte maximale est limitée à la prime nette payée, bornant le risque.
Le produit convient à une vue haussière modérée, avec budget de prime réduit.
Le profil reste directionnel, mais avec un plafond de gain défini par l’écart des strikes.
Le payoff final monte entre K1 et K2 puis se stabilise une fois le plafond atteint.""",
                "put_spread_graph": """Un bear put spread achète un put à strike haut K1 et vend un put à strike plus bas K2.
La jambe courte réduit le coût, mais le gain potentiel est plafonné sous K2.
Le delta est négatif mais plus faible en valeur absolue qu’un put nu, et le gamma est modéré.
La vega est atténuée, car une partie du risque de volatilité est cédée via la jambe vendue.
Le theta peut être moins négatif, la prime encaissée compensant l’érosion de la jambe longue.
La perte maximale est la prime nette payée, tandis que le gain maximal est (K1-K2) moins la prime.
Le produit cible une baisse modérée du sous-jacent plutôt qu’un scénario extrême.
La sensibilité au skew put est importante car les deux strikes peuvent se pricier différemment.
Le profil est borné : perte limitée au-dessus de K1, gain plafonné sous K2.
Le payoff final descend entre K1 et K2 puis se stabilise une fois le plancher atteint.""",
                "butterfly_graph": """Une butterfly call classique combine deux calls vendus au strike central et deux calls achetés aux ailes.
Le profil crée un pic de gain autour du strike médian, avec un coût relativement faible.
Le delta est proche de zéro au centre, tandis que le gamma est élevé près du strike central.
La vega est négative, car on est net vendeur de volatilité sur la zone centrale.
Le theta peut être positif autour du strike central, car la vente des options centrales s’érode plus vite.
Le risque est limité à la prime nette payée si le spot s’éloigne fortement.
Le gain maximal est atteint si le spot termine autour du strike médian à l’échéance.
Le produit convient pour parier sur un range étroit et une volatilité en baisse.
La largeur des ailes détermine l’ampleur du plateau de risque et de gain.
Le payoff final forme une tente autour du strike central, avec des pertes limitées en dehors.""",
                "condor_graph": """Un condor utilise quatre options à strikes échelonnés pour créer un plateau de gain plus large qu’une butterfly.
Les deux options centrales sont vendues, tandis que les deux ailes sont achetées pour borner le risque.
Le profil est plus plat et plus étendu au centre qu’une butterfly, avec un pic de gain moins aigu.
Le delta reste proche de zéro autour du centre, et le gamma est plus doux.
La vega est généralement négative, car la construction reste globalement short vol.
Le theta peut être positif dans la zone centrale, la valeur temps des jambes courtes se dégradant plus vite.
Le gain est limité entre les strikes intermédiaires, la perte est limitée aux extrêmes.
La construction peut être en crédit ou en débit selon le placement des strikes et le niveau de vol.
Le produit vise un range modéré plus large qu’une butterfly, avec un risque borné.
Le payoff final offre un plateau central avec des pentes plus douces vers les pertes aux ailes.""",
                "iron_condor_graph": """Un iron condor combine un put spread vendu et un call spread vendu pour encaisser un crédit initial.
Le plateau de gain se situe entre les strikes courts, tant que le spot reste dans le corridor.
La perte est limitée en dehors, bornée par la largeur des ailes longues achetées.
Le delta est proche de zéro au centre, le gamma modéré, ce qui convient aux vues de range.
La vega est négative, car la position est globalement short vol, et le theta est souvent positif au départ.
Le placement des ailes longues et courtes ajuste le compromis entre crédit encaissé et largeur de range.
Une hausse de volatilité ou un mouvement violent peut menacer le plateau et réduire le gain attendu.
Le trade peut être débouclé avant échéance si la vol se comprime et que le spot reste dans le range.
Le produit convient pour monétiser une attente de stagnation avec risque borné.
Le payoff final présente un plateau central de profit et des pertes plafonnées de part et d’autre.""",
                "iron_butterfly_graph": """Un iron butterfly vend un straddle au strike central et achète des ailes de protection plus éloignées.
Le crédit initial est maximal si le spot reste proche du strike central à l’échéance.
Les ailes longues bornent la perte en cas de mouvement extrême, rendant le risque limité.
Le delta est proche de zéro autour du centre, mais le gamma est élevé près du strike central.
La vega est négative et le theta positif, car la position vend de la valeur temps au cœur.
Le produit vise un range étroit et une volatilité en baisse, avec un crédit supérieur à l’iron condor.
Les break-even se situent autour du strike central plus ou moins la largeur des ailes moins le crédit net.
La gestion active peut être requise si le spot s’éloigne trop du centre.
La construction reste en crédit, avec une perte maximale bornée par l’écart ailes/centre.
Le payoff final est une tente inversée centrée sur le strike, avec profit au centre et pertes plafonnées aux extrêmes.""",
            }
            text = desc_map.get(key_suffix, "")
            if not text:
                return ""
            lines = text.strip("\n").splitlines()
            if len(lines) < 10:
                lines += [""] * (10 - len(lines))
            cleaned = [re.sub(r"^\s*\d+\)\s*", "", ln) for ln in lines[:10]]
            return "\n".join(cleaned)

        def _render_option_text(label: str, key_suffix: str):
            """Affiche un expander texte décrivant le payoff et la spécificité."""
            desc = _graph_desc(key_suffix)
            if not desc:
                return
            with st.expander(f"ℹ️ À propos – {label}", expanded=False):
                st.markdown(desc.replace("\n", "  \n"))

        def _render_payoff_dropdown(
            product: str,
            description: str,
            payoff_func,
            *,
            strike2_factor: float = 1.05,
            strike_lines_override: list[float] | Callable[[float, float], list[float]] | None = None,
            desc_key_suffix: str | None = None,
        ):
            with st.expander(f"🧭 Comprendre le payoff – {product}", expanded=False):
                K = float(st.session_state.get("common_strike_value", common_strike_value))
                K2 = float(st.session_state.get("common_strike_value", common_strike_value) * strike2_factor)
                xs = np.linspace(max(0.1, K * 0.5), K * 1.5, 100)
                ys = [payoff_func(x, K, K2) for x in xs]
                if strike_lines_override is None:
                    strikes_to_plot = [K]
                elif callable(strike_lines_override):
                    strikes_to_plot = strike_lines_override(K, K2)
                else:
                    strikes_to_plot = strike_lines_override
                fig = _payoff_plot(xs, ys, f"Payoff {product}", strike_lines=strikes_to_plot)
                st.plotly_chart(fig, width="stretch", key=_k(f"payoff_{product}"))
                st.caption(description)
                desc_key = desc_key_suffix or f"{product.lower().replace(' ', '_').replace('-', '_').replace('/', '_')}_graph"
                desc = _graph_desc(desc_key)
                if desc:
                    st.markdown(desc.replace("\n", "  \n"))

        # Helper to avoid duplicate Streamlit keys across Call/Put tabs.
        def _k(base: str) -> str:
            return f"{base}_{option_label.lower()}"
        # Helper for advanced structures, reused across dedicated tabs.
        def _render_structure_panel(structure_name: str):
            ks = structure_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            def kk(suffix: str) -> str:
                return _k(f"{ks}_{suffix}")

            st.subheader(structure_name)
            if structure_name == "Iron Condor":
                render_method_explainer(
                    "🪂 Construction de l’iron condor",
                    (
                        "- Quatre jambes européennes : long put bas, short put plus proche du spot, short call plus proche du spot, long call haut.\n"
                        "- Prix obtenu en sommant les primes BSM (positions achetées > positif ; positions vendues > négatif).\n"
                        "- Risque limité à l’écart entre ailes et jambes courtes, profit maximal égal au crédit net encaissé."
                    ),
                )
                wing_inner = st.number_input(
                    "Écart des strikes courts (autour de K commun)",
                    value=max(1.0, common_strike_value * 0.05),
                    min_value=0.1,
                    step=0.1,
                    key=kk("ic_wing_inner"),
                )
                wing_outer = st.number_input(
                    "Largeur des ailes (écart entre strike court et aile longue)",
                    value=max(1.0, common_strike_value * 0.05),
                    min_value=0.1,
                    step=0.1,
                    key=kk("ic_wing_outer"),
                )
                if st.button("Calculer l'Iron Condor (BSM)", key=kk("btn_iron_condor")):
                    K_mid = float(common_strike_value)
                    k_put_long = max(0.01, K_mid - (wing_inner + wing_outer))
                    k_put_short = max(0.01, K_mid - wing_inner)
                    k_call_short = K_mid + wing_inner
                    k_call_long = K_mid + wing_inner + wing_outer
                    try:
                        premium_put_long = _vanilla_price_with_dividend("put", common_spot_value, k_put_long, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        premium_put_short = _vanilla_price_with_dividend("put", common_spot_value, k_put_short, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        premium_call_short = _vanilla_price_with_dividend("call", common_spot_value, k_call_short, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        premium_call_long = _vanilla_price_with_dividend("call", common_spot_value, k_call_long, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        net_premium = premium_put_long - premium_put_short - premium_call_short + premium_call_long
                        credit = max(-net_premium, 0.0)
                        width = float(wing_outer)
                        max_profit = credit
                        max_loss = max(0.0, width - credit)
                        be_low = k_put_short - credit
                        be_high = k_call_short + credit
                        st.success(
                            f"Prime nette (achat>+ / vente>−) = {net_premium:.6f} "
                            f"{'(crédit)' if net_premium < 0 else '(débit)'}\n\n"
                            f"Max profit ≈ {max_profit:.6f} | Max perte ≈ {max_loss:.6f}\n"
                            f"Strikes : Put long {k_put_long:.2f} / Put short {k_put_short:.2f} / "
                            f"Call short {k_call_short:.2f} / Call long {k_call_long:.2f}\n"
                            f"Break-even bas ≈ {be_low:.4f} | Break-even haut ≈ {be_high:.4f}"
                        )
                    except Exception as exc:
                        st.error(f"Erreur lors du calcul Iron Condor : {exc}")
                return

            if structure_name == "Digital (cash-or-nothing)":
                payout = st.number_input("Payout", value=1.0, min_value=0.0, step=0.1, key=kk("payout"))
                if st.button("Pricer le digital", key=kk("btn")):
                    price = _digital_cash_or_nothing_price(
                        option_type=option_char,
                        S0=common_spot_value,
                        K=common_strike_value,
                        T=common_maturity_value,
                        r=common_rate_value,
                        dividend=float(d_common),
                        sigma=common_sigma_value,
                        payout=payout,
                    )
                    st.success(f"Prix digital ({option_label}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Digital (cash-or-nothing)",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_digital"),
                        spot=common_spot_value,
                        misc={
                            "payout": payout,
                            "style": "cash_or_nothing",
                        },
                    )
                return

            if structure_name == "Asset-or-nothing":
                if st.button("Pricer l'asset-or-nothing", key=kk("btn")):
                    price = _asset_or_nothing_price(
                        option_type=option_char,
                        S0=common_spot_value,
                        K=common_strike_value,
                        T=common_maturity_value,
                        r=common_rate_value,
                        dividend=float(d_common),
                        sigma=common_sigma_value,
                    )
                    st.success(f"Prix asset-or-nothing ({option_label}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Asset-or-nothing",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_asset_on"),
                        spot=common_spot_value,
                        misc={
                            "style": "asset_or_nothing",
                        },
                    )
                return

            if structure_name == "Forward-start option":
                t_start = st.number_input("T start (années)", value=float(common_maturity_value * 0.25), min_value=0.0, max_value=float(common_maturity_value * 0.9), step=0.05, key=kk("t_start"))
                k_fs = st.number_input("Facteur de strike (k)", value=1.0, min_value=0.1, step=0.05, key=kk("k"))
                n_paths_fs = st.number_input("Trajectoires MC", value=5000, min_value=500, step=500, key=kk("paths"))
                n_steps_fs = st.number_input("Pas de temps", value=200, min_value=20, step=10, key=kk("steps"))
                with st.expander(f"📈 Prix forward-start ({option_label})", expanded=False):
                    with st.spinner("Simulation Monte Carlo..."):
                        price = _forward_start_price_mc(
                            S0=common_spot_value,
                            r=common_rate_value,
                            q=float(d_common),
                            sigma=common_sigma_value,
                            T_start=t_start,
                            T_end=common_maturity_value,
                            k=k_fs,
                            n_paths=int(n_paths_fs),
                            n_steps=int(n_steps_fs),
                            option_type=option_char,
                        )
                    st.success(f"Prix forward-start ({option_label}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Forward-start",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_forward_start"),
                        spot=common_spot_value,
                        misc={
                            "T_start": float(t_start),
                            "k_factor": float(k_fs),
                            "n_paths": int(n_paths_fs),
                            "n_steps": int(n_steps_fs),
                        },
                    )
                return

            if structure_name == "Chooser option":
                t_choice = st.number_input("Date de choix (années)", value=float(common_maturity_value * 0.5), min_value=0.0, max_value=float(max(0.01, common_maturity_value)), step=0.05, key=kk("t"))
                if st.button("Pricer le chooser", key=kk("btn")):
                    price = _chooser_option_price(
                        S0=common_spot_value,
                        K=common_strike_value,
                        T=common_maturity_value,
                        t_choice=t_choice,
                        r=common_rate_value,
                        dividend=float(d_common),
                        sigma=common_sigma_value,
                    )
                    st.success(f"Prix chooser = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Chooser",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_chooser"),
                        spot=common_spot_value,
                        misc={
                            "t_choice": float(t_choice),
                        },
                    )
                return

            if structure_name == "Straddle":
                pre_price = st.session_state.get(_k("straddle_pre_price"))
                price = pre_price if pre_price is not None else (
                    _vanilla_price_with_dividend("call", common_spot_value, common_strike_value, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    + _vanilla_price_with_dividend("put", common_spot_value, common_strike_value, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                )
                st.success(f"Prix straddle (K={common_strike_value:.2f}) = {float(price):.6f}")
                render_add_to_dashboard_button(
                    product_label="Straddle",
                    option_char=option_char,
                    price_value=float(price),
                    strike=common_strike_value,
                    maturity=common_maturity_value,
                    key_prefix=kk("save_straddle"),
                    spot=common_spot_value,
                    legs=[
                        {"option_type": "call", "strike": common_strike_value},
                        {"option_type": "put", "strike": common_strike_value},
                    ],
                )
                return

            if structure_name == "Strangle":
                wing = st.number_input("Écart strike strangle", value=max(1.0, common_strike_value * 0.05), min_value=0.01, step=0.1, key=kk("wing"))
                if st.button("Pricer le strangle", key=kk("btn")):
                    k_put = max(0.01, common_strike_value - wing)
                    k_call = common_strike_value + wing
                    price = _vanilla_price_with_dividend("put", common_spot_value, k_put, common_maturity_value, common_rate_value, float(d_common), common_sigma_value) + _vanilla_price_with_dividend("call", common_spot_value, k_call, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix strangle (Put {k_put:.2f} / Call {k_call:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Strangle",
                        option_char=option_char,
                        price_value=price,
                        strike=k_put,
                        strike2=k_call,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_strangle"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "put", "strike": k_put},
                            {"option_type": "call", "strike": k_call},
                        ],
                        misc={
                            "strike_put": k_put,
                            "strike_call": k_call,
                            "wing": wing,
                        },
                    )
                return

            if structure_name == "Call spread":
                width = st.number_input("Écart strikes (vertical call spread)", value=max(1.0, common_strike_value * 0.05), min_value=0.01, step=0.1, key=kk("width"))
                if st.button("Pricer le call spread", key=kk("btn")):
                    k_long = common_strike_value
                    k_short = common_strike_value + width
                    price = _vanilla_price_with_dividend("call", common_spot_value, k_long, common_maturity_value, common_rate_value, float(d_common), common_sigma_value) - _vanilla_price_with_dividend("call", common_spot_value, k_short, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix call spread (long {k_long:.2f}, short {k_short:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Call spread",
                        option_char=option_char,
                        price_value=price,
                        strike=k_long,
                        strike2=k_short,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_call_spread"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "call", "strike": k_long, "side": "long"},
                            {"option_type": "call", "strike": k_short, "side": "short"},
                        ],
                        misc={"width": width},
                    )
                return

            if structure_name == "Put spread":
                width = st.number_input("Écart strikes (vertical put spread)", value=max(1.0, common_strike_value * 0.05), min_value=0.01, step=0.1, key=kk("width"))
                if st.button("Pricer le put spread", key=kk("btn")):
                    k_short = max(0.01, common_strike_value - width)
                    k_long = common_strike_value
                    price = _vanilla_price_with_dividend("put", common_spot_value, k_long, common_maturity_value, common_rate_value, float(d_common), common_sigma_value) - _vanilla_price_with_dividend("put", common_spot_value, k_short, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix put spread (long {k_long:.2f}, short {k_short:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Put spread",
                        option_char=option_char,
                        price_value=price,
                        strike=k_long,
                        strike2=k_short,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_put_spread"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "put", "strike": k_long, "side": "long"},
                            {"option_type": "put", "strike": k_short, "side": "short"},
                        ],
                        misc={"width": width},
                    )
                return

            if structure_name == "Butterfly":
                wing = st.number_input("Largeur des ailes (butterfly)", value=max(1.0, common_strike_value * 0.05), min_value=0.01, step=0.1, key=kk("wing"))
                if st.button("Pricer le butterfly", key=kk("btn")):
                    k1 = max(0.01, common_strike_value - wing)
                    k2 = common_strike_value
                    k3 = common_strike_value + wing
                    price = (
                        _vanilla_price_with_dividend("call", common_spot_value, k1, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        - 2 * _vanilla_price_with_dividend("call", common_spot_value, k2, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        + _vanilla_price_with_dividend("call", common_spot_value, k3, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    )
                    st.success(f"Prix butterfly (K1={k1:.2f}, K2={k2:.2f}, K3={k3:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Butterfly",
                        option_char=option_char,
                        price_value=price,
                        strike=k1,
                        strike2=k3,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_bfly"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "call", "strike": k1, "side": "long"},
                            {"option_type": "call", "strike": k2, "side": "short", "qty": 2},
                            {"option_type": "call", "strike": k3, "side": "long"},
                        ],
                        misc={"wing": wing},
                    )
                return

            if structure_name == "Condor":
                wing_inner = st.number_input("Écart strikes intérieurs", value=max(1.0, common_strike_value * 0.03), min_value=0.01, step=0.1, key=kk("inner"))
                wing_outer = st.number_input("Largeur d'aile condor", value=max(1.0, common_strike_value * 0.06), min_value=0.01, step=0.1, key=kk("outer"))
                if st.button("Pricer le condor", key=kk("btn")):
                    K1 = max(0.01, common_strike_value - (wing_inner + wing_outer))
                    K2 = max(0.01, common_strike_value - wing_inner)
                    K3 = common_strike_value + wing_inner
                    K4 = common_strike_value + wing_inner + wing_outer
                    price = (
                        _vanilla_price_with_dividend("call", common_spot_value, K1, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        - _vanilla_price_with_dividend("call", common_spot_value, K2, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        - _vanilla_price_with_dividend("call", common_spot_value, K3, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        + _vanilla_price_with_dividend("call", common_spot_value, K4, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    )
                    st.success(f"Prix condor (K1={K1:.2f}, K2={K2:.2f}, K3={K3:.2f}, K4={K4:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Condor",
                        option_char=option_char,
                        price_value=price,
                        strike=K1,
                        strike2=K4,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_condor"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "call", "strike": K1, "side": "long"},
                            {"option_type": "call", "strike": K2, "side": "short"},
                            {"option_type": "call", "strike": K3, "side": "short"},
                            {"option_type": "call", "strike": K4, "side": "long"},
                        ],
                        misc={
                            "wing_inner": wing_inner,
                            "wing_outer": wing_outer,
                        },
                    )
                return

            if structure_name == "Iron Butterfly":
                wing = st.number_input("Largeur des ailes (iron fly)", value=max(1.0, common_strike_value * 0.05), min_value=0.01, step=0.1, key=kk("wing"))
                if st.button("Pricer l'iron butterfly", key=kk("btn")):
                    K_mid = common_strike_value
                    K_low = max(0.01, K_mid - wing)
                    K_high = K_mid + wing
                    price = (
                        _vanilla_price_with_dividend("put", common_spot_value, K_low, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        - _vanilla_price_with_dividend("put", common_spot_value, K_mid, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        - _vanilla_price_with_dividend("call", common_spot_value, K_mid, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        + _vanilla_price_with_dividend("call", common_spot_value, K_high, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    )
                    st.success(f"Prix iron butterfly (K={K_low:.2f}/{K_mid:.2f}/{K_high:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Iron Butterfly",
                        option_char=option_char,
                        price_value=price,
                        strike=K_low,
                        strike2=K_high,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_iron_bfly"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "put", "strike": K_low, "side": "long"},
                            {"option_type": "put", "strike": K_mid, "side": "short"},
                            {"option_type": "call", "strike": K_mid, "side": "short"},
                            {"option_type": "call", "strike": K_high, "side": "long"},
                        ],
                        misc={
                            "wing": wing,
                            "k_mid": K_mid,
                        },
                    )
                return

            if structure_name == "Calendar spread":
                T_short = st.number_input("Maturité courte", value=float(max(0.1, common_maturity_value * 0.5)), min_value=0.01, key=kk("t_short"))
                T_long = st.number_input("Maturité longue", value=float(common_maturity_value), min_value=T_short + 0.01, key=kk("t_long"))
                opt_kind = st.selectbox("Type", ["call", "put"], key=kk("type"))
                if st.button("Pricer le calendar", key=kk("btn")):
                    long_leg = _vanilla_price_with_dividend(opt_kind, common_spot_value, common_strike_value, T_long, common_rate_value, float(d_common), common_sigma_value)
                    short_leg = _vanilla_price_with_dividend(opt_kind, common_spot_value, common_strike_value, T_short, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix calendar ({opt_kind}) = {long_leg - short_leg:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Calendar spread",
                        option_char=option_char,
                        price_value=long_leg - short_leg,
                        strike=common_strike_value,
                        maturity=T_long,
                        key_prefix=kk("save_calendar"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": opt_kind, "strike": common_strike_value, "side": "long", "tenor": T_long},
                            {"option_type": opt_kind, "strike": common_strike_value, "side": "short", "tenor": T_short},
                        ],
                        misc={
                            "T_short": float(T_short),
                            "T_long": float(T_long),
                            "opt_kind": opt_kind,
                        },
                    )
                return

            if structure_name == "Diagonal spread":
                T_short = st.number_input("Maturité courte", value=float(max(0.1, common_maturity_value * 0.5)), min_value=0.01, key=kk("t_short"))
                T_long = st.number_input("Maturité longue", value=float(common_maturity_value), min_value=T_short + 0.01, key=kk("t_long"))
                k_short = st.number_input("Strike court", value=float(round(common_spot_value)), min_value=0.01, key=kk("k_short"))
                k_long = st.number_input("Strike long", value=float(round(common_spot_value) * 1.05), min_value=0.01, key=kk("k_long"))
                opt_kind = st.selectbox("Type", ["call", "put"], key=kk("type"))
                if st.button("Pricer le diagonal", key=kk("btn")):
                    long_leg = _vanilla_price_with_dividend(opt_kind, common_spot_value, k_long, T_long, common_rate_value, float(d_common), common_sigma_value)
                    short_leg = _vanilla_price_with_dividend(opt_kind, common_spot_value, k_short, T_short, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix diagonal ({opt_kind}) = {long_leg - short_leg:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Diagonal spread",
                        option_char=option_char,
                        price_value=long_leg - short_leg,
                        strike=k_short,
                        strike2=k_long,
                        maturity=T_long,
                        key_prefix=kk("save_diagonal"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": opt_kind, "strike": k_long, "side": "long", "tenor": T_long},
                            {"option_type": opt_kind, "strike": k_short, "side": "short", "tenor": T_short},
                        ],
                        misc={
                            "T_short": float(T_short),
                            "T_long": float(T_long),
                            "k_short": float(k_short),
                            "k_long": float(k_long),
                            "opt_kind": opt_kind,
                        },
                    )
                return

            if structure_name == "Binary barrier (digital)":
                barrier_type = st.selectbox("Barrière", ["up", "down"], key=kk("barrier_type"))
                direction = st.selectbox("Knock", ["out", "in"], key=kk("direction"))
                payout = st.number_input("Payout", value=1.0, min_value=0.0, step=0.1, key=kk("payout"))
                base_level = common_spot_value * (1.1 if barrier_type == "up" else 0.9)
                barrier_level = st.number_input("Niveau barrière", value=float(base_level), min_value=0.0001, key=kk("level"))
                n_paths_bb = st.number_input("Trajectoires MC", value=5000, min_value=500, step=500, key=kk("paths"))
                n_steps_bb = st.number_input("Pas de temps", value=200, min_value=20, step=10, key=kk("steps"))
                if st.button("Pricer la binary barrière", key=kk("btn")):
                    price = _binary_barrier_mc(
                        option_type=option_char,
                        barrier_type=barrier_type,
                        direction=direction,
                        S0=common_spot_value,
                        K=common_strike_value,
                        barrier=barrier_level,
                        T=common_maturity_value,
                        r=common_rate_value,
                        dividend=float(d_common),
                        sigma=common_sigma_value,
                        payout=payout,
                        n_paths=int(n_paths_bb),
                        n_steps=int(n_steps_bb),
                    )
                    st.success(f"Prix binary barrière = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label=f"Binary barrier {barrier_type}-{direction}",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_binary_barrier"),
                        spot=common_spot_value,
                        legs=[{"option_type": option_char, "strike": common_strike_value, "barrier": barrier_level}],
                        misc={
                            "barrier_type": barrier_type,
                            "direction": direction,
                            "barrier_level": barrier_level,
                            "payout": payout,
                            "n_paths": int(n_paths_bb),
                            "n_steps": int(n_steps_bb),
                        },
                    )
                return

            if structure_name == "Asian géométrique":
                n_obs_geo = st.number_input("Observations", value=12, min_value=1, step=1, key=kk("obs"))
                with st.expander(f"📈 Prix Asian géométrique ({option_label})", expanded=False):
                    with st.spinner("Calcul en cours..."):
                        price = asian_geometric_closed_form(
                            spot=common_spot_value,
                            strike=common_strike_value,
                            rate=common_rate_value,
                            sigma=common_sigma_value,
                            maturity=common_maturity_value,
                            n_obs=int(n_obs_geo),
                            option_type="call" if option_char == "c" else "put",
                        )
                    st.success(f"Prix asian géométrique ({option_label}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Asian géométrique",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_asian_geo"),
                        spot=common_spot_value,
                        misc={
                            "method": "closed_form_geom",
                            "n_obs": int(n_obs_geo),
                            "sigma": float(common_sigma_value),
                            "r": float(common_rate_value),
                            "q": float(d_common),
                        },
                    )
                return

            if structure_name == "Lookback fixed (MC)":
                n_paths_lb = st.number_input("Trajectoires MC", value=5000, min_value=500, step=500, key=kk("paths"))
                n_steps_lb = st.number_input("Pas de temps", value=200, min_value=10, step=10, key=kk("steps"))
                with st.expander(f"📈 Prix Lookback fixed (MC) ({option_label})", expanded=False):
                    with st.spinner("Simulation Monte Carlo..."):
                        dt = common_maturity_value / n_steps_lb
                        drift = (common_rate_value - float(d_common) - 0.5 * common_sigma_value**2) * dt
                        diff = common_sigma_value * math.sqrt(dt)
                        disc = math.exp(-common_rate_value * common_maturity_value)
                        payoffs = []
                        for _ in range(int(n_paths_lb)):
                            s = common_spot_value
                            s_max = s_min = s
                            for _ in range(int(n_steps_lb)):
                                z = np.random.normal()
                                s *= math.exp(drift + diff * z)
                                s_max = max(s_max, s)
                                s_min = min(s_min, s)
                            if option_char == "c":
                                payoff = max(s_max - common_strike_value, 0.0)
                            else:
                                payoff = max(common_strike_value - s_min, 0.0)
                            payoffs.append(payoff)
                        price = disc * float(np.mean(payoffs)) if payoffs else 0.0
                    st.success(f"Prix lookback fixed ({option_label}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Lookback fixed",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_lookback"),
                        spot=common_spot_value,
                        misc={
                            "n_paths": int(n_paths_lb),
                            "n_steps": int(n_steps_lb),
                        },
                    )
                return

            if structure_name == "Cliquet / Ratchet (MC)":
                n_periods = st.number_input("Nombre de périodes", value=12, min_value=1, step=1, key=kk("periods"))
                cap = st.number_input("Cap par période", value=0.05, min_value=-1.0, step=0.01, key=kk("cap"))
                floor = st.number_input("Floor par période", value=0.0, min_value=-1.0, step=0.01, key=kk("floor"))
                n_paths_cliq = st.number_input("Trajectoires MC", value=3000, min_value=500, step=500, key=kk("paths"))
                with st.expander("📈 Prix cliquet / ratchet (MC)", expanded=False):
                    with st.spinner("Simulation Monte Carlo..."):
                        price = _cliquet_mc(
                            S0=common_spot_value,
                            r=common_rate_value,
                            q=float(d_common),
                            sigma=common_sigma_value,
                            T=common_maturity_value,
                            n_periods=int(n_periods),
                            cap=float(cap),
                            floor=float(floor),
                            n_paths=int(n_paths_cliq),
                        )
                    st.success(f"Prix cliquet/ratchet ≈ {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Cliquet / Ratchet",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_cliquet"),
                        spot=common_spot_value,
                        misc={
                            "n_periods": int(n_periods),
                            "cap": float(cap),
                            "floor": float(floor),
                            "n_paths": int(n_paths_cliq),
                        },
                    )
                return

            if structure_name == "Quanto option":
                sigma_fx = st.number_input("Vol FX", value=0.1, min_value=0.0, step=0.01, key=kk("sigma_fx"))
                rho_qt = st.number_input("Corrélation S/FX", value=0.0, min_value=-1.0, max_value=1.0, step=0.05, key=kk("rho"))
                opt_kind = st.selectbox("Type", ["call", "put"], key=kk("type"))
                if st.button("Pricer la quanto", key=kk("btn")):
                    price = _quanto_vanilla_price(
                        option_type=opt_kind,
                        S0=common_spot_value,
                        K=common_strike_value,
                        T=common_maturity_value,
                        r_dom=common_rate_value,
                        q_for=float(d_common),
                        sigma_asset=common_sigma_value,
                        sigma_fx=sigma_fx,
                        rho=rho_qt,
                    )
                    st.success(f"Prix quanto ({opt_kind}) = {price:.6f}")
                return

            if structure_name == "Rainbow option":
                S0_b = st.number_input("Spot actif B", value=float(common_spot_value), min_value=0.01, key=kk("S0b"))
                sigma_b = st.number_input("Vol B", value=float(common_sigma_value), min_value=0.0001, key=kk("sigb"))
                rho_ab = st.number_input("Corrélation A/B", value=0.2, min_value=-1.0, max_value=1.0, step=0.05, key=kk("rho"))
                payoff_on = st.selectbox("Sous-jacent du payoff", ["max", "min"], key=kk("payoff"))
                opt_kind = st.selectbox("Type", ["call", "put"], key=kk("type"))
                n_paths_r = st.number_input("Trajectoires MC", value=4000, min_value=500, step=500, key=kk("paths"))
                n_steps_r = st.number_input("Pas de temps", value=150, min_value=10, step=10, key=kk("steps"))
                if st.button("Pricer le rainbow", key=kk("btn")):
                    price = _rainbow_two_asset_mc(
                        payoff_on=payoff_on,
                        S0_a=common_spot_value,
                        S0_b=S0_b,
                        sigma_a=common_sigma_value,
                        sigma_b=sigma_b,
                        rho=rho_ab,
                        K=common_strike_value,
                        T=common_maturity_value,
                        r=common_rate_value,
                        q_a=float(d_common),
                        q_b=float(d_common),
                        n_paths=int(n_paths_r),
                        n_steps=int(n_steps_r),
                        option_type=opt_kind,
                    )
                    st.success(f"Prix rainbow ({payoff_on}) = {price:.6f}")
                return

        # Helper to render the relevant heatmap for the current Call/Put tab.
        def _render_heatmaps_for_current_option(label: str, call_matrix, put_matrix, x_vals, y_vals):
            if option_char == "c":
                st.write(f"Heatmap Call ({label})")
                _render_heatmap(call_matrix, x_vals, y_vals, f"Call ({label})")
            else:
                st.write(f"Heatmap Put ({label})")
                _render_heatmap(put_matrix, x_vals, y_vals, f"Put ({label})")
        # Heston Carr–Madan pricer helpers
        def _heston_params_from_state() -> HestonParams:
            return HestonParams(
                torch.tensor(float(st.session_state.get("heston_kappa_common", 2.0)), device=HES_DEVICE),
                torch.tensor(float(st.session_state.get("heston_theta_common", 0.04)), device=HES_DEVICE),
                torch.tensor(float(st.session_state.get("heston_eta_common", 0.5)), device=HES_DEVICE),
                torch.tensor(float(st.session_state.get("heston_rho_common", -0.7)), device=HES_DEVICE),
                torch.tensor(float(st.session_state.get("heston_v0_common", 0.04)), device=HES_DEVICE),
            )

        def _carr_madan_price(S0: float, K: float, T: float, r: float, q: float, opt_char: str, params: HestonParams) -> float:
            call_price = float(carr_madan_call_torch(S0, r, q, T, params, K))
            if opt_char == "c":
                return call_price
            # Put via parité call-put
            return float(call_price - S0 * math.exp(-q * T) + K * math.exp(-r * T))
        (
            tab_grp_vanilla,
            tab_grp_path,
            tab_grp_barrier,
            tab_grp_spreads,
            tab_grp_calendar,
            tab_grp_exotics,
            tab_grp_basket,
        ) = st.tabs(
            [
                "Vanilla / Early exercise",
                "Path-dependent",
                "Barrières",
                "Spreads & Wings",
                "Calendriers",
                "Exotiques",
                "Basket",
            ]
        )

        with tab_grp_vanilla:
            tab_heston, tab_american, tab_bermudan = st.tabs(["Européenne par Heston", "Américaine", "Bermuda"])

        with tab_grp_path:
            (
                tab_asian,
                tab_asian_geo,
                tab_lookback,
                tab_lookback_fixed,
                tab_forward_start,
                tab_cliquet,
            ) = st.tabs(["Asian", "Asian géométrique", "Lookback", "Lookback fixed", "Forward-start", "Cliquet / Ratchet"])

        with tab_grp_barrier:
            st.subheader("Barrières (vanilla / binaire) – vue Notebook")
            s0_ref = float(common_spot_value)
            hist_tkr = (
                st.session_state.get("common_underlying")
                or st.session_state.get("tkr_common")
                or st.session_state.get("heston_cboe_ticker")
                or "SPY"
            ).strip().upper()
            spy_close = load_close_series_for_ticker(hist_tkr, fallback_value=s0_ref)

            strike_anchor_bar = float(common_spot_value)
            col1, col2, col3 = st.columns(3)
            with col1:
                strike_b = st.slider(
                    "Strike",
                    min_value=0.6 * strike_anchor_bar,
                    max_value=1.4 * strike_anchor_bar,
                    value=float(round(strike_anchor_bar)),
                    step=0.5,
                    key=_k("barrier_all_strike"),
                )
                barrier_b = st.slider(
                    "Barrière",
                    min_value=0.5 * strike_anchor_bar,
                    max_value=1.8 * strike_anchor_bar,
                    value=float(round(strike_anchor_bar)),
                    step=0.5,
                    key=_k("barrier_all_level"),
                )
                call_put_b = st.selectbox("Type", ["call", "put"], key=_k("barrier_all_type"))
            with col2:
                direction_b = st.selectbox("Direction", ["up", "down"], key=_k("barrier_all_dir"))
                knock_b = st.selectbox("Knock", ["out", "in"], key=_k("barrier_all_knock"))
                binary_b = st.checkbox("Binaire ?", value=False, key=_k("barrier_all_binary"))
                payout_b = st.slider(
                    "Payout (si binaire)",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    key=_k("barrier_all_payout"),
                )
            with col3:
                r_b = float(common_rate_value)
                T_b = st.slider("T (années)", min_value=0.05, max_value=2.0, value=common_maturity_value, step=0.05, key=_k("barrier_all_T"))
            iv_bar = _get_cached_iv_for(strike_b, T_b, call_put_b)
            sigma_b = float(iv_bar) if iv_bar is not None and np.isfinite(iv_bar) and iv_bar > 0 else float(common_sigma_value)
            if iv_bar is not None and np.isfinite(iv_bar) and iv_bar > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_bar:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            with st.spinner("Calcul..."):
                view_dyn = view_barrier(
                    s0_ref,
                    strike_b,
                    barrier_b,
                    direction=direction_b,
                    knock=knock_b,
                    option_type=call_put_b,
                    payout=payout_b,
                    binary=binary_b,
                    r=r_b,
                    q=0.0,
                    sigma=sigma_b,
                    T=T_b,
                )
                premium = float(view_dyn.get("premium", 0.0))
                s_grid = view_dyn["s_grid"]
                payoff_grid = view_dyn["payoff"]
                pnl_grid = view_dyn["pnl"]
                payoff_s0 = float(np.interp(s0_ref, s_grid, payoff_grid))
                pnl_s0 = payoff_s0 - premium

            fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
            ax_ts.plot(spy_close.index, spy_close.values, label=f"{hist_tkr} close (1y)")
            ax_ts.axhline(strike_b, color="gray", linestyle="--", label=f"Strike = {strike_b:.2f}")
            ax_ts.axhline(barrier_b, color="firebrick", linestyle=":", label=f"Barriere = {barrier_b:.2f}")
            ax_ts.set_ylabel("Prix")
            ax_ts.set_title(f"Clôtures {hist_tkr} (strike/barrière)")
            ax_ts.legend(loc="best")
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts, clear_figure=True)

            fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
            ax_pay.plot(s_grid, payoff_grid, label="Payoff")
            ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax_pay.axvline(barrier_b, color="firebrick", linestyle=":", label=f"Barriere = {barrier_b:.2f}")
            ax_pay.axvline(strike_b, color="gray", linestyle="--", label=f"K = {strike_b:.2f}")
            ax_pay.axvline(s0_ref, color="crimson", linestyle="-.", label=f"S0 = {s0_ref:.2f}")
            ax_pay.axhline(0, color="black", linewidth=0.8)
            ax_pay.legend(loc="best")
            ax_pay.set_xlabel("Spot")
            ax_pay.set_ylabel("Payoff / P&L")
            ax_pay.set_title(f"Barrier {'binaire' if binary_b else 'vanilla'} ({direction_b} / {knock_b})")
            st.pyplot(fig_pay, clear_figure=True)

            price = float(premium)
            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${abs(price):.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_b or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("barrier_all_qty"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("barrier_all_side"))

            if st.button("Ajouter au dashboard", key=_k("barrier_all_add"), type="primary"):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": call_put_b,
                    "product_type": f"Barrier {'binary' if binary_b else 'vanilla'}",
                    "type": f"Barrier {'binary' if binary_b else 'vanilla'}",
                    "strike": float(strike_b),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(s0_ref),
                    "maturity_years": float(T_b),
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "barrier": float(barrier_b),
                        "barrier_type": direction_b,
                        "direction": direction_b,
                        "knock": knock_b,
                        "binary": bool(binary_b),
                        "payout": float(payout_b),
                        "spot_at_pricing": float(s0_ref),
                        "sigma_used": float(sigma_b),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(f"Barrière enregistrée dans le dashboard (id: {option_id}).")
                except Exception as exc:
                    st.error(f"Erreur lors de l'enregistrement : {exc}")
                    st.stop()

        with tab_grp_spreads:
            (
                tab_straddle,
                tab_strangle,
                tab_call_spread,
                tab_put_spread,
                tab_butterfly,
                tab_condor,
                tab_iron_condor,
                tab_iron_bfly,
            ) = st.tabs(["Straddle", "Strangle", "Call spread", "Put spread", "Butterfly", "Condor", "Iron Condor", "Iron Butterfly"])

        with tab_grp_calendar:
            tab_calendar, tab_diagonal = st.tabs(["Calendar spread", "Diagonal spread"])

        with tab_grp_exotics:
            tab_digital, tab_asset_on, tab_chooser, tab_quanto, tab_rainbow = st.tabs(["Digital", "Asset-or-nothing", "Chooser", "Quanto", "Rainbow"])

        with tab_grp_basket:
            opt_label_basket, opt_char_basket = _choose_option_select("opt_choice_basket", option_char)
            option_label, option_char = opt_label_basket, opt_char_basket
            ui_basket_surface(
                spot_common=common_spot_value,
                maturity_common=common_maturity_value,
                rate_common=common_rate_value,
                strike_common=common_strike_value,
                key_prefix=_k("basket"),
            )
        with tab_european:
            st.info("Onglet Européenne (BSM) masqué pour le moment.")

        with tab_heston:
            st.header("Option européenne – Heston")
            _render_option_text("Option européenne (Heston)", "european_graph_heston")
            opt_label_local, opt_char_local = _choose_option_select("opt_choice_heston_tab", option_char)
            option_label, option_char = opt_label_local, opt_char_local
            params_heston = _heston_params_from_state()
            calib_T_target = st.session_state.get("heston_calib_T_target")
            S0_h = float(common_spot_value)
            K_h_common = float(common_strike_value)
            r_h = float(common_rate_value)
            d_h = float(d_common)
            K_slider_h = st.slider(
                "K (strike – visualisation Heston)",
                min_value=max(0.1, S0_h - 20.0),
                max_value=S0_h + 20.0,
                value=float(round(S0_h)),
                step=max(0.01, K_h_common * 0.01),
                key=_k("eu_k_slider_heston"),
            )
            t_band_h = float(st.session_state.get("heston_cboe_calib_band", 0.4))
            t_center_h = float(calib_T_target) if calib_T_target is not None else float(common_maturity_value)
            t_min = max(0.01, t_center_h - t_band_h)
            t_max = max(t_min + 0.001, t_center_h + t_band_h)
            T_slider_h = st.slider(
                "T (années – visualisation Heston)",
                min_value=float(t_min),
                max_value=float(t_max),
                value=float(min(max(t_center_h, t_min), t_max)),
                step=0.01,
                key=_k("eu_T_slider_heston"),
            )
            st.session_state["eu_T_slider_val"] = T_slider_h

            opt_type_h = "call" if opt_char_local == "c" else "put"
            iv_h = _get_cached_iv_for(K_slider_h, T_slider_h, opt_type_h)
            if iv_h is not None and np.isfinite(iv_h) and iv_h > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_h:.4f}")
            else:
                st.caption("IV non trouvée dans le cache.")
            price_heston_display: float | None = None
            if st.session_state.get("carr_madan_calibrated", False):
                try:
                    price_heston_display = price_heston_carr_madan(
                        S0=float(S0_h),
                        K=float(K_slider_h),
                        T=float(T_slider_h),
                        r=float(r_h),
                        q=float(d_h),
                        kappa=float(st.session_state.get("heston_kappa_common", 0.0)),
                        theta=float(st.session_state.get("heston_theta_common", 0.0)),
                        sigma=float(st.session_state.get("heston_eta_common", 0.0)),
                        rho=float(st.session_state.get("heston_rho_common", 0.0)),
                        v0=float(st.session_state.get("heston_v0_common", 0.0)),
                        option_type=opt_type_h,
                    )
                    st.session_state["eu_price_heston"] = price_heston_display
                except Exception as exc:
                    st.error(f"Erreur Heston (Carr–Madan) : {exc}")

            premium_h = price_heston_display
            s_grid = np.linspace(max(0.1, K_slider_h * 0.4), K_slider_h * 1.6, 200)
            payoff_grid = np.maximum(s_grid - K_slider_h, 0.0) if opt_type_h == "call" else np.maximum(K_slider_h - s_grid, 0.0)
            pnl_grid = payoff_grid - premium_h if premium_h is not None else None
            payoff_s0 = float(np.interp(S0_h, s_grid, payoff_grid))
            pnl_s0 = payoff_s0 - premium_h if premium_h is not None else None
            st.caption(
                f"Paramètres utilisés pour le prix Heston : "
                f"S0={S0_h:.4f}, K={float(K_slider_h):.4f}, "
                f"T={float(T_slider_h):.4f}, r={r_h:.4f}, "
                f"q={float(d_h):.4f}"
            )
            fig_h, ax_h = plt.subplots(figsize=(7, 4))
            ax_h.plot(s_grid, payoff_grid, label="Payoff")
            if pnl_grid is not None and premium_h is not None:
                ax_h.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax_h.axvline(K_slider_h, color="gray", linestyle="--", label=f"K = {K_slider_h:.2f}")
            ax_h.axvline(S0_h, color="crimson", linestyle="-.", label=f"S0 = {S0_h:.2f}")
            ax_h.axhline(0, color="black", linewidth=0.8)
            ax_h.set_xlabel("Spot")
            ax_h.set_ylabel("Payoff / P&L")
            ax_h.set_title(f"Vanilla {opt_type_h.capitalize()} (Heston)")
            ax_h.legend(loc="best")
            st.pyplot(fig_h, clear_figure=True)

            misc_heston = {
                "heston_params": {
                    "kappa": float(st.session_state.get("heston_kappa_common", 2.0)),
                    "theta": float(st.session_state.get("heston_theta_common", 0.04)),
                    "eta": float(st.session_state.get("heston_eta_common", 0.5)),
                    "rho": float(st.session_state.get("heston_rho_common", -0.7)),
                    "v0": float(st.session_state.get("heston_v0_common", 0.04)),
                }
            }
            st.subheader("Heston (référence)")
            st.divider()
            heston_params_table = {
                "kappa": float(st.session_state.get("heston_kappa_common", 2.0)),
                "theta": float(st.session_state.get("heston_theta_common", 0.04)),
                "eta": float(st.session_state.get("heston_eta_common", 0.5)),
                "rho": float(st.session_state.get("heston_rho_common", -0.7)),
                "v0": float(st.session_state.get("heston_v0_common", 0.04)),
            }
            if st.session_state.get("carr_madan_calibrated", False):
                st.dataframe(pd.Series(heston_params_table, name="Paramètre").to_frame(), use_container_width=True, hide_index=False)
            else:
                st.info("Calibre Heston pour afficher les paramètres.")
            st.caption(
                f"Paramètres de calibration utilisés : Ticker={st.session_state.get('heston_cboe_ticker', '')} | "
                f"S0={S0_h:.4f} | K slider={K_slider_h:.4f} | T slider={T_slider_h:.4f} | "
                f"r={r_h:.4f} | q={d_h:.4f} | Bande T=±{t_band_h:.2f}"
            )
            st.subheader("🎯 Calibration NN Carr-Madan")
            span_mc = float(st.session_state.get("heatmap_span_value", 20.0))
            calls_df = st.session_state.get("heston_calls_df")
            puts_df = st.session_state.get("heston_puts_df")
            S0_ref = st.session_state.get("heston_S0_ref", common_spot_value)
            rf_rate = float(st.session_state.get("common_rate", 0.02))
            div_yield = float(st.session_state.get("common_dividend", 0.0))
            if calls_df is None or puts_df is None or getattr(calls_df, "empty", True):
                st.warning("Charge d’abord les données CBOE (Refresh) pour activer la calibration Heston.")
                calib_T_target = None
            else:
                st.caption(f"Device calibration NN : **{HES_DEVICE_LABEL}**")
                calib_T_target = st.session_state.get("heston_calib_T_target")
                col_nn, col_modes = st.columns(2)
                with col_nn:
                    calib_T_band_default = float(st.session_state.get("heston_cboe_calib_band", 0.02))

                    unique_T = sorted(calls_df["T"].round(2).unique().tolist())
                    if unique_T:
                        if calib_T_target is None:
                            target_guess = st.session_state.get("eu_T_slider_val", common_maturity_value)
                            idx_default = int(np.argmin(np.abs(np.array(unique_T) - float(target_guess))))
                        else:
                            try:
                                idx_default = unique_T.index(calib_T_target)
                            except ValueError:
                                idx_default = 0

                        idx_default = max(0, min(idx_default, len(unique_T) - 1))
                        calib_T_target = st.selectbox(
                            "Maturité T cible pour la calibration (Time to Maturity)",
                            unique_T,
                            index=idx_default,
                            format_func=lambda x: f"{x:.2f}",
                            key=_k("heston_cboe_calib_target"),
                            help="Maturité autour de laquelle la calibration Heston est centrée.",
                        )
                        st.session_state.heston_calib_T_target = calib_T_target
                    else:
                        st.warning("Pas de maturités disponibles dans les données CBOE.")
                        calib_T_target = None

                    calib_T_band = st.number_input(
                        "Largeur bande T (±)",
                        value=calib_T_band_default,
                        min_value=0.0001,
                        max_value=0.5,
                        step=0.01,
                        format="%.4f",
                        key=_k("heston_cboe_calib_band"),
                        help="Largeur de la bande de maturités autour de la cible utilisée pour la calibration.",
                    )
                    st.session_state["heston_cboe_calib_band"] = calib_T_band

                with col_modes:
                    st.subheader("⚙️ Modes de calibration NN")
                    mode = st.radio(
                        "Choisir un mode",
                        ["Rapide", "Bonne", "Excellente", "Custom"],
                        index=3,
                        horizontal=True,
                        key=_k("heston_cboe_mode"),
                        help="Choisit un compromis entre vitesse de calibration et précision de l’ajustement.",
                    )
                    if mode == "Rapide":
                        max_iters = 300
                        learning_rate = 0.01
                    elif mode == "Bonne":
                        max_iters = 1000
                        learning_rate = 0.005
                    elif mode == "Excellente":
                        max_iters = 2000
                        learning_rate = 0.001
                    else:
                        max_iters = st.number_input(
                            "Itérations NN (custom)",
                            min_value=50,
                            value=500,
                            step=50,
                            key=_k("heston_cboe_max_iters_custom"),
                        )
                        learning_rate = st.number_input(
                            "Learning rate (custom)",
                            min_value=1e-5,
                            max_value=0.1,
                            value=0.005,
                            step=0.0005,
                            format="%.5f",
                            key=_k("heston_cboe_lr_custom"),
                        )
                    st.markdown(
                        f"**Itérations NN** : `{max_iters}`  \n"
                        f"**Learning rate** : `{learning_rate}`"
                    )

                calib_band_range = (
                    max(MIN_IV_MATURITY, calib_T_target - calib_T_band),
                    calib_T_target + calib_T_band,
                ) if calib_T_target is not None else None

                run_disabled = (
                    bool(st.session_state.get("heston_calibrating", False))
                    or calls_df is None
                    or getattr(calls_df, "empty", True)
                )
                run_button = st.button(
                    "🚀 Lancer l'analyse",
                    type="primary",
                    width="stretch",
                    key=_k("heston_cboe_run"),
                    disabled=run_disabled,
                )
                st.divider()

                if run_button:
                    st.session_state["heston_calibrating"] = True
                    if calls_df is None or getattr(calls_df, "empty", True):
                        st.error("Pas de données CBOE en cache. Charge-les via Refresh (pas de téléchargement auto en calibration).")
                        st.stop()
                    if calib_band_range is None or calib_T_target is None:
                        st.error("Veuillez choisir une maturité T cible après avoir chargé les données.")
                        st.stop()

                    df_selected = calls_df if opt_type_h == "call" else puts_df
                    if df_selected is None or df_selected.empty:
                        df_selected = calls_df
                    if df_selected is None or df_selected.empty:
                        st.error("Pas de points pour la calibration (calls/puts).")
                        st.stop()
                    df_selected = df_selected.copy()
                    if opt_type_h == "put" and "C_mkt" not in df_selected.columns and "P_mkt" in df_selected.columns:
                        df_selected["C_mkt"] = df_selected["P_mkt"]

                    calib_slice = df_selected[
                        (df_selected["T"].round(2).between(*calib_band_range))
                        & (df_selected["K"].between(S0_ref - span_mc, S0_ref + span_mc))
                        & (df_selected["C_mkt"] > 0.05)
                        & (df_selected["iv_market"] > 0)
                    ]
                    if len(calib_slice) < 5:
                        calib_slice = df_selected.copy()

                    st.info(f"📡 Données CBOE chargées pour {st.session_state.get('heston_cboe_ticker', '')} (cache)")
                    st.success(f"{len(calls_df)} calls, {len(puts_df)} puts | S0 ≈ {S0_ref:.2f} | Calibration sur {'calls' if opt_type_h == 'call' else 'puts'}")
                    st.write(f"Maturité T cible pour la calibration : {calib_T_target:.2f} ans")

                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    loss_log: list[float] = []
                    start_time = time.time()

                    def progress_cb(current: int, total: int, loss_val: float) -> None:
                        ratio = current / total if total else 0.0
                        progress_bar.progress(min(1.0, max(0.0, ratio)))
                        status_text.text(f"⏳ Iter {current}/{total} | Loss = {loss_val:.6f}")
                        loss_log.append(loss_val)

                    try:
                        params_cm = calibrate_heston_nn(
                            calib_slice,
                            r=rf_rate,
                            q=div_yield,
                            max_iters=int(max_iters),
                            lr=learning_rate,
                            spot_override=S0_ref,
                            progress_callback=progress_cb,
                        )
                        elapsed = time.time() - start_time
                        if elapsed < 20:
                            status_text.text("Finalisation calibration... (stabilisation de l'affichage)")
                            time.sleep(20 - elapsed)
                        progress_bar.empty()
                        status_text.empty()
                        params_dict = {
                            "kappa": float(params_cm.kappa.detach()),
                            "theta": float(params_cm.theta.detach()),
                            "sigma": float(params_cm.sigma.detach()),
                            "rho": float(params_cm.rho.detach()),
                            "v0": float(params_cm.v0.detach()),
                        }
                        st.session_state["heston_kappa_common"] = params_dict["kappa"]
                        st.session_state["heston_theta_common"] = params_dict["theta"]
                        st.session_state["heston_eta_common"] = params_dict["sigma"]
                        st.session_state["heston_rho_common"] = params_dict["rho"]
                        st.session_state["heston_v0_common"] = params_dict["v0"]
                        st.session_state["carr_madan_calibrated"] = True
                        # Persist Heston params dans un JSON par ticker
                        ticker_local = (
                            st.session_state.get("heston_cboe_ticker")
                            or st.session_state.get("tkr_common")
                            or st.session_state.get("common_underlying")
                            or ""
                        ).strip().upper()
                        params_to_save = params_dict | {
                            "ticker": ticker_local,
                            "S0_ref": float(S0_ref),
                            "rf_rate": float(rf_rate),
                            "dividend_yield": float(div_yield),
                            "calib_band_T": float(calib_T_band),
                            "calib_target_T": float(calib_T_target) if calib_T_target is not None else None,
                            "span_mc": float(span_mc),
                        }
                        save_heston_params_to_json(ticker_local, params_to_save)
                        st.success("✓ Calibration terminée")
                        st.dataframe(pd.Series(params_dict, name="Paramètre").to_frame())
                        st.balloons()
                        st.success("🎉 Analyse terminée")
                        params_heston = _heston_params_from_state()
                    except Exception as exc:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Erreur : {exc}")
                        import traceback

                        st.code(traceback.format_exc())
                    finally:
                        st.session_state["heston_calibrating"] = False
                        st.rerun()

                if st.session_state.get("heston_calibrating", False):
                    st.info("🧠 Calibration Heston en cours... (le reste de l'onglet réapparaîtra après).")
                    st.stop()

            if not st.session_state.get("carr_madan_calibrated", False):
                st.error("🔴 Calibre Heston (bouton « Lancer l'analyse ») pour afficher la suite.")
                st.stop()

            render_inputs_explainer(
                "🔧 Paramètres utilisés – Heston européen",
                (
                    "- **\"K (strike)\"** : strike de référence saisi dans la barre latérale.\n"
                    "- **\"T (maturité, années)\"** : maturité commune pour les surfaces.\n"
                    "- **\"Taux sans risque r\"** et **\"Dividende continu d\"** : paramètres de taux.\n"
                    "- **\"Ticker (sous-jacent)\"** : code CBOE utilisé pour la collecte des options.\n"
                    "- **\"Largeur bande T (±)\" et \"Maturité T cible\" : bornes de calibration.\n"
                ),
            )
            st.caption("Pricing direct avec Carr–Madan (Heston calibré).")
            price_cm = price_heston_display
            if price_cm is None and st.session_state.get("carr_madan_calibrated", False):
                try:
                    with st.spinner("Calcul Heston Carr–Madan..."):
                        price_cm = price_heston_carr_madan(
                            S0=float(S0_h),
                            K=float(K_slider_h),
                            T=float(T_slider_h),
                            r=float(r_h),
                            q=float(d_h),
                            kappa=float(st.session_state.get("heston_kappa_common", 0.0)),
                            theta=float(st.session_state.get("heston_theta_common", 0.0)),
                            sigma=float(st.session_state.get("heston_eta_common", 0.0)),
                            rho=float(st.session_state.get("heston_rho_common", 0.0)),
                            v0=float(st.session_state.get("heston_v0_common", 0.0)),
                            option_type=opt_type_h,
                        )
                    st.session_state["eu_price_heston"] = price_cm
                except Exception as exc:
                    st.error(f"Erreur Carr–Madan : {exc}")
                    price_cm = None
            if st.session_state.get("carr_madan_calibrated", False):
                try:
                    heatmap_status = st.info("Calcul surface IV Heston…")
                    # Centre la grille autour des sliders K/T actuels
                    k_vals = _heatmap_axis(float(S0_h), 15.0)
                    t_vals = _heatmap_axis(float(T_slider_h), t_band_h)

                    call_matrix = np.zeros((len(t_vals), len(k_vals)), dtype=float)
                    put_matrix = np.zeros_like(call_matrix)
                    for i_t, t_val in enumerate(t_vals):
                        for j_k, k_val in enumerate(k_vals):
                            call_matrix[i_t, j_k] = _carr_madan_price(
                                S0=float(S0_h),
                                K=float(k_val),
                                T=float(t_val),
                                r=float(r_h),
                                q=float(d_h),
                                opt_char="c",
                                params=params_heston,
                            )
                            put_matrix[i_t, j_k] = _carr_madan_price(
                                S0=float(S0_h),
                                K=float(k_val),
                                T=float(t_val),
                                r=float(r_h),
                                q=float(d_h),
                                opt_char="p",
                                params=params_heston,
                            )

                    k_grid, t_grid = np.meshgrid(k_vals, t_vals)
                    # Utilise toujours la matrice cohérente avec le type Call/Put sélectionné dans l'onglet Heston.
                    price_grid = call_matrix if opt_char_local == "c" else put_matrix
                    iv_grid = np.full_like(price_grid, np.nan, dtype=float)
                    for i_t, t_val in enumerate(t_vals):
                        for j_k, k_val in enumerate(k_vals):
                            iv_grid[i_t, j_k] = implied_vol_option(
                                price=float(price_grid[i_t, j_k]),
                                S=float(S0_h),
                                K=float(k_val),
                                T=float(t_val),
                                r=float(r_h),
                                option_type="call" if opt_char_local == "c" else "put",
                            )
                    heatmap_status.empty()
                    try:
                        iv_masked = np.nan_to_num(iv_grid, nan=0.0, posinf=0.0, neginf=0.0)
                        fig_iv = go.Figure(
                            data=[
                                go.Surface(
                                    x=k_vals,
                                    y=t_vals,
                                    z=iv_masked,
                                    colorscale="Viridis",
                                    showscale=True,
                                )
                            ]
                        )
                        fig_iv.update_layout(
                            scene=dict(
                                xaxis_title="Strike K",
                                yaxis_title="Maturité T",
                                zaxis_title="IV",
                                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                                xaxis=dict(autorange=True),
                                yaxis=dict(autorange=True),
                                zaxis=dict(autorange=True),
                            ),
                            scene_dragmode="orbit",
                            height=500,
                            margin=dict(l=0, r=0, b=0, t=0),
                            hovermode=False,
                        )
                        st.plotly_chart(
                            fig_iv,
                            use_container_width=True,
                            key=_k("heston_iv_surface"),
                            config={
                                "scrollZoom": False,
                                "modeBarButtonsToRemove": ["zoom3d", "pan3d", "resetCameraDefault3d", "resetCameraLastSave3d"],
                                "displaylogo": False,
                            },
                        )
                        # Surface de prix Heston (mêmes axes, Z = prix)
                        price_masked = np.nan_to_num(price_grid, nan=0.0, posinf=0.0, neginf=0.0)
                        fig_px = go.Figure(
                            data=[
                                go.Surface(
                                    x=k_vals,
                                    y=t_vals,
                                    z=price_masked,
                                    colorscale="Plasma",
                                    showscale=True,
                                )
                            ]
                        )
                        fig_px.update_layout(
                            scene=dict(
                                xaxis_title="Strike K",
                                yaxis_title="Maturité T",
                                zaxis_title="Prix Heston",
                                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                                xaxis=dict(autorange=True),
                                yaxis=dict(autorange=True),
                                zaxis=dict(autorange=True),
                            ),
                            scene_dragmode="orbit",
                            height=500,
                            margin=dict(l=0, r=0, b=0, t=0),
                            hovermode=False,
                        )
                        st.plotly_chart(
                            fig_px,
                            use_container_width=True,
                            key=_k("heston_price_surface"),
                            config={
                                "scrollZoom": False,
                                "modeBarButtonsToRemove": ["zoom3d", "pan3d", "resetCameraDefault3d", "resetCameraLastSave3d"],
                                "displaylogo": False,
                            },
                        )
                    except Exception as _surf_exc:
                        st.warning(f"Impossible d'afficher la surface IV : {_surf_exc}")
                except Exception as exc:
                    st.error(f"Erreur calcul heatmap / surface IV Heston : {exc}")
            else:
                st.info("Calibre Heston pour afficher la heatmap de prix et la surface IV.")

            if st.session_state.get("carr_madan_calibrated", False):
                final_price = price_cm if price_cm is not None else None
                render_add_to_dashboard_button(
                    product_label="Vanilla (Heston CM)",
                    option_char=opt_char_local,
                    price_value=final_price,
                    strike=common_strike_value,
                    maturity=common_maturity_value,
                    key_prefix=_k("save_heston_cm_final"),
                    spot=common_spot_value,
                    misc=misc_heston,
                )

            st.divider()

        with tab_american:
            opt_label_local_am, opt_char_local_am = _choose_option_select("opt_choice_am_tab", option_char)
            option_label, option_char = opt_label_local_am, opt_char_local_am
            st.header("Option américaine")
            cpflag_am = option_label
            cpflag_am_char = option_char
            _render_option_text("Option américaine", "american_payoff")

            st.subheader("Arbre binomial CRR")
            render_method_explainer(
                "🌳 Arbre CRR",
                (
                    "- Discrétisation de l’horizon en `n_tree_am` pas.\n"
                    "- Recursion backward avec exercice optimal.\n"
                ),
            )
            k_min_am = max(0.1, S0_common - 20.0)
            k_max_am = S0_common + 20.0
            k_default_am = float(round(S0_common))
            k_default_am = float(min(max(k_default_am, k_min_am), k_max_am))
            t_default_am = _pick_default_T_near_one(k_default_am)
            t_min_am = 0.05
            t_max_am = float(max(0.5, t_default_am + 1.0, T_common + 0.5))
            t_default_am = float(min(max(t_default_am, t_min_am), t_max_am))
            K_slider_am = st.slider(
                "K (strike – Américain)",
                min_value=k_min_am,
                max_value=k_max_am,
                value=k_default_am,
                step=max(0.01, float(S0_common) * 0.01),
                key=_k("am_k_slider"),
            )
            T_slider_am = st.slider(
                "T (années – Américain)",
                min_value=float(t_min_am),
                max_value=float(t_max_am),
                value=float(t_default_am),
                step=0.01,
                key=_k("am_T_slider"),
            )
            iv_am = _get_cached_iv_for(K_slider_am, T_slider_am, "call" if option_char == "c" else "put")
            sigma_am = float(iv_am) if iv_am is not None and np.isfinite(iv_am) and iv_am > 0 else sigma_common
            if iv_am is not None and np.isfinite(iv_am) and iv_am > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_am:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
            n_tree_am = st.number_input(
                "Nombre de pas de l'arbre",
                value=10,
                min_value=5,
                key=_k("n_tree_am"),
                help="Nombre de pas de temps utilisés dans l’arbre binomial CRR.",
            )
            option_am_crr = Option(s0=S0_common, T=T_slider_am, K=K_slider_am, call=cpflag_am == "Call")
            int_n_tree = int(n_tree_am)
            if int_n_tree > 10:
                st.info("L'affichage peut devenir difficile à lire pour un nombre de pas supérieur à 10.")
            with st.spinner("Calcul du prix CRR"):
                try:
                    option_am_single = Option(
                        s0=S0_common,
                        T=T_slider_am,
                        K=K_slider_am,
                        call=(cpflag_am == 'Call'),
                    )
                    price_crr_single = crr_pricing(
                        r=r_common,
                        sigma=sigma_am,
                        option=option_am_single,
                        n=int_n_tree,
                    )
                    st.success(f"Prix américain CRR ({cpflag_am}) ≈ {price_crr_single:.6f} (avec {int_n_tree} pas)")
                    render_add_to_dashboard_button(
                        product_label="American (CRR)",
                        option_char=option_char,
                        price_value=price_crr_single,
                        strike=K_slider_am,
                        maturity=T_slider_am,
                        key_prefix=_k("save_am_crr"),
                        spot=S0_common,
                    )
                except Exception as exc:
                    st.error(f"Erreur CRR : {exc}")
            with st.spinner("Construction de l'arbre CRR"):
                spot_tree, value_tree = _build_crr_tree(
                    option=option_am_crr, r=r_common, sigma=sigma_common, n_steps=int_n_tree
                )
            st.write("**Représentation graphique**")
            fig_tree = _plot_crr_tree(spot_tree, value_tree)
            st.pyplot(fig_tree)
            plt.close(fig_tree)

            with st.spinner("Calcul de la heatmap CRR"):
                call_heatmap_crr, put_heatmap_crr = _compute_american_crr_heatmaps(
                heatmap_spot_values,
                heatmap_strike_values,
                T_slider_am,
                r_common,
                sigma_am,
                int_n_tree,
            )
            _render_heatmaps_for_current_option(
                "CRR",
                call_heatmap_crr,
                put_heatmap_crr,
                heatmap_spot_values,
                heatmap_strike_values,
            )
            st.caption(
                f"Paramètres utilisés pour CRR : "
                f"S0={S0_common:.4f}, K={K_slider_am:.4f}, T={T_slider_am:.4f}, "
                f"r={r_common:.4f}, σ={sigma_am:.4f}, n={int_n_tree}"
            )

        with tab_bermudan:
            opt_label_local_bmd, opt_char_local_bmd = _choose_option_select("opt_choice_bmd_tab", option_char)
            option_label, option_char = opt_label_local_bmd, opt_char_local_bmd
            st.header("Option bermudéenne")
            _render_option_text("Option bermudéenne", "bermuda_payoff")
            k_min_bmd = max(0.1, S0_common - 20.0)
            k_max_bmd = S0_common + 20.0
            k_default_bmd = float(round(S0_common))
            k_default_bmd = float(min(max(k_default_bmd, k_min_bmd), k_max_bmd))
            t_default_bmd = _pick_default_T_near_one(k_default_bmd)
            t_min_bmd = 0.05
            t_max_bmd = float(max(0.5, t_default_bmd + 1.0, T_common + 0.5))
            t_default_bmd = float(min(max(t_default_bmd, t_min_bmd), t_max_bmd))
            K_slider_bmd = st.slider(
                "K (strike – Bermudan)",
                min_value=k_min_bmd,
                max_value=k_max_bmd,
                value=k_default_bmd,
                step=max(0.01, float(S0_common) * 0.01),
                key=_k("bmd_k_slider"),
            )
            T_slider_bmd = st.slider(
                "T (années – Bermudan)",
                min_value=float(t_min_bmd),
                max_value=float(t_max_bmd),
                value=float(t_default_bmd),
                step=0.01,
                key=_k("bmd_T_slider"),
            )
            iv_bmd = _get_cached_iv_for(K_slider_bmd, T_slider_bmd, "call" if option_char == "c" else "put")
            sigma_bmd = float(iv_bmd) if iv_bmd is not None and np.isfinite(iv_bmd) and iv_bmd > 0 else sigma_common
            sigma_source_msg = (
                f"σ implicite (cache) ≈ {sigma_bmd:.4f}" if sigma_bmd != sigma_common else f"σ par défaut ≈ {sigma_common:.4f}"
            )
            if iv_bmd is not None and np.isfinite(iv_bmd) and iv_bmd > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_bmd:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            st.subheader("Longstaff–Schwartz (GBM)")
            render_method_explainer(
                "🧮 L-S Bermudan (GBM)",
                (
                    "- Simulation GBM sous la mesure risque-neutre avec `N_paths` trajectoires et `M` pas de temps.\n"
                    "- Régression backward sur `n_ex_dates` dates d’exercice pour décider de l’exercice optimal.\n"
                    "- Polynôme de degré `degree` pour approximer la valeur de continuation."
                ),
            )
            col_mc1, col_mc2 = st.columns(2)
            with col_mc1:
                n_paths_bmd = st.number_input(
                    "Trajectoires Monte Carlo",
                    value=5000,
                    min_value=500,
                    step=500,
                    key=_k("bmd_paths"),
                )
                n_steps_bmd = st.number_input(
                    "Pas de temps",
                    value=50,
                    min_value=10,
                    step=5,
                    key=_k("bmd_steps"),
                )
            with col_mc2:
                n_ex_dates_bmd = st.number_input(
                    "Dates d’exercice Bermudan",
                    value=6,
                    min_value=2,
                    max_value=80,
                    step=1,
                    key=_k("bmd_exdates"),
                )
                degree_bmd = st.selectbox(
                    "Degré polynôme (régression)",
                    [2, 3, 4, 5],
                    index=1,
                    key=_k("bmd_degree"),
                )

            with st.spinner("Calcul Bermudan LSMC..."):
                try:
                    price_bmd_mc = price_bermudan_lsmc(
                        S0=S0_common,
                        K=K_slider_bmd,
                        T=T_slider_bmd,
                        r=r_common,
                        q=d_common,
                        sigma=sigma_bmd,
                        cpflag=option_char,
                        M=int(n_steps_bmd),
                        N_paths=int(n_paths_bmd),
                        degree=int(degree_bmd),
                        n_ex_dates=int(n_ex_dates_bmd),
                        seed=12345,
                    )
                    st.success(f"Prix Bermudan L-S ({option_label}) = {price_bmd_mc:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Bermudan (LSMC)",
                        option_char=option_char,
                        price_value=price_bmd_mc,
                        strike=K_slider_bmd,
                        maturity=T_slider_bmd,
                        key_prefix=_k("save_bmd_lsmc"),
                        spot=S0_common,
                        misc={
                            "method": "lsmc_gbm",
                            "n_paths": int(n_paths_bmd),
                            "n_steps": int(n_steps_bmd),
                            "n_ex_dates": int(n_ex_dates_bmd),
                            "degree": int(degree_bmd),
                            "sigma_used": float(sigma_bmd),
                        },
                    )
                except Exception as exc:
                    st.error(f"Erreur Bermudan LSMC : {exc}")
            st.divider()

            st.subheader("Crank–Nicolson (PDE)")
            render_method_explainer(
                "🧮 Crank–Nicolson Bermudan",
                (
                    "- Résolution de la PDE Black–Scholes en log(S) avec dates d’exercice discrètes.\n"
                    "- Les grecs (Δ, Γ, Θ) sont obtenus par bumping autour des paramètres courants.\n"
                    "- Ajuste la finesse de la grille via `n_points` (espace) et `n_time`."
                ),
            )
            col_cn1, col_cn2 = st.columns(2)
            with col_cn1:
                n_spatial_bmd = st.number_input(
                    "Points spatiaux (grille logS)",
                    value=200,
                    min_value=80,
                    step=20,
                    key=_k("bmd_cn_nspace"),
                )
                n_time_bmd = st.number_input(
                    "Points temporels",
                    value=220,
                    min_value=80,
                    step=20,
                    key=_k("bmd_cn_time"),
                )
            with col_cn2:
                n_ex_dates_cn = st.number_input(
                    "Dates d’exercice (PDE)",
                    value=int(n_ex_dates_bmd),
                    min_value=2,
                    max_value=120,
                    step=1,
                    key=_k("bmd_cn_exdates"),
                )
                exercise_step_bmd = st.number_input(
                    "Exercise step (0 = désactivé)",
                    value=0,
                    min_value=0,
                    max_value=365,
                    step=1,
                    key=_k("bmd_cn_step"),
                )
                exercise_step_bmd = None if exercise_step_bmd <= 0 else int(exercise_step_bmd)

            try:
                cn_kwargs = {
                    "Typeflag": "Bmd",
                    "cpflag": option_char,
                    "S0": S0_common,
                    "K": K_slider_bmd,
                    "T": T_slider_bmd,
                    "vol": sigma_bmd,
                    "r": r_common,
                    "d": d_common,
                    "n_spatial": int(n_spatial_bmd),
                    "n_time": int(n_time_bmd),
                }
                if exercise_step_bmd is not None:
                    cn_kwargs["exercise_step"] = exercise_step_bmd
                else:
                    cn_kwargs["n_exercise_dates"] = int(n_ex_dates_cn)
                solver_bmd = CrankNicolsonBS(**cn_kwargs)
                price_bmd_cn, delta_bmd, gamma_bmd, theta_bmd = solver_bmd.CN_option_info()
                st.success(f"Prix Bermudan PDE ({option_label}) = {price_bmd_cn:.6f}")
                st.caption(f"Δ={delta_bmd:.4f} | Γ={gamma_bmd:.4f} | Θ={theta_bmd:.4f}")
                render_add_to_dashboard_button(
                    product_label="Bermudan (PDE)",
                    option_char=option_char,
                    price_value=price_bmd_cn,
                    strike=K_slider_bmd,
                    maturity=T_slider_bmd,
                    key_prefix=_k("save_bmd_cn"),
                    spot=S0_common,
                    misc={
                        "method": "crank_nicolson",
                        "n_spatial": int(n_spatial_bmd),
                        "n_time": int(n_time_bmd),
                        "n_ex_dates": None if exercise_step_bmd is not None else int(n_ex_dates_cn),
                        "exercise_step": exercise_step_bmd,
                        "sigma_used": float(sigma_bmd),
                    },
                )
            except Exception as exc:
                st.error(f"Erreur Bermudan PDE : {exc}")

        # Données de clôture 1 an pour les onglets path-dependent (valeur de référence S0)
        s0_path = float(common_spot_value)
        hist_tkr = (
            st.session_state.get("common_underlying")
            or st.session_state.get("tkr_common")
            or st.session_state.get("heston_cboe_ticker")
            or "SPY"
        ).strip().upper()
        spy_close_path = load_close_series_for_ticker(hist_tkr, fallback_value=s0_path)

        with tab_lookback:
            st.subheader("Lookback floating – vue Notebook")
            col1, col2 = st.columns(2)
            with col1:
                option_type_lb = st.selectbox("Type", ["call", "put"], key=_k("lb_type"))
                min_lb = st.slider(
                    "Min path",
                    min_value=0.8 * s0_path,
                    max_value=1.0 * s0_path,
                    value=float(round(s0_path)),
                    step=0.5,
                    key=_k("lb_min"),
                )
                max_lb = st.slider(
                    "Max path",
                    min_value=1.0 * s0_path,
                    max_value=1.2 * s0_path,
                    value=float(round(s0_path)),
                    step=0.5,
                    key=_k("lb_max"),
                )
                strike_lb = st.slider(
                    "Strike (référence)",
                    min_value=0.8 * s0_path,
                    max_value=1.2 * s0_path,
                    value=float(round(s0_path)),
                    step=0.5,
                    key=_k("lb_k"),
                )
            with col2:
                span_lb = st.slider("Span payoff (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.05, key=_k("lb_span"))
                T_lb = st.slider("T (années)", min_value=0.05, max_value=2.0, value=float(common_maturity_value), step=0.05, key=_k("lb_T"))

            view_dyn = view_lookback(
                s0_path,
                min_lb,
                max_lb,
                option_type=option_type_lb,
                span=span_lb,
                k_ref=float(strike_lb),
                T=float(T_lb),
            )
            premium = float(view_dyn.get("premium", 0.0))
            price_display = abs(premium)
            price_display = abs(premium)
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]
            payoff_s0 = float(np.interp(s0_path, s_grid, payoff_grid))
            pnl_s0 = payoff_s0 - premium

            fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
            ax_ts.plot(spy_close_path.index, spy_close_path.values, label=f"{hist_tkr} close (1y)")
            ax_ts.axhline(min_lb, color="teal", linestyle=":", label=f"Min = {min_lb:.2f}")
            ax_ts.axhline(max_lb, color="gray", linestyle="--", label=f"Max = {max_lb:.2f}")
            ax_ts.axhline(s0_path, color="firebrick", linestyle="-.", label=f"S0 = {s0_path:.2f}")
            ax_ts.set_ylabel("Prix")
            ax_ts.set_title(f"Clôtures {hist_tkr} (Lookback)")
            ax_ts.legend(loc="best")
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts, clear_figure=True)

            fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
            ax_pay.plot(s_grid, payoff_grid, label="Payoff")
            ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax_pay.axvline(s0_path, color="crimson", linestyle="-.", label=f"S0 = {s0_path:.2f}")
            ax_pay.axhline(0, color="black", linewidth=0.8)
            ax_pay.legend(loc="best")
            ax_pay.set_xlabel("Spot")
            ax_pay.set_ylabel("Payoff / P&L")
            ax_pay.set_title(f"Lookback floating ({option_type_lb})")
            st.pyplot(fig_pay, clear_figure=True)

            price = float(premium)
            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((common_maturity_value or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("lb_qty"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("lb_side"))

            if st.button("Ajouter au dashboard", key=_k("lb_add"), type="primary"):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": option_type_lb,
                    "product_type": "Lookback floating",
                    "type": "Lookback floating",
                    "strike": float(strike_lb),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(s0_path),
                    "maturity_years": float(T_lb),
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "min_path": float(min_lb),
                        "max_path": float(max_lb),
                        "strike_ref": float(strike_lb),
                        "span": float(span_lb),
                        "T": float(T_lb),
                        "spot_at_pricing": float(s0_path),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(f"Lookback floating ajouté au dashboard (id: {option_id}).")
                except Exception as exc:
                    st.error(f"Erreur lors de l'enregistrement : {exc}")


        with tab_asian:
            st.subheader("Asian arithmétique – vue Notebook")
            avg_close = float(spy_close_path.mean()) if spy_close_path is not None else s0_path
            col1, col2 = st.columns(2)
            with col1:
                option_type_as = st.selectbox("Type", ["call", "put"], key=_k("asian_type"))
                strike_as = st.slider(
                    "Strike",
                    min_value=0.6 * s0_path,
                    max_value=1.4 * s0_path,
                    value=float(round(s0_path)),
                    step=0.5,
                    key=_k("asian_k"),
                )
                avg_as = st.slider(
                    "Moyenne (ref)",
                    min_value=0.5 * s0_path,
                    max_value=1.5 * s0_path,
                    value=float(round(s0_path)),
                    step=0.5,
                    key=_k("asian_avg"),
                )
            with col2:
                # r est récupéré du cache commun (common_rate)
                r_as = float(common_rate_value)
                T_as = st.slider("T (années)", min_value=0.05, max_value=2.0, value=common_maturity_value, step=0.05, key=_k("asian_T"))
            iv_as = _get_cached_iv_for(strike_as, T_as, option_type_as)
            sigma_as = float(iv_as) if iv_as is not None and np.isfinite(iv_as) and iv_as > 0 else float(common_sigma_value)
            if iv_as is not None and np.isfinite(iv_as) and iv_as > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_as:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            view_dyn = view_asian_arith(
                s0_path,
                strike_as,
                avg_as,
                option_type=option_type_as,
                r=r_as,
                q=0.0,
                sigma=sigma_as,
                T=T_as,
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]
            payoff_s0 = float(np.interp(s0_path, s_grid, payoff_grid))
            pnl_s0 = payoff_s0 - premium

            fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
            ax_ts.plot(spy_close_path.index, spy_close_path.values, label=f"{hist_tkr} close (1y)")
            ax_ts.axhline(avg_as, color="purple", linestyle=":", label=f"Moyenne = {avg_as:.2f}")
            ax_ts.axhline(strike_as, color="gray", linestyle="--", label=f"K = {strike_as:.2f}")
            ax_ts.set_ylabel("Prix")
            ax_ts.set_title(f"Clôtures {hist_tkr} (Asian arith)")
            ax_ts.legend(loc="best")
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts, clear_figure=True)

            fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
            ax_pay.plot(s_grid, payoff_grid, label="Payoff")
            ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax_pay.axvline(strike_as, color="gray", linestyle="--", label=f"K = {strike_as:.2f}")
            ax_pay.axvline(s0_path, color="crimson", linestyle="-.", label=f"S0 = {s0_path:.2f}")
            ax_pay.axhline(0, color="black", linewidth=0.8)
            ax_pay.legend(loc="best")
            ax_pay.set_xlabel("Spot")
            ax_pay.set_ylabel("Payoff / P&L")
            ax_pay.set_title(f"Asian arithmétique ({option_type_as})")
            st.pyplot(fig_pay, clear_figure=True)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${premium:.6f}")
            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_as or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("asian_qty"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("asian_side"))

            if st.button("Ajouter au dashboard", key=_k("asian_add"), type="primary"):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": option_type_as,
                    "product_type": "Asian arithmétique",
                    "type": "Asian arithmétique",
                    "strike": float(strike_as),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": float(premium),
                    "side": side,
                    "S0": float(s0_path),
                    "maturity_years": float(T_as),
                    "T_0": today.isoformat(),
                    "price": float(premium),
                    "misc": {
                        "avg_ref": float(avg_as),
                        "sigma_used": float(sigma_as),
                        "r": float(r_as),
                        "spot_at_pricing": float(s0_path),
                    },
                }
                try:
                    option_id = add_option_to_dashboard(payload)
                    st.success(f"Asian arithmétique ajoutée (id: {option_id})")
                except Exception as exc:
                    st.error(f"Erreur lors de l'enregistrement : {exc}")

        with tab_iron_condor:
            k_center = st.slider(
                "Strike central (iron condor)",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=float(common_spot_value),
                step=0.5,
                key=_k("iron_condor_center"),
            )
            inner = st.slider(
                "Écart strikes courts",
                min_value=max(0.1, 0.02 * float(common_spot_value)),
                max_value=max(1.0, 0.5 * float(common_spot_value)),
                value=max(0.5, 0.05 * float(common_spot_value)),
                step=0.1,
                key=_k("iron_condor_inner"),
            )
            outer_raw = st.slider(
                "Écart strikes ailes",
                min_value=max(0.2, 0.03 * float(common_spot_value)),
                max_value=max(1.5, 0.7 * float(common_spot_value)),
                value=max(0.9, 0.1 * float(common_spot_value)),
                step=0.1,
                key=_k("iron_condor_outer"),
            )
            outer = max(outer_raw, inner + max(0.1, 0.01 * float(common_spot_value)))

            k_put_long = max(0.01, k_center - outer)
            k_put_short = k_center - inner
            k_call_short = k_center + inner
            k_call_long = k_center + outer
            T_iron_condor = st.slider(
                "T (années)",
                min_value=0.05,
                max_value=2.0,
                value=float(common_maturity_value),
                step=0.05,
                key=_k("iron_condor_T"),
            )
            ivs_ic = [
                _get_cached_iv_for(k_put_long, T_iron_condor, "put"),
                _get_cached_iv_for(k_put_short, T_iron_condor, "put"),
                _get_cached_iv_for(k_call_short, T_iron_condor, "call"),
                _get_cached_iv_for(k_call_long, T_iron_condor, "call"),
            ]
            sigma_put_long_ic = float(ivs_ic[0]) if ivs_ic[0] is not None and np.isfinite(ivs_ic[0]) and ivs_ic[0] > 0 else float(common_sigma_value)
            sigma_put_short_ic = float(ivs_ic[1]) if ivs_ic[1] is not None and np.isfinite(ivs_ic[1]) and ivs_ic[1] > 0 else float(common_sigma_value)
            sigma_call_short_ic = float(ivs_ic[2]) if ivs_ic[2] is not None and np.isfinite(ivs_ic[2]) and ivs_ic[2] > 0 else float(common_sigma_value)
            sigma_call_long_ic = float(ivs_ic[3]) if ivs_ic[3] is not None and np.isfinite(ivs_ic[3]) and ivs_ic[3] > 0 else float(common_sigma_value)
            iv_vals_ic = [v for v in ivs_ic if v is not None and np.isfinite(v) and v > 0]
            if iv_vals_ic:
                iv_txt = " | ".join(
                    f"K={k:.2f}: {v:.4f}" if v is not None and np.isfinite(v) and v > 0 else f"K={k:.2f}: n/a"
                    for k, v in zip([k_put_long, k_put_short, k_call_short, k_call_long], ivs_ic)
                )
                st.caption(f"IV récupérées (cache) ≈ {iv_txt}")
                st.caption(
                    "σ utilisées : "
                    f"put long {sigma_put_long_ic:.4f} | put short {sigma_put_short_ic:.4f} | "
                    f"call short {sigma_call_short_ic:.4f} | call long {sigma_call_long_ic:.4f}"
                )
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            view_dyn = view_iron_condor(
                float(common_spot_value),
                k_put_long,
                k_put_short,
                k_call_short,
                k_call_long,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                sigma_put_long=float(sigma_put_long_ic),
                sigma_put_short=float(sigma_put_short_ic),
                sigma_call_short=float(sigma_call_short_ic),
                sigma_call_long=float(sigma_call_long_ic),
                T=float(T_iron_condor),
            )
            premium = float(view_dyn.get("premium", 0.0))
            price_display = abs(premium)

            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Iron Condor (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("iron_condor_pre_price")] = premium
            price = float(price_display)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price_display:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_iron_condor or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("iron_condor_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("iron_condor_side_inline"))
            st.caption(
                f"K put long: {k_put_long:.4f} | K put short: {k_put_short:.4f} | "
                f"K call short: {k_call_short:.4f} | K call long: {k_call_long:.4f}"
            )
            st.caption(f"T (maturité, années): {float(T_iron_condor):.4f}")

            if st.button("Ajouter au dashboard", key=_k("iron_condor_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": "call" if option_char.lower() == "c" else "put",
                    "product_type": "Iron Condor",
                    "type": "Iron Condor",
                    "strike": float(k_put_long),
                    "strike2": float(k_call_long),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price_display,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_iron_condor),
                    "legs": [
                        {"option_type": "put", "strike": float(k_put_long)},
                        {"option_type": "put", "strike": float(k_put_short)},
                        {"option_type": "call", "strike": float(k_call_short)},
                        {"option_type": "call", "strike": float(k_call_long)},
                    ],
                    "T_0": today.isoformat(),
                    "price": price_display,
                    "misc": {
                        "structure": "Iron Condor",
                        "legs": [
                            {"option_type": "put", "strike": float(k_put_long)},
                            {"option_type": "put", "strike": float(k_put_short)},
                            {"option_type": "call", "strike": float(k_call_short)},
                            {"option_type": "call", "strike": float(k_call_long)},
                        ],
                        "premium_raw": float(premium),
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_put_long_used": float(sigma_put_long_ic),
                        "sigma_put_short_used": float(sigma_put_short_ic),
                        "sigma_call_short_used": float(sigma_call_short_ic),
                        "sigma_call_long_used": float(sigma_call_long_ic),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_iron_condor),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Iron condor ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_digital:
            opt_label_dig, opt_char_dig = _choose_option_select("opt_choice_digital", option_char)
            option_label, option_char = opt_label_dig, opt_char_dig
            strike = st.slider(
                "Strike",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=float(common_spot_value),
                step=0.5,
                key=_k("digital_k"),
            )
            T_dig = st.slider("T (années)", min_value=0.05, max_value=2.0, value=float(common_maturity_value), step=0.05, key=_k("digital_T"))
            opt_type = "call" if opt_char_dig == "c" else "put"
            iv_dig = _get_cached_iv_for(strike, T_dig, opt_type)
            sigma_dig = float(iv_dig) if iv_dig is not None and np.isfinite(iv_dig) and iv_dig > 0 else float(common_sigma_value)
            if iv_dig is not None and np.isfinite(iv_dig) and iv_dig > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_dig:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
            view_dyn = view_digital(
                float(common_spot_value),
                strike,
                T=float(T_dig),
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(sigma_dig),
                option_type=opt_type,
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Digital (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("digital_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_dig or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("digital_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("digital_side_inline"))
            st.caption(f"K: {strike:.4f}")
            st.caption(f"T (maturité, années): {float(T_dig):.4f}")

            if st.button("Ajouter au dashboard", key=_k("digital_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": opt_char_dig,
                    "product_type": "Digital",
                    "type": "Digital",
                    "strike": float(strike),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_dig),
                    "legs": [
                        {"option_type": opt_char_rain, "strike": float(strike), "payout": 1.0, "digital": True},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Digital",
                        "strike": float(strike),
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_used": float(sigma_dig),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_dig),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Digital ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_asset_on:
            opt_label_aon, opt_char_aon = _choose_option_select("opt_choice_asset_on", option_char)
            option_label, option_char = opt_label_aon, opt_char_aon
            strike = st.slider(
                "Strike",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=float(common_spot_value),
                step=0.5,
                key=_k("asset_on_k"),
            )
            T_aon = st.slider("T (années)", min_value=0.05, max_value=2.0, value=float(common_maturity_value), step=0.05, key=_k("asset_on_T"))
            opt_type = "call" if opt_char_aon == "c" else "put"
            iv_aon = _get_cached_iv_for(strike, T_aon, opt_type)
            sigma_aon = float(iv_aon) if iv_aon is not None and np.isfinite(iv_aon) and iv_aon > 0 else float(common_sigma_value)
            if iv_aon is not None and np.isfinite(iv_aon) and iv_aon > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_aon:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
            view_dyn = view_asset_or_nothing(
                float(common_spot_value),
                strike,
                T=float(T_aon),
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(sigma_aon),
                option_type=opt_type,
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Asset-or-nothing (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("asset_on_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_aon or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("asset_on_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("asset_on_side_inline"))
            st.caption(f"K: {strike:.4f}")
            st.caption(f"T (maturité, années): {float(T_aon):.4f}")

            if st.button("Ajouter au dashboard", key=_k("asset_on_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": opt_char_aon,
                    "product_type": "Asset-or-nothing",
                    "type": "Asset-or-nothing",
                    "strike": float(strike),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_aon),
                    "legs": [
                        {"option_type": opt_char_aon, "strike": float(strike), "asset_or_nothing": True},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Asset-or-nothing",
                        "strike": float(strike),
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_used": float(sigma_aon),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_aon),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Asset-or-nothing ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_forward_start:
            fs_label, fs_char = _choose_option_select("opt_choice_forward_start", option_char)
            option_label, option_char = fs_label, fs_char
            spot_start = st.slider(
                "Spot de départ (S_start)",
                min_value=0.5 * float(s0_path),
                max_value=1.5 * float(s0_path),
                value=float(round(s0_path)),
                step=0.5,
                key=_k("forward_start_s_start"),
            )
            strike_fs = st.slider(
                "Strike (K = m × S_start)",
                min_value=0.8 * float(s0_path),
                max_value=1.2 * float(s0_path),
                value=float(round(s0_path)),
                step=0.5,
                key=_k("forward_start_k"),
            )
            m_factor = float(strike_fs / spot_start) if spot_start else 1.0
            T_fs = st.slider(
                "T (années)",
                min_value=0.05,
                max_value=2.0,
                value=float(common_maturity_value),
                step=0.05,
                key=_k("forward_start_T"),
            )
            opt_type = "call" if option_char.lower() == "c" else "put"
            strike_forward = m_factor * spot_start
            fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
            ax_ts.plot(spy_close_path.index, spy_close_path.values, label=f"{hist_tkr} close (1y)")
            ax_ts.axhline(spot_start, color="gray", linestyle="--", label=f"S_start = {spot_start:.2f}")
            ax_ts.axhline(strike_forward, color="firebrick", linestyle=":", label=f"K = m*S_start = {strike_forward:.2f}")
            ax_ts.set_ylabel("Prix")
            ax_ts.set_title(f"Clôtures {hist_tkr} (Forward-start)")
            ax_ts.legend(loc="best")
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts, clear_figure=True)
            view_dyn = view_forward_start(
                s0_path,
                spot_start,
                m=m_factor,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                T=float(T_fs),
                option_type=opt_type,
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(s0_path), color="crimson", linestyle="-.", label=f"S_0 = {float(s0_path):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Forward-start (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("forward_start_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_fs or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("forward_start_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("forward_start_side_inline"))
            st.caption(f"S_start: {spot_start:.4f} | m: {m_factor:.4f}")
            st.caption(f"T (maturité, années): {float(T_fs):.4f}")

            if st.button("Ajouter au dashboard", key=_k("forward_start_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": opt_char_rain,
                    "product_type": "Forward-start",
                    "type": "Forward-start",
                    "strike": float(m_factor * spot_start),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_fs),
                    "legs": [
                        {"option_type": opt_char_rain, "strike": float(m_factor * spot_start), "forward_start": True, "S_start": float(spot_start), "m": float(m_factor)},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Forward-start",
                        "spot_start": float(spot_start),
                        "m_factor": float(m_factor),
                        "T": float(T_fs),
                        "spot_at_pricing": float(common_spot_value),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Forward-start ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_chooser:
            opt_label_chooser, opt_char_chooser = _choose_option_select("opt_choice_chooser", option_char)
            option_label, option_char = opt_label_chooser, opt_char_chooser
            strike = st.slider(
                "Strike",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=float(common_spot_value),
                step=0.5,
                key=_k("chooser_k"),
            )
            T_chooser = st.slider("T (années)", min_value=0.05, max_value=2.0, value=float(common_maturity_value), step=0.05, key=_k("chooser_T"))
            iv_chooser = _get_cached_iv_for(strike, T_chooser, "call" if opt_char_chooser == "c" else "put")
            sigma_chooser = float(iv_chooser) if iv_chooser is not None and np.isfinite(iv_chooser) and iv_chooser > 0 else float(common_sigma_value)
            if iv_chooser is not None and np.isfinite(iv_chooser) and iv_chooser > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_chooser:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
            view_dyn = view_chooser(
                float(common_spot_value),
                strike,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(sigma_chooser),
                T=float(T_chooser),
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Chooser (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("chooser_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_chooser or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("chooser_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("chooser_side_inline"))
            st.caption(f"K: {strike:.4f}")
            st.caption(f"T (maturité, années): {float(T_chooser):.4f}")

            if st.button("Ajouter au dashboard", key=_k("chooser_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": opt_char_chooser,
                    "product_type": "Chooser",
                    "type": "Chooser",
                    "strike": float(strike),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_chooser),
                    "legs": [
                        {"option_type": "chooser", "strike": float(strike)},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Chooser",
                        "strike": float(strike),
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_used": float(sigma_chooser),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_chooser),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Chooser ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_straddle:
            strike_slider = st.slider(
                "Strike",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=float(common_spot_value),
                step=0.5,
                key=_k("straddle_k"),
            )
            T_straddle = st.slider(
                "T (années)",
                min_value=0.05,
                max_value=2.0,
                value=float(common_maturity_value),
                step=0.05,
                key=_k("straddle_T"),
            )
            iv_straddle = _get_cached_iv_for(strike_slider, T_straddle, "call")
            iv_straddle_put = _get_cached_iv_for(strike_slider, T_straddle, "put")
            sigma_call_straddle = float(iv_straddle) if iv_straddle is not None and np.isfinite(iv_straddle) and iv_straddle > 0 else float(common_sigma_value)
            sigma_put_straddle = float(iv_straddle_put) if iv_straddle_put is not None and np.isfinite(iv_straddle_put) and iv_straddle_put > 0 else float(common_sigma_value)
            if any(v is not None and np.isfinite(v) and v > 0 for v in (iv_straddle, iv_straddle_put)):
                iv_call_txt = f"{iv_straddle:.4f}" if iv_straddle is not None and np.isfinite(iv_straddle) and iv_straddle > 0 else "n/a"
                iv_put_txt = f"{iv_straddle_put:.4f}" if iv_straddle_put is not None and np.isfinite(iv_straddle_put) and iv_straddle_put > 0 else "n/a"
                st.caption(f"IV récupérées (cache) ≈ call {iv_call_txt} | put {iv_put_txt}")
                st.caption(f"σ utilisées : call {sigma_call_straddle:.4f} | put {sigma_put_straddle:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            view_dyn = view_straddle(
                float(common_spot_value),
                strike_slider,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                sigma_call=float(sigma_call_straddle),
                sigma_put=float(sigma_put_straddle),
                T=float(T_straddle),
            )
            premium = float(view_dyn.get("premium", 0.0))

            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]
            pnl_at_s0 = float(payoff_grid[np.searchsorted(s_grid, float(common_spot_value), side="left") - 1] - premium)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Straddle (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            # Stocke la prime et affiche directement le formulaire d'ajout (équivalent dropdown)
            st.session_state[_k("straddle_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_straddle or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("straddle_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("straddle_side_inline"))
            st.caption(f"K (strike commun): {strike_slider:.4f}")
            st.caption(f"T (maturité, années): {float(T_straddle):.4f}")

            if st.button("Ajouter au dashboard", key=_k("straddle_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": "call" if option_char.lower() == "c" else "put",
                    "product_type": "Straddle",
                    "type": "Straddle",
                    "strike": float(strike_slider),
                    "strike2": float(strike_slider),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_straddle),
                    "legs": [
                        {"option_type": "call", "strike": float(strike_slider)},
                        {"option_type": "put", "strike": float(strike_slider)},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Straddle",
                        "legs": [
                            {"option_type": "call", "strike": float(strike_slider)},
                            {"option_type": "put", "strike": float(strike_slider)},
                        ],
                        "strike2": float(strike_slider),
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_call_used": float(sigma_call_straddle),
                        "sigma_put_used": float(sigma_put_straddle),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_straddle),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Straddle ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_strangle:
            k_put_raw = st.slider(
                "Strike put",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=0.95 * float(common_spot_value),
                step=0.5,
                key=_k("strangle_k_put"),
            )
            k_call_raw = st.slider(
                "Strike call",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=1.05 * float(common_spot_value),
                step=0.5,
                key=_k("strangle_k_call"),
            )
            k_put = min(k_put_raw, k_call_raw)
            k_call = max(k_put_raw, k_call_raw)
            T_strangle = st.slider(
                "T (années)",
                min_value=0.05,
                max_value=2.0,
                value=float(common_maturity_value),
                step=0.05,
                key=_k("strangle_T"),
            )
            iv_put = _get_cached_iv_for(k_put, T_strangle, "put")
            iv_call = _get_cached_iv_for(k_call, T_strangle, "call")
            sigma_put_strangle = float(iv_put) if iv_put is not None and np.isfinite(iv_put) and iv_put > 0 else float(common_sigma_value)
            sigma_call_strangle = float(iv_call) if iv_call is not None and np.isfinite(iv_call) and iv_call > 0 else float(common_sigma_value)
            if any(v is not None and np.isfinite(v) and v > 0 for v in (iv_put, iv_call)):
                iv_put_txt = f"{iv_put:.4f}" if iv_put is not None and np.isfinite(iv_put) and iv_put > 0 else "n/a"
                iv_call_txt = f"{iv_call:.4f}" if iv_call is not None and np.isfinite(iv_call) and iv_call > 0 else "n/a"
                st.caption(f"IV récupérées (cache) ≈ put {iv_put_txt} | call {iv_call_txt}")
                st.caption(f"σ utilisées : put {sigma_put_strangle:.4f} | call {sigma_call_strangle:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            view_dyn = view_strangle(
                float(common_spot_value),
                k_put,
                k_call,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                sigma_call=float(sigma_call_strangle),
                sigma_put=float(sigma_put_strangle),
                T=float(T_strangle),
            )
            premium = float(view_dyn.get("premium", 0.0))

            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Strangle (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("strangle_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_strangle or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("strangle_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("strangle_side_inline"))
            st.caption(f"K_put: {k_put:.4f} | K_call: {k_call:.4f}")
            st.caption(f"T (maturité, années): {float(T_strangle):.4f}")

            if st.button("Ajouter au dashboard", key=_k("strangle_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": "call" if option_char.lower() == "c" else "put",
                    "product_type": "Strangle",
                    "type": "Strangle",
                    "strike": float(k_put),
                    "strike2": float(k_call),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_strangle),
                    "legs": [
                        {"option_type": "put", "strike": float(k_put)},
                        {"option_type": "call", "strike": float(k_call)},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Strangle",
                        "legs": [
                            {"option_type": "put", "strike": float(k_put)},
                            {"option_type": "call", "strike": float(k_call)},
                        ],
                        "strike_put": float(k_put),
                        "strike_call": float(k_call),
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_put_used": float(sigma_put_strangle),
                        "sigma_call_used": float(sigma_call_strangle),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_strangle),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Strangle ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_call_spread:
            k_long_raw = st.slider(
                "Strike call long",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=0.95 * float(common_spot_value),
                step=0.5,
                key=_k("call_spread_k_long"),
            )
            k_short_raw = st.slider(
                "Strike call short",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=1.05 * float(common_spot_value),
                step=0.5,
                key=_k("call_spread_k_short"),
            )
            k_long = min(k_long_raw, k_short_raw)
            k_short = max(k_long_raw, k_short_raw)
            T_call_spread = st.slider(
                "T (années)",
                min_value=0.05,
                max_value=2.0,
                value=float(common_maturity_value),
                step=0.05,
                key=_k("call_spread_T"),
            )
            iv_long = _get_cached_iv_for(k_long, T_call_spread, "call")
            iv_short = _get_cached_iv_for(k_short, T_call_spread, "call")
            sigma_long_cs = float(iv_long) if iv_long is not None and np.isfinite(iv_long) and iv_long > 0 else float(common_sigma_value)
            sigma_short_cs = float(iv_short) if iv_short is not None and np.isfinite(iv_short) and iv_short > 0 else float(common_sigma_value)
            if any(v is not None and np.isfinite(v) and v > 0 for v in (iv_long, iv_short)):
                iv_long_txt = f"{iv_long:.4f}" if iv_long is not None and np.isfinite(iv_long) and iv_long > 0 else "n/a"
                iv_short_txt = f"{iv_short:.4f}" if iv_short is not None and np.isfinite(iv_short) and iv_short > 0 else "n/a"
                st.caption(f"IV récupérées (cache) ≈ long {iv_long_txt} | short {iv_short_txt}")
                st.caption(f"σ utilisées : long {sigma_long_cs:.4f} | short {sigma_short_cs:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            view_dyn = view_call_spread(
                float(common_spot_value),
                k_long,
                k_short,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                sigma_long=float(sigma_long_cs),
                sigma_short=float(sigma_short_cs),
                T=float(T_call_spread),
            )
            premium = float(view_dyn.get("premium", 0.0))

            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Call spread (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("call_spread_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_call_spread or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("call_spread_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("call_spread_side_inline"))
            st.caption(f"K long: {k_long:.4f} | K short: {k_short:.4f}")
            st.caption(f"T (maturité, années): {float(T_call_spread):.4f}")

            if st.button("Ajouter au dashboard", key=_k("call_spread_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": "call" if option_char.lower() == "c" else "put",
                    "product_type": "Call Spread",
                    "type": "Call spread",
                    "strike": float(k_long),
                    "strike2": float(k_short),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_call_spread),
                    "legs": [
                        {"option_type": "call", "strike": float(k_long)},
                        {"option_type": "call", "strike": float(k_short)},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Call spread",
                        "legs": [
                            {"option_type": "call", "strike": float(k_long)},
                            {"option_type": "call", "strike": float(k_short)},
                        ],
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_call_long_used": float(sigma_long_cs),
                        "sigma_call_short_used": float(sigma_short_cs),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_call_spread),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Call spread ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_put_spread:
            k_long_raw = st.slider(
                "Strike put long",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=1.05 * float(common_spot_value),
                step=0.5,
                key=_k("put_spread_k_long"),
            )
            k_short_raw = st.slider(
                "Strike put short",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=0.95 * float(common_spot_value),
                step=0.5,
                key=_k("put_spread_k_short"),
            )
            k_long = max(k_long_raw, k_short_raw)
            k_short = min(k_long_raw, k_short_raw)
            T_put_spread = st.slider(
                "T (années)",
                min_value=0.05,
                max_value=2.0,
                value=float(common_maturity_value),
                step=0.05,
                key=_k("put_spread_T"),
            )
            iv_long_p = _get_cached_iv_for(k_long, T_put_spread, "put")
            iv_short_p = _get_cached_iv_for(k_short, T_put_spread, "put")
            sigma_long_ps = float(iv_long_p) if iv_long_p is not None and np.isfinite(iv_long_p) and iv_long_p > 0 else float(common_sigma_value)
            sigma_short_ps = float(iv_short_p) if iv_short_p is not None and np.isfinite(iv_short_p) and iv_short_p > 0 else float(common_sigma_value)
            if any(v is not None and np.isfinite(v) and v > 0 for v in (iv_long_p, iv_short_p)):
                iv_long_txt = f"{iv_long_p:.4f}" if iv_long_p is not None and np.isfinite(iv_long_p) and iv_long_p > 0 else "n/a"
                iv_short_txt = f"{iv_short_p:.4f}" if iv_short_p is not None and np.isfinite(iv_short_p) and iv_short_p > 0 else "n/a"
                st.caption(f"IV récupérées (cache) ≈ long {iv_long_txt} | short {iv_short_txt}")
                st.caption(f"σ utilisées : long {sigma_long_ps:.4f} | short {sigma_short_ps:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            view_dyn = view_put_spread(
                float(common_spot_value),
                k_long,
                k_short,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                sigma_long=float(sigma_long_ps),
                sigma_short=float(sigma_short_ps),
                T=float(T_put_spread),
            )
            premium = float(view_dyn.get("premium", 0.0))

            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Put spread (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("put_spread_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_put_spread or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("put_spread_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("put_spread_side_inline"))
            st.caption(f"K long: {k_long:.4f} | K short: {k_short:.4f}")
            st.caption(f"T (maturité, années): {float(T_put_spread):.4f}")

            if st.button("Ajouter au dashboard", key=_k("put_spread_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": "call" if option_char.lower() == "c" else "put",
                    "product_type": "Put Spread",
                    "type": "Put spread",
                    "strike": float(k_long),
                    "strike2": float(k_short),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_put_spread),
                    "legs": [
                        {"option_type": "put", "strike": float(k_long)},
                        {"option_type": "put", "strike": float(k_short)},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Put spread",
                        "legs": [
                            {"option_type": "put", "strike": float(k_long)},
                            {"option_type": "put", "strike": float(k_short)},
                        ],
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_put_long_used": float(sigma_long_ps),
                        "sigma_put_short_used": float(sigma_short_ps),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_put_spread),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Put spread ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_butterfly:
            k_center = st.slider(
                "Strike central",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=float(common_spot_value),
                step=0.5,
                key=_k("butterfly_k_center"),
            )
            wing = st.slider(
                "Écart ailes",
                min_value=max(0.1, 0.02 * float(common_spot_value)),
                max_value=max(1.0, 0.5 * float(common_spot_value)),
                value=max(0.5, 0.05 * float(common_spot_value)),
                step=0.1,
                key=_k("butterfly_wing"),
            )
            k1 = max(0.01, k_center - wing)
            k2 = k_center
            k3 = k_center + wing
            T_bfly = st.slider(
                "T (années)",
                min_value=0.05,
                max_value=2.0,
                value=float(common_maturity_value),
                step=0.05,
                key=_k("butterfly_T"),
            )
            ivs_bfly = [
                _get_cached_iv_for(k1, T_bfly, "call"),
                _get_cached_iv_for(k2, T_bfly, "call"),
                _get_cached_iv_for(k3, T_bfly, "call"),
            ]
            sigma_k1_bfly = float(ivs_bfly[0]) if ivs_bfly[0] is not None and np.isfinite(ivs_bfly[0]) and ivs_bfly[0] > 0 else float(common_sigma_value)
            sigma_k2_bfly = float(ivs_bfly[1]) if ivs_bfly[1] is not None and np.isfinite(ivs_bfly[1]) and ivs_bfly[1] > 0 else float(common_sigma_value)
            sigma_k3_bfly = float(ivs_bfly[2]) if ivs_bfly[2] is not None and np.isfinite(ivs_bfly[2]) and ivs_bfly[2] > 0 else float(common_sigma_value)
            iv_vals_bfly = [v for v in ivs_bfly if v is not None and np.isfinite(v) and v > 0]
            sigma_bfly = float(np.mean(iv_vals_bfly)) if iv_vals_bfly else float(common_sigma_value)
            if iv_vals_bfly:
                iv_txt = " | ".join(
                    f"K={k:.2f}: {v:.4f}" if v is not None and np.isfinite(v) and v > 0 else f"K={k:.2f}: n/a"
                    for k, v in zip([k1, k2, k3], ivs_bfly)
                )
                st.caption(f"IV récupérées (cache) ≈ {iv_txt}")
                st.caption(f"σ utilisées : K1 {sigma_k1_bfly:.4f} | K2 {sigma_k2_bfly:.4f} | K3 {sigma_k3_bfly:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            view_dyn = view_butterfly(
                float(common_spot_value),
                k1,
                k2,
                k3,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                sigma_k1=float(sigma_k1_bfly),
                sigma_k2=float(sigma_k2_bfly),
                sigma_k3=float(sigma_k3_bfly),
                T=float(T_bfly),
            )
            premium = float(view_dyn.get("premium", 0.0))

            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Butterfly (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("butterfly_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_bfly or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("butterfly_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("butterfly_side_inline"))
            st.caption(f"K1: {k1:.4f} | K2: {k2:.4f} | K3: {k3:.4f}")
            st.caption(f"T (maturité, années): {float(T_bfly):.4f}")

            if st.button("Ajouter au dashboard", key=_k("butterfly_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": "call" if option_char.lower() == "c" else "put",
                    "product_type": "Butterfly",
                    "type": "Butterfly",
                    "strike": float(k1),
                    "strike2": float(k3),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_bfly),
                    "legs": [
                        {"option_type": "call", "strike": float(k1)},
                        {"option_type": "call", "strike": float(k2)},
                        {"option_type": "call", "strike": float(k2)},
                        {"option_type": "call", "strike": float(k3)},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Butterfly",
                        "legs": [
                            {"option_type": "call", "strike": float(k1)},
                            {"option_type": "call", "strike": float(k2)},
                            {"option_type": "call", "strike": float(k2)},
                            {"option_type": "call", "strike": float(k3)},
                        ],
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_k1_used": float(sigma_k1_bfly),
                        "sigma_k2_used": float(sigma_k2_bfly),
                        "sigma_k3_used": float(sigma_k3_bfly),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_bfly),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Butterfly ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_condor:
            k_center = st.slider(
                "Strike central (condor)",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=float(common_spot_value),
                step=0.5,
                key=_k("condor_center"),
            )
            inner = st.slider(
                "Écart strikes courts",
                min_value=max(0.1, 0.02 * float(common_spot_value)),
                max_value=max(1.0, 0.5 * float(common_spot_value)),
                value=max(0.5, 0.05 * float(common_spot_value)),
                step=0.1,
                key=_k("condor_inner"),
            )
            outer_raw = st.slider(
                "Écart strikes ailes",
                min_value=max(0.2, 0.03 * float(common_spot_value)),
                max_value=max(1.5, 0.7 * float(common_spot_value)),
                value=max(0.9, 0.1 * float(common_spot_value)),
                step=0.1,
                key=_k("condor_outer"),
            )
            outer = max(outer_raw, inner + max(0.1, 0.01 * float(common_spot_value)))

            k1 = max(0.01, k_center - outer)
            k2 = k_center - inner
            k3 = k_center + inner
            k4 = k_center + outer
            T_condor = st.slider(
                "T (années)",
                min_value=0.05,
                max_value=2.0,
                value=float(common_maturity_value),
                step=0.05,
                key=_k("condor_T"),
            )
            ivs_condor = [
                _get_cached_iv_for(k1, T_condor, "call"),
                _get_cached_iv_for(k2, T_condor, "call"),
                _get_cached_iv_for(k3, T_condor, "call"),
                _get_cached_iv_for(k4, T_condor, "call"),
            ]
            sigma_k1_condor = float(ivs_condor[0]) if ivs_condor[0] is not None and np.isfinite(ivs_condor[0]) and ivs_condor[0] > 0 else float(common_sigma_value)
            sigma_k2_condor = float(ivs_condor[1]) if ivs_condor[1] is not None and np.isfinite(ivs_condor[1]) and ivs_condor[1] > 0 else float(common_sigma_value)
            sigma_k3_condor = float(ivs_condor[2]) if ivs_condor[2] is not None and np.isfinite(ivs_condor[2]) and ivs_condor[2] > 0 else float(common_sigma_value)
            sigma_k4_condor = float(ivs_condor[3]) if ivs_condor[3] is not None and np.isfinite(ivs_condor[3]) and ivs_condor[3] > 0 else float(common_sigma_value)
            iv_vals_condor = [v for v in ivs_condor if v is not None and np.isfinite(v) and v > 0]
            sigma_condor = float(np.mean(iv_vals_condor)) if iv_vals_condor else float(common_sigma_value)
            if iv_vals_condor:
                iv_txt = " | ".join(
                    f"K={k:.2f}: {v:.4f}" if v is not None and np.isfinite(v) and v > 0 else f"K={k:.2f}: n/a"
                    for k, v in zip([k1, k2, k3, k4], ivs_condor)
                )
                st.caption(f"IV récupérées (cache) ≈ {iv_txt}")
                st.caption(
                    f"σ utilisées : "
                    f"K1 {sigma_k1_condor:.4f} | K2 {sigma_k2_condor:.4f} | "
                    f"K3 {sigma_k3_condor:.4f} | K4 {sigma_k4_condor:.4f}"
                )
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            view_dyn = view_condor(
                float(common_spot_value),
                k1,
                k2,
                k3,
                k4,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                sigma_k1=float(sigma_k1_condor),
                sigma_k2=float(sigma_k2_condor),
                sigma_k3=float(sigma_k3_condor),
                sigma_k4=float(sigma_k4_condor),
                T=float(T_condor),
            )
            premium = float(view_dyn.get("premium", 0.0))

            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Condor (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("condor_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_condor or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("condor_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("condor_side_inline"))
            st.caption(f"K1: {k1:.4f} | K2: {k2:.4f} | K3: {k3:.4f} | K4: {k4:.4f}")
            st.caption(f"T (maturité, années): {float(T_condor):.4f}")

            if st.button("Ajouter au dashboard", key=_k("condor_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": "call" if option_char.lower() == "c" else "put",
                    "product_type": "Condor",
                    "type": "Condor",
                    "strike": float(k1),
                    "strike2": float(k4),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_condor),
                    "legs": [
                        {"option_type": "call", "strike": float(k1)},
                        {"option_type": "call", "strike": float(k2)},
                        {"option_type": "call", "strike": float(k3)},
                        {"option_type": "call", "strike": float(k4)},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Condor",
                        "legs": [
                            {"option_type": "call", "strike": float(k1)},
                            {"option_type": "call", "strike": float(k2)},
                            {"option_type": "call", "strike": float(k3)},
                            {"option_type": "call", "strike": float(k4)},
                        ],
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_k1_used": float(sigma_k1_condor),
                        "sigma_k2_used": float(sigma_k2_condor),
                        "sigma_k3_used": float(sigma_k3_condor),
                        "sigma_k4_used": float(sigma_k4_condor),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_condor),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Condor ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_iron_bfly:
            k_center = st.slider(
                "Strike central (iron butterfly)",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=float(common_spot_value),
                step=0.5,
                key=_k("iron_bfly_center"),
            )
            wing = st.slider(
                "Écart ailes",
                min_value=max(0.1, 0.02 * float(common_spot_value)),
                max_value=max(1.0, 0.5 * float(common_spot_value)),
                value=max(0.5, 0.05 * float(common_spot_value)),
                step=0.1,
                key=_k("iron_bfly_wing"),
            )
            k_put_long = max(0.01, k_center - wing)
            k_call_long = k_center + wing

            # Iron butterfly uses its own pricer (different from iron condor)
            T_iron_bfly = st.slider(
                "T (années)",
                min_value=0.05,
                max_value=2.0,
                value=float(common_maturity_value),
                step=0.05,
                key=_k("iron_bfly_T"),
            )
            ivs_ib = [
                _get_cached_iv_for(k_put_long, T_iron_bfly, "put"),
                _get_cached_iv_for(k_center, T_iron_bfly, "call"),
                _get_cached_iv_for(k_call_long, T_iron_bfly, "call"),
            ]
            sigma_put_long_ib = float(ivs_ib[0]) if ivs_ib[0] is not None and np.isfinite(ivs_ib[0]) and ivs_ib[0] > 0 else float(common_sigma_value)
            sigma_call_center_ib = float(ivs_ib[1]) if ivs_ib[1] is not None and np.isfinite(ivs_ib[1]) and ivs_ib[1] > 0 else float(common_sigma_value)
            sigma_call_long_ib = float(ivs_ib[2]) if ivs_ib[2] is not None and np.isfinite(ivs_ib[2]) and ivs_ib[2] > 0 else float(common_sigma_value)
            sigma_put_center_ib = sigma_call_center_ib  # même strike central, on réutilise la même IV pour le put central
            iv_vals_ib = [v for v in ivs_ib if v is not None and np.isfinite(v) and v > 0]
            sigma_iron_bfly = float(np.mean(iv_vals_ib)) if iv_vals_ib else float(common_sigma_value)
            if iv_vals_ib:
                iv_txt = " | ".join(
                    f"K={k:.2f}: {v:.4f}" if v is not None and np.isfinite(v) and v > 0 else f"K={k:.2f}: n/a"
                    for k, v in zip([k_put_long, k_center, k_call_long], ivs_ib)
                )
                st.caption(f"IV récupérées (cache) ≈ {iv_txt}")
                st.caption(
                    "σ utilisées : "
                    f"put long {sigma_put_long_ib:.4f} | "
                    f"put center {sigma_put_center_ib:.4f} | "
                    f"call center {sigma_call_center_ib:.4f} | "
                    f"call long {sigma_call_long_ib:.4f}"
                )
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
            view_dyn = view_iron_butterfly(
                float(common_spot_value),
                k_put_long,
                k_center,
                k_call_long,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                sigma_put_long=float(sigma_put_long_ib),
                sigma_put_center=float(sigma_put_center_ib),
                sigma_call_center=float(sigma_call_center_ib),
                sigma_call_long=float(sigma_call_long_ib),
                T=float(T_iron_bfly),
            )
            premium_raw = price_iron_butterfly_bs(
                float(common_spot_value),
                k_put_long,
                k_center,
                k_call_long,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                sigma_put_long=float(sigma_put_long_ib),
                sigma_put_center=float(sigma_put_center_ib),
                sigma_call_center=float(sigma_call_center_ib),
                sigma_call_long=float(sigma_call_long_ib),
                T=float(T_iron_bfly),
            )
            premium = float(premium_raw)
            price_display = abs(premium)

            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["payoff"] - premium

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Iron Butterfly (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("iron_bfly_pre_price")] = premium
            price = float(price_display)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price_display:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_iron_bfly or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("iron_bfly_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("iron_bfly_side_inline"))
            st.caption(f"K_put_long: {k_put_long:.4f} | K_centre: {k_center:.4f} | K_call_long: {k_call_long:.4f}")
            st.caption(f"T (maturité, années): {float(T_iron_bfly):.4f}")

            if st.button("Ajouter au dashboard", key=_k("iron_bfly_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": "call" if option_char.lower() == "c" else "put",
                    "product_type": "Iron Butterfly",
                    "type": "Iron Butterfly",
                    "strike": float(k_put_long),
                    "strike2": float(k_call_long),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price_display,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_iron_bfly),
                    "legs": [
                        {"option_type": "put", "strike": float(k_put_long)},
                        {"option_type": "put", "strike": float(k_center)},
                        {"option_type": "call", "strike": float(k_center)},
                        {"option_type": "call", "strike": float(k_call_long)},
                    ],
                    "T_0": today.isoformat(),
                    "price": price_display,
                    "misc": {
                        "structure": "Iron butterfly",
                        "legs": [
                            {"option_type": "put", "strike": float(k_put_long)},
                            {"option_type": "put", "strike": float(k_center)},
                            {"option_type": "call", "strike": float(k_center)},
                            {"option_type": "call", "strike": float(k_call_long)},
                        ],
                        "premium_raw": float(premium),
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_put_long_used": float(sigma_put_long_ib),
                        "sigma_put_center_used": float(sigma_put_center_ib),
                        "sigma_call_center_used": float(sigma_call_center_ib),
                        "sigma_call_long_used": float(sigma_call_long_ib),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_iron_bfly),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Iron butterfly ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_calendar:
            st.subheader("Calendar spread – vue Notebook")
            s0_ref = float(common_spot_value)
            hist_tkr = (
                st.session_state.get("common_underlying")
                or st.session_state.get("tkr_common")
                or st.session_state.get("heston_cboe_ticker")
                or "SPY"
            ).strip().upper()
            spy_close = load_close_series_for_ticker(hist_tkr, fallback_value=s0_ref)

            strike_anchor_cal = float(common_spot_value)
            col1, col2 = st.columns(2)
            with col1:
                option_type_cal = st.selectbox("Type", ["call", "put"], key=_k("calendar_type"))
                strike_cal = st.slider(
                    "Strike",
                    min_value=0.6 * strike_anchor_cal,
                    max_value=1.4 * strike_anchor_cal,
                    value=float(round(strike_anchor_cal)),
                    step=0.5,
                    key=_k("calendar_strike"),
                )
                t_short = st.slider("T court (années)", min_value=0.05, max_value=1.0, value=0.25, step=0.05, key=_k("calendar_t_short"))
                t_long_raw = st.slider("T long (années)", min_value=0.1, max_value=2.0, value=0.75, step=0.05, key=_k("calendar_t_long"))
                t_long = max(t_long_raw, t_short + 0.01)
                if t_long != t_long_raw:
                    st.caption(f"T long ajusté à {t_long:.2f} pour rester après T court.")
            with col2:
                span_cal = st.slider("Span payoff (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.05, key=_k("calendar_span"))

            iv_cal = _get_cached_iv_for(strike_cal, t_long, option_type_cal)
            sigma_cal = float(iv_cal) if iv_cal is not None and np.isfinite(iv_cal) and iv_cal > 0 else float(common_sigma_value)
            if iv_cal is not None and np.isfinite(iv_cal) and iv_cal > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_cal:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
            r_cal = float(common_rate_value)

            view_dyn = view_calendar_spread(
                s0_ref,
                strike_cal,
                T_short=t_short,
                T_long=t_long,
                option_type=option_type_cal,
                r=r_cal,
                q=0.0,
                sigma=sigma_cal,
                span=span_cal,
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]
            payoff_s0 = float(np.interp(s0_ref, s_grid, payoff_grid))
            pnl_s0 = payoff_s0 - premium

            forward_start_date = datetime.date.today() + datetime.timedelta(days=int(t_short * 365))
            fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
            ax_ts.plot(spy_close.index, spy_close.values, label=f"{hist_tkr} close (1y)")
            ax_ts.axhline(strike_cal, color="gray", linestyle="--", label=f"K = {strike_cal:.2f}")
            ax_ts.axvline(forward_start_date, color="purple", linestyle=":", label=f"Start ~ {forward_start_date.isoformat()}")
            ax_ts.set_ylabel("Prix")
            ax_ts.set_title(f"Clôtures {hist_tkr} (strike / forward start)")
            ax_ts.legend(loc="best")
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts, clear_figure=True)

            fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
            ax_pay.plot(s_grid, payoff_grid, label="Payoff")
            ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax_pay.axvline(strike_cal, color="gray", linestyle="--", label=f"K = {strike_cal:.2f}")
            ax_pay.axvline(s0_ref, color="crimson", linestyle="-.", label=f"S0 = {s0_ref:.2f}")
            ax_pay.axhline(0, color="black", linewidth=0.8)
            ax_pay.legend(loc="best")
            ax_pay.set_xlabel("Spot")
            ax_pay.set_ylabel("Payoff / P&L")
            ax_pay.set_title(f"Calendar spread ({option_type_cal})")
            st.pyplot(fig_pay, clear_figure=True)

            price = float(premium)
            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((t_long or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("calendar_qty"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("calendar_side"))

            if st.button("Ajouter au dashboard", key=_k("calendar_add"), type="primary"):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": option_type_cal,
                    "product_type": "Calendar spread",
                    "type": "Calendar spread",
                    "strike": float(strike_cal),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(s0_ref),
                    "maturity_years": float(t_long),
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "T_short": float(t_short),
                        "T_long": float(t_long),
                        "sigma": float(sigma_cal),
                        "r": float(r_cal),
                        "span": float(span_cal),
                        "spot_at_pricing": float(s0_ref),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(f"Calendar spread ajouté au dashboard (id: {option_id}).")
                except Exception as exc:
                    st.error(f"Erreur lors de l'enregistrement : {exc}")

        with tab_diagonal:
            st.subheader("Diagonal spread – vue Notebook")
            s0_ref = float(common_spot_value)
            hist_tkr = (
                st.session_state.get("common_underlying")
                or st.session_state.get("tkr_common")
                or st.session_state.get("heston_cboe_ticker")
                or "SPY"
            ).strip().upper()
            spy_close = load_close_series_for_ticker(hist_tkr, fallback_value=s0_ref)

            strike_anchor_diag = float(common_spot_value)
            col1, col2 = st.columns(2)
            with col1:
                option_type_diag = st.selectbox("Type", ["call", "put"], key=_k("diag_type"))
                k_near = st.slider(
                    "Strike near",
                    min_value=0.6 * strike_anchor_diag,
                    max_value=1.4 * strike_anchor_diag,
                    value=float(round(strike_anchor_diag)),
                    step=0.5,
                    key=_k("diag_k_near"),
                )
                k_far = st.slider(
                    "Strike far",
                    min_value=0.6 * strike_anchor_diag,
                    max_value=1.6 * strike_anchor_diag,
                    value=float(round(strike_anchor_diag) * 1.02),
                    step=0.5,
                    key=_k("diag_k_far"),
                )
                t_near = st.slider("T near (années)", min_value=0.05, max_value=1.0, value=0.25, step=0.05, key=_k("diag_t_near"))
                t_far_raw = st.slider("T far (années)", min_value=0.1, max_value=2.0, value=0.75, step=0.05, key=_k("diag_t_far"))
                t_far = max(t_far_raw, t_near + 0.01)
                if t_far != t_far_raw:
                    st.caption(f"T far ajusté à {t_far:.2f} pour rester après T near.")
            with col2:
                span_diag = st.slider("Span payoff (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.05, key=_k("diag_span"))
            r_diag = float(common_rate_value)
            iv_diag = _get_cached_iv_for(k_far, t_far, option_type_diag)
            sigma_diag = float(iv_diag) if iv_diag is not None and np.isfinite(iv_diag) and iv_diag > 0 else float(common_sigma_value)
            if iv_diag is not None and np.isfinite(iv_diag) and iv_diag > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_diag:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            view_dyn = view_diagonal_spread(
                s0_ref,
                k_near,
                k_far,
                T_near=t_near,
                T_far=t_far,
                option_type=option_type_diag,
                r=r_diag,
                q=0.0,
                sigma=sigma_diag,
                span=span_diag,
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]
            payoff_s0 = float(np.interp(s0_ref, s_grid, payoff_grid))
            pnl_s0 = payoff_s0 - premium

            forward_start_date = datetime.date.today() + datetime.timedelta(days=int(t_near * 365))
            fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
            ax_ts.plot(spy_close.index, spy_close.values, label=f"{hist_tkr} close (1y)")
            ax_ts.axhline(k_near, color="gray", linestyle="--", label=f"K near = {k_near:.2f}")
            ax_ts.axhline(k_far, color="firebrick", linestyle=":", label=f"K far = {k_far:.2f}")
            ax_ts.axvline(forward_start_date, color="purple", linestyle=":", label=f"Start near ~ {forward_start_date.isoformat()}")
            ax_ts.set_ylabel("Prix")
            ax_ts.set_title(f"Clôtures {hist_tkr} (strikes / start)")
            ax_ts.legend(loc="best")
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts, clear_figure=True)

            fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
            ax_pay.plot(s_grid, payoff_grid, label="Payoff")
            ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax_pay.axvline(k_near, color="gray", linestyle="--", label=f"K near = {k_near:.2f}")
            ax_pay.axvline(k_far, color="firebrick", linestyle=":", label=f"K far = {k_far:.2f}")
            ax_pay.axvline(s0_ref, color="crimson", linestyle="-.", label=f"S0 = {s0_ref:.2f}")
            ax_pay.axhline(0, color="black", linewidth=0.8)
            ax_pay.legend(loc="best")
            ax_pay.set_xlabel("Spot")
            ax_pay.set_ylabel("Payoff / P&L")
            ax_pay.set_title(f"Diagonal spread ({option_type_diag})")
            st.pyplot(fig_pay, clear_figure=True)

            price = float(premium)
            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((t_far or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("diag_qty"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("diag_side"))

            if st.button("Ajouter au dashboard", key=_k("diag_add"), type="primary"):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": option_type_diag,
                    "product_type": "Diagonal spread",
                    "type": "Diagonal spread",
                    "strike": float(k_near),
                    "strike2": float(k_far),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(s0_ref),
                    "maturity_years": float(t_far),
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "T_near": float(t_near),
                        "T_far": float(t_far),
                        "sigma_used": float(sigma_diag),
                        "r": float(r_diag),
                        "span": float(span_diag),
                        "spot_at_pricing": float(s0_ref),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(f"Diagonal spread ajoutée au dashboard (id: {option_id}).")
                except Exception as exc:
                    st.error(f"Erreur lors de l'enregistrement : {exc}")

        with tab_asian_geo:
            st.subheader("Asian géométrique – vue Notebook")
            avg_close = float(spy_close_path.mean()) if spy_close_path is not None else s0_path
            col1, col2 = st.columns(2)
            with col1:
                option_type_ag = st.selectbox("Type", ["call", "put"], key=_k("asian_geo_type"))
                strike_ag = st.slider(
                    "Strike",
                    min_value=0.6 * s0_path,
                    max_value=1.4 * s0_path,
                    value=float(round(s0_path)),
                    step=0.5,
                    key=_k("asian_geo_k"),
                )
                avg_ag = st.slider(
                    "Moyenne (ref)",
                    min_value=0.5 * s0_path,
                    max_value=1.5 * s0_path,
                    value=float(round(s0_path)),
                    step=0.5,
                    key=_k("asian_geo_avg"),
                )
            with col2:
                r_ag = float(common_rate_value)
                T_ag = st.slider("T (années)", min_value=0.05, max_value=2.0, value=common_maturity_value, step=0.05, key=_k("asian_geo_T"))
            iv_ag = _get_cached_iv_for(strike_ag, T_ag, option_type_ag)
            sigma_ag = float(iv_ag) if iv_ag is not None and np.isfinite(iv_ag) and iv_ag > 0 else float(common_sigma_value)
            if iv_ag is not None and np.isfinite(iv_ag) and iv_ag > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_ag:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")

            view_dyn = view_asian_geom(
                s0_path,
                strike_ag,
                avg_ag,
                option_type=option_type_ag,
                r=r_ag,
                q=0.0,
                sigma=sigma_ag,
                T=T_ag,
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]
            payoff_s0 = float(np.interp(s0_path, s_grid, payoff_grid))
            pnl_s0 = payoff_s0 - premium

            fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
            ax_ts.plot(spy_close_path.index, spy_close_path.values, label=f"{hist_tkr} close (1y)")
            ax_ts.axhline(avg_ag, color="purple", linestyle=":", label=f"Moyenne = {avg_ag:.2f}")
            ax_ts.axhline(strike_ag, color="gray", linestyle="--", label=f"K = {strike_ag:.2f}")
            ax_ts.set_ylabel("Prix")
            ax_ts.set_title(f"Clôtures {hist_tkr} (Asian géo)")
            ax_ts.legend(loc="best")
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts, clear_figure=True)

            fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
            ax_pay.plot(s_grid, payoff_grid, label="Payoff")
            ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax_pay.axvline(strike_ag, color="gray", linestyle="--", label=f"K = {strike_ag:.2f}")
            ax_pay.axvline(s0_path, color="crimson", linestyle="-.", label=f"S0 = {s0_path:.2f}")
            ax_pay.axhline(0, color="black", linewidth=0.8)
            ax_pay.legend(loc="best")
            ax_pay.set_xlabel("Spot")
            ax_pay.set_ylabel("Payoff / P&L")
            ax_pay.set_title(f"Asian géométrique ({option_type_ag})")
            st.pyplot(fig_pay, clear_figure=True)

            price = float(premium)
            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")
            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_ag or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("asian_geo_qty"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("asian_geo_side"))

            if st.button("Ajouter au dashboard", key=_k("asian_geo_add"), type="primary"):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": option_type_ag,
                    "product_type": "Asian géométrique",
                    "type": "Asian géométrique",
                    "strike": float(strike_ag),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(s0_path),
                    "maturity_years": float(T_ag),
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "avg_ref": float(avg_ag),
                        "sigma_used": float(sigma_ag),
                        "r": float(r_ag),
                        "spot_at_pricing": float(s0_path),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(f"Asian géométrique ajoutée au dashboard (id: {option_id}).")
                except Exception as exc:
                    st.error(f"Erreur lors de l'enregistrement : {exc}")


        with tab_lookback_fixed:
            st.subheader("Lookback fixed – vue Notebook")
            col1, col2 = st.columns(2)
            with col1:
                option_type_lbf = st.selectbox("Type", ["call", "put"], key=_k("lbf_type"))
                min_lbf = st.slider(
                    "Min path",
                    min_value=0.8 * s0_path,
                    max_value=1.0 * s0_path,
                    value=float(round(s0_path)),
                    step=0.5,
                    key=_k("lbf_min"),
                )
                max_lbf = st.slider(
                    "Max path",
                    min_value=1.0 * s0_path,
                    max_value=1.2 * s0_path,
                    value=float(round(s0_path)),
                    step=0.5,
                    key=_k("lbf_max"),
                )
            with col2:
                strike_lbf = st.slider(
                    "Strike",
                    min_value=0.8 * s0_path,
                    max_value=1.2 * s0_path,
                    value=float(round(s0_path)),
                    step=0.5,
                    key=_k("lbf_k"),
                )
                span_lbf = st.slider("Span payoff (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.05, key=_k("lbf_span"))
                T_lbf = st.slider("T (années)", min_value=0.05, max_value=2.0, value=float(common_maturity_value), step=0.05, key=_k("lbf_T"))

            view_dyn = view_lookback_fixed(
                s0_path,
                min_lbf,
                max_lbf,
                strike_lbf,
                option_type=option_type_lbf,
                span=span_lbf,
                T=float(T_lbf),
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]
            payoff_s0 = float(np.interp(s0_path, s_grid, payoff_grid))
            pnl_s0 = payoff_s0 - premium

            fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
            ax_ts.plot(spy_close_path.index, spy_close_path.values, label=f"{hist_tkr} close (1y)")
            ax_ts.axhline(min_lbf, color="teal", linestyle=":", label=f"Min = {min_lbf:.2f}")
            ax_ts.axhline(max_lbf, color="gray", linestyle="--", label=f"Max = {max_lbf:.2f}")
            ax_ts.axhline(strike_lbf, color="firebrick", linestyle="-.", label=f"K = {strike_lbf:.2f}")
            ax_ts.set_ylabel("Prix")
            ax_ts.set_title(f"Clôtures {hist_tkr} (Lookback fixed)")
            ax_ts.legend(loc="best")
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts, clear_figure=True)

            fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
            ax_pay.plot(s_grid, payoff_grid, label="Payoff")
            ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax_pay.axvline(s0_path, color="crimson", linestyle="-.", label=f"S0 = {s0_path:.2f}")
            ax_pay.axhline(0, color="black", linewidth=0.8)
            ax_pay.legend(loc="best")
            ax_pay.set_xlabel("Spot")
            ax_pay.set_ylabel("Payoff / P&L")
            ax_pay.set_title(f"Lookback fixed ({option_type_lbf})")
            st.pyplot(fig_pay, clear_figure=True)

            price = float(premium)
            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")
            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_lbf or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("lbf_qty"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("lbf_side"))

            if st.button("Ajouter au dashboard", key=_k("lbf_add"), type="primary"):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": option_type_lbf,
                    "product_type": "Lookback fixed",
                    "type": "Lookback fixed",
                    "strike": float(strike_lbf),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(s0_path),
                    "maturity_years": float(T_lbf),
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "min_path": float(min_lbf),
                        "max_path": float(max_lbf),
                        "span": float(span_lbf),
                        "T": float(T_lbf),
                        "spot_at_pricing": float(s0_path),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(f"Lookback fixed ajoutée au dashboard (id: {option_id}).")
                except Exception as exc:
                    st.error(f"Erreur lors de l'enregistrement : {exc}")


        with tab_cliquet:
            clq_label, clq_char = _choose_option_select("opt_choice_cliquet_tab", option_char)
            option_label, option_char = clq_label, clq_char
            st.subheader("Cliquet / Ratchet – vue Notebook")
            k_cliquet_anchor = float(common_spot_value)
            strike_clq = st.slider(
                "Strike / niveau de référence",
                min_value=0.6 * k_cliquet_anchor,
                max_value=1.4 * k_cliquet_anchor,
                value=float(round(k_cliquet_anchor)),
                step=0.5,
                key=_k("cliquet_k"),
            )
            floor_val = st.slider("Floor", min_value=-0.5, max_value=0.5, value=0.0, step=0.01, key=_k("cliquet_floor"))
            cap_val = st.slider("Cap", min_value=0.0, max_value=0.5, value=0.1, step=0.01, key=_k("cliquet_cap"))
            T_clq = st.slider("T (années)", min_value=0.05, max_value=2.0, value=float(common_maturity_value), step=0.05, key=_k("cliquet_T"))

            view_dyn = view_cliquet(
                s0_path,
                floor=floor_val,
                cap=cap_val,
                T=float(T_clq),
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(common_sigma_value),
                n_periods=12,
                n_paths=4000,
                k_ref=float(strike_clq),
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]
            payoff_s0 = float(np.interp(s0_path, s_grid, payoff_grid))
            pnl_s0 = payoff_s0 - premium

            fig_ts, ax_ts = plt.subplots(figsize=(8, 3))
            ax_ts.plot(spy_close_path.index, spy_close_path.values, label=f"{hist_tkr} close (1y)")
            ax_ts.axhline(s0_path, color="gray", linestyle="--", label=f"S0 = {s0_path:.2f}")
            ax_ts.set_ylabel("Prix")
            ax_ts.set_title(f"Clôtures {hist_tkr} (Cliquet)")
            ax_ts.legend(loc="best")
            fig_ts.autofmt_xdate()
            st.pyplot(fig_ts, clear_figure=True)

            fig_pay, ax_pay = plt.subplots(figsize=(7, 4))
            ax_pay.plot(s_grid, payoff_grid, label="Payoff cliquet")
            ax_pay.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax_pay.axvline(s0_path, color="crimson", linestyle="-.", label=f"S0 = {s0_path:.2f}")
            ax_pay.axhline(0, color="black", linewidth=0.8)
            ax_pay.legend(loc="best")
            ax_pay.set_xlabel("Spot")
            ax_pay.set_ylabel("Payoff / P&L")
            ax_pay.set_title("Cliquet / Ratchet (approx)")
            st.pyplot(fig_pay, clear_figure=True)

            price = float(premium)
            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")
            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_clq or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("cliquet_qty"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("cliquet_side"))

            if st.button("Ajouter au dashboard", key=_k("cliquet_add"), type="primary"):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": option_char,
                    "product_type": "Cliquet / Ratchet",
                    "type": "Cliquet / Ratchet",
                    "strike": float(strike_clq),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(s0_path),
                    "maturity_years": float(T_clq),
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "floor": float(floor_val),
                        "cap": float(cap_val),
                        "strike_ref": float(strike_clq),
                        "T": float(T_clq),
                        "spot_at_pricing": float(s0_path),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(f"Cliquet / Ratchet ajouté au dashboard (id: {option_id}).")
                except Exception as exc:
                    st.error(f"Erreur lors de l'enregistrement : {exc}")


        with tab_quanto:
            opt_label_quanto, opt_char_quanto = _choose_option_select("opt_choice_quanto", option_char)
            option_label, option_char = opt_label_quanto, opt_char_quanto
            strike = st.slider(
                "Strike",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=float(common_spot_value),
                step=0.5,
                key=_k("quanto_k"),
            )
            fx_rate = st.slider(
                "Taux FX (payout)",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
                key=_k("quanto_fx"),
            )
            opt_type = "call" if option_char.lower() == "c" else "put"
            T_quanto = st.slider("T (années)", min_value=0.05, max_value=2.0, value=float(common_maturity_value), step=0.05, key=_k("quanto_T"))
            iv_quanto = _get_cached_iv_for(strike, T_quanto, opt_type)
            sigma_quanto = float(iv_quanto) if iv_quanto is not None and np.isfinite(iv_quanto) and iv_quanto > 0 else float(common_sigma_value)
            if iv_quanto is not None and np.isfinite(iv_quanto) and iv_quanto > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_quanto:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
            view_dyn = view_quanto(
                float(common_spot_value),
                strike,
                fx_rate=fx_rate,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(sigma_quanto),
                T=float(T_quanto),
                option_type=opt_type,
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Quanto (payoff & P&L avec prime BS)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("quanto_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l'entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_quanto or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("quanto_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("quanto_side_inline"))
            st.caption(f"K: {strike:.4f} | FX: {fx_rate:.4f}")
            st.caption(f"T (maturité, années): {float(T_quanto):.4f}")

            if st.button("Ajouter au dashboard", key=_k("quanto_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": opt_char_quanto,
                    "product_type": "Quanto",
                    "type": "Quanto",
                    "strike": float(strike),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_quanto),
                    "legs": [
                        {"option_type": opt_char_rain, "strike": float(strike), "fx_rate": float(fx_rate), "quanto": True},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Quanto",
                        "strike": float(strike),
                        "fx_rate": float(fx_rate),
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_used": float(sigma_quanto),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_quanto),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Quanto ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

        with tab_rainbow:
            opt_label_rain, opt_char_rain = _choose_option_select("opt_choice_rainbow", option_char)
            option_label, option_char = opt_label_rain, opt_char_rain
            strike = st.slider(
                "Strike",
                min_value=0.5 * float(common_spot_value),
                max_value=1.5 * float(common_spot_value),
                value=float(common_spot_value),
                step=0.5,
                key=_k("rainbow_k"),
            )
            ticker_b = st.text_input(
                "Ticker second sous-jacent (CBOE)",
                value=st.session_state.get(_k("rainbow_ticker_b"), ""),
                key=_k("rainbow_ticker_b_input"),
            )
            if st.button("📡 Récupérer le spot CBOE (B)", key=_k("rainbow_fetch_b")) and ticker_b.strip():
                t_b = ticker_b.strip().upper()
                try:
                    _, _, spot_b_cboe, _, _ = load_cboe_data(t_b)
                    st.session_state[_k("rainbow_s0b_cboe")] = float(spot_b_cboe)
                    st.session_state[_k("rainbow_ticker_b")] = t_b
                    st.success(f"Spot CBOE pour {t_b} : {float(spot_b_cboe):.4f}")
                except Exception as exc:
                    st.error(f"Impossible de récupérer le spot CBOE pour {t_b} : {exc}")

            spot_b_default = st.session_state.get(_k("rainbow_s0b_cboe"))
            if spot_b_default is None:
                st.info("Récupère d'abord le spot du second sous-jacent (bouton ci-dessus) pour afficher le pricing.")
                return
            spot_b = float(spot_b_default)
            opt_type = "call" if option_char.lower() == "c" else "put"
            T_rainbow = st.slider("T (années)", min_value=0.05, max_value=2.0, value=float(common_maturity_value), step=0.05, key=_k("rainbow_T"))
            iv_rain = _get_cached_iv_for(strike, T_rainbow, opt_type)
            sigma_rain = float(iv_rain) if iv_rain is not None and np.isfinite(iv_rain) and iv_rain > 0 else float(common_sigma_value)
            if iv_rain is not None and np.isfinite(iv_rain) and iv_rain > 0:
                st.caption(f"IV récupérée (cache) ≈ {iv_rain:.4f}")
            else:
                st.caption("IV non trouvée dans le cache, usage de σ par défaut.")
            view_dyn = view_rainbow(
                float(common_spot_value),
                float(spot_b),
                strike,
                r=float(common_rate_value),
                q=float(d_common),
                sigma=float(sigma_rain),
                sigma_b=float(sigma_rain),
                T=float(T_rainbow),
                option_type=opt_type,
            )
            premium = float(view_dyn.get("premium", 0.0))
            s_grid = view_dyn["s_grid"]
            payoff_grid = view_dyn["payoff"]
            pnl_grid = view_dyn["pnl"]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(s_grid, payoff_grid, label="Payoff brut")
            ax.plot(s_grid, pnl_grid, label="P&L net", color="darkorange")
            ax.axvline(float(common_spot_value), color="crimson", linestyle="-.", label=f"S_0 = {float(common_spot_value):.2f}")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Spot (sous-jacent principal)")
            ax.set_ylabel("Payoff / P&L")
            ax.set_title("Rainbow (payoff & P&L avec prime BS approximative)")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)

            st.session_state[_k("rainbow_pre_price")] = premium
            price = float(premium)

            st.markdown("### Ajouter au dashboard")
            st.metric("Prix calculé", f"${price:.6f}")

            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent A: {underlying or 'N/A'} | Spot A: {float(common_spot_value):.4f}")
            st.caption(
                f"Sous-jacent B: {(st.session_state.get(_k('rainbow_ticker_b'), 'N/A') or 'N/A').upper()} | Spot: {spot_b:.4f}"
            )
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((T_rainbow or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=_k("rainbow_qty_inline"))
            side = st.selectbox("Sens", ["long", "short"], index=0, key=_k("rainbow_side_inline"))
            st.caption(f"K: {strike:.4f}")
            st.caption(f"T (maturité, années): {float(T_rainbow):.4f}")

            if st.button("Ajouter au dashboard", key=_k("rainbow_add_inline")):
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": opt_char_rain,
                    "product_type": "Rainbow",
                    "type": "Rainbow",
                    "strike": float(strike),
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": price,
                    "side": side,
                    "S0": float(common_spot_value),
                    "maturity_years": float(T_rainbow),
                    "legs": [
                        {"option_type": opt_char_rain, "strike": float(strike), "secondary_spot": float(spot_b), "rainbow": True},
                    ],
                    "T_0": today.isoformat(),
                    "price": price,
                    "misc": {
                        "structure": "Rainbow",
                        "strike": float(strike),
                        "secondary_spot": float(spot_b),
                        "spot_at_pricing": float(common_spot_value),
                        "sigma_used": float(sigma_rain),
                        "r": float(common_rate_value),
                        "q": float(d_common),
                        "maturity": float(T_rainbow),
                    },
                }
                try:
                    st.caption(f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}")
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"Rainbow ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}")

    render_option_tabs_for_type("Call", "c")

# Alpaca API Setup
key = os.getenv("APCA_API_KEY_ID") or "PKRQ4GPVDAPCYIH6QGR4HI5USK"
secret_key = os.getenv("APCA_API_SECRET_KEY") or "3mENa9jXaLhESSekQzvz4cRh758awvBppB7Dfs9o1LJw"
BASE_URL = os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets/"
api = tradeapi.REST(str(key), str(secret_key), str(BASE_URL), api_version="v2")

# Page config
st.set_page_config(page_title="AI Trading Bot", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 90vw !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper functions
@st.cache_data(ttl=10)
def fetch_portfolio():
    """
    Fetch live equity positions from Alpaca and project them into a UI-friendly list.

    Returns:
        list[dict]: Each entry contains Symbol, Quantity, prices, P/L, and side.
    Side effects:
        - Network call to Alpaca REST API.
        - Displays Streamlit error on failure.
    """
    try:
        positions = api.list_positions()
        portfolio = []
        for pos in positions:
            portfolio.append({
                'Symbol': pos.symbol,
                'Quantity': pos.qty,
                'Entry Price': f"${float(pos.avg_entry_price):.2f}",
                'Current Price': f"${float(pos.current_price):.2f}",
                'Unrealized P/L': f"${float(pos.unrealized_pl):.2f}",
                'Side': 'buy'
            })
        return portfolio
    except Exception as e:
        st.error(f"Error fetching portfolio: {e}")
        return []

@st.cache_data(ttl=10)
def fetch_open_orders():
    """
    Fetch open orders from Alpaca for display.

    Returns:
        list[dict]: Orders with symbol, qty, limit price (or Market), and side.
    Side effects:
        - Network call to Alpaca REST API.
        - Displays Streamlit error on failure.
    """
    try:
        orders = api.list_orders(status='open')
        open_orders = []
        for order in orders:
            open_orders.append({
                'Symbol': order.symbol,
                'Quantity': order.qty,
                'Limit Price': f"${float(order.limit_price):.2f}" if order.limit_price else "Market",
                'Side': order.side
            })
        return open_orders
    except Exception as e:
        st.error(f"Error fetching orders: {e}")
        return []

def get_data(symbol):
    """
    Retrieve a spot price for a ticker using Alpaca first, then yfinance as fallback.

    Args:
        symbol (str): Ticker symbol to quote.
    Returns:
        dict: {"price": float} with -1 if no source returned a valid price.
    Side effects:
        - Network calls to Alpaca and yfinance.
    """
    symbol = (symbol or "").strip().upper()
    # 1) Alpaca live trade
    try:
        barset = api.get_latest_trade(symbol)
        return {"price": float(barset.price)}
    except Exception:
        pass
    # 2) yfinance fallback (last close)
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="1d")
        if not hist.empty and "Close" in hist.columns:
            return {"price": float(hist["Close"].iloc[-1])}
    except Exception:
        pass
    return {"price": -1}


@st.cache_data(ttl=120)
def get_spot_cboe_cached(symbol: str) -> float | None:
    """
    Fetch delayed CBOE spot for a ticker (cached to avoid repeated downloads).
    Falls back to None on error.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    try:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/{sym}.json"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        payload = resp.json().get("data", {})
        spot = payload.get("current_price")
        if spot is None:
            # Some responses nest under quotes
            spot = payload.get("last_sale", {}).get("price")
        return float(spot) if spot is not None else None
    except Exception:
        return None


def load_cached_option_history() -> tuple[str | None, pd.DataFrame | None]:
    """Load cached 1y close history for Options tab."""
    if CACHE_OPTIONS_HISTORY_FILE.exists():
        try:
            df = pd.read_csv(CACHE_OPTIONS_HISTORY_FILE, parse_dates=["Date"], index_col="Date")
            return None, df
        except Exception:
            return None, None
    return None, None


def get_last_cached_option_ticker() -> str | None:
    """Retourne le dernier ticker utilisé pour les options (via meta cache)."""
    meta = load_options_meta()
    tkr = meta.get("ticker")
    if tkr:
        return str(tkr).strip().upper()
    return None


def save_cached_option_history(ticker: str, df: pd.DataFrame) -> None:
    """Persist 1y close history for reuse across sessions."""
    try:
        CACHE_OPTIONS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_OPTIONS_HISTORY_FILE, index_label="Date")
    except Exception:
        pass


def save_cached_option_chain(ticker: str, calls_df: pd.DataFrame, puts_df: pd.DataFrame, S0_ref: float, r: float, q: float) -> None:
    """Persist last downloaded CBOE chain for reuse across sessions."""
    try:
        CACHE_OPTIONS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if calls_df is not None and not calls_df.empty:
            calls_df.to_csv(CACHE_OPTIONS_CALLS_FILE, index=False)
        if puts_df is not None and not puts_df.empty:
            puts_df.to_csv(CACHE_OPTIONS_PUTS_FILE, index=False)
        meta = {"ticker": ticker, "S0_ref": S0_ref, "r": r, "q": q}
        with open(CACHE_OPTIONS_META_FILE, "w") as f:
            json.dump(meta, f)
    except Exception:
        pass


def load_cached_option_chain(ticker: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, float | None, float | None, float | None]:
    """Load cached CBOE chain if it matches the requested ticker."""
    tkr = (ticker or "").strip().upper()
    try:
        meta = load_options_meta()
        if meta.get("ticker", "").upper() != tkr or not tkr:
            return None, None, None, None, None
        calls_df = pd.read_csv(CACHE_OPTIONS_CALLS_FILE) if CACHE_OPTIONS_CALLS_FILE.exists() else None
        puts_df = pd.read_csv(CACHE_OPTIONS_PUTS_FILE) if CACHE_OPTIONS_PUTS_FILE.exists() else None
        return calls_df, puts_df, float(meta.get("S0_ref") or 0.0), float(meta.get("r") or 0.0), float(meta.get("q") or 0.0)
    except Exception:
        return None, None, None, None, None


def load_options_meta() -> dict:
    """Charge le meta cache options (options_last_meta.json) si disponible, avec migration depuis l'ancien emplacement."""
    # Migration : ancien fichier (DATASETS_DIR) vers JSON_DIR
    try:
        if LEGACY_OPTIONS_META_FILE.exists() and not CACHE_OPTIONS_META_FILE.exists():
            CACHE_OPTIONS_META_FILE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(LEGACY_OPTIONS_META_FILE, CACHE_OPTIONS_META_FILE)
    except Exception:
        pass
    try:
        with open(CACHE_OPTIONS_META_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        # Fallback legacy
        try:
            with open(LEGACY_OPTIONS_META_FILE, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def load_heston_params_from_json(ticker: str) -> dict | None:
    """Charge les paramètres Heston persistés ; compatibilité avec l'ancien format par ticker."""
    _migrate_heston_params_cache()
    tkr = (ticker or "").strip().upper()
    if not tkr or not HESTON_PARAMS_FILE.exists():
        return None
    try:
        with HESTON_PARAMS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Nouveau format : dictionnaire plat (avec éventuellement la clé "ticker")
        if isinstance(data, dict) and {"kappa", "theta", "sigma", "rho", "v0"}.issubset(data.keys()):
            if not data.get("ticker") or str(data.get("ticker")).strip().upper() == tkr:
                return data
            # ticker différent -> on ne retourne rien
            return None
        # Ancien format : mapping {ticker: params}
        params = data.get(tkr) if isinstance(data, dict) else None
        if not isinstance(params, dict):
            return None
        return params
    except Exception:
        return None


def save_heston_params_to_json(ticker: str, params: dict) -> None:
    """Persiste les paramètres Heston dans un JSON plat (un seul set de paramètres)."""
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return
    _migrate_heston_params_cache()
    try:
        HESTON_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(params) if isinstance(params, dict) else {}
        payload["ticker"] = tkr
        with HESTON_PARAMS_FILE.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def fetch_option_history_to_cache(ticker: str) -> pd.DataFrame:
    """
    Download 1y daily closes for ticker via CLI helper and persist to cache CSV.
    Returns the DataFrame (may be empty on failure).
    """
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return pd.DataFrame()
    cli_path = SCRIPTS_DIR / "fetch_history_cli.py"
    hist_df = pd.DataFrame()
    try:
        result = subprocess.run(
            [sys.executable, str(cli_path), "--ticker", tkr, "--period", "1y", "--interval", "1d"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            hist_df = pd.read_csv(io.StringIO(result.stdout))
            if "Date" in hist_df.columns:
                hist_df["Date"] = pd.to_datetime(hist_df["Date"])
                hist_df.set_index("Date", inplace=True)
            save_cached_option_history(tkr, hist_df)
    except Exception:
        pass
    return hist_df
def load_equities():
    """Load configured trading systems (equities) from disk; returns {} on failure."""
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_equities(equities):
    """Persist trading systems (equities) to disk."""
    with open(DATA_FILE, 'w') as f:
        json.dump(equities, f, indent=2)

def load_portfolio():
    """Load the spot portfolio snapshot from disk; returns {} on failure."""
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_portfolio(portfolio):
    """Persist the spot portfolio snapshot to disk."""
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)


def load_sell_systems():
    """Load automated sell systems configuration from disk; returns {} on failure."""
    try:
        with open(SELL_SYSTEMS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sell_systems(sell_systems):
    """Persist automated sell systems configuration to disk."""
    with open(SELL_SYSTEMS_FILE, 'w') as f:
        json.dump(sell_systems, f, indent=2)


def _parse_dashboard_refresh_date(ts: str | None) -> datetime.date | None:
    """Convert ISO datetime strings to date objects for cache freshness checks."""
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(str(ts)).date()
    except Exception:
        return None


def load_dashboard_cache() -> dict:
    """Read Dashboard_cache.json; returns {} on failure."""
    try:
        with open(DASHBOARD_CACHE_FILE, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    except Exception:
        return {}


def save_dashboard_cache(cache: dict) -> None:
    """Persist dashboard price cache to disk (best-effort)."""
    try:
        DASHBOARD_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DASHBOARD_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def dashboard_cache_last_refresh(cache: dict | None = None) -> datetime.date | None:
    """Return the last refresh date stored in the dashboard cache."""
    cache = cache or load_dashboard_cache()
    return _parse_dashboard_refresh_date(cache.get("last_refresh"))


def collect_dashboard_tickers(
    portfolio: dict | None = None,
    forwards: dict | None = None,
    custom_options: dict | None = None,
) -> set[str]:
    """Aggregate all tickers shown on the Dashboard tab."""
    tickers: set[str] = set()
    today = datetime.date.today()
    if isinstance(portfolio, dict):
        tickers.update((sym or "").strip().upper() for sym in portfolio.keys())
    if isinstance(forwards, dict):
        for fwd in forwards.values():
            matur_str = fwd.get("maturity")
            try:
                matur_dt = datetime.date.fromisoformat(matur_str) if matur_str else None
            except Exception:
                matur_dt = None
            if matur_dt is not None and matur_dt < today:
                continue
            tickers.add((fwd.get("symbol") or "").strip().upper())
    if isinstance(custom_options, dict):
        for opt in custom_options.values():
            if str(opt.get("status", "open")).lower() != "open":
                continue
            tickers.add((opt.get("underlying") or "").strip().upper())
    return {t for t in tickers if t}


def refresh_dashboard_cache(tickers: set[str], cache: dict | None = None) -> dict:
    """Fetch live prices for provided tickers and persist them with a timestamp."""
    base_cache = cache or load_dashboard_cache()
    prices = {k: v for k, v in (base_cache.get("prices") or {}).items() if k in tickers}
    for sym in sorted(tickers):
        sym_clean = (sym or "").strip().upper()
        if not sym_clean:
            continue
        try:
            live_price = float(get_data(sym_clean).get("price", -1.0) or -1.0)
        except Exception:
            continue
        if live_price > 0:
            prices[sym_clean] = live_price
    payload = {
        "last_refresh": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prices": prices,
    }
    save_dashboard_cache(payload)
    return payload


def load_options_book():
    """
    Load the unified options book from disk, preferring the current file and
    falling back to the legacy file if needed.

    Returns:
        dict: Map of option_id -> option entry; {} if nothing is found/parseable.
    """
    for path in (OPTIONS_BOOK_FILE, OPTIONS_BOOK_FILE_LEGACY):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return {}


def load_options_book_legacy_only() -> dict:
    """Direct read of the legacy options_book.json (used for display/debug)."""
    try:
        with open(OPTIONS_BOOK_FILE_LEGACY, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_options_book(book):
    """Persist the unified options book (and legacy mirror) to disk."""
    try:
        OPTIONS_BOOK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OPTIONS_BOOK_FILE, 'w') as f:
            json.dump(book, f, indent=2)
        if OPTIONS_BOOK_FILE_LEGACY != OPTIONS_BOOK_FILE:
            try:
                with open(OPTIONS_BOOK_FILE_LEGACY, 'w') as f_legacy:
                    json.dump(book, f_legacy, indent=2)
            except Exception:
                # Legacy mirror is best-effort; ignore failures there.
                pass
    except Exception as exc:
        # Surface file-system issues (permissions, etc.) in the UI.
        try:
            st.error(f"Erreur lors de l'écriture du fichier options : {exc}")
        except Exception:
            pass
        raise


def _split_options_book(book: dict):
    """
    Partition the unified book into open vs expired slices.

    Args:
        book (dict): Full options book keyed by option id.
    Returns:
        tuple(dict, dict): (active, expired) maps.
    """
    active = {k: v for k, v in book.items() if v.get("status", "open") == "open"}
    expired = {k: v for k, v in book.items() if v.get("status") == "expired"}
    return active, expired


def load_options_portfolio():
    """Return only active option positions from the unified book."""
    active, _ = _split_options_book(load_options_book())
    return active


def save_options_portfolio(options_portfolio):
    """
    Persist active options merged with any expired entries back to disk.

    Args:
        options_portfolio (dict): Active/open options keyed by id.
    Returns:
        dict: Full merged book (active + expired) written to disk.
    """
    book = load_options_book()
    _, expired = _split_options_book(book)
    merged = dict(expired)
    for option_id, entry in options_portfolio.items():
        entry_copy = dict(entry)
        entry_copy.setdefault("status", "open")
        merged[option_id] = entry_copy
    save_options_book(merged)
    return merged


def load_expired_options():
    """Return expired/closed options from the unified book (migration safe)."""
    migrate_legacy_expired_options()
    _, expired = _split_options_book(load_options_book())
    return expired


def save_expired_options(expired_options):
    """
    Persist expired options merged with active ones, marking each entry as expired.

    Args:
        expired_options (dict): Expired option entries keyed by id.
    Returns:
        dict: Full merged book (active + expired) written to disk.
    """
    active = load_options_portfolio()
    book = dict(active)
    for option_id, entry in expired_options.items():
        entry_copy = dict(entry)
        entry_copy["status"] = "expired"
        book[option_id] = entry_copy
    save_options_book(book)
    return book


def migrate_legacy_expired_options():
    """Move legacy expired_options.json entries into the unified book schema."""
    if not LEGACY_EXPIRED_FILE.exists():
        return
    try:
        with open(LEGACY_EXPIRED_FILE, "r") as f:
            legacy = json.load(f)
    except Exception:
        return
    if not isinstance(legacy, dict) or not legacy:
        try:
            LEGACY_EXPIRED_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        return
    book = load_options_book()
    changed = False
    for option_id, entry in legacy.items():
        entry_copy = dict(entry)
        entry_copy["status"] = "expired"
        entry_copy.setdefault("option_type", entry_copy.get("type", ""))
        entry_copy.setdefault("product", entry_copy.get("product_type", entry_copy.get("structure", "vanilla")))
        entry_copy.setdefault("side", entry_copy.get("side", "long"))
        entry_copy.setdefault("quantity", entry_copy.get("quantity", entry_copy.get("qty", 0)))
        entry_copy.setdefault("avg_price", entry_copy.get("avg_price", entry_copy.get("T_0_price", 0.0)))
        book[option_id] = entry_copy
        changed = True
    if changed:
        save_options_book(book)
    try:
        LEGACY_EXPIRED_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _compute_leg_payoff(leg: dict, spot: float) -> float:
    """
    Compute payoff contribution for a single leg (call/put, long/short).

    Args:
        leg: Leg definition containing option_type, strike, qty, side.
        spot: Underlying level at evaluation.
    Returns:
        Signed payoff scaled by quantity.
    """
    option_type = (leg.get("option_type") or leg.get("type") or "call").lower()
    strike = float(leg.get("strike", 0.0) or 0.0)
    qty = float(leg.get("qty", 1.0) or 1.0)
    payoff = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if (leg.get("side") or "long").lower() == "short":
        payoff *= -1.0
    return payoff * qty


def compute_option_payoff(option: dict, spot: float) -> float:
    """
    Evaluate payoff for a stored option/structure at a given spot.

    Supports multi-leg structures, Asian, digital, barrier, straddle/strangle,
    vanilla spreads, and vanilla fallback.

    Args:
        option: Stored option dict (dashboard schema).
        spot: Underlying level to evaluate payoff.
    Returns:
        float: Payoff (positive for long exposure, negative if encoded in legs).
    """
    legs = option.get("legs") or []
    if legs:
        return sum(_compute_leg_payoff(leg, spot) for leg in legs)

    product = (option.get("product_type") or option.get("product") or option.get("structure") or option.get("type") or "").lower()
    option_type_raw = (
        option.get("option_type")
        or option.get("cpflag")
        or option.get("cp_flag")
        or option.get("cp")
        or ""
    )
    if not option_type_raw:
        t_val = str(option.get("type", "")).lower()
        if t_val in {"call", "put"}:
            option_type_raw = t_val
    option_type = str(option_type_raw).lower()
    strike = float(option.get("strike", 0.0) or 0.0)
    strike2 = float(option.get("strike2", option.get("strike_upper", 0.0) or 0.0) or 0.0)
    misc = option.get("misc") if isinstance(option.get("misc"), dict) else {}
    closing_prices = misc.get("closing_prices") if isinstance(misc, dict) else None

    # Path-dependent: Asian arithmetic / geometric via provided path
    if "asian" in product and closing_prices:
        if "geom" in product:
            vals = [v for v in closing_prices if v and v > 0]
            if not vals:
                avg_level = spot
            else:
                avg_level = float(np.exp(np.mean(np.log(vals))))
        else:
            avg_level = float(np.mean(closing_prices))
        if option_type == "put":
            return max(strike - avg_level, 0.0)
        return max(avg_level - strike, 0.0)

    # Digital (cash-or-nothing) payoff
    if "digital" in product:
        payout = float(misc.get("payout", 1.0) or 1.0)
        if option_type == "put":
            return payout if spot < strike else 0.0
        return payout if spot > strike else 0.0

    # Barrier (simple in/out check using closing_prices if provided)
    if "barrier" in product:
        barrier = float(misc.get("barrier", misc.get("barrier_level", 0.0)) or 0.0)
        barrier_type = str(misc.get("barrier_type", "up")).lower()
        knock = str(misc.get("knock", misc.get("direction", "out"))).lower()
        path_vals = closing_prices or [spot]
        hit = False
        for p in path_vals:
            if barrier_type == "up" and p >= barrier:
                hit = True
                break
            if barrier_type == "down" and p <= barrier:
                hit = True
                break
        if knock == "out" and hit:
            return 0.0
        if knock == "in" and not hit:
            return 0.0
        # otherwise vanilla payoff

    if product in {"straddle"}:
        return max(spot - strike, 0.0) + max(strike - spot, 0.0)
    if product in {"strangle"}:
        k_put = min(strike, strike2 or strike)
        k_call = max(strike, strike2 or strike)
        return max(k_put - spot, 0.0) + max(spot - k_call, 0.0)
    if product in {"call_spread", "bull_call_spread", "callspread"}:
        k1 = strike
        k2 = strike2 or strike
        return max(spot - k1, 0.0) - max(spot - k2, 0.0)
    if product in {"put_spread", "bear_put_spread", "putspread"}:
        k_long = strike
        k_short = strike2 or strike
        return max(k_long - spot, 0.0) - max(k_short - spot, 0.0)

    # Vanilla fallback
    if option_type == "put" or product == "put":
        return max(strike - spot, 0.0)
    return max(spot - strike, 0.0)


def compute_option_pnl(option: dict, spot_at_event: float, mark_price: float | None = None) -> dict:
    """
    Compute PnL metadata for an option position given an event spot or mark price.

    Args:
        option: Stored option dict with avg_price/side/quantity.
        spot_at_event: Underlying level at expiration/event.
        mark_price: Optional mark-to-market override.
    Returns:
        dict: payoff_per_unit, pnl_per_unit, pnl_total.
    """
    premium = float(option.get("avg_price", 0.0) or 0.0)
    quantity = float(option.get("quantity", 1) or 0.0)
    side = (option.get("side") or "long").lower()
    payoff_value = mark_price if mark_price is not None else compute_option_payoff(option, spot_at_event)

    payoff_total = payoff_value * quantity
    premium_total = premium * quantity

    if side == "long":
        pnl_total = payoff_total - premium_total
    else:
        pnl_total = premium_total - payoff_total

    pnl_per_unit = pnl_total / quantity if quantity else 0.0

    return {
        "payoff_per_unit": payoff_value,
        "pnl_per_unit": pnl_per_unit,
        "pnl_total": pnl_total,
    }


def describe_expired_option_payoff(option: dict, spot_at_event: float | None = None) -> dict:
    """Short metadata for the payoff computation of an expired option."""
    product_raw = (
        option.get("product_type")
        or option.get("product")
        or option.get("structure")
        or option.get("type")
        or ""
    )
    product = str(product_raw).lower()
    option_type_raw = (
        option.get("option_type")
        or option.get("cpflag")
        or option.get("cp")
        or ""
    )
    option_type = str(option_type_raw or "call").lower()
    strike = float(option.get("strike", 0.0) or 0.0)
    strike2 = float(option.get("strike2", option.get("strike_upper", 0.0) or 0.0) or 0.0)
    quantity = float(option.get("quantity", 0.0) or 0.0)
    side = (option.get("side") or "long").lower()
    misc = option.get("misc") if isinstance(option.get("misc"), dict) else {}
    closing_prices = misc.get("closing_prices") if isinstance(misc, dict) else None
    barrier_level = float(misc.get("barrier", misc.get("barrier_level", 0.0)) or 0.0)
    barrier_type = str(misc.get("barrier_type", "up")).lower() if isinstance(misc, dict) else ""
    knock = str(misc.get("knock", misc.get("direction", "out"))).lower() if isinstance(misc, dict) else ""
    payout = float(misc.get("payout", 1.0)) if isinstance(misc, dict) else 1.0
    legs = option.get("legs") or []

    spot_close = float(
        spot_at_event
        if spot_at_event is not None
        else option.get("underlying_close", option.get("S_T", 0.0))
        or 0.0
    )

    method_label = "Vanilla"
    description = "Payoff européen: call=max(S-K,0) ; put=max(K-S,0)."
    params: dict = {"strike": strike, "option_type": option_type, "underlying_close": spot_close}

    prod_low = product.lower()
    if legs:
        method_label = "Structure multi-jambes"
        description = "Somme des payoffs jambe par jambe avec signe long/short et quantités propres à chaque jambe."
        params["legs"] = legs
    elif "asian" in prod_low:
        if closing_prices:
            params["closing_prices"] = closing_prices
        method_label = "Asiatique"
        if "geom" in prod_low:
            description = "Moyenne géométrique des prix observés, puis payoff vanilla sur cette moyenne."
            params["average"] = "geometric"
        else:
            description = "Moyenne arithmétique des prix observés, puis payoff vanilla sur cette moyenne."
            params["average"] = "arithmetic"
        params["strike"] = strike
    elif "digital" in prod_low:
        method_label = "Digital (cash-or-nothing)"
        description = "Paiement fixe si la condition sur S_T est vraie, 0 sinon."
        params.update({"strike": strike, "payout": payout, "option_type": option_type})
    elif "barrier" in prod_low:
        method_label = "Barrière"
        description = "Vérifie le franchissement de la barrière (knock in/out) avant d'appliquer le payoff vanilla."
        params.update(
            {
                "strike": strike,
                "barrier": barrier_level,
                "barrier_type": barrier_type,
                "direction": knock,
                "closing_prices": closing_prices or [],
            }
        )
    elif "straddle" in prod_low:
        method_label = "Straddle"
        description = "Somme call+put au même strike: max(S-K,0)+max(K-S,0)."
        params["strike"] = strike
    elif "strangle" in prod_low:
        method_label = "Strangle"
        description = "Put strike bas + Call strike haut: max(K_put-S,0)+max(S-K_call,0)."
        params.update({"strike_put": strike, "strike_call": strike2})
    elif any(k in prod_low for k in ["spread", "butterfly", "condor", "iron butterfly", "iron condor", "diagonal", "calendar"]):
        method_label = "Combinaison de spreads"
        description = "Combinaison linéaire de payoffs vanilla (long/short) sur différents strikes/échéances."
        params.update({"strike": strike, "strike2": strike2, "legs": legs or []})

    payoff_unit = option.get("payoff_per_unit")
    if payoff_unit is None:
        payoff_unit = compute_option_payoff(option, spot_close)
    side_mult = -1.0 if side == "short" else 1.0
    payoff_signed_unit = payoff_unit * side_mult
    payoff_total = payoff_signed_unit * quantity

    return {
        "function": "compute_option_payoff",
        "method_label": method_label,
        "description": description,
        "params": params,
        "underlying_close": spot_close,
        "quantity": quantity,
        "side": side,
        "payoff_unit": payoff_unit,
        "payoff_signed_unit": payoff_signed_unit,
        "payoff_total": payoff_total,
    }


def mark_option_market_value(option: dict, chain_entry: dict | None = None) -> tuple[float, float | None, float, float, str]:
    """
    Derive a model mark for an option position.

    Resolution order:
    - If chain_entry provided (CBOE row), use its spot/IV/T.
    - If misc contains Heston params, price via Carr-Madan; else BSM; else intrinsic payoff.

    Args:
        option: Stored option dict (dashboard schema).
        chain_entry: Optional CBOE chain row with spot/iv/T hints.
    Returns:
        tuple: (spot, T_years, sigma_used, mark_price, method_label).
    """
    underlying = option.get("underlying")
    spot_data = get_data(underlying) if underlying else {"price": 0}
    spot = float(spot_data.get("price", 0.0) or 0.0)
    # Fallback to provided S0 if live spot not available
    if spot <= 0:
        spot = float(option.get("S0", 0.0) or 0.0)

    sigma = float(option.get("sigma", 0.2) or 0.2)
    option_type = (option.get("option_type") or option.get("type") or "").lower()
    strike = float(option.get("strike", 0.0) or 0.0)
    strike2 = float(option.get("strike2", option.get("strike_upper", 0.0) or 0.0) or 0.0)
    product = (
        option.get("product_type")
        or option.get("product")
        or option.get("structure")
        or option.get("type")
        or ""
    ).lower()
    legs = option.get("legs") or []
    q_val = 0.0

    T_years = None
    try:
        expiry_date = datetime.date.fromisoformat(option.get("expiration"))
        days = (expiry_date - datetime.date.today()).days
        T_years = max(days, 0) / 365.0
    except Exception:
        T_years = None

    try:
        q_val = float(get_q(underlying) or option.get("q") or 0.0) if underlying else float(option.get("q", 0.0) or 0.0)
    except Exception:
        q_val = float(option.get("q", 0.0) or 0.0)

    if chain_entry:
        spot = float(chain_entry.get("spot", spot) or spot)
        sigma = float(chain_entry.get("iv", sigma) or sigma)
        T_years = float(chain_entry.get("T", T_years) or (T_years or 0.0))

    try:
        r_val = float(get_r(T_years) or option.get("r") or 0.0) if T_years else float(option.get("r", 0.0) or 0.0)
    except Exception:
        r_val = float(option.get("r", 0.0) or 0.0)

    mark_price = None
    method = "intrinsic"
    # Heston Carr–Madan if params are provided in misc
    misc = option.get("misc") if isinstance(option.get("misc"), dict) else {}
    heston_misc = misc.get("heston_params") if isinstance(misc, dict) else None
    # fallback maturity from stored field if expiration missing
    if T_years is None:
        try:
            T_years = float(option.get("maturity_years") or 0.0)
        except Exception:
            T_years = None
    if (
        heston_misc
        and spot > 0
        and strike > 0
        and T_years is not None
        and T_years > 0
        and option_type in {"call", "put"}
    ):
        try:
            params = HestonParams(
                torch.tensor(float(heston_misc.get("kappa")), device=HES_DEVICE),
                torch.tensor(float(heston_misc.get("theta")), device=HES_DEVICE),
                torch.tensor(float(heston_misc.get("eta")), device=HES_DEVICE),
                torch.tensor(float(heston_misc.get("rho")), device=HES_DEVICE),
                torch.tensor(float(heston_misc.get("v0")), device=HES_DEVICE),
            )
            r_val = float(get_r(T_years) or 0.0)
            mark_price = _carr_madan_price(
                S0=spot,
                K=strike,
                T=T_years,
                r=r_val,
                q=q_val,
                opt_char="c" if option_type == "call" else "p",
                params=params,
            )
            method = "Heston CM (stored params)"
        except Exception:
            mark_price = None

    def _bs_price_from_legs() -> float | None:
        if not legs:
            return None
        total = 0.0
        used = False
        for leg in legs:
            try:
                leg_type = str(leg.get("option_type", "")).lower()
                k_leg = float(leg.get("strike", 0.0) or 0.0)
                t_leg = float(leg.get("tenor", T_years if T_years is not None else 0.0) or 0.0)
                sigma_leg = float(leg.get("sigma", sigma) or sigma)
                qty_leg = float(leg.get("qty", leg.get("quantity", 1.0)) or 1.0)
                side_mult = -1.0 if str(leg.get("side", "long")).lower() == "short" else 1.0
                if spot <= 0 or k_leg <= 0 or t_leg <= 0:
                    continue
                used = True
                if leg_type == "put":
                    leg_price = bs_price_put(spot, k_leg, r=r_val, q=q_val, sigma=sigma_leg, T=t_leg)
                else:
                    leg_price = bs_price_call(spot, k_leg, r=r_val, q=q_val, sigma=sigma_leg, T=t_leg)
                total += side_mult * qty_leg * leg_price
            except Exception:
                continue
        return total if used else None

    if mark_price is None and spot > 0 and T_years is not None and T_years > 0:
        prod_low = product
        sigma_eff = max(sigma, 1e-6)
        try:
            if "straddle" in prod_low and strike > 0:
                mark_price = price_straddle_bs(spot, strike, r=r_val, q=q_val, sigma=sigma_eff, T=T_years)
                method = "pricing.py straddle (BS)"
            elif "strangle" in prod_low:
                k_put = strike if strike > 0 else None
                k_call = strike2 if strike2 > 0 else None
                for leg in legs:
                    leg_type = str(leg.get("option_type", "")).lower()
                    if leg_type == "put" and leg.get("strike") is not None:
                        k_put = float(leg.get("strike") or k_put or 0.0)
                    if leg_type == "call" and leg.get("strike") is not None:
                        k_call = float(leg.get("strike") or k_call or 0.0)
                if k_put and k_call:
                    if k_put > k_call:
                        k_put, k_call = k_call, k_put
                    mark_price = price_strangle_bs(spot, k_put, k_call, r=r_val, q=q_val, sigma=sigma_eff, T=T_years)
                    method = "pricing.py strangle (BS)"
            elif "call spread" in prod_low:
                k_long = strike if strike > 0 else None
                k_short = strike2 if strike2 > 0 else None
                for leg in legs:
                    if str(leg.get("option_type", "")).lower() != "call":
                        continue
                    s_leg = str(leg.get("side", "long")).lower()
                    if s_leg == "long" and leg.get("strike") is not None:
                        k_long = float(leg.get("strike") or k_long or 0.0)
                    if s_leg == "short" and leg.get("strike") is not None:
                        k_short = float(leg.get("strike") or k_short or 0.0)
                if k_long and k_short:
                    mark_price = price_call_spread_bs(spot, k_long, k_short, r=r_val, q=q_val, sigma=sigma_eff, T=T_years)
                    method = "pricing.py call spread (BS)"
            elif "put spread" in prod_low:
                k_long = strike if strike > 0 else None
                k_short = strike2 if strike2 > 0 else None
                for leg in legs:
                    if str(leg.get("option_type", "")).lower() != "put":
                        continue
                    s_leg = str(leg.get("side", "long")).lower()
                    if s_leg == "long" and leg.get("strike") is not None:
                        k_long = float(leg.get("strike") or k_long or 0.0)
                    if s_leg == "short" and leg.get("strike") is not None:
                        k_short = float(leg.get("strike") or k_short or 0.0)
                if k_long and k_short:
                    mark_price = price_put_spread_bs(spot, k_long, k_short, r=r_val, q=q_val, sigma=sigma_eff, T=T_years)
                    method = "pricing.py put spread (BS)"
            elif "butterfly" in prod_low and "iron" not in prod_low:
                strikes = sorted({float(leg.get("strike")) for leg in legs if leg and leg.get("strike") is not None})
                if len(strikes) >= 3:
                    k1, k2, k3 = strikes[0], strikes[1], strikes[-1]
                elif strike > 0 and strike2 > 0:
                    k1, k3 = strike, strike2
                    k2 = (k1 + k3) / 2.0
                else:
                    k1 = k2 = k3 = 0.0
                if k1 and k2 and k3:
                    mark_price = price_butterfly_bs(spot, k1, k2, k3, r=r_val, q=q_val, sigma=sigma_eff, T=T_years)
                    method = "pricing.py butterfly (BS)"
            elif "condor" in prod_low and "iron" not in prod_low:
                strikes = sorted({float(leg.get("strike")) for leg in legs if leg and leg.get("strike") is not None})
                if len(strikes) >= 4:
                    k1, k2, k3, k4 = strikes[0], strikes[1], strikes[2], strikes[-1]
                    mark_price = price_condor_bs(spot, k1, k2, k3, k4, r=r_val, q=q_val, sigma=sigma_eff, T=T_years)
                    method = "pricing.py condor (BS)"
            elif "iron butterfly" in prod_low:
                k_put_long = None
                k_center = None
                k_call_long = None
                for leg in legs:
                    ltype = str(leg.get("option_type", "")).lower()
                    side_leg = str(leg.get("side", "long")).lower()
                    if ltype == "put" and side_leg == "long" and leg.get("strike") is not None:
                        k_put_long = float(leg.get("strike"))
                    elif ltype == "call" and side_leg == "long" and leg.get("strike") is not None:
                        k_call_long = float(leg.get("strike"))
                    elif side_leg == "short" and leg.get("strike") is not None:
                        k_center = float(leg.get("strike"))
                if not k_center and strike > 0:
                    k_center = strike
                if k_put_long and k_call_long and k_center:
                    mark_price = price_iron_butterfly_bs(
                        spot,
                        k_put_long,
                        k_center,
                        k_call_long,
                        r=r_val,
                        q=q_val,
                        sigma=sigma_eff,
                        T=T_years,
                    )
                    method = "pricing.py iron butterfly (BS)"
            elif "iron condor" in prod_low:
                k_put_long = k_put_short = k_call_short = k_call_long = None
                for leg in legs:
                    ltype = str(leg.get("option_type", "")).lower()
                    side_leg = str(leg.get("side", "long")).lower()
                    if ltype == "put" and side_leg == "long" and leg.get("strike") is not None:
                        k_put_long = float(leg.get("strike"))
                    elif ltype == "put" and side_leg == "short" and leg.get("strike") is not None:
                        k_put_short = float(leg.get("strike"))
                    elif ltype == "call" and side_leg == "short" and leg.get("strike") is not None:
                        k_call_short = float(leg.get("strike"))
                    elif ltype == "call" and side_leg == "long" and leg.get("strike") is not None:
                        k_call_long = float(leg.get("strike"))
                if all(v is not None for v in [k_put_long, k_put_short, k_call_short, k_call_long]):
                    mark_price = price_iron_condor_bs(
                        spot,
                        k_put_long,
                        k_put_short,
                        k_call_short,
                        k_call_long,
                        r=r_val,
                        q=q_val,
                        sigma=sigma_eff,
                        T=T_years,
                    )
                    method = "pricing.py iron condor (BS)"
        except Exception:
            mark_price = None

    if mark_price is None:
        leg_price = _bs_price_from_legs()
        if leg_price is not None:
            mark_price = leg_price
            method = "pricing.py legs (BS sum)"

    # BSM fallback
    if mark_price is None and spot > 0 and strike > 0 and T_years is not None and option_type in {"call", "put"}:
        mark_price = black_scholes_price(
            S=spot,
            K=strike,
            T=T_years,
            r=r_val,
            sigma=max(sigma, 1e-6),
            option_type=option_type,
            q=q_val,
        )
        method = "BSM (CBOE IV)" if chain_entry else "BSM"

    if mark_price is None and spot > 0:
        mark_price = compute_option_payoff(option, spot)
        method = "payoff"

    if mark_price is None:
        mark_price = 0.0

    return spot, T_years, sigma, float(mark_price), method


def _bsm_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> dict:
    """Compute BSM Greeks (delta, gamma, vega, theta, rho) for a call/put."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    r_eff = float(r) - float(q or 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r_eff + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1 * d1)
    if option_type == "call":
        delta = math.exp(-q * T) * norm_cdf(d1)
        theta = (
            - (S * math.exp(-q * T) * pdf_d1 * sigma) / (2 * sqrtT)
            - r * K * math.exp(-r * T) * norm_cdf(d2)
            + q * S * math.exp(-q * T) * norm_cdf(d1)
        )
        rho = K * T * math.exp(-r * T) * norm_cdf(d2)
    else:
        delta = -math.exp(-q * T) * norm_cdf(-d1)
        theta = (
            - (S * math.exp(-q * T) * pdf_d1 * sigma) / (2 * sqrtT)
            + r * K * math.exp(-r * T) * norm_cdf(-d2)
            - q * S * math.exp(-q * T) * norm_cdf(-d1)
        )
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)
    gamma = math.exp(-q * T) * pdf_d1 / (S * sigma * sqrtT)
    vega = S * math.exp(-q * T) * pdf_d1 * sqrtT
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def compute_option_greeks(option: dict, chain_entry: dict | None = None) -> dict:
    """
    Compute Greeks lazily for an option entry using BSM inputs.
    Heavy calls (rates/dividends) happen only when invoked.
    """
    spot, T_years, sigma_used, _, _ = mark_option_market_value(option, chain_entry=chain_entry)
    strike = float(option.get("strike", 0.0) or 0.0)
    option_type = (option.get("option_type") or option.get("type") or "").lower()
    if spot <= 0 or strike <= 0 or T_years is None or T_years <= 0 or sigma_used <= 0 or option_type not in {"call", "put"}:
        return {
            "delta": None,
            "gamma": None,
            "vega": None,
            "theta": None,
            "rho": None,
            "spot": spot,
            "T": T_years,
            "sigma": sigma_used,
        }
    try:
        r_val = float(get_r(T_years) or 0.0)
    except Exception:
        r_val = 0.0
    try:
        q_val = float(get_q(option.get("underlying", "")) or 0.0) if option.get("underlying") else 0.0
    except Exception:
        q_val = 0.0
    greeks = _bsm_greeks(
        S=spot,
        K=strike,
        T=T_years,
        r=r_val,
        sigma=max(sigma_used, 1e-6),
        option_type=option_type,
        q=q_val,
    )
    greeks.update({"spot": spot, "T": T_years, "sigma": sigma_used, "r": r_val, "q": q_val})
    return greeks


def add_option_to_dashboard(record: dict) -> str:
    """
    Normalize and persist an option/structure into the unified options book.

    Args:
        record: Raw option payload (pricing UI or chain selection).
    Returns:
        str: Assigned option id.
    Side effects:
        - Writes to options_portfolio.json (and legacy mirror).
    """
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    book = load_options_book()
    option_id = record.get("id") or record.get("contract_symbol") or record.get("label") or f"opt_{int(time.time() * 1000)}"
    entry = dict(record)
    entry["id"] = option_id
    entry.setdefault("status", "open")
    entry.setdefault("created_at", now)
    entry["last_updated"] = now

    def _f(val):
        return float(val) if val not in (None, "") else None

    # Canonical fields for storage (uniform schema)
    entry["underlying"] = entry.get("underlying") or entry.get("ticker") or entry.get("symbol") or ""
    product_val_raw = entry.get("product") or entry.get("product_type") or entry.get("structure") or "vanilla"
    product_val = str(product_val_raw)
    entry["product"] = product_val
    entry["product_type"] = product_val
    entry["type"] = entry.get("type") or "vanilla"
    pv_lower = product_val.lower()
    entry["option_type"] = (
        entry.get("option_type")
        or entry.get("cpflag")
        or entry.get("cp")
        or ("call" if pv_lower.startswith("call") else "put")
    )
    entry["side"] = entry.get("side") or "long"
    entry["strike"] = _f(entry.get("strike"))
    entry["strike2"] = _f(entry.get("strike2"))
    entry["expiration"] = entry.get("expiration") or None
    entry["quantity"] = _f(entry.get("quantity"))
    entry["S0"] = _f(entry.get("S0"))
    entry["legs"] = entry.get("legs")

    t0_price = entry.get("T_0_price")
    if t0_price is None:
        t0_price = entry.get("avg_price", entry.get("price"))
    entry["T_0_price"] = _f(t0_price)
    entry["avg_price"] = _f(entry["T_0_price"])

    misc_val = entry.get("misc")
    if not isinstance(misc_val, dict):
        misc_val = None
    # Normalization for misc keys
    if isinstance(misc_val, dict) and "product" in entry and "strangle" in str(entry["product"]).lower():
        # rename legacy keys
        if "k_call" in misc_val:
            misc_val["strike_call"] = misc_val.pop("k_call")
        if "k_put" in misc_val:
            misc_val["strike_put"] = misc_val.pop("k_put")
    if isinstance(misc_val, dict) and "product" in entry and "barrier" in str(entry["product"]).lower():
        # ensure knock/direction are both populated for barrier structures
        if "knock" not in misc_val and "direction" in misc_val:
            misc_val["knock"] = misc_val["direction"]
        if "direction" not in misc_val and "knock" in misc_val:
            misc_val["direction"] = misc_val["knock"]
    entry["misc"] = misc_val

    book[option_id] = entry
    save_options_book(book)
    return option_id


def load_custom_options():
    """Load custom options book (pricing-saved) from disk; returns {} on failure."""
    try:
        with open(CUSTOM_OPTIONS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_custom_options(custom_options):
    """Persist custom options to disk."""
    with open(CUSTOM_OPTIONS_FILE, 'w') as f:
        json.dump(custom_options, f, indent=2)


def load_forwards():
    """Load forward positions from disk; returns {} on failure."""
    try:
        with open(FORWARDS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_forwards(forwards):
    """Persist forward positions to disk."""
    with open(FORWARDS_FILE, 'w') as f:
        json.dump(forwards, f, indent=2)


def floor_3(v: float) -> float:
    """
    Floor a numeric value to 3 decimal places (no rounding).
    Example: 1.999999 -> 1.999, 1.994 -> 1.994
    """
    v = float(v or 0.0)
    return math.floor(v * 1000.0) / 1000.0

def buy_asset(symbol, quantity, price):
    """
    Update spot portfolio for a buy (or reducing a short).

    Args:
        symbol: Ticker symbol.
        quantity: Units to add.
        price: Execution price.
    Returns:
        dict | None: Updated position or None if closed.
    """
    portfolio = load_portfolio()
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    position = portfolio.get(symbol)

    if position:
        old_qty = position['quantity']
        old_avg = position['avg_price']
        side = position.get('side', 'long')

        if side == 'long':
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            portfolio[symbol] = {
                'quantity': new_qty,
                'avg_price': round(new_avg, 2),
                'side': 'long',
                'last_updated': now
            }
        else:
            if quantity < old_qty:
                new_qty = old_qty - quantity
                portfolio[symbol] = {
                    'quantity': new_qty,
                    'avg_price': old_avg,
                    'side': 'short',
                    'last_updated': now
                }
            elif quantity == old_qty:
                del portfolio[symbol]
            else:
                new_long_qty = quantity - old_qty
                portfolio[symbol] = {
                    'quantity': new_long_qty,
                    'avg_price': round(price, 2),
                    'side': 'long',
                    'last_updated': now
                }
    else:
        portfolio[symbol] = {
            'quantity': quantity,
            'avg_price': round(price, 2),
            'side': 'long',
            'last_updated': now
        }
    save_portfolio(portfolio)
    return portfolio.get(symbol)


def sell_asset(symbol, quantity, price):
    """
    Update spot portfolio for a sell/short action.

    Args:
        symbol: Ticker symbol.
        quantity: Units to sell.
        price: Execution price.
    Returns:
        bool: True if applied.
    """
    portfolio = load_portfolio()
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    position = portfolio.get(symbol)

    if position:
        old_qty = position['quantity']
        old_avg = position['avg_price']
        side = position.get('side', 'long')

        if side == 'long':
            if quantity < old_qty:
                new_qty = old_qty - quantity
                portfolio[symbol] = {
                    'quantity': new_qty,
                    'avg_price': old_avg,
                    'side': 'long',
                    'last_updated': now
                }
            elif quantity == old_qty:
                del portfolio[symbol]
            else:
                new_short_qty = quantity - old_qty
                portfolio[symbol] = {
                    'quantity': new_short_qty,
                    'avg_price': round(price, 2),
                    'side': 'short',
                    'last_updated': now
                }
        else:
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            portfolio[symbol] = {
                'quantity': new_qty,
                'avg_price': round(new_avg, 2),
                'side': 'short',
                'last_updated': now
            }
    else:
        portfolio[symbol] = {
            'quantity': quantity,
            'avg_price': round(price, 2),
            'side': 'short',
            'last_updated': now
        }

    save_portfolio(portfolio)
    return True


def process_sell_systems():
    """
    Run automated sell systems against current market data.

    - Loads configured systems and current portfolio.
    - Triggers sells when price <= configured levels.
    - Persists level state and portfolio updates.
    Side effects: network price fetches, JSON writes, Streamlit messages.
    """
    sell_systems = load_sell_systems()
    if not sell_systems:
        st.info("No sell systems configured.")
        return

    portfolio = load_portfolio()
    any_executed = False

    for symbol, config in sell_systems.items():
        if config.get("status") != "On":
            continue
        if symbol not in portfolio:
            continue

        current_price_data = get_data(symbol)
        current_price = current_price_data['price']
        if current_price <= 0:
            continue

        levels = config.get("levels", {})
        for level_key, level in levels.items():
            if level.get("triggered"):
                continue

            trigger_price = level.get("price")
            level_qty = int(level.get("quantity", 0))
            if trigger_price is None or level_qty <= 0:
                continue

            current_position_qty = portfolio.get(symbol, {}).get("quantity", 0)
            if current_position_qty <= 0:
                break

            if current_price <= trigger_price:
                qty_to_sell = min(level_qty, current_position_qty)
                if qty_to_sell <= 0:
                    continue

                if sell_asset(symbol, qty_to_sell, current_price):
                    level["triggered"] = True
                    any_executed = True
                    st.success(
                        f"Auto-sell executed for {symbol}: sold {qty_to_sell} units "
                        f"around ${current_price:.2f} (level {level_key})"
                    )
                    portfolio = load_portfolio()

        config["levels"] = levels

    if any_executed:
        save_sell_systems(sell_systems)
    else:
        st.info("No sell levels were triggered based on current market prices.")


def trade_option_contract(
    contract_symbol,
    underlying_symbol,
    option_type,
    strike,
    expiration,
    side,
    quantity,
    price,
    spot_at_trade=None,
):
    """
    Update options portfolio positions with a trade (buy/sell/close).

    Args:
        contract_symbol: OCC or custom id.
        underlying_symbol: Underlying ticker.
        option_type: "call" or "put".
        strike: Strike price.
        expiration: Expiration ISO date.
        side: "long" or "short".
        quantity: Units traded.
        price: Execution price.
        spot_at_trade: Optional underlying level at trade time.
    Returns:
        dict | None: Updated position or None if closed.
    """
    options_portfolio = load_options_portfolio()
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    position = options_portfolio.get(contract_symbol)

    if position:
        old_qty = position['quantity']
        old_avg = position['avg_price']
        old_side = position.get('side', 'long')
        s0 = position.get('S0')

        if old_side == side:
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            options_portfolio[contract_symbol] = {
                'underlying': underlying_symbol,
                'type': option_type,
                'strike': strike,
                'expiration': expiration,
                'quantity': new_qty,
                'avg_price': floor_3(new_avg),
                'S0': s0 if s0 is not None else (floor_3(spot_at_trade) if spot_at_trade is not None else None),
                'side': side,
                'last_updated': now,
            }
        else:
            if quantity < old_qty:
                new_qty = old_qty - quantity
                options_portfolio[contract_symbol] = {
                    'underlying': underlying_symbol,
                    'type': option_type,
                    'strike': strike,
                    'expiration': expiration,
                    'quantity': new_qty,
                    'avg_price': old_avg,
                    'S0': s0,
                    'side': old_side,
                    'last_updated': now,
                }
            elif quantity == old_qty:
                del options_portfolio[contract_symbol]
            else:
                new_qty = quantity - old_qty
                options_portfolio[contract_symbol] = {
                    'underlying': underlying_symbol,
                    'type': option_type,
                    'strike': strike,
                    'expiration': expiration,
                    'quantity': new_qty,
                    'avg_price': floor_3(price),
                    'S0': s0 if s0 is not None else (floor_3(spot_at_trade) if spot_at_trade is not None else None),
                    'side': side,
                    'last_updated': now,
                }
    else:
        options_portfolio[contract_symbol] = {
            'underlying': underlying_symbol,
            'type': option_type,
            'strike': strike,
            'expiration': expiration,
            'quantity': quantity,
            'avg_price': floor_3(price),
            'S0': floor_3(spot_at_trade) if spot_at_trade is not None else None,
            'side': side,
            'last_updated': now,
        }

    save_options_portfolio(options_portfolio)
    return options_portfolio.get(contract_symbol)


def fetch_options_chain(symbol):
    """
    Retrieve delayed CBOE option chain for a given ticker.

    Args:
        symbol: Underlying ticker symbol.
    Returns:
        list[dict]: Chain rows with symbol, strike, expiration, T, price, iv.
    Side effects:
        - Network I/O to cdn.cboe.com.
        - Streamlit error message on failure.
    """
    try:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol.upper()}.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
        data_block = payload.get("data", {})
        options = data_block.get("options", [])
        raw_spot = data_block.get("current_price", 0.0) or 0.0
        spot = floor_3(raw_spot)

        today = datetime.date.today()
        chain = []

        for c in options:
            opt_symbol = c.get("option")
            if not opt_symbol or len(opt_symbol) < 15:
                continue

            # OCC symbology: root + YYMMDD + C/P + 8-digit strike
            date_code = opt_symbol[-15:-9]  # YYMMDD
            cp_flag = opt_symbol[-9]        # C or P
            strike_code = opt_symbol[-8:]   # 8 digits

            try:
                year = 2000 + int(date_code[0:2])
                month = int(date_code[2:4])
                day = int(date_code[4:6])
                expiration = datetime.date(year, month, day)
            except Exception:
                continue

            try:
                strike = int(strike_code) / 1000.0
                strike = floor_3(strike)
            except Exception:
                continue

            days_to_expiry = (expiration - today).days
            T = max(days_to_expiry, 0) / 365.0

            bid = c.get("bid", 0.0) or 0.0
            ask = c.get("ask", 0.0) or 0.0
            last = c.get("last_trade_price", 0.0) or 0.0

            if bid > 0 and ask > 0:
                price = (bid + ask) / 2.0
            elif last > 0:
                price = last
            else:
                price = max(bid, ask)

            price = floor_3(price)

            iv = c.get("iv", 0.0) or 0.0

            chain.append(
                {
                    "symbol": opt_symbol,
                    "underlying": data_block.get("symbol", symbol.upper()),
                    "spot": spot,
                    "type": "call" if cp_flag.upper() == "C" else "put",
                    "strike": strike,
                    "expiration": expiration.isoformat(),
                    "T": T,
                    "price": price,
                    "iv": iv,
                }
            )

        return chain
    except Exception as e:
        st.error(f"Error fetching options from CBOE for {symbol}: {e}")
        return []


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> float:
    """
    Black-Scholes price for a European option.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to maturity in years.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: "call" or "put".
        q: Continuous dividend yield.
    Returns:
        float: BSM theoretical price; falls back to intrinsic if inputs degenerate.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    r_eff = float(r) - float(q or 0.0)
    d1 = (math.log(S / K) + (r_eff + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def get_underlying_close_on_date(symbol: str, date: datetime.date) -> float:
    """
    Try to get the underlying daily close on a given date using Alpaca data.
    Falls back to current price if historical data is unavailable.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return 0.0

    # Primary: yfinance daily close for the specific date
    try:
        start = date.isoformat()
        end = (date + datetime.timedelta(days=1)).isoformat()
        hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1] or 0.0)
    except Exception:
        pass

    # Fallback: latest available price (API or yfinance) to avoid noisy warnings
    price_data = get_data(symbol)
    return float(price_data.get("price", 0.0) or 0.0)


def process_expired_options():
    """
    Mark expired options inside the unified book, computing payoff and realized PnL
    using the underlying close on expiration.
    """
    book = load_options_book()
    if not book:
        return

    today = datetime.date.today()

    changed = False
    for option_id, pos in list(book.items()):
        exp_str = pos.get("expiration")
        if not exp_str:
            continue

        try:
            exp_date = datetime.date.fromisoformat(exp_str)
        except Exception:
            continue

        if exp_date > today or pos.get("status", "open") != "open":
            continue

        underlying = pos.get("underlying")
        quantity = float(pos.get("quantity", 0) or 0)
        if quantity <= 0:
            continue

        S_T = get_underlying_close_on_date(underlying, exp_date) if underlying else 0.0
        pnl = compute_option_pnl(pos, S_T)
        payoff_unit = float(pnl.get("payoff_per_unit", 0.0))

        book[option_id] = {
            **pos,
            **pnl,
            "status": "expired",
            "underlying_close": S_T,
            "closed_at": today.isoformat(),
            "payoff_per_unit": payoff_unit,
        }
        changed = True

    if changed:
        save_options_book(book)

def chatgpt_response(message: str):
    """
    Call OpenAI ChatGPT with portfolio/orders context and user question.

    Args:
        message: User prompt/question.
    Returns:
        str: Assistant response or error string.
    Side effects:
        - Network call to OpenAI API.
        - Reads OPENAI_API_KEY from environment.
    """
    try:
        # Coerce env values to plain strings before passing to OpenAI/httpx
        api_key_env = os.getenv("OPENAI_API_KEY")
        base_url_env = os.getenv("OPENAI_BASE_URL")
        if api_key_env is not None and not isinstance(api_key_env, str):
            api_key_env = os.fspath(api_key_env)
        if base_url_env:
            base_url_env = os.fspath(base_url_env) if not isinstance(base_url_env, str) else base_url_env
        else:
            base_url_env = None

        client = OpenAI(api_key=str(api_key_env) if api_key_env else None, base_url=str(base_url_env) if base_url_env else None)
        portfolio_data = json.dumps(fetch_portfolio(), indent=2)
        open_orders = json.dumps(fetch_open_orders(), indent=2)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI Portfolio Manager specializing in portfolio analysis, "
                    "risk management, and strategic market insights. "
                    "Your goal is to provide data-driven evaluations and professional recommendations."
                )
            },
            {
                "role": "user",
                "content": f"""
                    Here is my current portfolio:
                    {portfolio_data}

                    Here are my open orders:
                    {open_orders}

                    Your tasks:
                    1. Evaluate the risk exposures of my current holdings.
                    2. Analyze the potential impact of open orders.
                    3. Provide insights into portfolio health, diversification, and trade adjustments.
                    4. Speculate on market outlook given current conditions.
                    5. Identify potential risks and suggest mitigation strategies.

                    Finally, answer this specific question:
                    {message}
                    """
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_max_entry_price(symbol):
    """
    Return the maximum filled average price for a symbol from recent orders.

    Args:
        symbol: Ticker symbol to inspect.
    Returns:
        float: Highest filled_avg_price or -1 if none.
    Side effects: Alpaca list_orders call; Streamlit error on failure.
    """
    try:
        orders = api.list_orders(status="filled", limit=50)
        prices = [float(order.filled_avg_price) for order in orders if order.filled_avg_price and order.symbol == symbol]
        return max(prices) if prices else -1
    except Exception as e:
        st.error(f"Error fetching orders: {e}")
        return 0

def place_initial_order(symbol):
    """
    Submit a market buy order (qty=1) via Alpaca.

    Args:
        symbol: Ticker to buy.
    Returns:
        bool: True if submitted, False on error.
    Side effects: Network order placement and Streamlit messaging.
    """
    try:
        api.submit_order(
            symbol=symbol,
            qty=1,
            side="buy",
            type="market",
            time_in_force="gtc"
        )
        st.success(f"Initial order placed for {symbol}")
        time.sleep(2)
        return True
    except Exception as e:
        st.error(f"Error placing initial order: {e}")
        return False

def place_limit_order(symbol, price):
    """
    Submit a limit buy order (qty=1) via Alpaca.

    Args:
        symbol: Ticker to buy.
        price: Limit price.
    Returns:
        bool: True on success, False on error.
    Side effects: Network order placement and Streamlit messaging.
    """
    try:
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='buy',
            type='limit',
            time_in_force='gtc',
            limit_price=price
        )
        st.success(f"Placed limit order for {symbol} @ ${price}")
        return True
    except Exception as e:
        st.error(f"Error placing order: {e}")
        return False

# Always keep cached data fresh on each rerun
st.cache_data.clear()

# Process any expired options and realize their PnL
process_expired_options()

# Streamlit UI
st.title("📈 AI assisted Trading system")
st.markdown("---")

with st.expander("📘 Tutoriel d'utilisation de l'outil"):
    st.markdown("""
    ### 📘 Mise en situation : ce que vous essayez de faire
    
    Imaginez que vous êtes un investisseur particulier qui en a assez de trader "au feeling" : 
    vous voulez arrêter de courir après le marché, structurer vos décisions et savoir exactement 
    pourquoi vous entrez, renforcez ou sortez d'une position.
    
    Cet outil est là pour vous aider à **transformer votre trading en processus**. 
    Vous cherchez principalement à optimiser trois choses :
    
    - **Votre risque** : limiter la profondeur des pertes (drawdown) que vous êtes prêt à accepter
    - **Votre prix moyen d'entrée** : profiter des baisses pour améliorer vos points d'achat au lieu de paniquer
    - **Votre temps et votre charge mentale** : automatiser ce qui peut l'être, et garder votre énergie pour les décisions importantes
    
    L'idée n'est pas de prédire le futur, mais de mettre un cadre autour de votre comportement : 
    ouvrir l'application, voir en quelques secondes si tout est sous contrôle, puis décider calmement 
    s'il y a une action à prendre aujourd'hui ou non.
    
    ---
    
    ### 🧭 Comment lire l'application
    
    L'application est construite comme un **parcours logique de trader discipliné** :
    
    1. Vous **observez** votre situation (ce que vous possédez, comment ça évolue, ce que vos systèmes ont fait)
    2. Vous **décidez** où mettre du capital, ce que vous voulez renforcer ou alléger
    3. Vous **exécutez** des ordres manuels quand vous voulez intervenir directement
    4. Vous **automatisez** certaines parties de votre stratégie avec des systèmes basés sur le drawdown et des niveaux de prix
    5. Vous **analysez** avec l'IA pour challenger vos idées, comprendre vos risques et clarifier votre stratégie
    
    Chaque onglet correspond à une étape de ce parcours. 
    Dans les sections "📚 Comment utiliser ..." au bas de chaque onglet, 
    vous trouverez une explication détaillée de **ce que vous êtes en train d'y faire** 
    (qu'est-ce que vous optimisez, quels sont les enjeux, où se situe la difficulté mentale).
    
    ---
    
    ### 🎯 Comment bien démarrer
    
    Pour une première utilisation, vous pouvez suivre ce mini-scénario :
    
    1. Ouvrez l'application et prenez un moment pour comprendre que le but n'est pas de "trader plus", 
       mais de **trader mieux, avec un plan**.
    2. Allez ensuite dans chaque onglet, l'un après l'autre, sans forcément passer d'ordres au début :
       contentez-vous de lire le texte d'aide en bas de page et de repérer les boutons qui déclenchent de vraies actions.
    3. Quand vous vous sentez à l'aise, commencez petit : 
       un premier achat manuel, un premier système dans Trading Systems, une première question à l'IA.
    4. Revenez quelques jours plus tard pour voir ce qui s'est passé : 
       avez-vous respecté votre plan ? vos systèmes ont-ils réagi comme prévu ? qu'avez-vous appris ?
    
    Utilisé de cette manière, cet outil devient **un cadre d'apprentissage et d'optimisation** : 
    à chaque utilisation, vous comprenez un peu mieux votre propre comportement de trader, 
    et vous ajustez votre manière d'investir pour qu'elle soit plus cohérente, plus sereine et plus alignée avec votre tolérance au risque.
    """)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Data is refreshed automatically on each interaction")

    st.divider()
    st.subheader("🧹 Maintenance / Reset")

    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def _safe_truncate(path: Path) -> None:
        try:
            if path.exists():
                path.write_text("")
        except Exception:
            pass

    def _reset_all():
        try:
            # Options caches
            for p in [
                CACHE_OPTIONS_HISTORY_FILE,
                CACHE_OPTIONS_CALLS_FILE,
                CACHE_OPTIONS_PUTS_FILE,
                CACHE_OPTIONS_META_FILE,
                LEGACY_OPTIONS_META_FILE,
                TRAIN_FILE,
                TEST_FILE,
                BASKET_CLOSING_FILE,
                CLOSING_CACHE_FILE,
                CACHE_CSV_DIR / "closing_prices.csv",
            ]:
                _safe_truncate(p)
            # Books & expirées
            save_options_book({})
            _safe_truncate(LEGACY_EXPIRED_FILE)
            # Forwards / trading systems / portfolio
            save_forwards({})
            save_sell_systems({})
            save_portfolio({})
            _safe_truncate(JSON_DIR / "equities.json")
            _safe_truncate(HESTON_PARAMS_FILE)
            st.cache_data.clear()
            st.success("Reset complet : caches options, book, expirées, forwards, trading systems, portfolio, equities.json, heston_params.json.")
        except Exception as exc:
            st.error(f"Reset complet impossible : {exc}")

    if st.button("🗑️ Vider cache Options (calls/puts/meta)", key="btn_clear_opt_cache"):
        for p in [
            CACHE_OPTIONS_HISTORY_FILE,
            CACHE_OPTIONS_CALLS_FILE,
            CACHE_OPTIONS_PUTS_FILE,
            CACHE_OPTIONS_META_FILE,
            LEGACY_OPTIONS_META_FILE,
            TRAIN_FILE,
            TEST_FILE,
            BASKET_CLOSING_FILE,
            CLOSING_CACHE_FILE,
            CACHE_CSV_DIR / "closing_prices.csv",
        ]:
            _safe_truncate(p)
        try:
            pass
        except Exception:
            pass
        st.success("Caches options (calls/puts/meta/history/train/test/closing) vidés.")

    if st.button("🗑️ Vider book Options (open + expired)", key="btn_clear_opt_book"):
        try:
            save_options_book({})
            _safe_truncate(LEGACY_EXPIRED_FILE)
            st.success("Options (open + expired) réinitialisées.")
        except Exception as exc:
            st.error(f"Impossible de vider le book options : {exc}")

    if st.button("🗑️ Vider options expirées uniquement", key="btn_clear_opt_expired"):
        try:
            active = load_options_portfolio()
            save_options_book(active)
            _safe_truncate(LEGACY_EXPIRED_FILE)
            st.success("Options expirées supprimées (open conservées).")
        except Exception as exc:
            st.error(f"Impossible de vider les expirées : {exc}")

    if st.button("🗑️ Vider forwards", key="btn_clear_forwards"):
        try:
            save_forwards({})
            st.success("Forwards réinitialisés.")
        except Exception as exc:
            st.error(f"Impossible de vider les forwards : {exc}")

    if st.button("🗑️ Vider trading systems", key="btn_clear_trading_systems"):
        try:
            save_sell_systems({})
            _safe_truncate(JSON_DIR / "equities.json")
            st.success("Trading systems réinitialisés.")
        except Exception as exc:
            st.error(f"Impossible de vider les trading systems : {exc}")

    if st.button("🗑️ Vider portfolio (spot)", key="btn_clear_portfolio"):
        try:
            save_portfolio({})
            st.success("Portfolio spot réinitialisé.")
        except Exception as exc:
            st.error(f"Impossible de vider le portfolio : {exc}")

    if st.button("🧨 Reset total (tout vider)", key="btn_reset_all"):
        _reset_all()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "💰 Buy/Sell",
    "📋 Trading Systems",
    "📄 Forwards",
    "📜 Options",
])
# Rendre la barre d'onglets principale sticky pour qu'elle reste visible au scroll.
st.markdown(
    """
    <style>
    .sticky-main-tabs {
        position: sticky;
        top: 0;
        z-index: 999;
        background: var(--background-color, #fff);
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    }
    </style>
    <script>
    const lists = window.parent.document.querySelectorAll('div[data-baseweb="tab-list"]');
    if (lists.length > 0) {
        const mainTabs = lists[0];
        if (mainTabs && !mainTabs.classList.contains('sticky-main-tabs')) {
            mainTabs.classList.add('sticky-main-tabs');
        }
    }
    </script>
    """,
    unsafe_allow_html=True,
)
if st.session_state.pop("_switch_to_dashboard", False):
    st.markdown(
        """
        <script>
        const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
        for (const t of tabs) {
          const label = (t.innerText || "").trim();
          if (label.includes("📊 Dashboard") || label.includes("Dashboard")) {
            t.click();
            break;
          }
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

# Tab 1: Dashboard
with tab1:
    # Préchargement des données Dashboard et du cache prix
    my_portfolio = load_portfolio()
    forwards_dash = load_forwards()
    custom_opts = {k: v for k, v in load_options_book().items() if v.get("status", "open") == "open"}
    expired_options_data = load_expired_options()
    dashboard_tickers = collect_dashboard_tickers(
        portfolio=my_portfolio,
        forwards=forwards_dash,
        custom_options=custom_opts,
    )
    dashboard_cache = load_dashboard_cache()
    today = datetime.date.today()
    cached_prices = dashboard_cache.get("prices") or {}
    cached_symbols = set(cached_prices.keys())
    cache_refresh_date = dashboard_cache_last_refresh(dashboard_cache)
    if cache_refresh_date != today or dashboard_tickers - cached_symbols:
        dashboard_cache = refresh_dashboard_cache(dashboard_tickers, cache=dashboard_cache)
        cached_prices = dashboard_cache.get("prices") or {}
    dashboard_prices = cached_prices
    last_refresh_ts = dashboard_cache.get("last_refresh")

    def _dashboard_price(sym: str, fallback: float = 0.0) -> float:
        """Return cached dashboard price when available, otherwise live price."""
        sym_clean = (sym or "").strip().upper()
        if sym_clean and dashboard_prices:
            cached_val = dashboard_prices.get(sym_clean)
            try:
                if cached_val is not None:
                    return float(cached_val)
            except Exception:
                pass
        live_price = float(get_data(sym_clean).get("price", fallback) or fallback)
        if live_price > 0 and sym_clean:
            dashboard_prices[sym_clean] = live_price
        return live_price

    # Quick P&L synthesis (spot, options, forwards) before AI widget
    spot_total_pnl = 0.0
    spot_total_notional = 0.0
    for symbol, data in my_portfolio.items():
        avg_price = float(data.get("avg_price", 0.0) or 0.0)
        quantity = float(data.get("quantity", 0.0) or 0.0)
        side = (data.get("side") or "long").lower()
        current_price = _dashboard_price(symbol, avg_price)
        position_value = quantity * current_price
        notional = avg_price * quantity
        if side == "long":
            pnl = (current_price - avg_price) * quantity
        else:
            pnl = (avg_price - current_price) * quantity
        spot_total_pnl += pnl
        spot_total_notional += notional

    # Options P&L : uniquement réalisé (options expirées), aucun pricing live au chargement
    open_options_pnl = 0.0
    realized_options_pnl = sum(
        float(opt.get("pnl_total", 0.0) or 0.0) for opt in expired_options_data.values()
    )
    total_options_pnl = realized_options_pnl

    # Forwards P&L
    fwd_positions = forwards_dash
    total_forward_pnl = 0.0
    total_forward_notional = 0.0
    for fwd in fwd_positions.values():
        qty = float(fwd.get("quantity", 0.0) or 0.0)
        if qty <= 0:
            continue
        side = (fwd.get("side") or "long").lower()
        mult = 1.0 if side == "long" else -1.0
        sym = fwd.get("symbol", "")
        spot_now = _dashboard_price(sym, 0.0) if sym else 0.0
        price_fwd = float(fwd.get("forward_price", 0.0) or 0.0)
        total_forward_pnl += mult * (spot_now - price_fwd) * qty
        total_forward_notional += abs(price_fwd * qty)

    st.markdown("### 📊 Synthèse P&L")
    col_spot, col_opt, col_fwd = st.columns(3)
    with col_spot:
        delta_spot = f"{(spot_total_pnl / spot_total_notional * 100):.2f}%" if spot_total_notional > 0 else None
        st.metric("P&L Spot", f"${spot_total_pnl:.2f}", delta=delta_spot)
    with col_opt:
        st.metric("P&L Options (open+réalisé)", f"${total_options_pnl:.2f}", delta=total_options_pnl)
        st.caption(f"Réalisé (options expirées uniquement) {realized_options_pnl:+.2f}")
    with col_fwd:
        delta_fwd = f"{(total_forward_pnl / total_forward_notional * 100):.2f}%" if total_forward_notional > 0 else None
        st.metric("P&L Forwards", f"${total_forward_pnl:.2f}", delta=delta_fwd)

    # AI Assistant section moved here
    st.subheader("🤖 AI Portfolio Assistant")
    st.markdown("Ask questions about your portfolio and get AI-powered insights")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your portfolio..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatgpt_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown("---")

    with st.expander("📘 Comprendre le Dashboard"):
        st.markdown("""
        ### 📊 Ce que vous faites dans le Dashboard
        
        Le Dashboard est votre **vue d’ensemble** : en haut, vous discutez avec l’IA à propos de votre portfolio; en dessous,
        vous voyez vos positions chiffrées de façon froide et objective.
        
        La section *AI Portfolio Manager* sert à poser vos questions stratégiques : risques, diversification, idées de gestion,
        explications de concepts. L’IA répond en tenant compte de vos positions et de vos ordres ouverts.
        
        Le bloc *My Portfolio* liste chaque actif (long ou short), avec quantité, prix moyen, prix spot, valeur et P&L.
        Plus bas, la section *Configured Trading Systems* montre vos robots actifs ou en attente.
        """)
    st.subheader("💰 My Portfolio")
    
    if my_portfolio:
        portfolio_data = []
        total_value = 0
        total_pnl = 0
        total_notional = 0

        for symbol, data in my_portfolio.items():
            avg_price = data['avg_price']
            quantity = data['quantity']
            side = data.get('side', 'long')

            current_price = _dashboard_price(symbol, avg_price)
            position_value = quantity * current_price

            if side == 'long':
                pnl = (current_price - avg_price) * quantity
            else:
                pnl = (avg_price - current_price) * quantity

            notional = avg_price * quantity
            pnl_pct = (pnl / notional * 100) if notional > 0 else 0

            total_value += position_value
            total_pnl += pnl
            total_notional += notional
            
            portfolio_data.append({
                'Symbol': symbol,
                'Side': side.capitalize(),
                'Quantity': quantity,
                'S_0 Price': f"${avg_price:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Value': f"${position_value:.2f}",
                'P&L': f"${pnl:.2f}",
                'P&L %': f"{pnl_pct:.2f}%",
                '_value_raw': position_value,
            })
        
        # Allocate % of each position relative to total portfolio value
        for row in portfolio_data:
            alloc = (row.get("_value_raw", 0.0) / total_value * 100) if total_value > 0 else 0.0
            row["Allocation %"] = f"{alloc:.2f}%"
        df_my_portfolio = pd.DataFrame(portfolio_data)
        # Drop helper column used for allocation and reorder columns
        df_my_portfolio = df_my_portfolio.drop(columns=[c for c in df_my_portfolio.columns if str(c).startswith("_")], errors="ignore")
        col_order = [
            "Allocation %",
            "Symbol",
            "Side",
            "Quantity",
            "S_0 Price",
            "Current Price",
            "Value",
            "P&L",
            "P&L %",
        ]
        df_my_portfolio = df_my_portfolio[[c for c in col_order if c in df_my_portfolio.columns]]
        st.dataframe(df_my_portfolio, width="stretch", hide_index=True)

        refresh_cols = st.columns([1, 3])
        with refresh_cols[0]:
            if st.button("🔄 Rafraîchir les prix Dashboard", key="btn_refresh_dashboard_prices"):
                refreshed_cache = refresh_dashboard_cache(
                    collect_dashboard_tickers(my_portfolio, forwards_dash, custom_opts)
                )
                dashboard_prices = refreshed_cache.get("prices") or {}
                last_refresh_ts = refreshed_cache.get("last_refresh")
                st.success("Prix du dashboard rafraîchis.")
                try:
                    st.rerun()
                except Exception:
                    pass
        with refresh_cols[1]:
            refresh_label = "jamais" if not last_refresh_ts else str(last_refresh_ts)
            try:
                if last_refresh_ts:
                    dt_refresh = datetime.datetime.fromisoformat(str(last_refresh_ts))
                    refresh_label = dt_refresh.strftime("%Y-%m-%d %H:%M")
            except Exception:
                refresh_label = str(last_refresh_ts)
            st.caption(f"Dernier refresh: {refresh_label}")

        # Forward portfolio section
        st.markdown("---")
        st.markdown("### 🚀 Forward Portfolio")
        if forwards_dash:
            fwd_rows = []
            today_dash = datetime.date.today()
            for key, fwd in forwards_dash.items():
                sym = fwd.get("symbol", "")
                qty = int(fwd.get("quantity", 0) or 0)
                price_fwd = float(fwd.get("forward_price", 0.0) or 0.0)
                side = fwd.get("side", "long")
                maturity_str = fwd.get("maturity")
                try:
                    maturity_dt = datetime.date.fromisoformat(maturity_str)
                    days_to_mat = (maturity_dt - today_dash).days
                except Exception:
                    maturity_dt = None
                    days_to_mat = None
                spot_now = _dashboard_price(sym, 0.0) if sym else 0.0
                mult = 1.0 if side == "long" else -1.0
                pnl_unit = mult * (spot_now - price_fwd)
                pnl_total = pnl_unit * qty
                fwd_rows.append(
                    {
                        "Symbol": sym,
                        "Side": side.capitalize(),
                        "Quantity": qty,
                        "Forward Price": price_fwd,
                        "Spot Now": round(spot_now, 4),
                        "Maturity": maturity_str,
                        "Days to mat": days_to_mat,
                        "P&L/unit": round(pnl_unit, 4),
                        "P&L total": round(pnl_total, 2),
                    }
                )
            df_fwds = pd.DataFrame(fwd_rows)
            st.dataframe(df_fwds, width="stretch", hide_index=True)
        else:
            st.info("No forward positions yet.")

        st.markdown("---")
        st.markdown("### 🧾 Custom option purchases (acheté depuis pricing)")
        if custom_opts:
            def _price_from_pricing_py(pos: dict, chain_entry: dict | None = None):
                """Pricing via notebooks/scripts/pricing.py helpers (BS approximations)."""
                product_raw = (
                    pos.get("product_type")
                    or pos.get("structure")
                    or pos.get("product")
                    or pos.get("type")
                    or ""
                )
                product = str(product_raw).lower()
                underlying = pos.get("underlying")
                spot = float(pos.get("S0", 0.0) or 0.0)
                if chain_entry:
                    spot = float(chain_entry.get("spot", spot) or spot)
                if spot <= 0 and underlying:
                    try:
                        spot = _dashboard_price(underlying, 0.0)
                    except Exception:
                        spot = 0.0

                strike = float(pos.get("strike", 0.0) or 0.0)
                strike2 = float(pos.get("strike2", 0.0) or 0.0) if pos.get("strike2") else None
                sigma = float(pos.get("sigma", 0.2) or 0.2)
                if chain_entry:
                    sigma = float(chain_entry.get("iv", sigma) or sigma)
                try:
                    r_val = float(pos.get("r", get_r(None) or 0.0))
                except Exception:
                    r_val = 0.0
                try:
                    q_val = float(pos.get("q", get_q(underlying) or 0.0) if underlying else pos.get("q", 0.0) or 0.0)
                except Exception:
                    q_val = float(pos.get("q", 0.0) or 0.0)

                T_years = None
                if pos.get("expiration"):
                    try:
                        expiry_date = datetime.date.fromisoformat(pos.get("expiration"))
                        days = (expiry_date - datetime.date.today()).days
                        T_years = max(days, 0) / 365.0
                    except Exception:
                        T_years = None
                if T_years is None:
                    try:
                        T_years = float(pos.get("maturity_years") or 0.0)
                    except Exception:
                        T_years = None
                if chain_entry:
                    T_years = float(chain_entry.get("T", T_years or 0.0) or (T_years or 0.0))
                if T_years is None or T_years <= 0:
                    T_years = float(common_maturity_value if 'common_maturity_value' in globals() else 1.0)

                method = "pricing.py"
                mark = None

                legs = pos.get("legs") or []

                if "straddle" in product and strike > 0:
                    mark = price_straddle_bs(spot, strike, r=r_val, q=q_val, sigma=sigma, T=T_years)
                    method = "pricing.py straddle (BS)"
                elif "strangle" in product:
                    k_put = strike if strike > 0 else None
                    k_call = strike2 if strike2 and strike2 > 0 else None
                    for leg in legs:
                        if str(leg.get("option_type", "")).lower() == "put" and leg.get("strike") is not None:
                            k_put = float(leg.get("strike"))
                        if str(leg.get("option_type", "")).lower() == "call" and leg.get("strike") is not None:
                            k_call = float(leg.get("strike"))
                    if k_put and k_call:
                        if k_put > k_call:
                            k_put, k_call = k_call, k_put
                        mark = price_strangle_bs(spot, k_put, k_call, r=r_val, q=q_val, sigma=sigma, T=T_years)
                        method = "pricing.py strangle (BS)"
                elif "call spread" in product:
                    k_long = strike
                    k_short = strike2 if strike2 else strike
                    for leg in legs:
                        if str(leg.get("option_type", "")).lower() == "call":
                            if str(leg.get("side", "long")).lower() == "long":
                                k_long = float(leg.get("strike", k_long))
                            else:
                                k_short = float(leg.get("strike", k_short))
                    if k_long and k_short:
                        mark = price_call_spread_bs(spot, k_long, k_short, r=r_val, q=q_val, sigma=sigma, T=T_years)
                        method = "pricing.py call spread (BS)"
                elif "put spread" in product:
                    k_long = strike
                    k_short = strike2 if strike2 else strike
                    for leg in legs:
                        if str(leg.get("option_type", "")).lower() == "put":
                            if str(leg.get("side", "long")).lower() == "long":
                                k_long = float(leg.get("strike", k_long))
                            else:
                                k_short = float(leg.get("strike", k_short))
                    if k_long and k_short:
                        mark = price_put_spread_bs(spot, k_long, k_short, r=r_val, q=q_val, sigma=sigma, T=T_years)
                        method = "pricing.py put spread (BS)"
                elif "butterfly" in product and "iron" not in product:
                    strikes = sorted({float(leg.get("strike")) for leg in legs if leg.get("strike") is not None})
                    if len(strikes) >= 3:
                        k1, k2, k3 = strikes[0], strikes[1], strikes[-1]
                        mark = price_butterfly_bs(spot, k1, k2, k3, r=r_val, q=q_val, sigma=sigma, T=T_years)
                        method = "pricing.py butterfly (BS)"
                elif "condor" in product and "iron" not in product:
                    strikes = sorted({float(leg.get("strike")) for leg in legs if leg.get("strike") is not None})
                    if len(strikes) >= 4:
                        k1, k2, k3, k4 = strikes[0], strikes[1], strikes[2], strikes[-1]
                        mark = price_condor_bs(spot, k1, k2, k3, k4, r=r_val, q=q_val, sigma=sigma, T=T_years)
                        method = "pricing.py condor (BS)"
                elif "iron butterfly" in product:
                    k_put_long = k_call_long = k_center = None
                    for leg in legs:
                        ltype = str(leg.get("option_type", "")).lower()
                        side_leg = str(leg.get("side", "long")).lower()
                        if ltype == "put" and side_leg == "long":
                            k_put_long = float(leg.get("strike"))
                        elif ltype == "call" and side_leg == "long":
                            k_call_long = float(leg.get("strike"))
                        elif side_leg == "short":
                            k_center = float(leg.get("strike"))
                    if k_put_long and k_call_long and k_center:
                        mark = price_iron_butterfly_bs(spot, k_put_long, k_center, k_call_long, r=r_val, q=q_val, sigma=sigma, T=T_years)
                        method = "pricing.py iron butterfly (BS)"
                elif "iron condor" in product:
                    k_put_long = k_put_short = k_call_short = k_call_long = None
                    for leg in legs:
                        ltype = str(leg.get("option_type", "")).lower()
                        side_leg = str(leg.get("side", "long")).lower()
                        if ltype == "put" and side_leg == "long":
                            k_put_long = float(leg.get("strike"))
                        elif ltype == "put" and side_leg == "short":
                            k_put_short = float(leg.get("strike"))
                        elif ltype == "call" and side_leg == "short":
                            k_call_short = float(leg.get("strike"))
                        elif ltype == "call" and side_leg == "long":
                            k_call_long = float(leg.get("strike"))
                    if all(v is not None for v in [k_put_long, k_put_short, k_call_short, k_call_long]):
                        mark = price_iron_condor_bs(spot, k_put_long, k_put_short, k_call_short, k_call_long, r=r_val, q=q_val, sigma=sigma, T=T_years)
                        method = "pricing.py iron condor (BS)"
                elif "digital" in product:
                    view = view_digital(spot, strike, T=T_years, r=r_val, q=q_val, sigma=sigma, option_type=option_type, payout=float((pos.get("misc") or {}).get("payout", 1.0) if isinstance(pos.get("misc"), dict) else 1.0))
                    mark = float(view.get("premium", 0.0))
                    method = "pricing.py digital (BS)"
                elif "asset-or-nothing" in product:
                    view = view_asset_or_nothing(spot, strike, T=T_years, r=r_val, q=q_val, sigma=sigma, option_type=option_type)
                    mark = float(view.get("premium", 0.0))
                    method = "pricing.py asset-or-nothing (BS)"
                elif "chooser" in product:
                    view = view_chooser(spot, strike, T=T_years, r=r_val, q=q_val, sigma=sigma)
                    mark = float(view.get("premium", 0.0))
                    method = "pricing.py chooser (BS)"
                elif "quanto" in product:
                    fx = float((pos.get("misc") or {}).get("fx_rate", 1.0) if isinstance(pos.get("misc"), dict) else 1.0)
                    view = view_quanto(spot, strike, fx_rate=fx, option_type=option_type or "call", r=r_val, q=q_val, sigma=sigma, T=T_years)
                    mark = float(view.get("premium", 0.0))
                    method = "pricing.py quanto (BS approx)"
                elif "rainbow" in product:
                    s2 = float((pos.get("misc") or {}).get("S2", 0.0) if isinstance(pos.get("misc"), dict) else 0.0) or spot
                    view = view_rainbow(spot, s2, strike, option_type=option_type or "call", r=r_val, q=q_val, sigma=sigma, T=T_years)
                    mark = float(view.get("premium", 0.0))
                    method = "pricing.py rainbow (best-of approx)"
                elif "calendar" in product:
                    T_short = float((pos.get("misc") or {}).get("T_short", max(T_years * 0.5, 0.05)) if isinstance(pos.get("misc"), dict) else max(T_years * 0.5, 0.05))
                    T_long = float((pos.get("misc") or {}).get("T_long", T_years) if isinstance(pos.get("misc"), dict) else T_years)
                    view = view_calendar_spread(spot, strike, T_short, T_long, r=r_val, q=q_val, sigma=sigma, option_type=option_type or "call")
                    mark = float(view.get("premium", 0.0))
                    method = "pricing.py calendar spread (BS)"
                elif "diagonal" in product:
                    T_near = float((pos.get("misc") or {}).get("T_near", max(T_years * 0.5, 0.05)) if isinstance(pos.get("misc"), dict) else max(T_years * 0.5, 0.05))
                    T_far = float((pos.get("misc") or {}).get("T_far", T_years) if isinstance(pos.get("misc"), dict) else T_years)
                    k_far = strike2 if strike2 else strike
                    view = view_diagonal_spread(spot, strike, k_far, T_near, T_far, r=r_val, q=q_val, sigma=sigma, option_type=option_type or "call")
                    mark = float(view.get("premium", 0.0))
                    method = "pricing.py diagonal spread (BS)"
                elif "barrier" in product:
                    misc = pos.get("misc") if isinstance(pos.get("misc"), dict) else {}
                    barrier = float(misc.get("barrier") or misc.get("barrier_level") or (strike * 1.1 if strike else spot * 1.1))
                    direction = str(misc.get("direction", misc.get("barrier_type", "up"))).lower()
                    knock = str(misc.get("knock", "out")).lower()
                    view = view_barrier(spot, strike, barrier, direction=direction, knock=knock, option_type=option_type or "call", payout=float(misc.get("payout", 1.0) or 1.0), binary=False, r=r_val, q=q_val, sigma=sigma, T=T_years)
                    mark = float(view.get("premium", 0.0))
                    method = "pricing.py barrier (proxy)"

                return spot, T_years, sigma, mark, method

            chains_by_underlying_custom: dict[str, list[dict]] = {}
            def _pricing_metadata(prod: str) -> tuple[str, list[str]]:
                prod_low = (prod or "").lower()
                if "barrier" in prod_low:
                    return "barrier_pricer", ["option_type", "barrier_type", "knock", "direction", "barrier_level", "strike", "expiration", "sigma", "S0", "r", "q"]
                if "asian" in prod_low:
                    return "asian_pricer", ["option_type", "strike", "expiration", "sigma", "S0", "r", "q", "n_obs"]
                if "digital" in prod_low:
                    return "digital_pricer", ["option_type", "strike", "expiration", "sigma", "S0", "r", "q", "payout"]
                if "asset-or-nothing" in prod_low:
                    return "asset_or_nothing_pricer", ["option_type", "strike", "expiration", "sigma", "S0", "r", "q"]
                if "forward-start" in prod_low:
                    return "forward_start_pricer", ["option_type", "strike", "T_start", "expiration", "sigma", "S0", "r", "q"]
                if "chooser" in prod_low:
                    return "chooser_pricer", ["option_type", "strike", "expiration", "t_choice", "sigma", "S0", "r", "q"]
                if "straddle" in prod_low:
                    return "straddle_pricer", ["S0", "strike", "expiration", "sigma", "r", "q"]
                if "strangle" in prod_low:
                    return "strangle_pricer", ["S0", "strike_put", "strike_call", "expiration", "sigma", "r", "q"]
                if "spread" in prod_low or "butterfly" in prod_low or "condor" in prod_low:
                    return "spread_pricer", ["legs (strike/side/type)", "expiration", "sigma", "S0", "r", "q"]
                if "lookback" in prod_low:
                    return "lookback_pricer", ["option_type", "strike", "expiration", "sigma", "S0", "r", "q", "n_paths", "n_steps"]
                if "cliquet" in prod_low or "ratchet" in prod_low:
                    return "cliquet_pricer", ["n_periods", "cap", "floor", "sigma", "S0", "r", "q", "n_paths"]
                return "vanilla_bsm", ["option_type", "strike", "expiration", "sigma", "S0", "r", "q"]
            rows_custom = []
            def _safe_f(val, default=0.0):
                try:
                    return float(val)
                except Exception:
                    return default
            for key, pos in custom_opts.items():
                try:
                    underlying = pos.get("underlying")
                    option_type_raw = (
                        pos.get("option_type")
                        or pos.get("cpflag")
                        or pos.get("cp_flag")
                        or pos.get("cp")
                        or ""
                    )
                    if not option_type_raw:
                        t_val = str(pos.get("type", "")).lower()
                        if t_val in {"call", "put"}:
                            option_type_raw = t_val
                    option_type = str(option_type_raw).lower()
                    strike = _safe_f(pos.get("strike", 0.0), 0.0)
                    strike2_val = pos.get("strike2")
                    strike2 = _safe_f(strike2_val, None) if strike2_val not in (None, "") else None
                    quantity = _safe_f(pos.get("quantity", 0), 0.0)
                    avg_price = _safe_f(pos.get("avg_price", pos.get("T_0_price", 0.0)), 0.0)
                    side = pos.get("side", "long").lower()
                    product = (
                        pos.get("product_type")
                        or pos.get("structure")
                        or pos.get("product")
                        or pos.get("type")
                        or "vanilla"
                    )
                    misc = pos.get("misc")

                    rows_custom.append({
                        "ID/Contract": key,
                        "Product": product,
                        "Underlying": underlying,
                        "Type": option_type.capitalize(),
                        "Side": side.capitalize(),
                        "Strike": strike,
                        "Strike2": strike2,
                        "Expiration": pos.get("expiration"),
                        "Quantity": quantity,
                        "Misc (json)": json.dumps(misc or {}, ensure_ascii=False),
                    })
                except Exception as exc:
                    rows_custom.append({
                        "ID/Contract": key,
                        "Product": pos.get("product"),
                        "Error": str(exc),
                    })

        if "rows_custom" not in locals():
            rows_custom = []
        df_custom = pd.DataFrame(rows_custom) if rows_custom else pd.DataFrame(
            columns=["ID/Contract", "Product", "Underlying", "Type", "Side", "Strike", "Strike2", "Expiration", "Quantity", "Misc (json)"]
        )
        st.dataframe(df_custom, width="stretch", hide_index=True)

        st.markdown("#### Profils payoff / P&L (options ouvertes)")
        spot_cache: dict[str, float | None] = {}

        def _spot_live(sym: str) -> float | None:
            if not sym:
                return None
            sym_clean = (sym or "").strip().upper()
            if sym_clean in spot_cache:
                return spot_cache[sym_clean]
            if dashboard_prices.get(sym_clean) is not None:
                try:
                    cached_val = float(dashboard_prices[sym_clean])
                    spot_cache[sym_clean] = cached_val
                    return cached_val
                except Exception:
                    pass
            spot_val = get_spot_cboe_cached(sym_clean)
            if spot_val is None or spot_val <= 0:
                spot_val = _dashboard_price(sym_clean, 0.0)
            spot_cache[sym_clean] = spot_val
            return spot_val

        if not custom_opts:
            st.info("Aucune option custom enregistrée.")

        for key, pos in custom_opts.items():
            underlying = (pos.get("underlying") or "").upper()
            avg_price = float(pos.get("avg_price", pos.get("T_0_price", 0.0)) or 0.0)
            side = (pos.get("side") or "long").lower()
            current_spot = _spot_live(underlying) or float(pos.get("S0", 0.0) or 0.0)
            strike = float(pos.get("strike", 0.0) or 0.0)
            base_ref = current_spot if current_spot > 0 else (strike if strike > 0 else 1.0)
            s_grid = np.linspace(max(0.01, 0.5 * base_ref), 1.5 * base_ref, 180)
            pay_grid = np.array([compute_option_payoff(pos, s) for s in s_grid], dtype=float)
            if side == "short":
                pay_grid *= -1.0
            pnl_grid = pay_grid - avg_price if side == "long" else avg_price + pay_grid

            _, _, _, mark_price, method = _price_from_pricing_py(pos, None)

            with st.expander(f"{key} – {pos.get('product_type', pos.get('product')) or 'Option'} ({underlying})"):
                st.caption(f"Spot (CBOE/cache): {current_spot:.4f}" if current_spot else "Spot indisponible")
                st.metric("Payoff actuel", f"${(pay_grid[np.searchsorted(s_grid, current_spot, side='left') - 1] if current_spot else 0.0):.4f}", help=f"Méthode: {method}")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(s_grid, pay_grid, label="Payoff (signé)")
                ax.plot(s_grid, pnl_grid, label="P&L (vs. prime)", color="darkorange")
                if strike > 0:
                    ax.axvline(strike, color="gray", linestyle="--", label=f"K = {strike:.2f}")
                if current_spot:
                    ax.axvline(current_spot, color="crimson", linestyle="-.", label=f"S_now = {current_spot:.2f}")
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Spot")
                ax.set_ylabel("Payoff / P&L")
                ax.legend(loc="best")
                ax.grid(alpha=0.3, linestyle="--")
                st.pyplot(fig, clear_figure=True)

            with st.expander("Manage custom options", expanded=False):
                for key, pos in custom_opts.items():
                    snap = st.session_state.get(f"pricing_result_{key}", {})
                    spot = float(snap.get("spot") or 0.0)
                    mark_price = float(snap.get("mark_price") or 0.0)
                    has_price = mark_price is not None and mark_price > 0
                    pnl_total = float(snap.get("pnl") or 0.0)
                    pnl_per_unit = float(snap.get("pnl_per_unit") or 0.0)
                    method = snap.get("method", "n/a")
                    sigma_used = float(snap.get("sigma") or 0.0)
                    T_years = snap.get("T")

                    strike = pos.get("strike")
                    qty = pos.get("quantity")
                    if qty is None:
                        qty = 0
                    avg_price = pos.get("avg_price")
                    if avg_price is None:
                        avg_price = pos.get("T_0_price", 0.0)
                    avg_price = float(avg_price or 0.0)
                    side = pos.get("side", "long").lower()
                    product = (
                        pos.get("product_type")
                        or pos.get("structure")
                        or pos.get("product")
                        or pos.get("type")
                        or "vanilla"
                    )
                    underlying = pos.get("underlying")
                    option_type_raw = (
                        pos.get("option_type")
                        or pos.get("cpflag")
                        or pos.get("cp_flag")
                        or pos.get("cp")
                        or ""
                    )
                    if not option_type_raw:
                        t_val = str(pos.get("type", "")).lower()
                        if t_val in {"call", "put"}:
                            option_type_raw = t_val
                    option_type = str(option_type_raw).lower()
                    strike2_val = pos.get("strike2")
                    strike2 = float(strike2_val) if strike2_val not in (None, "") else None
                    misc = pos.get("misc")
                    pricing_fn, needed = _pricing_metadata(product)
                    product_low = product.lower() if product else ""
                    available = set(k for k in [
                        "option_type",
                        "strike" if strike is not None else None,
                        "strike2" if strike2 is not None else None,
                        "expiration" if pos.get("expiration") else None,
                        "quantity" if qty else None,
                        "sigma" if sigma_used else None,
                        "S0" if spot else None,
                        "legs (strike/side/type)" if pos.get("legs") else None,
                    ] if k)
                    # We can compute r/q from rates_utils
                    available.update({"r", "q"})
                    # For spreads/structures without explicit legs, consider strike/strike2 as leg info
                    if any(keyword in product_low for keyword in ["spread", "butterfly", "condor", "straddle", "strangle"]):
                        available.add("legs (strike/side/type)")
                    if isinstance(misc, dict):
                        available.update(misc.keys())
                        if "barrier" in product_low and "knock" in misc and "direction" not in available:
                            available.add("direction")
                    missing = set(needed) - available
                    extra = available - set(needed)
                    btn_label = "Close"
                    close_max = max(1, int(qty)) if qty and qty > 0 else 1

                    with st.expander(f"{key} ({product})"):
                        side_label = side.upper() if side else "N/A"
                        opt_label = option_type.upper() if option_type else "N/A"
                        st.caption(f"Produit reconnu: {product} | Option type: {opt_label} | Side: {side_label}")
                        state_key = f"pricing_launched_{key}"
                        launched = st.session_state.get(state_key, False)
                        if st.button("🚀 Lancer pricing", key=f"run_pricing_{key}"):
                            try:
                                chain_state_key = f"chain_cache_{underlying}"
                                chain_list = st.session_state.get(chain_state_key)
                                if chain_list is None:
                                    chain_list = fetch_options_chain(underlying) if underlying else []
                                    st.session_state[chain_state_key] = chain_list
                                chain_entry = None
                                if chain_list and pos.get("expiration") and strike:
                                    try:
                                        expiry_date = datetime.date.fromisoformat(pos.get("expiration"))
                                        days_to_expiry = (expiry_date - datetime.date.today()).days
                                        target_T = max(days_to_expiry, 0) / 365.0
                                    except Exception:
                                        target_T = None
                                    if target_T is not None:
                                        best = None
                                        best_score = float("inf")
                                        for c in chain_list:
                                            cT = float(c.get("T", 0.0) or 0.0)
                                            cK = float(c.get("strike", 0.0) or 0.0)
                                            scale = max(strike, 1.0)
                                            score = abs(cT - target_T) + abs(cK - strike) / scale
                                            if score < best_score:
                                                best_score = score
                                                best = c
                                        chain_entry = best
                                spot_calc, T_calc, sigma_calc, new_mark, new_method = _price_from_pricing_py(pos, chain_entry=chain_entry)
                                if new_mark is None or new_mark <= 0:
                                    spot_calc, T_calc, sigma_calc, new_mark, new_method = mark_option_market_value(pos, chain_entry=chain_entry)
                                # Save pricing snapshot for subsequent renders
                                st.session_state[f"pricing_result_{key}"] = {
                                    "spot": spot_calc,
                                    "T": T_calc,
                                    "sigma": sigma_calc,
                                    "mark_price": new_mark,
                                    "method": new_method,
                                    "pnl": (new_mark - avg_price) * qty if side == "long" else (avg_price - new_mark) * qty,
                                    "pnl_per_unit": (new_mark - avg_price) if side == "long" else (avg_price - new_mark),
                                }
                                # Compute and cache Greeks in the same action
                                greeks = compute_option_greeks(pos, chain_entry=chain_entry)
                                st.session_state[f"greeks_{key}"] = greeks
                                st.success(f"Pricing lancé ({pricing_fn}) → prix courant ≈ {new_mark:.4f} via {new_method}")
                                st.session_state[state_key] = True
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Erreur lors du pricing : {exc}")
                                st.session_state[state_key] = False
                        if not st.session_state.get(state_key, False):
                            st.info("Lance le pricing pour afficher les détails.")
                            continue
                        greeks_vals = st.session_state.get(f"greeks_{key}")
                        close_qty = st.selectbox(
                            "Quantity to close",
                            options=list(range(1, close_max + 1)),
                            index=close_max - 1 if close_max > 0 else 0,
                            key=f"close_qty_{key}",
                        )
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Prix actuel", f"${mark_price:.4f}")
                            st.caption(
                                f"S = {spot:.4f} | K = {strike}"
                                + (f" | K2 = {strike2}" if strike2 is not None else "")
                            )
                            st.caption(
                                f"T = {T_years:.4f}" if T_years is not None else "T inconnu"
                            )
                            st.caption(f"Méthode: {method} | σ={sigma_used:.4f}")
                        with col_b:
                            st.metric("T_0 Price", f"${avg_price:.4f}")
                            st.metric("Qty", f"{qty}")
                        with col_c:
                            st.metric("PnL if closed", f"${pnl_total:.2f}", delta=f"{pnl_per_unit:.4f}")
                            if greeks_vals:
                                st.caption(
                                    f"Δ {greeks_vals.get('delta') if greeks_vals.get('delta') is not None else '-':.2f}"
                                    f" | Γ {greeks_vals.get('gamma') if greeks_vals.get('gamma') is not None else '-':.2f}"
                                    f" | Vega {greeks_vals.get('vega') if greeks_vals.get('vega') is not None else '-':.2f}"
                                )
                        if isinstance(misc, dict) and misc:
                            st.caption(f"Misc: {misc}")
                        st.caption(f"Pricing fn: {pricing_fn}")
                        st.caption(f"Paramètres requis: {needed}")
                        st.caption(f"Paramètres disponibles: {sorted(available)}")
                        if greeks_vals:
                            col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
                            col_g1.metric("Delta", f"{greeks_vals.get('delta', '-'):.2f}")
                            col_g2.metric("Gamma", f"{greeks_vals.get('gamma', '-'):.2f}")
                            col_g3.metric("Vega", f"{greeks_vals.get('vega', '-'):.2f}")
                            col_g4.metric("Theta", f"{greeks_vals.get('theta', '-'):.2f}")
                            col_g5.metric("Rho", f"{greeks_vals.get('rho', '-'):.2f}")
                        # Afficher les valeurs connues pour chaque paramètre requis
                        def _param_value(k: str):
                            if k == "option_type":
                                return option_type
                            if k == "strike":
                                return strike
                            if k == "strike2":
                                return strike2
                            if k == "expiration":
                                return pos.get("expiration")
                            if k == "quantity":
                                return qty
                            if k == "sigma":
                                return sigma_used
                            if k == "S0":
                                return spot
                            if k == "r":
                                return (misc or {}).get("r")
                            if k == "q":
                                return (misc or {}).get("q")
                            if k == "n_obs":
                                return (misc or {}).get("n_obs")
                            if k == "n_paths":
                                return (misc or {}).get("n_paths")
                            if k == "n_steps":
                                return (misc or {}).get("n_steps")
                            if k == "barrier_level":
                                return (misc or {}).get("barrier") or (misc or {}).get("barrier_level")
                            if k == "barrier_type":
                                return (misc or {}).get("barrier_type")
                            if k == "knock" or k == "direction":
                                return (misc or {}).get("knock") or (misc or {}).get("direction")
                            if k == "legs (strike/side/type)":
                                return pos.get("legs")
                            if k == "T_start":
                                return (misc or {}).get("T_start")
                            if k == "t_choice":
                                return (misc or {}).get("t_choice")
                            if k == "cap":
                                return (misc or {}).get("cap")
                            if k == "floor":
                                return (misc or {}).get("floor")
                            return (misc or {}).get(k) or pos.get(k)

                        if needed:
                            st.caption("Valeurs actuelles des paramètres requis :")
                            param_lines = [f"- `{k}` : { _param_value(k) }" for k in needed]
                            st.markdown("\n".join(param_lines))
                        if missing:
                            st.markdown(f"🔴 Paramètres manquants: {sorted(missing)}")
                        elif extra:
                            st.markdown(f"🟨 Paramètres en trop: {sorted(extra)}")
                        else:
                            st.markdown("🟩 Paramètres OK pour pricer")

                        price_issue = not has_price or mark_price <= 0
                        if price_issue:
                            st.markdown("🔴 PnL non calculable : prix de marché/spot manquant.")

                        if st.button(f"✅ {btn_label}", key=f"close_custom_{key}", disabled=bool(missing or price_issue)):
                            book = load_options_book()
                            entry = dict(book.get(key, pos))
                            entry["option_type"] = option_type
                            entry["type"] = option_type

                            close_entry = dict(entry)
                            close_entry["quantity"] = close_qty
                            close_entry.update({
                                "status": "expired" if close_qty >= qty else "open",
                                "closed_at": datetime.date.today().isoformat(),
                                "underlying_close": spot,
                                "mark_close": mark_price,
                            })
                            pnl_vals = compute_option_pnl(close_entry, spot, mark_price=mark_price)
                            close_entry.update(pnl_vals)

                            if close_qty >= qty:
                                book[key] = close_entry
                            else:
                                # partial close: reduce remaining quantity
                                remaining = max(qty - close_qty, 0)
                                entry["quantity"] = remaining
                                book[key] = entry
                                book[f"{key}_closed_{int(time.time())}"] = close_entry

                            save_options_book(book)
                            st.success(f"Closed {close_qty} of {key} ({product}), realized PnL ≈ ${pnl_vals['pnl_total']:.2f}")
                            time.sleep(1)
                            st.rerun()
        else:
            st.info("Aucune option custom à afficher pour le moment.")
    else:
        st.info("Aucune option custom dans la liste.")
    
    # Trading Systems Section
    st.markdown("---")
    st.subheader("🎯 Configured Trading Systems")
    equities = load_equities()
    sell_systems = load_sell_systems()
    
    if equities:
        systems_data = []
        for symbol, data in equities.items():
            direction = data.get('direction', 'long')
            systems_data.append({
                'Symbol': symbol,
                'Direction': direction.capitalize(),
                'Position': data['position'],
                'Entry Price': f"${data['entry_price']:.2f}",
                'Drawdown': f"{data['drawdown']*100:.1f}%",
                'Levels': len(data['levels']),
                'Status': data['status']
            })
        
        df_systems = pd.DataFrame(systems_data)
        st.dataframe(df_systems, width="stretch", hide_index=True)

        st.markdown("#### 🔎 Niveaux d'achat / vente par système")
        for symbol, data in equities.items():
            buy_levels = data.get("levels", {}) or {}
            sell_cfg = sell_systems.get(symbol, {}) if isinstance(sell_systems, dict) else {}
            sell_levels = sell_cfg.get("levels", {}) if isinstance(sell_cfg, dict) else {}
            with st.expander(f"{symbol} – niveaux d'achat/vente", expanded=False):
                col_buy, col_sell = st.columns(2)
                with col_buy:
                    if buy_levels:
                        rows_buy = []
                        for lvl_key, price in sorted(buy_levels.items(), key=lambda kv: str(kv[0])):
                            rows_buy.append({"Level": str(lvl_key), "Buy @": price})
                        st.table(pd.DataFrame(rows_buy))
                    else:
                        st.caption("Aucun niveau d'achat configuré.")
                with col_sell:
                    if sell_levels:
                        rows_sell = []
                        for lvl_key, lvl in sorted(sell_levels.items(), key=lambda kv: str(kv[0])):
                            rows_sell.append({
                                "Level": str(lvl_key),
                                "Sell @": lvl.get("price"),
                                "Qty": lvl.get("quantity"),
                                "Triggered": bool(lvl.get("triggered")),
                            })
                        st.table(pd.DataFrame(rows_sell))
                    else:
                        st.caption("Aucun niveau de vente configuré.")
        
        # Quick remove section
        st.markdown("**Quick Remove:**")
        remove_cols = st.columns(len(equities) if len(equities) <= 5 else 5)
        for idx, symbol in enumerate(list(equities.keys())[:5]):
            with remove_cols[idx]:
                if st.button(f"🗑️ {symbol}", key=f"dash_remove_{symbol}"):
                    del equities[symbol]
                    save_equities(equities)
                    st.success(f"Removed {symbol}")
                    time.sleep(0.5)
                    st.rerun()
    else:
        st.info("No trading systems configured. Add one in the 'Trading Systems' tab.")

    # Expired forwards
    st.markdown("---")
    st.markdown("### ⏳ Forwards expirés / clôturés")
    forwards_all = load_forwards()
    today = datetime.date.today()
    expired_fwds = {}
    for fwd_id, fwd in forwards_all.items():
        matur_str = fwd.get("maturity")
        try:
            matur_dt = datetime.date.fromisoformat(matur_str) if matur_str else None
        except Exception:
            matur_dt = None
        if matur_dt is not None and matur_dt < today:
            expired_fwds[fwd_id] = fwd

    if expired_fwds:
        rows_fwd_exp: list[dict] = []
        for key, fwd in expired_fwds.items():
            sym = fwd.get("symbol", "")
            qty = int(fwd.get("quantity", 0) or 0)
            price_fwd = float(fwd.get("forward_price", 0.0) or 0.0)
            side = (fwd.get("side") or "long").lower()
            maturity_str = fwd.get("maturity")
            try:
                maturity_dt = datetime.date.fromisoformat(maturity_str) if maturity_str else None
            except Exception:
                maturity_dt = None
            days_to_mat = (maturity_dt - today).days if maturity_dt else None
            spot_now = _dashboard_price(sym, 0.0) if sym else 0.0
            mult = 1.0 if side == "long" else -1.0
            pnl_unit = mult * (spot_now - price_fwd)
            pnl_total = pnl_unit * qty
            rows_fwd_exp.append({
                "ID": key,
                "Symbol": sym,
                "Side": side.capitalize(),
                "Quantity": qty,
                "Forward Price": price_fwd,
                "Spot Now": round(spot_now, 4),
                "Maturity": maturity_str,
                "Days to mat": days_to_mat,
                "P&L/unit": round(pnl_unit, 4),
                "P&L total": round(pnl_total, 2),
            })
        df_fwd_exp = pd.DataFrame(rows_fwd_exp)
        st.dataframe(df_fwd_exp, width="stretch", hide_index=True)

        st.markdown("#### Détail / suppression")
        for key, fwd in expired_fwds.items():
            with st.expander(f"{key} — {fwd.get('symbol','N/A')} ({fwd.get('side','')})", expanded=False):
                sym = fwd.get("symbol", "")
                qty = int(fwd.get("quantity", 0) or 0)
                price_fwd = float(fwd.get("forward_price", 0.0) or 0.0)
                side = (fwd.get("side") or "long").lower()
                spot_now = _dashboard_price(sym, 0.0) if sym else 0.0
                mult = 1.0 if side == "long" else -1.0
                pnl_unit = mult * (spot_now - price_fwd)
                pnl_total = pnl_unit * qty
                st.json(fwd, expanded=False)
                st.caption(f"Spot actuel: {spot_now:.4f} | P&L/unité: {pnl_unit:.4f} | P&L total: {pnl_total:.2f}")
                if st.button(f"🗑️ Supprimer {key}", key=f"del_fwd_exp_{key}"):
                    forwards_all.pop(key, None)
                    save_forwards(forwards_all)
                    st.success(f"Forward {key} supprimé.")
                    time.sleep(0.5)
                    st.rerun()
    else:
        st.info("Aucun forward expiré pour le moment.")

    # Expired options at the end of the dashboard
    st.markdown("---")
    st.markdown("### ✅ Options expirées / clôturées")
    expired_options = dict(expired_options_data)
    legacy_book = load_options_book_legacy_only()
    for opt_id, entry in legacy_book.items():
        if opt_id not in expired_options and entry.get("status") in {"expired", "closed"}:
            expired_options[opt_id] = entry

    if expired_options:
        exp_rows = []
        for key, opt in expired_options.items():
            exp_rows.append({
                "Qty": opt.get("quantity"),
                "Underlying": opt.get("underlying"),
                "Side": str(opt.get("side", "")).capitalize(),
                "Type": str(opt.get("option_type") or opt.get("type", "")).capitalize(),
                "T_0 Price": opt.get("avg_price"),
                "Strike": opt.get("strike"),
                "Closing asset price": opt.get("underlying_close"),
                "Payoff/unit": opt.get("payoff_per_unit"),
                "PnL total": opt.get("pnl_total"),
                "Expiration": opt.get("expiration"),
            })
        df_exp = pd.DataFrame(exp_rows)
        desired_cols = [
            "Expiration",
            "Qty",
            "Underlying",
            "Side",
            "Type",
            "Strike",
            "Closing asset price",
            "T_0 Price",
            "Payoff/unit",
            "PnL total",
        ]
        df_exp = df_exp[[c for c in desired_cols if c in df_exp.columns]]
        st.dataframe(df_exp, width="stretch", hide_index=True)
        if legacy_book:
            st.caption("Source merge : options_portfolio.json + options_book.json (legacy).")

        st.markdown("#### Profils payoff / P&L (options expirées)")
        spot_cache_exp: dict[str, float | None] = {}

        def _spot_live_exp(sym: str) -> float | None:
            if not sym:
                return None
            sym_clean = (sym or "").strip().upper()
            if sym_clean in spot_cache_exp:
                return spot_cache_exp[sym_clean]
            if dashboard_prices.get(sym_clean) is not None:
                try:
                    cached_val = float(dashboard_prices[sym_clean])
                    spot_cache_exp[sym_clean] = cached_val
                    return cached_val
                except Exception:
                    pass
            spot_val = get_spot_cboe_cached(sym_clean)
            if spot_val is None or spot_val <= 0:
                spot_val = _dashboard_price(sym_clean, 0.0)
            spot_cache_exp[sym_clean] = spot_val
            return spot_val

        for key, opt in expired_options.items():
            underlying = (opt.get("underlying") or "").upper()
            side = (opt.get("side") or "long").lower()
            avg_price = float(opt.get("avg_price", opt.get("T_0_price", 0.0)) or 0.0)
            qty = float(opt.get("quantity", 0.0) or 0.0)
            strike = float(opt.get("strike", 0.0) or 0.0)
            spot_close = float(opt.get("underlying_close", opt.get("S_T", 0.0)) or 0.0)
            spot_now = _spot_live_exp(underlying)
            ref = spot_close if spot_close > 0 else (spot_now if spot_now and spot_now > 0 else strike if strike > 0 else 1.0)
            s_grid = np.linspace(max(0.01, 0.5 * ref), 1.5 * ref, 180)
            pay_grid = np.array([compute_option_payoff(opt, s) for s in s_grid], dtype=float)
            if side == "short":
                pay_grid *= -1.0
            pnl_grid = pay_grid - avg_price if side == "long" else avg_price + pay_grid
            pay_close = compute_option_payoff(opt, ref)
            if side == "short":
                pay_close *= -1.0
            pnl_total_line = pay_close * qty - avg_price * qty if side == "long" else avg_price * qty + pay_close * qty

            with st.expander(f"{key} – {opt.get('product_type') or opt.get('product') or opt.get('type') or 'Option'} ({underlying})", expanded=False):
                st.caption(f"Qty: {qty:.0f} | Side: {side} | Strike: {strike:.4f} | Spot réf: {ref:.4f}")
                st.metric("Payoff @ réf (expirée)", f"${pay_close:.4f}")
                st.metric("P&L total (ligne)", f"${pnl_total_line:.4f}")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(s_grid, pay_grid, label="Payoff (signé)")
                ax.plot(s_grid, pnl_grid, label="P&L (vs. prime)", color="darkorange")
                if strike > 0:
                    ax.axvline(strike, color="gray", linestyle="--", label=f"K = {strike:.2f}")
                ax.axvline(ref, color="crimson", linestyle="-.", label=f"Ref = {ref:.2f}")
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Spot")
                ax.set_ylabel("Payoff / P&L")
                ax.legend(loc="best")
                ax.grid(alpha=0.3, linestyle="--")
                st.pyplot(fig, clear_figure=True)
                st.caption(f"Payoff @ ref: {pay_close:.4f} | Avg price: {avg_price:.4f}")

        with st.expander("🔎 Détails payoff / méthode de calcul (toutes les options)", expanded=False):
            for key, opt in expired_options.items():
                info = describe_expired_option_payoff(opt)
                title = f"{key} – {opt.get('product_type') or opt.get('product') or opt.get('type') or ''}"
                st.markdown(f"**{title}**")
                st.write(f"**Méthode** : {info['method_label']} (fonction `{info['function']}`)")
                st.caption(info["description"])
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"Payoff/unité: {info['payoff_unit']:.4f}")
                    st.write(f"Payoff/unité signé (side): {info['payoff_signed_unit']:.4f}")
                with c2:
                    st.write(f"Payoff total (qty x side): {info['payoff_total']:.4f}")
                    st.write(f"Sous-jacent à l'échéance: {info['underlying_close']}")
                    st.write(f"Side: {info['side']} | Qty: {info['quantity']}")
                st.write("Paramètres utilisés :")
                st.json(info["params"])
                st.markdown("---")
    else:
        st.info("Aucune option expirée/close pour l’instant.")

# Tab 2: Buy/Sell
with tab2:
    with st.expander("📘 Comprendre Buy/Sell"):
        st.markdown("""
        ### 💰 Ce que vous faites dans Buy/Sell
        
        Cet onglet est votre **poste d’exécution manuelle** : c’est ici que vous décidez
        consciemment d’entrer, renforcer, réduire ou retourner une position, en contrôlant précisément prix et quantité.
        
        Le bloc *Buy / Cover Asset* permet soit d’acheter pour être ou rester **Long**, soit d’acheter pour **couvrir un short**.
        Vous choisissez la *Direction* (Long/Short), le symbole, la quantité et le prix d’exécution.
        
        Le bloc *Sell / Short Asset* sert à gérer les positions existantes : vendre une partie d’un long, le clôturer entièrement,
        ou vendre au-delà de votre quantité actuelle pour devenir **net short** sur un actif.
        
        À droite, vous voyez à chaque fois la position en place (quantité, prix moyen, sens long/short) et le P&L estimé du trade
        avant de cliquer, ce qui vous aide à visualiser l’impact concret de l’ordre sur votre portefeuille.
        
        Utilisez cet onglet pour **intervenir manuellement** malgré vos systèmes automatiques : prendre des profits, couper une perte,
        inverser une position ou initier un short tactique, tout en gardant en tête le P&L et le risque global de votre compte.
        """)
    st.subheader("💰 Buy/Sell Assets")
    
    col1, col2 = st.columns(2)
    
    # BUY Section
    with col1:
        st.markdown("### 📈 Buy / Cover Asset")
        buy_side = st.radio(
            "Direction",
            options=["Long", "Short"],
            index=0,
            horizontal=True,
            key="buy_side"
        )
        buy_symbol = st.text_input("Symbol to Buy", placeholder="e.g., AAPL", key="buy_symbol").upper()
        buy_quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="buy_qty")
        
        if buy_symbol:
            price_data = get_data(buy_symbol)
            if price_data['price'] > 0:
                st.info(f"Current price: ${price_data['price']:.2f}")
                buy_price = st.number_input("Buy Price", min_value=0.01, value=float(price_data['price']), step=0.01, key="buy_price")
                total_cost = buy_quantity * buy_price
                st.metric("Total Cost", f"${total_cost:.2f}")
                
                if st.button("✅ Execute Order", type="primary", key="exec_buy"):
                    if buy_side == "Long":
                        result = buy_asset(buy_symbol, buy_quantity, buy_price)
                        st.success(f"Bought {buy_quantity} units of {buy_symbol} @ ${buy_price:.2f}")
                        if result:
                            side = result.get('side', 'long').upper()
                            st.info(
                                f"New position: {result['quantity']} units @ avg ${result['avg_price']:.2f} "
                                f"({side})"
                            )
                        else:
                            st.info("Position fully closed.")
                    else:
                        if sell_asset(buy_symbol, buy_quantity, buy_price):
                            st.success(f"Shorted {buy_quantity} units of {buy_symbol} @ ${buy_price:.2f}")
                            portfolio_after = load_portfolio()
                            new_pos = portfolio_after.get(buy_symbol)
                            if new_pos:
                                side = new_pos.get("side", "short").upper()
                                st.info(
                                    f"New position: {new_pos['quantity']} units @ avg ${new_pos['avg_price']:.2f} "
                                    f"({side})"
                                )
                    time.sleep(1)
                    st.rerun()
            else:
                st.error(f"Could not fetch price for {buy_symbol}")
    
    # SELL / SHORT Section
    with col2:
        st.markdown("### 📉 Sell / Short Asset")
        my_portfolio = load_portfolio()
        
        if my_portfolio:
            sell_symbol = st.selectbox("Symbol to Sell/Short", options=list(my_portfolio.keys()), key="sell_symbol")
            
            if sell_symbol:
                position = my_portfolio[sell_symbol]
                current_qty = position['quantity']
                avg_price = position['avg_price']
                side = position.get('side', 'long')
                
                st.info(
                    f"Current position: {current_qty} units @ avg ${avg_price:.2f} "
                    f"({side.upper()})"
                )
                
                sell_quantity = st.number_input(
                    "Quantity to Sell (you can sell more than you hold to go net short)",
                    min_value=1,
                    value=1,
                    step=1,
                    key="sell_qty"
                )

                action_options = ["Sell/Short more"]
                if side == "short":
                    action_options.append("Buy to cover")
                action = st.radio(
                    "Action",
                    options=action_options,
                    index=0,
                    horizontal=True,
                    key="sell_action",
                )

                price_data = get_data(sell_symbol)
                if price_data['price'] > 0:
                    market_price = float(price_data['price'])
                    st.info(f"Current market price used: ${market_price:.2f}")

                    if action == "Buy to cover":
                        cash_flow = -sell_quantity * market_price  # cash out
                        pnl = (avg_price - market_price) * sell_quantity
                        notional = avg_price * sell_quantity
                        pnl_pct = (pnl / notional * 100) if notional > 0 else 0.0
                        st.metric("Cash Outlay", f"${-cash_flow:.2f}")
                        st.metric("P&L (per this cover)", f"${pnl:.2f}", delta=f"{pnl_pct:.2f}%")

                        if st.button("✅ Buy to cover", type="primary", key="exec_cover"):
                            result = buy_asset(sell_symbol, sell_quantity, market_price)
                            st.success(f"Bought {sell_quantity} units of {sell_symbol} @ market ${market_price:.2f} to cover short")
                            if result:
                                side_new = result.get('side', 'long').upper()
                                st.info(
                                    f"New position: {result['quantity']} units @ avg ${result['avg_price']:.2f} "
                                    f"({side_new})"
                                )
                            else:
                                st.info("Position fully closed.")
                            time.sleep(1)
                            st.rerun()
                    else:
                        total_proceeds = sell_quantity * market_price
                        if side == 'long':
                            pnl = (market_price - avg_price) * sell_quantity
                        else:
                            pnl = (avg_price - market_price) * sell_quantity
                        notional = avg_price * sell_quantity
                        pnl_pct = (pnl / notional * 100) if notional > 0 else 0.0
                        
                        st.metric("Total Proceeds", f"${total_proceeds:.2f}")
                        st.metric("P&L (per this trade)", f"${pnl:.2f}", delta=f"{pnl_pct:.2f}%")
                        
                        if st.button("✅ Execute Sell / Short", type="primary", key="exec_sell"):
                            if sell_asset(sell_symbol, sell_quantity, market_price):
                                st.success(f"Sold {sell_quantity} units of {sell_symbol} @ market ${market_price:.2f}")
                                st.info(f"Trade P&L (approx.): ${pnl:.2f}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to execute sell order")
                else:
                    st.error(f"Could not fetch price for {sell_symbol}")
        else:
            st.info("No assets in portfolio to sell or short")

# Tab 3: Trading Systems
with tab3:
    with st.expander("📘 Comprendre Trading Systems"):
        st.markdown("""
        ### 📋 Ce que vous faites dans Trading Systems
        
        Ici, vous ne passez pas d’ordres immédiats : vous **concevez des systèmes automatiques** (long ou short)
        qui interviendront pour vous à différents niveaux de prix prédéfinis.
        
        Le champ *Symbol* sert à choisir l’actif que vous voulez suivre
        de façon structurée (indice, action, ETF, crypto, etc.).
        
        Le champ *Direction* vous permet de choisir si le système doit exploiter une **hausse** (Long) ou une **baisse** (Short) :
        - **Long** : le robot cherche à accumuler ou renforcer sur l’actif
        - **Short** : le robot cherche à construire ou renforcer une position vendeuse
        
        *Number of Levels* définit combien de paliers d’intervention vous voulez.
        Chaque niveau correspond à un prix où le système déclenchera automatiquement un ordre (dans le sens choisi).
        
        *Drawdown %* contrôle l’écart et la direction des niveaux :
        - **Valeur négative** : niveaux en dessous du prix d’entrée (buy the dip / rachat de short)
        - **Valeur positive** : niveaux au-dessus du prix d’entrée (short plus haut / pyramider sur une tendance haussière)
        
        Plus le pourcentage est faible, plus les niveaux sont serrés; plus il est élevé, plus les niveaux sont espacés.
        
        Quand vous cliquez sur *Add Equity*, l’outil calcule tous les niveaux autour du prix actuel, enregistre le système en mode *Off*,
        puis vous laisse l’activer et le surveiller dans la section *Manage Your Trading Systems* plus bas sur cette page.
        C’est ici que vous transformez une idée en robot.
        
        Utilisez cet onglet pour **planifier à l’avance** comment vous voulez que vos positions se construisent ou se réduisent,
        sans avoir à rester devant les écrans à chaque mouvement de marché.
        """)

    st.subheader("📋 Trading Systems")
    
    # Check current count
    equities = load_equities()
    current_count = len(equities)
    
    st.info(f"Trading Systems: {current_count}/10")
    
    if current_count >= 10:
        st.error("⚠️ Maximum limit reached! You cannot add more than 10 equities. Please remove one first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.text_input("Symbol", placeholder="e.g., AAPL").upper()
        
        with col2:
            direction = st.radio("Direction", options=["Long", "Short"], index=0, horizontal=True, key="add_equity_direction")
        
        with col3:
            levels = st.number_input("Number of Levels", min_value=1, max_value=10, value=5)
        
        with col4:
            drawdown = st.number_input("Drawdown %", min_value=-50.0, max_value=50.0, value=5.0, step=0.1)
        
        if st.button("➕ Add Equity", type="primary"):
            if symbol:
                equities = load_equities()
                
                # Double check limit
                if len(equities) >= 10:
                    st.error("Maximum limit of 10 equities reached!")
                elif symbol in equities:
                    st.warning(f"{symbol} already exists!")
                else:
                    price_data = get_data(symbol)
                    entry_price = price_data['price']
                    
                    if entry_price > 0:
                        drawdown_decimal = drawdown / 100
                        
                        # Si drawdown négatif: niveaux à la baisse (en dessous du prix d'entrée)
                        # Si drawdown positif: niveaux à la hausse (au dessus du prix d'entrée)
                        if drawdown_decimal < 0:
                            # Drawdown négatif → niveaux en dessous
                            level_prices = {str(i+1): round(entry_price * (1 + drawdown_decimal * (i+1)), 2) for i in range(levels)}
                        else:
                            # Drawdown positif → niveaux au dessus
                            level_prices = {str(i+1): round(entry_price * (1 + drawdown_decimal * (i+1)), 2) for i in range(levels)}
                        
                        stored_drawdown = drawdown_decimal
                        
                        equities[symbol] = {
                            "position": 0,
                            "entry_price": entry_price,
                            "levels": level_prices,
                            "drawdown": stored_drawdown,
                            "direction": direction.lower(),
                            "status": "Off"
                        }
                        
                        save_equities(equities)
                        st.success(f"✅ Added {symbol} ({direction}) at ${entry_price:.2f}")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Could not fetch price for {symbol}")
            else:
                st.error("Please enter a symbol")
    
    # Trading Systems Management (moved from separate tab)
    st.markdown("---")
    st.markdown("### 📋 Manage Your Trading Systems")
    st.markdown("""
    Pilotez vos systèmes automatiques déjà configurés : c'est votre salle de contrôle pour voir comment vos robots long/short sont positionnés.
    Le toggle *Active* active/désactive un système. Le tableau *Price Levels* montre tous les niveaux d'intervention.
    """)
    
    equities_list = load_equities()
    
    if equities_list:
        for symbol, data in equities_list.items():
            direction = data.get('direction', 'long')
            with st.expander(f"{symbol} ({direction.upper()}) - Status: {data['status']}", expanded=False):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Direction", direction.capitalize())
                
                with col2:
                    st.metric("Position", data['position'])
                
                with col3:
                    st.metric("Entry Price", f"${data['entry_price']:.2f}")
                
                with col4:
                    st.metric("Drawdown", f"{data['drawdown']*100:.1f}%")
                
                with col5:
                    current_status = data['status']
                    new_status = st.toggle(
                        "Active", 
                        value=current_status == "On",
                        key=f"toggle_manage_{symbol}"
                    )
                    
                    if (new_status and current_status == "Off") or (not new_status and current_status == "On"):
                        equities_list[symbol]['status'] = "On" if new_status else "Off"
                        save_equities(equities_list)
                        st.rerun()
                
                # Display levels
                st.markdown("**Price Levels:**")
                levels_df = pd.DataFrame([
                    {"Level": k, "Price": f"${v:.2f}"} 
                    for k, v in data['levels'].items()
                ])
                st.dataframe(levels_df, width="stretch", hide_index=True)
                
                # Remove button
                if st.button(f"🗑️ Remove {symbol}", key=f"remove_manage_{symbol}"):
                    del equities_list[symbol]
                    save_equities(equities_list)
                    st.success(f"Removed {symbol}")
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("No trading systems configured yet. Add an equity above to get started.")

# Tab 4: Forwards
with tab4:
    with st.expander("📘 Comprendre Forwards"):
        st.markdown(
            """
            ### 📄 Ce que vous faites dans Forwards
            
            Cet onglet vous permet d'**acheter (ou vendre) un forward** sur un sous-jacent : vous fixez aujourd'hui un prix d'échange futur.
            
            - Vous choisissez le sous-jacent, la date d'échéance, le sens (long/short) et le prix forward.
            - Le forward est enregistré dans un fichier JSON (`forwards.json`).
            - Il apparaît ensuite dans le Dashboard avec P&L mark-to-market basé sur le prix spot actuel.
            
            Un **long forward** gagne lorsque le prix spot est au-dessus du prix forward à l'échéance; un **short forward** gagne dans le cas inverse.
            """
        )

    st.subheader("🚀 Trade Forward")

    col_sym, col_fetch = st.columns([3, 1], vertical_alignment="bottom")
    with col_sym:
        fwd_symbol = st.text_input(
            "Underlying symbol",
            placeholder="e.g., AAPL",
            key="fwd_symbol",
        ).upper()
    with col_fetch:
        if st.button("🔍 Fetch spot", key="btn_fetch_forward_spot"):
            price_data = get_data(fwd_symbol) if fwd_symbol else {"price": 0}
            spot_now_fetch = float(price_data.get("price", 0.0) or 0.0)
            if spot_now_fetch > 0:
                st.session_state["fwd_spot_symbol"] = fwd_symbol
                st.session_state["fwd_spot_value"] = spot_now_fetch
                st.success(f"Spot {fwd_symbol} ≈ ${spot_now_fetch:.4f}")
            else:
                st.warning("Spot introuvable pour ce ticker.")

    today = datetime.date.today()
    default_maturity = today + datetime.timedelta(days=30)
    fwd_maturity = st.date_input(
        "Maturity date",
        value=default_maturity,
        min_value=today,
        key="fwd_maturity",
    )

    fwd_qty = st.number_input(
        "Notional (units)",
        min_value=1,
        value=1,
        step=1,
        key="fwd_qty",
    )

    spot_now = 0.0
    if fwd_symbol and st.session_state.get("fwd_spot_symbol") == fwd_symbol:
        spot_now = float(st.session_state.get("fwd_spot_value", 0.0) or 0.0)
    if spot_now <= 0 and fwd_symbol:
        price_data = get_data(fwd_symbol)
        spot_now = float(price_data.get("price", 0.0) or 0.0)
    if fwd_symbol:
        if spot_now > 0:
            days_to_mat = max((fwd_maturity - today).days, 0)
            T_years = days_to_mat / 365.0
            r_forward = get_r(T_years) if T_years > 0 else get_r(0.1)
            forward_price = spot_now * math.exp(r_forward * T_years)
            pill = (
                f"<div style='display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.6rem;'>"
                f"<span style='background:#e8f5e9;color:#1b5e20;padding:6px 14px;border-radius:999px;font-weight:600;'>"
                f"Spot {fwd_symbol} = {spot_now:.4f}</span>"
                f"<span style='background:#e3f2fd;color:#0d47a1;padding:6px 14px;border-radius:999px;font-weight:600;'>"
                f"r = {r_forward:.4f}</span>"
                f"<span style='background:#fffde7;color:#f57f17;padding:6px 14px;border-radius:999px;font-weight:600;'>"
                f"T = {T_years:.3f}y</span>"
                f"<span style='background:#e8f5e9;color:#1b5e20;padding:6px 14px;border-radius:999px;font-weight:700;'>"
                f"F ≈ {forward_price:.4f}</span></div>"
            )
            st.markdown(pill, unsafe_allow_html=True)
        else:
            st.warning("Impossible de récupérer le spot, tu peux quand même enregistrer mais le prix forward restera nul.")
            forward_price = 0.0
    else:
        forward_price = 0.0

    fwd_side_label = st.radio(
        "Position",
        options=["Long forward", "Short forward"],
        horizontal=True,
        key="fwd_side",
    )
    fwd_side = "long" if str(fwd_side_label).startswith("Long") else "short"

    if st.button("Enregistrer le forward", type="primary", key="btn_save_forward"):
        if fwd_symbol and forward_price > 0:
            forwards = load_forwards()
            uid = f"{fwd_symbol}_{fwd_maturity.isoformat()}_{int(time.time())}"
            forwards[uid] = {
                "symbol": fwd_symbol,
                "maturity": fwd_maturity.isoformat(),
                "forward_price": round(float(forward_price), 4),
                "quantity": int(fwd_qty),
                "side": fwd_side,
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "spot_at_trade": round(spot_now, 4),
            }
            save_forwards(forwards)
            st.success(f"Forward {fwd_side.upper()} sur {fwd_symbol} enregistré pour {fwd_maturity} à {forward_price:.4f}.")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Renseigne un symbole et assure-toi que le spot est disponible (>0) pour fixer le prix forward.")

    st.markdown("---")
    st.markdown("### 🚀 Forward Portfolio")
    forwards = load_forwards()
    if forwards:
        rows = []
        for key, fwd in forwards.items():
            sym = fwd.get("symbol", "")
            qty = int(fwd.get("quantity", 0) or 0)
            price_fwd = float(fwd.get("forward_price", 0.0) or 0.0)
            side = fwd.get("side", "long")
            maturity_str = fwd.get("maturity")
            try:
                maturity_dt = datetime.date.fromisoformat(maturity_str)
            except Exception:
                maturity_dt = None
            days_to_mat = (maturity_dt - today).days if maturity_dt else None
            spot_now = float(get_data(sym).get("price", 0.0) or 0.0) if sym else 0.0
            mult = 1.0 if side == "long" else -1.0
            pnl_unit = mult * (spot_now - price_fwd)
            pnl_total = pnl_unit * qty
            rows.append({
                "Symbol": sym,
                "Side": side.capitalize(),
                "Quantity": qty,
                "Forward Price": price_fwd,
                "Spot Now": round(spot_now, 4),
                "Maturity": maturity_str,
                "Days to mat": days_to_mat,
                "P&L/unit": round(pnl_unit, 4),
                "P&L total": round(pnl_total, 2),
            })
        if rows:
            df_fwd = pd.DataFrame(rows)
            st.dataframe(df_fwd, width="stretch", hide_index=True)
        if st.button("🧹 Clear forwards", key="clear_forwards"):
            save_forwards({})
            st.success("Forward portfolio cleared.")
            time.sleep(1)
            st.rerun()
    else:
        st.info("Aucun forward pour le moment.")

# Tab 5: app_options (copie intégrale)
with tab5:
    run_app_options()

# Footer
st.markdown("---")
