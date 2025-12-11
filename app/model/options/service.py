"""
Options service layer (business logic only, no UI).
Provides pricing/greeks/payoff helpers consumed by the controller.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from app.model.options.core.greeks import compute_option_greeks
from app.model.options.heatmaps import compute_crr_heatmaps, heatmap_axis
from app.model.options.engines.pricing import CrankNicolsonBS, bs_option_price
from app.model.options.data.iv_surface import fetch_iv_surface, interpolate_surface
from app.model.options.mc_engine import MCModel, price_european_mc


def compute_price(option_dict: dict, market: dict) -> float:
    """
    Core pricing via Crank-Nicolson wrapper.
    """
    opt = option_dict or {}
    mkt = market or {}
    cpflag = str(opt.get("option_char") or opt.get("type") or opt.get("option_type") or "c").lower()
    cpflag = "p" if cpflag.startswith("p") else "c"
    type_raw = str(opt.get("style") or opt.get("exercise") or "Eu").strip().lower()
    if type_raw.startswith("am"):
        typeflag = "Am"
    elif type_raw.startswith("bm"):
        typeflag = "Bmd"
    else:
        typeflag = "Eu"

    S0 = float(
        opt.get("S0")
        or opt.get("spot")
        or opt.get("underlying_close")
        or mkt.get("spot")
        or mkt.get("S0")
        or 0.0
    )
    K = float(opt.get("strike") or 0.0)
    T = float(opt.get("T") or opt.get("maturity") or opt.get("maturity_years") or 0.0)
    vol = float(opt.get("sigma") or opt.get("iv") or opt.get("vol") or 0.0)
    r = float(opt.get("r") or mkt.get("r") or 0.0)
    q = float(opt.get("q") or opt.get("dividend_yield") or mkt.get("q") or 0.0)

    solver = CrankNicolsonBS(typeflag, cpflag, S0=S0, K=K, T=T, vol=vol, r=r, d=q)
    price, _, _, _ = solver.CN_option_info()
    return float(price)


def compute_greeks(option_dict: dict, market: dict) -> dict:
    """
    Compute greeks using the model helper.
    """
    spot = None
    if isinstance(market, dict):
        spot = market.get("spot") or market.get("S0")
    return compute_option_greeks(option_dict or {}, spot=spot)


def compute_payoff_surface(
    option_dict: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return CRR heatmaps and axes.
    """
    opt = option_dict or {}
    S0 = float(opt.get("S0") or opt.get("spot") or 0.0)
    K = float(opt.get("strike") or 0.0)
    T = float(opt.get("T") or opt.get("maturity") or opt.get("maturity_years") or 0.0)
    r = float(opt.get("r") or 0.0)
    sigma = float(opt.get("sigma") or opt.get("iv") or opt.get("vol") or 0.0)
    n_steps = int(opt.get("n_steps") or 25)

    span_S = 0.25 * S0 if S0 else 0.0
    span_K = 0.25 * K if K else 0.0
    s_axis = heatmap_axis(S0, span_S, n_points=31)
    k_axis = heatmap_axis(K, span_K, n_points=31)
    call_matrix, put_matrix = compute_crr_heatmaps(s_axis, k_axis, T, r, sigma, n_steps)
    return call_matrix, put_matrix, s_axis, k_axis


def load_iv_surface(symbol: str):
    """
    Fetch and interpolate IV surface for a ticker.
    """
    df_iv = fetch_iv_surface(symbol)
    try:
        return interpolate_surface(df_iv)
    except Exception:
        return None, None, None


def price_and_greeks(option: dict, market: dict) -> dict:
    """
    Bundle price and greeks.
    """
    return {
        "price": compute_price(option, market),
        "greeks": compute_greeks(option, market),
    }


def price_option_mc(
    ticker: str,
    K: float,
    T: float,
    sigma: float,
    model: str = "bs",
    n_paths: int = 10000,
    n_steps: int = 252,
) -> float:
    """
    Thin wrapper over price_european_mc.
    Uses MCModel enum for validation, resolves ticker into S0 using fetch_spot_price if needed.
    """
    from app.model.market_data.market_data import fetch_spot_price
    from app.model.yieldcurve.rates_utils import get_r, get_q

    S0 = fetch_spot_price(ticker)
    if S0 is None:
        raise ValueError(f"Cannot determine spot for ticker {ticker!r}")

    try:
        mc_model = MCModel(model)
    except ValueError:
        mc_model = MCModel.BS

    return price_european_mc(
        S0=S0,
        K=K,
        T=T,
        sigma=sigma,
        model=mc_model.value,
        n_paths=n_paths,
        n_steps=n_steps,
        ticker=ticker,
    )


def price_european_from_cboe(
    ticker: str,
    K: float,
    T: float,
    option_type: str = "call",
    max_maturity_years: float = 2.0,
) -> dict:
    """
    Pricing européen basé sur la surface IV CBOE :
    - récupère la surface IV (K, T, iv) via fetch_iv_surface
    - choisit le point le plus proche de (K, T)
    - price via Black-Scholes (bs_option_price)
    Retourne un dict : {price, S0, K, T, iv, r, cp}.
    """
    from app.model.market_data.market_data import fetch_spot_price
    from app.model.yieldcurve.rates_utils import get_r

    sym = (ticker or "").strip().upper()
    if not sym:
        raise ValueError("Ticker manquant pour price_european_from_cboe().")

    df = fetch_iv_surface(sym, max_maturity_years=max_maturity_years)
    if df is None or df.empty:
        raise ValueError(f"Aucune surface IV CBOE pour {sym}.")

    cols = {c.lower(): c for c in df.columns}
    k_col = cols.get("k") or cols.get("strike")
    t_col = cols.get("t") or cols.get("maturity") or cols.get("tau")
    iv_col = cols.get("iv") or cols.get("sigma") or cols.get("vol")
    if not (k_col and t_col and iv_col):
        raise ValueError("Surface IV: colonnes K/T/iv manquantes.")

    df_clean = df.dropna(subset=[k_col, t_col, iv_col]).copy()
    cp = "c" if option_type.lower().startswith("c") else "p"
    if "type" in df_clean.columns:
        df_clean = df_clean[df_clean["type"].astype(str).str.lower().str.startswith(cp)]
        if df_clean.empty:
            df_clean = df.dropna(subset=[k_col, t_col, iv_col]).copy()

    if df_clean.empty:
        raise ValueError(f"Surface IV vide pour {sym} (type {option_type}).")

    df_clean["dK"] = (df_clean[k_col] - K).abs()
    df_clean["dT"] = (df_clean[t_col] - T).abs()
    K_scale = max(abs(K), 1.0)
    T_scale = max(abs(T), 1e-4)
    df_clean["score"] = df_clean["dK"] / K_scale + df_clean["dT"] / T_scale

    row = df_clean.sort_values("score").iloc[0]
    K_used = float(row[k_col])
    T_used = float(row[t_col])
    iv = float(row[iv_col])

    S0 = fetch_spot_price(sym)
    if S0 is None:
        S0 = float(row.get("S0", float("nan")))

    r = float(get_r(T_used))

    price = bs_option_price(
        0.0,
        float(S0),
        K_used,
        T_used,
        r,
        iv,
        "call" if cp == "c" else "put",
    )

    return {
        "price": float(price),
        "S0": float(S0),
        "K": K_used,
        "T": T_used,
        "iv": iv,
        "r": r,
        "cp": cp,
    }


__all__ = [
    "compute_price",
    "compute_greeks",
    "compute_payoff_surface",
    "load_iv_surface",
    "price_and_greeks",
    "price_option_mc",
    "price_european_from_cboe",
]
