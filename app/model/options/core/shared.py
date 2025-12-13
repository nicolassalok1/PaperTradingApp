"""
Shared option utilities (pure logic, UI-agnostic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils.paths import CACHE_CSV_DIR, CACHE_OHLC_DIR


def heatmap_axis(center: float, span: float, n_points: int = 11) -> np.ndarray:
    """Return a numeric axis around a center value."""
    lo = max(0.0, center - span)
    hi = max(lo, center + span)
    if np.isclose(lo, hi):
        return np.array([lo])
    return np.linspace(lo, hi, n_points)


def build_crr_tree(option=None, r: float = 0.0, sigma: float = 0.0, n_steps: int = 1):
    """
    Build a simple CRR tree of underlying prices and option values.
    Accepts an object with attributes s0, T, and payoff(s).
    """
    if option is None or n_steps <= 0:
        return np.zeros((1, 1)), np.zeros((1, 1))

    try:
        T = float(option.T)
        S0 = float(option.s0)
    except Exception:
        return np.zeros((1, 1)), np.zeros((1, 1))

    dt = T / max(1, n_steps)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    a = np.exp(r * dt)
    p = (a - d) / max(u - d, 1e-12)
    q = 1.0 - p

    spot_tree = np.full((n_steps + 1, n_steps + 1), np.nan)
    value_tree = np.full_like(spot_tree, np.nan)

    for level in range(n_steps + 1):
        for up_moves in range(level + 1):
            spot_tree[level, up_moves] = S0 * (u**up_moves) * (d ** (level - up_moves))

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


def compute_american_crr_heatmaps(
    s_values: np.ndarray,
    k_values: np.ndarray,
    maturity: float,
    rate: float,
    sigma: float,
    n_steps: int = 25,
):
    """
    Very simple CRR payoff matrices for visualization:
    - call_matrix: payoff(S_T, K)
    - put_matrix : payoff(K, S_T)
    """
    dt = maturity / max(1, n_steps)
    disc = np.exp(-rate * maturity)
    call_mat = np.zeros((len(k_values), len(s_values)))
    put_mat = np.zeros_like(call_mat)

    for i, strike in enumerate(k_values):
        for j, spot in enumerate(s_values):
            intrinsic_c = max(spot - strike, 0.0)
            intrinsic_p = max(strike - spot, 0.0)
            adj = 0.5 * sigma * np.sqrt(max(maturity, 0.0)) * (j + i + 1) / max(n_steps, 1)
            call_mat[i, j] = max(intrinsic_c, 0.0) * disc + adj
            put_mat[i, j] = max(intrinsic_p, 0.0) * disc + adj * 0.5

    return call_mat, put_mat


def get_cached_iv_for(
    *args,
    df_iv: pd.DataFrame | None = None,
    K_target: float | None = None,
    T_target: float | None = None,
    option_type: str | None = None,
    k_tol: float = 0.05,
    t_tol: float = 0.05,
    ticker: str | None = None,
):
    """
    Lightweight IV fetcher.
    - If a DataFrame is provided (explicitly or as first arg), use it to fetch ['K', 'T', 'iv_market'] (or 'iv').
    - Otherwise, gracefully return None (no cache available yet).
    Returns a float or None.
    """
    if args:
        first = args[0]
        if isinstance(first, pd.DataFrame):
            df_iv = first
            if len(args) > 1:
                K_target = args[1]
            if len(args) > 2:
                T_target = args[2]
            if len(args) > 3 and option_type is None:
                option_type = args[3]
        elif K_target is None:
            if len(args) >= 1:
                K_target = args[0]
            if len(args) >= 2:
                T_target = args[1]
            if len(args) >= 3 and option_type is None:
                option_type = args[2]

    if df_iv is None or getattr(df_iv, "empty", True):
        return _fallback_iv_from_ticker(ticker)
    if K_target is None or T_target is None:
        return _fallback_iv_from_ticker(ticker)

    df = df_iv.copy()
    cols = {str(c).lower(): c for c in df.columns}
    k_col = cols.get("k") or cols.get("strike") or "K"
    t_col = cols.get("t") or cols.get("maturity") or cols.get("tau") or "T"
    iv_col = cols.get("iv_market") or cols.get("iv") or cols.get("sigma") or cols.get("vol")
    type_col = cols.get("type")

    if iv_col is None or k_col not in df.columns or t_col not in df.columns:
        return _fallback_iv_from_ticker(ticker)

    df = df.dropna(subset=[k_col, t_col, iv_col]).copy()
    if df.empty:
        return _fallback_iv_from_ticker(ticker)

    if option_type and type_col and type_col in df.columns:
        cp = "c" if str(option_type).lower().startswith("c") else "p"
        df = df[df[type_col].astype(str).str.lower().str.startswith(cp)]
        if df.empty:
            return _fallback_iv_from_ticker(ticker)

    try:
        k_val = pd.to_numeric(df[k_col], errors="coerce")
        t_val = pd.to_numeric(df[t_col], errors="coerce")
        iv_val = pd.to_numeric(df[iv_col], errors="coerce")
        df = df.assign(**{k_col: k_val, t_col: t_val, iv_col: iv_val}).dropna(
            subset=[k_col, t_col, iv_col]
        )
    except Exception:
        pass

    if df.empty:
        return _fallback_iv_from_ticker(ticker)

    df["dk"] = np.abs(df[k_col] - K_target) / max(float(K_target), 1e-6)
    df["dt"] = np.abs(df[t_col] - T_target)

    df_filt = df[(df["dk"] <= k_tol) & (df["dt"] <= t_tol)]
    if df_filt.empty:
        return _fallback_iv_from_ticker(ticker)

    df_filt = df_filt.copy()
    df_filt["score"] = df_filt["dt"] + df_filt["dk"]
    row = df_filt.sort_values("score").iloc[0]
    try:
        return float(row.get(iv_col, np.nan))
    except Exception:
        return _fallback_iv_from_ticker(ticker)


def _fallback_iv_from_ticker(ticker: str | None) -> float | None:
    if not ticker:
        return None
    iv_proxy = _iv_from_stooq_cache(ticker)
    if iv_proxy is not None and np.isfinite(iv_proxy) and iv_proxy > 0:
        return float(iv_proxy)
    return None


def _iv_from_stooq_cache(ticker: str) -> float | None:
    """
    Derive a simple volatility proxy from stooq OHLC cache:
    annualized std of log returns.
    """
    tk = (ticker or "").strip().lower()
    if not tk:
        return None
    candidates = [
        CACHE_OHLC_DIR / f"stooq_{tk}.us_start_end_d.csv",
        CACHE_OHLC_DIR / f"stooq_{tk}.us_start_end_D.csv",
        CACHE_CSV_DIR / f"stooq_{tk}.us_start_end_d.csv",  # legacy
        CACHE_CSV_DIR / f"stooq_{tk}.us_start_end_D.csv",  # legacy
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    close_col = next((c for c in df.columns if str(c).lower() == "close"), None)
    if close_col is None:
        return None
    try:
        prices = pd.to_numeric(df[close_col], errors="coerce").dropna()
        if len(prices) > 252:
            prices = prices.tail(252)  # recent year
        if len(prices) < 2:
            return None
        log_ret = np.log(prices).diff().dropna()
        if len(log_ret) == 0:
            return None
        sigma = float(log_ret.std() * np.sqrt(252))
        return sigma if np.isfinite(sigma) and sigma > 0 else None
    except Exception:
        return None


def pick_default_T(
    k_ref: float, maturities: list[float] | np.ndarray | None = None, *, target: float = 1.0
):
    """
    Choose best T from a list:
    - Prefer maturities close to 'target' (default ~1Y)
    """
    if maturities is None or len(maturities) == 0:
        return float(target)

    mats = np.array([float(t) for t in maturities if t and t > 0], dtype=float)
    if mats.size == 0:
        return float(target)

    idx = np.argmin(np.abs(mats - target))
    return float(mats[idx])


__all__ = [
    "heatmap_axis",
    "compute_american_crr_heatmaps",
    "build_crr_tree",
    "get_cached_iv_for",
    "pick_default_T",
]
