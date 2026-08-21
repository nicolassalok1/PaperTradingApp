"""
Kalman Filters tab controller (MVC bridge, no streamlit here).

Bridges the view to:
  - app.model.market_data          : history bars + latest spot
  - app.model.volatility_models.kalman.ou_kalman : OU calibration, Kalman
    mean-level filter, OU forecast, band signals and backtests.

Everything returned to the view is JSON-safe (plain dict/list/float), so the
view can park results in st.session_state without holding model objects.
The live tick path works the same way: the filter state travels as a plain
dict (phi, mu, Q, R, x, P) through `kalman_step`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Bar granularities offered by the tab.
# label -> (yahoo interval, default range, seconds per bar)
BAR_CHOICES: Dict[str, tuple[str, str, int]] = {
    "1 min": ("1m", "5d", 60),
    "5 min": ("5m", "1mo", 300),
    "15 min": ("15m", "1mo", 900),
    "1 h": ("1h", "3mo", 3600),
    "1 jour": ("1d", "1y", 86400),
}


def bar_choice_labels() -> List[str]:
    return list(BAR_CHOICES.keys())


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy containers/scalars to plain python."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------
def load_history(ticker: str, bar_label: str, *, max_bars: int = 500) -> Dict[str, Any]:
    """
    Load OHLC bars for the tab. Intraday granularities hit the Yahoo chart
    endpoint (Stooq is daily-only); "1 jour" uses the cached daily path.
    Returns {"success", "message", "bars": [{t,o,h,l,c}], "bar_seconds"}.
    """
    from app.model.market_data.market_data import fetch_intraday_ohlc

    tk = (ticker or "").strip().upper()
    if not tk:
        return {"success": False, "message": "Ticker vide.", "bars": []}
    choice = BAR_CHOICES.get(bar_label) or BAR_CHOICES["5 min"]
    interval, period, bar_seconds = choice

    try:
        df = fetch_intraday_ohlc(tk, interval=interval, period=period)
    except Exception as exc:  # defensive: network/parse failures reach the UI as a message
        return {"success": False, "message": f"Chargement impossible : {exc}", "bars": []}

    if df is None or df.empty or "Close" not in df.columns:
        return {
            "success": False,
            "message": "Aucune barre reçue (ticker inconnu, marché fermé depuis trop longtemps, ou Yahoo indisponible).",
            "bars": [],
        }

    close_num = pd.to_numeric(df["Close"], errors="coerce")
    df = df.loc[np.isfinite(close_num.to_numpy(dtype=float))].tail(int(max_bars))
    bars: List[Dict[str, Any]] = []
    has_ohlc = all(col in df.columns for col in ("Open", "High", "Low"))

    def _finite_or(v: Any, default: float) -> float:
        try:
            fv = float(v)
        except Exception:
            return default
        return fv if np.isfinite(fv) else default

    for _, row in df.iterrows():
        ts = row.get("Date")
        try:
            t_iso = pd.Timestamp(ts).isoformat()
        except Exception:
            t_iso = str(ts)
        close = float(row["Close"])
        bars.append(
            {
                "t": t_iso,
                "o": _finite_or(row.get("Open"), close) if has_ohlc else close,
                "h": _finite_or(row.get("High"), close) if has_ohlc else close,
                "l": _finite_or(row.get("Low"), close) if has_ohlc else close,
                "c": close,
            }
        )
    if len(bars) < 5:
        return {"success": False, "message": f"Seulement {len(bars)} barre(s) exploitables (minimum 5).", "bars": bars}
    return {
        "success": True,
        "message": f"{len(bars)} barres chargées — dernière : {bars[-1]['t']}.",
        "bars": bars,
        "bar_seconds": bar_seconds,
        "interval": interval,
        "period": period,
    }


def latest_price(ticker: str) -> float | None:
    """
    Live spot for the tick path: Alpaca latest trade -> quote mid. None without
    Alpaca keys: the Stooq fallback of fetch_spot_price is a disk-cached daily
    close of unknown age, which must never be fed to the filter as a live tick.
    """
    from app.model.market_data.market_data import fetch_live_spot_price

    try:
        px = fetch_live_spot_price((ticker or "").strip().upper())
        return float(px) if px is not None and np.isfinite(float(px)) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Calibration + filter + bands + backtests
# ---------------------------------------------------------------------------
def run_pipeline(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    payload: {closes: [...], calib_window: int, noise_lever: float,
              band_k: float, forecast_steps: int, mu_mode: "window"|"ar1",
              calib_anchor: "head"|"tail"}
    """
    from app.model.volatility_models.kalman.ou_kalman import run_ou_kalman_pipeline

    data = payload or {}
    closes = data.get("closes")
    if closes is None:
        return {"success": False, "message": "Série de clôtures manquante."}
    try:
        y = np.asarray(closes, dtype=float)
    except Exception:
        return {"success": False, "message": "Clôtures invalides (numériques attendues)."}

    def _num(key: str, default: float) -> float:
        v = data.get(key)
        return default if v is None else float(v)

    try:
        res = run_ou_kalman_pipeline(
            y,
            calib_window=int(_num("calib_window", 60)),
            noise_lever=_num("noise_lever", 50.0),
            band_k=_num("band_k", 1.0),
            forecast_steps=int(_num("forecast_steps", 5)),
            mu_mode=str(data.get("mu_mode") or "window"),
            calib_anchor=str(data.get("calib_anchor") or "head"),
        )
    except Exception as exc:
        return {"success": False, "message": f"Échec du pipeline OU/Kalman : {exc}"}
    if res is None:
        return {
            "success": False,
            "message": "Pas assez de barres pour l'AR(1) (minimum 5). Augmente la fenêtre ou change la granularité.",
        }

    params = res["params"]
    run = res["kalman"]
    state = run.state
    out: Dict[str, Any] = {
        "success": True,
        "message": "OK",
        "params": {
            "phi": params.phi,
            "mu": params.mu,
            "sigma_eps": params.sigma_eps,
            "mu_window": params.mu_window,
            "mu_ar1": params.mu_ar1,
            "theta": params.theta,
            "half_life_bars": params.half_life_bars,
            "sigma_stat": params.sigma_stat,
            "n_bars": params.n_bars,
        },
        "obs_scale": res["obs_scale"],
        "sigma_stat": res["sigma_stat"],
        "band_halfwidth": res["band_halfwidth"],
        "calib_anchor": res["calib_anchor"],
        "calib_window": res["calib_window"],
        "n_bars": res["n_bars"],
        "x_filt": run.x_filt,
        "x_pred": run.x_pred,
        "P_filt": run.P_filt,
        "kalman_gain_last": float(run.gain[-1]) if run.gain.size else 0.0,
        "state": {
            "phi": state.phi,
            "mu": state.mu,
            "Q": state.Q,
            "R": state.R,
            "x": state.x,
            "P": state.P,
        },
        "forecast": res["forecast"],
        "backtests": {},
    }
    if res.get("backtests"):
        out["oos_start"] = res.get("oos_start")
        for name, bt in res["backtests"].items():
            out["backtests"][name] = {
                "positions": bt.positions,
                "pnl": bt.pnl,
                "equity": bt.equity,
                "trades": bt.trades,
                "stats": bt.stats,
            }
    return _json_safe(out)


def kalman_step(state: Dict[str, Any] | None, price: float, *, forecast_steps: int = 5) -> Dict[str, Any]:
    """
    One live tick: predict + update the mean level with the observed price.
    `state` is the plain dict produced by run_pipeline()["state"] (or by a
    previous kalman_step). Returns the updated state + fresh OU forecast.
    """
    from app.model.volatility_models.kalman.ou_kalman import KalmanOUState

    if not state:
        return {"success": False, "message": "État Kalman manquant (lance d'abord une calibration)."}
    try:
        st = KalmanOUState(
            phi=float(state["phi"]),
            mu=float(state["mu"]),
            Q=float(state["Q"]),
            R=float(state["R"]),
            x=float(state["x"]),
            P=float(state["P"]),
        )
        z = float(price)
    except Exception as exc:
        return {"success": False, "message": f"État ou prix invalide : {exc}"}
    if not np.isfinite(z):
        return {"success": False, "message": "Prix non fini."}
    if not all(np.isfinite(v) for v in (st.phi, st.mu, st.Q, st.R, st.x, st.P)):
        return {"success": False, "message": "État Kalman non fini."}
    if not (0.0 < st.phi < 1.0) or st.Q < 0.0 or st.R <= 0.0 or st.P < 0.0:
        return {"success": False, "message": "État Kalman hors domaine (phi dans (0,1), Q,P >= 0, R > 0)."}

    x_pred = st.phi * st.x + (1.0 - st.phi) * st.mu
    try:
        st.update(z)
        fc = st.forecast(int(forecast_steps))
    except Exception as exc:
        return {"success": False, "message": f"Tick impossible : {exc}"}
    return _json_safe(
        {
            "success": True,
            "state": {"phi": st.phi, "mu": st.mu, "Q": st.Q, "R": st.R, "x": st.x, "P": st.P},
            "x": st.x,
            "x_pred": x_pred,
            "P": st.P,
            "forecast_mean": fc,
        }
    )


__all__ = [
    "BAR_CHOICES",
    "bar_choice_labels",
    "load_history",
    "latest_price",
    "run_pipeline",
    "kalman_step",
]
