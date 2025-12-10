from __future__ import annotations

import json

from app.model.heston.pricing import price_heston_carr_madan
from app.utils.paths import JSON_DIR


HESTON_PARAMS_FILE = JSON_DIR / "options_h_params_calibrated.json"
HESTON_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_heston_params_from_json(ticker: str) -> dict | None:
    """Load persisted Heston params (flat schema with optional ticker)."""
    tkr = (ticker or "").strip().upper()
    if not tkr or not HESTON_PARAMS_FILE.exists():
        return None
    try:
        with HESTON_PARAMS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and {"kappa", "theta", "sigma", "rho", "v0"}.issubset(
            data.keys()
        ):
            if not data.get("ticker") or str(data.get("ticker")).strip().upper() == tkr:
                return data
            return None
        params = data.get(tkr) if isinstance(data, dict) else None
        if not isinstance(params, dict):
            return None
        return params
    except Exception:
        return None


def save_heston_params_to_json(ticker: str, params: dict) -> None:
    """Persist Heston params in a flat JSON with an optional ticker tag."""
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return
    try:
        HESTON_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(params) if isinstance(params, dict) else {}
        payload["ticker"] = tkr
        with HESTON_PARAMS_FILE.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def price_heston_european_call(
    S0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    option_type: str = "call",
) -> float:
    """
    Thin wrapper around the Carr-Madan Heston pricer.
    """
    return float(
        price_heston_carr_madan(
            S0=float(S0),
            K=float(K),
            T=float(T),
            r=float(r),
            q=float(q),
            kappa=float(kappa),
            theta=float(theta),
            sigma=float(sigma),
            rho=float(rho),
            v0=float(v0),
            option_type=option_type,
        )
    )
