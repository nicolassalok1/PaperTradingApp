"""
Calibration controller (thin wrapper over model calibration scaffold).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from app.model.calibration.logic import get_supported_models, run_calibration, validate_request
from app.model.calibration.market_surface import (
    default_grid,
    load_market_surface_csv,
    make_fixed_grid,
)
from app.model.calibration.heston_pricer import price_grid_from_params
from app.model.calibration.implied_vol import implied_vol_grid
from app.model.calibration.heston_nn import predict_params, WEIGHTS_PATH
from app.model.calibration.types import (
    CalibrationModelName,
    CalibrationRequest,
    CalibrationResult,
    MarketSurfaceSource,
)


class CalibrationController:
    """Placeholder for future model calibration (architecture only)."""

    def get_models(self) -> list[str]:
        return [m.value for m in get_supported_models()]

    def _coerce_model(self, raw: Any) -> CalibrationModelName:
        try:
            return CalibrationModelName(str(raw))
        except Exception:
            return CalibrationModelName.HESTON

    def _coerce_source(self, raw: Any) -> MarketSurfaceSource:
        try:
            return MarketSurfaceSource(str(raw))
        except Exception:
            return MarketSurfaceSource.API_PLACEHOLDER

    def submit(self, payload: Dict | None) -> Dict[str, Any]:
        data = payload or {}
        model = self._coerce_model(data.get("model"))
        source = self._coerce_source(data.get("source"))
        constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else None
        req = CalibrationRequest(
            model=model,
            source=source,
            ticker=str(data.get("ticker") or "").strip() or None,
            constraints=constraints,
            surface_path=data.get("surface_path"),
        )
        ok, msg = validate_request(req)
        if not ok:
            return {"success": False, "message": msg}
        result = run_calibration(req)
        if isinstance(result, CalibrationResult):
            return {
                "success": bool(result.success),
                "message": result.message,
                "details": result.details or {},
            }
        return {"success": False, "message": "Calibration non implémentée.", "details": {}}

    def _to_ndarray(self, arr: Any) -> np.ndarray:
        try:
            return np.asarray(arr, dtype=float)
        except Exception:
            return np.array([])

    def run_heston_nn_calibration(self, payload: Dict | None) -> Dict[str, Any]:
        data = payload or {}
        csv_bytes = data.get("csv_bytes")
        df = data.get("df")
        market_df = load_market_surface_csv(df if df is not None else csv_bytes)
        m_grid = self._to_ndarray(data.get("m_grid"))
        t_grid = self._to_ndarray(data.get("t_grid"))
        if m_grid.size == 0 or t_grid.size == 0:
            m_grid, t_grid = default_grid()

        iv_market, mask = make_fixed_grid(market_df, m_grid, t_grid)
        S0 = float(data.get("S0") or 100.0)
        r = float(data.get("r") or 0.02)
        q = float(data.get("q") or 0.0)

        pred = predict_params(iv_market, m_grid, t_grid, weights_path=WEIGHTS_PATH)
        if not pred.get("success"):
            return {"success": False, "message": pred.get("message", "Erreur prédiction."), "details": {}}

        params = pred.get("params") or {}
        params_tuple = (
            params.get("kappa"),
            params.get("theta"),
            params.get("sigma"),
            params.get("rho"),
            params.get("v0"),
        )
        if any(p is None for p in params_tuple):
            return {"success": False, "message": "Paramètres incomplets.", "details": {}}

        price_grid = price_grid_from_params(S0, m_grid, t_grid, r, q, params_tuple)
        iv_model = implied_vol_grid(price_grid, S0, m_grid, t_grid, r, q)
        iv_error = iv_model - iv_market

        return {
            "success": True,
            "message": pred.get("message", "OK"),
            "params": params,
            "m_grid": m_grid.tolist(),
            "t_grid": t_grid.tolist(),
            "iv_market": iv_market.tolist(),
            "iv_model": iv_model.tolist(),
            "iv_error": iv_error.tolist(),
            "mask": mask.tolist(),
        }


__all__ = ["CalibrationController"]
