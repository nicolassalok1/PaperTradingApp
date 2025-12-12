"""
Calibration controller (thin wrapper over model calibration scaffold).
"""

from __future__ import annotations

import io
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from app.model.calibration.logic import get_supported_models, run_calibration, validate_request
from app.model.calibration.yahoo_iv_parser import parse_yahoo_iv_csv
from app.model.calibration.market_surface import (
    load_csv as load_market_surface_csv_v2,
    build_fixed_grid,
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

    def run_heston_nn_from_yahoo(self, payload: Dict | None) -> Dict[str, Any]:
        data = payload or {}
        csv_bytes = data.get("csv_bytes") or data.get("file")
        S0 = float(data.get("S0") or 0.0)
        r = float(data.get("r") or 0.0)
        q = float(data.get("q") or 0.0)
        asof_date = data.get("asof_date")

        if csv_bytes is None and not data.get("surface_path"):
            return {"success": False, "message": "CSV Yahoo IV requis.", "details": {}}
        if S0 <= 0:
            return {"success": False, "message": "S0 doit être strictement positif.", "details": {}}

        try:
            market_df = parse_yahoo_iv_csv(
                csv_bytes or data.get("surface_path"), asof_date=asof_date, S0=S0
            )
        except Exception as exc:
            return {"success": False, "message": f"Erreur parsing CSV: {exc}", "details": {}}

        m_grid, t_grid = default_grid()
        iv_market, mask = make_fixed_grid(market_df, m_grid, t_grid)

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

        try:
            price_grid = price_grid_from_params(S0, m_grid, t_grid, r, q, params_tuple)
            iv_model = implied_vol_grid(price_grid, S0, m_grid, t_grid, r, q)
            iv_error = np.where(mask, iv_model - iv_market, np.nan)
        except Exception as exc:
            return {"success": False, "message": f"Erreur calcul surface modèle: {exc}", "details": {}}

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

    def run_heston_nn_from_surface(self, payload: Dict | None) -> Dict[str, Any]:
        """
        Pipeline for CSV format with columns: K, T, S0, iv, type (CALL only).
        """
        data = payload or {}
        csv_bytes = data.get("csv_bytes") or data.get("file")
        surface_path = data.get("surface_path")

        if csv_bytes is None and surface_path is None:
            return {"success": False, "message": "CSV surface requis.", "details": {}}

        try:
            raw_df = (
                pd.read_csv(io.BytesIO(csv_bytes))
                if isinstance(csv_bytes, (bytes, bytearray))
                else pd.read_csv(surface_path) if surface_path is not None else None
            )
        except Exception as exc:
            return {"success": False, "message": f"Lecture CSV échouée: {exc}", "details": {}}

        market_df = load_market_surface_csv_v2(raw_df if raw_df is not None else csv_bytes)
        if market_df is None or market_df.empty:
            return {"success": False, "message": "Surface IV vide après parsing.", "details": {}}

        # S0 default from CSV median if not provided
        S0_raw = data.get("S0")
        S0_val = None
        try:
            if S0_raw is not None:
                S0_val = float(S0_raw)
            elif raw_df is not None and "S0" in raw_df.columns:
                S0_vals = pd.to_numeric(raw_df["S0"], errors="coerce")
                S0_pos = S0_vals[S0_vals > 0]
                if not S0_pos.empty:
                    S0_val = float(S0_pos.median())
        except Exception:
            S0_val = None

        if S0_val is None or S0_val <= 0:
            return {"success": False, "message": "S0 invalide ou manquant.", "details": {}}

        r = float(data.get("r") or 0.0)
        q = float(data.get("q") or 0.0)

        iv_market, mask, m_grid, t_grid = build_fixed_grid(market_df)

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

        try:
            price_grid = price_grid_from_params(S0_val, m_grid, t_grid, r, q, params_tuple)
            iv_model = implied_vol_grid(price_grid, S0_val, m_grid, t_grid, r, q)
            iv_error = np.where(mask, iv_model - iv_market, np.nan)
        except Exception as exc:
            return {"success": False, "message": f"Erreur calcul surface modèle: {exc}", "details": {}}

        return {
            "success": True,
            "message": pred.get("message", "OK"),
            "params": params,
            "S0": S0_val,
            "r": r,
            "q": q,
            "m_grid": m_grid.tolist(),
            "t_grid": t_grid.tolist(),
            "iv_market": iv_market.tolist(),
            "iv_model": iv_model.tolist(),
            "iv_error": iv_error.tolist(),
            "mask": mask.tolist(),
        }

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
