"""
Calibration controller (thin wrapper over model calibration scaffold).
"""

from __future__ import annotations

from typing import Any, Dict

from app.model.calibration.logic import get_supported_models, run_calibration, validate_request
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


__all__ = ["CalibrationController"]
