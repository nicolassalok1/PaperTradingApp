"""
Calibration controller (thin wrapper over model calibration scaffold).
"""

from __future__ import annotations

import io
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from app.model.calibration.logic import get_supported_models, run_calibration, validate_request
from app.model.calibration.heston_calibrator import (
    PARAM_ORDER as HESTON_PARAM_ORDER,
    DEFAULT_BOUNDS as HESTON_DEFAULT_BOUNDS,
    build_bounds as build_heston_bounds,
    calibrate_heston_least_squares,
)
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
from app.model.calibration.heston_nn import (
    TORCH_AVAILABLE,
    TORCH_IMPORT_ERROR,
    WEIGHTS_PATH,
    predict_params,
    train_heston_surface_net,
)
from app.model.calibration.storage import (
    list_results as list_calibration_results,
    load_result as load_calibration_result,
    save_result as save_calibration_result,
)
from app.model.calibration.types import (
    CalibrationModelName,
    CalibrationRequest,
    CalibrationResult,
    MarketSurfaceSource,
)
from app.model.options.data.iv_surface import fetch_iv_surface as _fetch_iv_surface
from app.model.options.logic import download_options_alpaca as _download_options_alpaca


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
        """
        Convert to numpy array; return empty if input is None or scalar (avoids len() on 0-d arrays).
        """
        if arr is None:
            return np.array([])
        try:
            out = np.asarray(arr, dtype=float)
            if out.ndim == 0:  # scalar -> treat as empty so defaults kick in
                return np.array([])
            return out
        except Exception:
            return np.array([])

    def get_heston_nn_info(self) -> Dict[str, Any]:
        """Expose whether Heston NN weights are available on disk."""
        info: Dict[str, Any] = {
            "weights_path": str(WEIGHTS_PATH),
            "weights_exists": False,
            "torch_available": bool(TORCH_AVAILABLE),
            "torch_error": str(TORCH_IMPORT_ERROR or ""),
        }
        try:
            exists = bool(WEIGHTS_PATH.exists())
        except Exception:
            exists = False
        info["weights_exists"] = exists
        if exists:
            try:
                st = WEIGHTS_PATH.stat()
                info["size_bytes"] = int(st.st_size)
                info["mtime"] = float(st.st_mtime)
            except Exception:
                pass
        return info

    def train_heston_nn_weights(self, payload: Dict | None, progress_callback=None) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            msg = "PyTorch non installé → entraînement NN indisponible."
            if TORCH_IMPORT_ERROR:
                msg = f"{msg} ({TORCH_IMPORT_ERROR})"
            return {"success": False, "message": msg, "details": {}}

        data = payload or {}
        mode = str(data.get("mode") or "surface").lower().strip()

        # Surface-based CNN (legacy)
        if mode == "surface":
            n_samples = int(data.get("n_samples") or 2000)
            epochs = int(data.get("epochs") or 25)
            batch_size = int(data.get("batch_size") or 64)
            lr = float(data.get("lr") or 1e-3)
            device = str(data.get("device") or "cpu")
            seed = data.get("seed")
            try:
                seed_i = int(seed) if seed is not None else 42
            except Exception:
                seed_i = None
            u_max = float(data.get("u_max") or 50.0)
            n_integration = int(data.get("n_integration") or 800)
            S0 = float(data.get("S0") or 100.0)
            r = float(data.get("r") or 0.02)
            q = float(data.get("q") or 0.0)

            try:
                res = train_heston_surface_net(
                    n_samples=n_samples,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                    seed=seed_i,
                    u_max=u_max,
                    n_integration=n_integration,
                    S0=S0,
                    r=r,
                    q=q,
                    weights_path=WEIGHTS_PATH,
                    progress_epoch=progress_callback,
                )
            except Exception as exc:
                return {"success": False, "message": str(exc), "details": {}}

            if isinstance(res, dict) and not bool(res.get("success", True)):
                return {
                    "success": False,
                    "message": str(res.get("message") or "Entraînement NN échoué."),
                    "details": res,
                    "nn_info": self.get_heston_nn_info(),
                }

            return {
                "success": True,
                "message": "Poids NN entraînés.",
                "details": res,
                "nn_info": self.get_heston_nn_info(),
            }

        # Triplet (S0,K,T) → params, price RMSE loss
        return {
            "success": False,
            "message": "Mode NN par point (S0,K,T -> params) désactivé (paramètres globaux uniquement).",
            "details": {},
        }

    def get_heston_default_bounds(self) -> Dict[str, Any]:
        """Expose default calibration bounds for UI builder."""
        return {k: [float(v[0]), float(v[1])] for k, v in (HESTON_DEFAULT_BOUNDS or {}).items()}

    def fetch_yahoo_iv_surface(self, ticker: str, max_maturity_years: float = 2.0) -> pd.DataFrame:
        """
        Fetch/build an IV surface derived from the Yahoo option chain.
        Kept in the controller to avoid the view importing model modules directly.
        """
        return _fetch_iv_surface(ticker, max_maturity_years=float(max_maturity_years))

    def list_saved_calibrations(self, limit: int = 200) -> Dict[str, Any]:
        try:
            items = list_calibration_results(limit=int(limit))
        except Exception as exc:
            return {"success": False, "message": str(exc), "items": []}
        return {"success": True, "items": items}

    def save_calibration_result(self, result: Dict[str, Any], name: str | None = None, overwrite: bool = False) -> Dict[str, Any]:
        try:
            return save_calibration_result(result=result, name=name, overwrite=bool(overwrite))
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    def load_calibration_result(self, name_or_path: str) -> Dict[str, Any]:
        try:
            return load_calibration_result(name_or_path=name_or_path)
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    def download_alpaca_options_chain(self, ticker: str) -> Dict[str, Any]:
        """Download an Alpaca options snapshot chain for an underlying ticker."""
        sym = str(ticker or "").strip().upper()
        if not sym:
            return {"success": False, "message": "Ticker manquant.", "df": None}
        try:
            df = _download_options_alpaca(sym)
        except Exception as exc:
            return {"success": False, "message": f"Erreur Alpaca: {exc}", "df": None}
        if df is None or df.empty:
            return {"success": False, "message": f"Aucune option retournée pour {sym}.", "df": None}
        return {"success": True, "message": f"{len(df)} contrats chargés pour {sym}.", "ticker": sym, "df": df}

    def run_heston_nn_from_yahoo(self, payload: Dict | None) -> Dict[str, Any]:
        data = payload or {}
        csv_bytes = data.get("csv_bytes") or data.get("file")
        S0 = float(data.get("S0") or 0.0)
        r = float(data.get("r") or 0.0)
        q = float(data.get("q") or 0.0)
        asof_date = data.get("asof_date")
        constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else None

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
        lb, ub, err = build_heston_bounds(constraints)
        if not err and params:
            for i, name in enumerate(HESTON_PARAM_ORDER):
                if name in params and params[name] is not None:
                    try:
                        params[name] = float(np.clip(float(params[name]), float(lb[i]), float(ub[i])))
                    except Exception:
                        pass
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
            "metrics": self._iv_error_metrics(iv_error, mask),
            "mask": mask.tolist(),
        }

    def run_heston_nn_from_surface(self, payload: Dict | None) -> Dict[str, Any]:
        """
        Pipeline for CSV format with columns: K, T, S0, iv, type (CALL only).
        """
        data = payload or {}
        csv_bytes = data.get("csv_bytes") or data.get("file")
        surface_path = data.get("surface_path")
        df_in = data.get("df")
        constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else None
        ticker = str(data.get("ticker") or "").strip().upper() or None

        if not isinstance(df_in, pd.DataFrame) and csv_bytes is None and surface_path is None:
            return {"success": False, "message": "CSV surface requis.", "details": {}}

        try:
            raw_df = None
            if isinstance(df_in, pd.DataFrame):
                raw_df = df_in.copy()
            elif isinstance(csv_bytes, (bytes, bytearray)):
                raw_df = pd.read_csv(io.BytesIO(csv_bytes))
            elif surface_path is not None:
                raw_df = pd.read_csv(surface_path)
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
        lb, ub, err = build_heston_bounds(constraints)
        if not err and params:
            for i, name in enumerate(HESTON_PARAM_ORDER):
                if name in params and params[name] is not None:
                    try:
                        params[name] = float(np.clip(float(params[name]), float(lb[i]), float(ub[i])))
                    except Exception:
                        pass
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
            "ticker": ticker,
            "S0": S0_val,
            "r": r,
            "q": q,
            "m_grid": m_grid.tolist(),
            "t_grid": t_grid.tolist(),
            "iv_market": iv_market.tolist(),
            "iv_model": iv_model.tolist(),
            "iv_error": iv_error.tolist(),
            "metrics": self._iv_error_metrics(iv_error, mask),
            "mask": mask.tolist(),
        }

    def _iv_error_metrics(self, iv_error: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        try:
            err = np.asarray(iv_error, dtype=float)[np.asarray(mask, dtype=bool)]
            err = err[np.isfinite(err)]
        except Exception:
            return {}
        if err.size == 0:
            return {}
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        max_abs = float(np.max(np.abs(err)))
        return {"mae": mae, "rmse": rmse, "max_abs": max_abs}

    def _sanitize_heston_params(self, params: Dict[str, Any], constraints: Dict[str, Any] | None = None) -> tuple[Dict[str, float], tuple[float, float, float, float, float]]:
        lb, ub, err = build_heston_bounds(constraints)
        if err:
            raise ValueError(err)
        if not isinstance(params, dict):
            raise ValueError("Paramètres Heston invalides.")

        clean: Dict[str, float] = {}
        vals: list[float] = []
        for i, name in enumerate(HESTON_PARAM_ORDER):
            try:
                v = float(params.get(name))
            except Exception:
                v = float("nan")
            v = float(np.minimum(np.maximum(v, lb[i]), ub[i]))
            if not np.isfinite(v):
                raise ValueError(f"Paramètre {name} invalide.")
            clean[name] = v
            vals.append(v)

        if len(vals) != len(HESTON_PARAM_ORDER):
            raise ValueError("Paramètres Heston: longueur invalide (attendu 5).")

        kappa, theta, sigma, rho, v0 = vals
        if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0 or abs(rho) >= 1:
            raise ValueError("Paramètres Heston hors domaine (positivité ou |rho|>=1).")

        return clean, (kappa, theta, sigma, rho, v0)

    def run_heston_global_calibration(self, payload: Dict | None) -> Dict[str, Any]:
        """
        Global Heston calibration pipeline: fixed grid -> NN warm start -> optional LS refine -> IV surface.
        Always returns a single parameter vector (kappa, theta, sigma, rho, v0).
        """
        data = payload or {}
        csv_bytes = data.get("csv_bytes") or data.get("file")
        surface_path = data.get("surface_path")
        df_in = data.get("df")
        constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else None
        ticker = str(data.get("ticker") or "").strip().upper() or None

        if not isinstance(df_in, pd.DataFrame) and csv_bytes is None and surface_path is None:
            return {"success": False, "message": "CSV surface requis.", "details": {}}

        try:
            raw_df = None
            market_df = None
            if isinstance(df_in, pd.DataFrame):
                # If already on (moneyness, ttm, iv[, S0]) grid, keep it; else reparse below.
                cols = {str(c).strip().lower(): c for c in df_in.columns}
                if "moneyness" in cols and "ttm" in cols and "iv" in cols:
                    market_df = pd.DataFrame(
                        {
                            "moneyness": pd.to_numeric(df_in[cols["moneyness"]], errors="coerce"),
                            "ttm": pd.to_numeric(df_in[cols["ttm"]], errors="coerce"),
                            "iv": pd.to_numeric(df_in[cols["iv"]], errors="coerce"),
                        }
                    )
                    if "s0" in cols:
                        market_df["S0"] = pd.to_numeric(df_in[cols["s0"]], errors="coerce")
                    market_df = market_df.dropna(subset=["moneyness", "ttm", "iv"])
                else:
                    raw_df = df_in.copy()
            elif isinstance(csv_bytes, (bytes, bytearray)):
                raw_df = pd.read_csv(io.BytesIO(csv_bytes))
            elif surface_path is not None:
                raw_df = pd.read_csv(surface_path)
        except Exception as exc:
            return {"success": False, "message": f"Lecture CSV échouée: {exc}", "details": {}}

        if market_df is None:
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
                s0_vals = pd.to_numeric(raw_df["S0"], errors="coerce")
                s0_pos = s0_vals[s0_vals > 0]
                if not s0_pos.empty:
                    S0_val = float(s0_pos.median())
            elif "S0" in market_df.columns:
                s0_vals = pd.to_numeric(market_df["S0"], errors="coerce")
                s0_pos = s0_vals[s0_vals > 0]
                if not s0_pos.empty:
                    S0_val = float(s0_pos.median())
        except Exception:
            S0_val = None

        if S0_val is None or S0_val <= 0:
            return {"success": False, "message": "S0 invalide ou manquant.", "details": {}}

        r = float(data.get("r") or 0.0)
        q = float(data.get("q") or 0.0)
        fit_to_observed_only = bool(data.get("fit_to_observed_only", True))
        u_max = float(data.get("u_max") or 50.0)
        n_integration = int(data.get("n_integration") or 2000)
        max_nfev = int(data.get("max_nfev") or 60)
        n_starts = int(data.get("n_starts") or 1)
        seed = data.get("seed")
        try:
            seed = int(seed) if seed is not None else None
        except Exception:
            seed = None
        refine = bool(data.get("refine", True))

        m_grid = self._to_ndarray(data.get("m_grid"))
        t_grid = self._to_ndarray(data.get("t_grid"))
        if m_grid.size == 0 or t_grid.size == 0:
            iv_market, mask, m_grid, t_grid = build_fixed_grid(market_df)
        else:
            iv_market, mask = make_fixed_grid(market_df, m_grid, t_grid)
        mask_bool = np.asarray(mask, dtype=bool)

        pred = predict_params(iv_market, m_grid, t_grid, weights_path=WEIGHTS_PATH)
        if not pred.get("success"):
            return {"success": False, "message": pred.get("message", "Erreur prédiction."), "details": {"pred": pred}}

        try:
            params_nn, params_tuple_nn = self._sanitize_heston_params(pred.get("params") or {}, constraints)
        except Exception as exc:
            return {
                "success": False,
                "message": f"Paramètres NN invalides: {exc}",
                "details": {"pred": pred},
            }

        params_final = params_nn
        params_tuple = params_tuple_nn
        calib = None
        msg = str(pred.get("message") or "OK")

        if refine:
            calib = calibrate_heston_least_squares(
                S0=S0_val,
                r=r,
                q=q,
                m_grid=m_grid,
                t_grid=t_grid,
                iv_market=iv_market,
                mask=mask_bool,
                constraints=constraints,
                fit_to_observed_only=fit_to_observed_only,
                u_max=u_max,
                n_integration=n_integration,
                max_nfev=max_nfev,
                n_starts=n_starts,
                seed=seed,
                x0=params_tuple_nn,
            )
            if calib.get("success"):
                try:
                    params_final, params_tuple = self._sanitize_heston_params(calib.get("params") or params_nn, constraints)
                    msg = calib.get("message", msg)
                except Exception as exc:
                    msg = f"LS calibrée mais paramètres invalides: {exc}"
            else:
                msg = f"NN uniquement (LS: {calib.get('message', 'échec')})"

        try:
            price_grid = price_grid_from_params(S0_val, m_grid, t_grid, r, q, params_tuple)
            iv_model = implied_vol_grid(price_grid, S0_val, m_grid, t_grid, r, q)
        except Exception as exc:
            return {"success": False, "message": f"Erreur calcul surface modèle: {exc}", "details": {"calib": calib}}

        if iv_model.shape != iv_market.shape:
            return {
                "success": False,
                "message": "Shape IV modèle invalide (ne correspond pas au marché).",
                "details": {"iv_model_shape": iv_model.shape, "iv_market_shape": iv_market.shape},
            }

        iv_error = np.where(mask_bool, iv_model - iv_market, np.nan)
        metrics = self._iv_error_metrics(iv_error, mask_bool)

        result = {
            "success": True,
            "message": msg,
            "method": "nn_warm_start_least_squares" if refine else "nn_warm_start",
            "model": "heston_v1",
            "params": params_final,
            "metrics": metrics,
            "ticker": ticker,
            "S0": S0_val,
            "r": r,
            "q": q,
            "m_grid": m_grid.tolist(),
            "t_grid": t_grid.tolist(),
            "iv_market": iv_market.tolist(),
            "iv_model": iv_model.tolist(),
            "iv_error": iv_error.tolist(),
            "mask": mask_bool.tolist(),
            "details": {"pred": pred, "calibration": calib},
        }
        if len(params_final) != len(HESTON_PARAM_ORDER):
            raise ValueError("Paramètres Heston: longueur inattendue.")

        return result

    def run_heston_ls_from_surface(self, payload: Dict | None) -> Dict[str, Any]:
        """
        Least-squares Heston calibration from an IV surface (K,T,S0,iv,type).
        Does not require NN weights.
        """
        data = payload or {}
        csv_bytes = data.get("csv_bytes") or data.get("file")
        surface_path = data.get("surface_path")
        df_in = data.get("df")
        constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else None
        ticker = str(data.get("ticker") or "").strip().upper() or None

        if not isinstance(df_in, pd.DataFrame) and csv_bytes is None and surface_path is None:
            return {"success": False, "message": "CSV surface requis.", "details": {}}

        try:
            raw_df = None
            if isinstance(df_in, pd.DataFrame):
                raw_df = df_in.copy()
            elif isinstance(csv_bytes, (bytes, bytearray)):
                raw_df = pd.read_csv(io.BytesIO(csv_bytes))
            elif surface_path is not None:
                raw_df = pd.read_csv(surface_path)
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
                s0_vals = pd.to_numeric(raw_df["S0"], errors="coerce")
                s0_pos = s0_vals[s0_vals > 0]
                if not s0_pos.empty:
                    S0_val = float(s0_pos.median())
            elif "S0" in market_df.columns:
                s0_vals = pd.to_numeric(market_df["S0"], errors="coerce")
                s0_pos = s0_vals[s0_vals > 0]
                if not s0_pos.empty:
                    S0_val = float(s0_pos.median())
        except Exception:
            S0_val = None

        if S0_val is None or S0_val <= 0:
            return {"success": False, "message": "S0 invalide ou manquant.", "details": {}}

        r = float(data.get("r") or 0.0)
        q = float(data.get("q") or 0.0)
        fit_to_observed_only = bool(data.get("fit_to_observed_only", True))
        u_max = float(data.get("u_max") or 50.0)
        n_integration = int(data.get("n_integration") or 2000)
        max_nfev = int(data.get("max_nfev") or 50)
        n_starts = int(data.get("n_starts") or 1)
        seed = data.get("seed")
        try:
            seed = int(seed) if seed is not None else None
        except Exception:
            seed = None

        iv_market, mask, m_grid, t_grid = build_fixed_grid(market_df)

        calib = calibrate_heston_least_squares(
            S0=S0_val,
            r=r,
            q=q,
            m_grid=m_grid,
            t_grid=t_grid,
            iv_market=iv_market,
            mask=mask,
            constraints=constraints,
            fit_to_observed_only=fit_to_observed_only,
            u_max=u_max,
            n_integration=n_integration,
            max_nfev=max_nfev,
            n_starts=n_starts,
            seed=seed,
        )
        if not calib.get("success"):
            return {"success": False, "message": calib.get("message", "Calibration échouée."), "details": calib}

        params = calib.get("params") or {}
        params_tuple = (
            params.get("kappa"),
            params.get("theta"),
            params.get("sigma"),
            params.get("rho"),
            params.get("v0"),
        )
        if any(p is None for p in params_tuple):
            return {"success": False, "message": "Paramètres incomplets après calibration.", "details": calib}

        try:
            price_grid = price_grid_from_params(S0_val, m_grid, t_grid, r, q, params_tuple)
            iv_model = implied_vol_grid(price_grid, S0_val, m_grid, t_grid, r, q)
            iv_error = np.where(mask, iv_model - iv_market, np.nan)
            metrics = self._iv_error_metrics(iv_error, mask)
        except Exception as exc:
            return {"success": False, "message": f"Erreur calcul surface modèle: {exc}", "details": calib}

        msg = calib.get("message") or "OK"
        nfev = calib.get("nfev")
        if nfev:
            msg = f"{msg} (nfev={nfev})"
        if calib.get("n_starts") and int(calib.get("n_starts") or 0) > 1:
            msg = f"{msg} | starts={int(calib.get('n_starts') or 0)}"

        return {
            "success": True,
            "message": msg,
            "method": "least_squares",
            "converged": bool(calib.get("converged", False)),
            "params": params,
            "metrics": metrics,
            "ticker": ticker,
            "S0": S0_val,
            "r": r,
            "q": q,
            "m_grid": m_grid.tolist(),
            "t_grid": t_grid.tolist(),
            "iv_market": iv_market.tolist(),
            "iv_model": iv_model.tolist(),
            "iv_error": iv_error.tolist(),
            "mask": mask.tolist(),
            "details": calib,
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

    def get_advanced_models(self) -> list[Dict[str, Any]]:
        """
        Advanced calibration models exposed to the UI (new API).
        Kept separate from V1 to avoid changing existing behavior.
        """
        return [
            {"key": "heston_v1", "label": "Heston (least squares)", "pricing": "cf", "calibration": "least_squares", "expensive": False},
            {"key": "sabr", "label": "SABR (Hagan analytic)", "pricing": "analytic_iv", "calibration": "least_squares", "expensive": False},
            {"key": "merton_jump_diffusion", "label": "Jump Diffusion (Merton) via FFT", "pricing": "fft", "calibration": "least_squares", "expensive": False},
            {"key": "rheston", "label": "rHeston (Markovian approx) via FFT", "pricing": "fft", "calibration": "least_squares", "expensive": True},
            {"key": "rbergomi", "label": "rBergomi (MC + surrogate)", "pricing": "mc", "calibration": "mc_surrogate", "expensive": True},
            {"key": "volterra", "label": "Volterra SDE (MC proxy)", "pricing": "mc", "calibration": "mc_proxy", "expensive": True},
        ]

    def _json_safe(self, obj: Any) -> Any:
        import numpy as _np

        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, _np.generic):
            try:
                return obj.item()
            except Exception:
                return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]
        return str(obj)

    def run_advanced_surface_calibration(self, payload: Dict | None) -> Dict[str, Any]:
        """
        New unified calibration runner used by the Advanced Calibration tab.
        All numerics live in the model layer; the controller only orchestrates.
        """
        from app.model.calibration.base_calibrator import CalibratorSettings, SurfaceGrid
        from app.model.volatility_models.sabr.calibrator import SABRAnalyticCalibrator
        from app.model.volatility_models.jump_diffusion.calibrator import MertonJumpDiffusionCalibrator
        from app.model.volatility_models.heston.calibrator_legacy import HestonLegacyLeastSquaresCalibrator
        from app.model.volatility_models.rheston.calibrator_fft import RHestonFFTMarkovianCalibrator
        from app.model.volatility_models.rbergomi.calibrator_mc_surrogate import RBergomiMCSurrogateCalibrator
        from app.model.volatility_models.volterra.calibrator_mc import VolterraSDECalibrator

        data = payload or {}
        model_key = str(data.get("model") or "").strip() or "heston_v1"
        constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else None

        r_val = float(data.get("r") or 0.0)
        q_val = float(data.get("q") or 0.0)
        fit_to_observed_only = bool(data.get("fit_to_observed_only", True))
        max_nfev = int(data.get("max_nfev") or 80)
        n_starts = int(data.get("n_starts") or 1)
        seed_raw = data.get("seed")
        try:
            seed = int(seed_raw) if seed_raw is not None else None
        except Exception:
            seed = None

        settings = CalibratorSettings(
            fit_to_observed_only=fit_to_observed_only,
            max_nfev=max_nfev,
            n_starts=n_starts,
            seed=seed,
        )

        csv_bytes = data.get("csv_bytes")
        df = data.get("df")
        market_df = load_market_surface_csv_v2(df if df is not None else csv_bytes)
        if market_df is None or getattr(market_df, "empty", True):
            return {"success": False, "message": "Surface vide / illisible.", "details": {}}

        # Grids
        m_grid = self._to_ndarray(data.get("m_grid"))
        t_grid = self._to_ndarray(data.get("t_grid"))
        if m_grid.size == 0 or t_grid.size == 0:
            m_grid, t_grid = default_grid()

        iv_market, mask = make_fixed_grid(market_df, m_grid, t_grid)

        # Spot estimate (prefer explicit S0, else median from csv)
        S0_val = data.get("S0")
        if S0_val is None:
            try:
                if "S0" in market_df.columns:
                    s0s = pd.to_numeric(market_df["S0"], errors="coerce")
                    s0s = s0s[s0s > 0]
                    S0_val = float(s0s.median()) if not s0s.empty else None
            except Exception:
                S0_val = None
        S0_val = float(S0_val or 100.0)

        surface = SurfaceGrid(
            S0=S0_val,
            r=r_val,
            q=q_val,
            m_grid=np.asarray(m_grid, dtype=float),
            t_grid=np.asarray(t_grid, dtype=float),
            iv_market=np.asarray(iv_market, dtype=float),
            mask=np.asarray(mask, dtype=bool),
        )

        if model_key == "heston_v1":
            try:
                return self._json_safe(
                    self.run_heston_global_calibration(
                        {
                            "df": market_df,
                            "S0": float(surface.S0),
                            "r": float(surface.r),
                            "q": float(surface.q),
                            "m_grid": surface.m_grid,
                            "t_grid": surface.t_grid,
                            "constraints": constraints,
                            "fit_to_observed_only": fit_to_observed_only,
                            "max_nfev": max_nfev,
                            "n_starts": n_starts,
                            "seed": seed,
                            "ticker": data.get("ticker"),
                            "refine": True,
                        }
                    )
                )
            except Exception as exc:
                return {"success": False, "message": f"Erreur calibration Heston: {exc}", "details": {}}

        calibrator_map = {
            "heston_v1": HestonLegacyLeastSquaresCalibrator(),
            "sabr": SABRAnalyticCalibrator(),
            "merton_jump_diffusion": MertonJumpDiffusionCalibrator(),
            "rheston": RHestonFFTMarkovianCalibrator(),
            "rbergomi": RBergomiMCSurrogateCalibrator(),
            "volterra": VolterraSDECalibrator(),
        }
        calibrator = calibrator_map.get(model_key)
        if calibrator is None:
            return {"success": False, "message": f"Modèle inconnu: {model_key}", "details": {}}

        try:
            result = calibrator.calibrate(surface, constraints=constraints, settings=settings)
        except Exception as exc:
            return {"success": False, "message": f"Erreur calibration: {exc}", "details": {}}

        return self._json_safe(
            {
                "success": bool(result.success),
                "message": str(result.message),
                "model": str(result.model),
                "method": str(result.method),
                "params": result.params,
                "metrics": result.metrics or {},
                "S0": float(surface.S0),
                "r": float(surface.r),
                "q": float(surface.q),
                "m_grid": surface.m_grid,
                "t_grid": surface.t_grid,
                "iv_market": surface.iv_market,
                "iv_model": result.iv_model,
                "iv_error": result.iv_error,
                "mask": surface.mask,
                "details": result.details or {},
            }
        )

    def kalman_smooth(self, payload: Dict | None) -> Dict[str, Any]:
        """
        State-space estimation helper (no pricing): smooth parameter sequences.
        """
        from app.model.volatility_models.kalman.kalman_filter import smooth_parameters_random_walk

        data = payload or {}
        y = data.get("y")
        if y is None:
            return {"success": False, "message": "Missing 'y' sequence.", "details": {}}
        try:
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            return {"success": False, "message": "Invalid 'y' (must be numeric).", "details": {}}

        q_noise = float(data.get("q") or 1e-4)
        r_noise = float(data.get("r") or 1e-2)
        out = smooth_parameters_random_walk(y=y_arr, q=q_noise, r=r_noise)
        return self._json_safe({"success": True, "message": "OK", "details": out})


__all__ = ["CalibrationController"]
