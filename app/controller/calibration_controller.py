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
from app.model.options.data.iv_surface import (
    fetch_iv_surface as _fetch_iv_surface,
    list_cached_iv_surface_tickers as _list_cached_iv_surface_tickers,
)
from app.model.options.logic import download_options_alpaca as _download_options_alpaca
from app.model.calibration.implied_vol import bs_call_price as _bs_call_price
from app.model.calibration.loss_surface import compute_bs_vega_grid, iv_error_metrics_weighted

#: Dispatch key of the spec-4.10 joint ``(H, eta, rho)`` rough Bergomi calibrator.
#: Deliberately distinct from ``"rbergomi"``, which is the MC-surrogate calibrator.
ROUGH_VOL_MODEL_KEY = "rbergomi_joint_h"

#: Ordered stages of the rough-volatility pipeline, with the label the UI shows.
#: The last one is the only expensive one; everything before it is closed-form or
#: quadrature and runs in well under a second on a Yahoo chain.
ROUGH_VOL_STEPS: tuple[tuple[str, str], ...] = (
    ("chain", "Nettoyage des chaînes d'options (4.1)"),
    ("forward", "Courbe forward par parité call-put (4.2)"),
    ("surface", "Surface OTM en log-moneyness (4.2)"),
    ("variance_swap", "Strikes de swap de variance K_var (4.3)"),
    ("forward_variance", "Courbe de variance forward ξ₀ (4.4)"),
    ("hurst", "Estimation initiale de H par le skew ATM (4.5)"),
    ("initializer", "Point de départ (H₀, η₀, ρ₀) (4.9)"),
    ("calibration", "Calibration jointe (H, η, ρ) à ξ₀ figé (4.10/4.11)"),
)

#: The two stages ``run_rbergomi_hurst_pipeline`` accepts.
ROUGH_VOL_STAGE_PREPARE = "prepare"
ROUGH_VOL_STAGE_FULL = "full"


class CalibrationController:
    """Placeholder for future model calibration (architecture only)."""

    def get_models(self) -> list[str]:
        return [m.value for m in get_supported_models()]

    @staticmethod
    def bs_call_price(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Expose BS call price to the UI layer without importing model modules directly."""
        return float(_bs_call_price(S0, K, T, r, q, sigma))

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

    def default_calibration_grid(self) -> tuple[list[float], list[float]]:
        """(moneyness nodes, maturity nodes) every surface is put on before fitting."""
        m_grid, t_grid = default_grid()
        return [float(m) for m in m_grid], [float(t) for t in t_grid]

    def list_cached_yahoo_surface_tickers(self) -> list[str]:
        """Tickers whose Yahoo IV surface is already on disk (loadable offline)."""
        try:
            return list(_list_cached_iv_surface_tickers())
        except Exception:
            return []

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
        vega_weights = compute_bs_vega_grid(S0_val, m_grid, t_grid, r, q, iv_market)
        metrics_vw = iv_error_metrics_weighted(iv_error, mask_bool, vega_weights)

        result = {
            "success": True,
            "message": msg,
            "method": "nn_warm_start_least_squares" if refine else "nn_warm_start",
            "model": "heston_v1",
            "params": params_final,
            "metrics": metrics,
            "metrics_vw": metrics_vw,
            "ticker": ticker,
            "S0": S0_val,
            "r": r,
            "q": q,
            "m_grid": m_grid.tolist(),
            "t_grid": t_grid.tolist(),
            "iv_market": iv_market.tolist(),
            "iv_model": iv_model.tolist(),
            "iv_error": iv_error.tolist(),
            "vega_weights": vega_weights.tolist(),
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
            # Spec 4.10/4.11 joint (H, eta, rho) fit with xi0 FROZEN. Distinct key from
            # "rbergomi" on purpose: that one is the surrogate calibrator and is pinned by
            # tests/quant/test_advanced_calibration_roundtrip.py. This one cannot run from an
            # IV grid alone — it needs constraints["xi0_curve"] (spec 4.4), which only the
            # rough-volatility pipeline can build — so it lives in its own tab, not in the
            # per-model tabs of "Calibration avancée".
            {
                "key": ROUGH_VOL_MODEL_KEY,
                "label": "rBergomi (H joint, MC — ξ₀ figé)",
                "pricing": "mc",
                "calibration": "joint_h_mc",
                "expensive": True,
                "requires_constraints": ["xi0_curve"],
                "ui": "tab_rough_vol",
                "entry_point": "run_rbergomi_hurst_pipeline",
            },
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
        from app.model.calibration.base_calibrator import CalibratorSettings, SurfaceGrid, apply_degeneracy_guard
        from app.model.volatility_models.sabr.calibrator import SABRAnalyticCalibrator
        from app.model.volatility_models.jump_diffusion.calibrator import MertonJumpDiffusionCalibrator
        from app.model.volatility_models.heston.calibrator_legacy import HestonLegacyLeastSquaresCalibrator
        from app.model.volatility_models.rheston.calibrator_fft import RHestonFFTMarkovianCalibrator
        from app.model.volatility_models.rbergomi.calibrator_mc_surrogate import RBergomiMCSurrogateCalibrator
        from app.model.volatility_models.rbergomi.calibrator_joint_mc import RBergomiJointHCalibrator
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

        nn_unavailable_msg: str | None = None
        if model_key == "heston_v1":
            try:
                heston_res = self.run_heston_global_calibration(
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
                        "u_max": data.get("u_max"),
                        "n_integration": data.get("n_integration"),
                        "ticker": data.get("ticker"),
                        "refine": True,
                    }
                )
            except Exception as exc:
                return {"success": False, "message": f"Erreur calibration Heston: {exc}", "details": {}}
            details = heston_res.get("details") if isinstance(heston_res, dict) else None
            nn_failed = (
                isinstance(heston_res, dict)
                and not heston_res.get("success")
                and isinstance(details, dict)
                and "pred" in details
            )
            if not nn_failed:
                return self._json_safe(heston_res)
            # The NN warm start could not run (torch or weights missing): fall back to the
            # torch-free least-squares Heston calibrator below instead of giving up.
            nn_unavailable_msg = str(heston_res.get("message") or "NN indisponible")

        calibrator_map = {
            "heston_v1": HestonLegacyLeastSquaresCalibrator(),
            "sabr": SABRAnalyticCalibrator(),
            "merton_jump_diffusion": MertonJumpDiffusionCalibrator(),
            "rheston": RHestonFFTMarkovianCalibrator(),
            "rbergomi": RBergomiMCSurrogateCalibrator(),
            ROUGH_VOL_MODEL_KEY: RBergomiJointHCalibrator(),
            "volterra": VolterraSDECalibrator(),
        }
        calibrator = calibrator_map.get(model_key)
        if calibrator is None:
            return {"success": False, "message": f"Modèle inconnu: {model_key}", "details": {}}

        try:
            result = calibrator.calibrate(surface, constraints=constraints, settings=settings)
        except Exception as exc:
            return {"success": False, "message": f"Erreur calibration: {exc}", "details": {}}

        apply_degeneracy_guard(result)

        message = str(result.message)
        if nn_unavailable_msg and result.success:
            message = f"{message} — repli moindres carrés ({nn_unavailable_msg})"

        return self._json_safe(
            {
                "success": bool(result.success),
                "message": message,
                "model": str(result.model),
                "method": str(result.method),
                "params": result.params,
                "metrics": result.metrics or {},
                "metrics_vw": result.metrics_vw or {},
                "S0": float(surface.S0),
                "r": float(surface.r),
                "q": float(surface.q),
                "m_grid": surface.m_grid,
                "t_grid": surface.t_grid,
                "iv_market": surface.iv_market,
                "iv_model": result.iv_model,
                "iv_error": result.iv_error,
                "vega_weights": result.vega_weights,
                "mask": surface.mask,
                "details": result.details or {},
            }
        )

    # ------------------------------------------------------------------
    # Rough-volatility pipeline (spec 4.1 -> 4.11) — Phase 5
    # ------------------------------------------------------------------

    def get_rough_vol_steps(self) -> list[Dict[str, str]]:
        """Ordered pipeline stages, so the view can name them without importing the model."""
        return [{"step": str(s), "label_fr": str(label)} for s, label in ROUGH_VOL_STEPS]

    def get_rough_vol_flag_labels(self) -> Dict[str, Dict[str, Any]]:
        """
        Every flag the joint calibrator can raise, with its French label and
        whether it is *blocking* (i.e. on its own enough to make ``success``
        False).

        The view must be able to tell "cette surface n'identifie pas H" from a
        mere advisory without importing ``app.model`` — the MVC gate forbids it.
        """
        from app.model.volatility_models.rbergomi.calibrator_joint_mc import (
            BLOCKING_FLAGS,
            JOINT_CALIBRATION_LABELS_FR,
        )

        blocking = {str(f) for f in BLOCKING_FLAGS}
        return {
            str(flag): {"label_fr": str(label), "blocking": bool(str(flag) in blocking)}
            for flag, label in JOINT_CALIBRATION_LABELS_FR.items()
        }

    def rbergomi_joint_cost_estimate(self, payload: Dict | None) -> Dict[str, Any]:
        """
        Monte-Carlo budget of ONE joint calibration, **before** it is launched.

        Returns evaluation counts and cumulated path counts per stage. It never
        returns a wall time: no per-evaluation constant is measured here, and
        inventing one would be a fabricated number. The view turns this into
        seconds only once it holds a *measured* ``mean_evaluation_seconds`` from
        a previous run on the same machine.

        Two things this makes explicit, both raised by the Phase-4 panel:

        * ``tab_advanced_calibration`` estimates a run as
          ``per_eval * max_nfev * n_starts``. That is the Stage-2 term ONLY: it
          ignores the Stage-1 design, the spec-4.11 profiles, the eta/rho valley,
          the noise floor, the grid-refinement bias and the final high-accuracy
          repricing. ``ratio_vs_local_stage_only`` states by how much.
        * ``CalibratorSettings.max_nfev`` defaults to ``80`` and the joint
          calibrator applies its own budget (``local_nfev_per_param * n_free``)
          whenever it sees that exact value — so a caller who *deliberately*
          passes 80 is indistinguishable from one who passed nothing.
          ``max_nfev_is_ambiguous`` says when that is the case.
        """
        from app.model.calibration.base_calibrator import CalibratorSettings
        from app.model.volatility_models.rbergomi.calibrator_joint_mc import (
            PARAM_ORDER,
            JointMCConfig,
            _config_from_mapping,
            resolve_bounds,
        )

        data = payload or {}
        constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else {}
        mc_override = data.get("mc_cfg")
        if not isinstance(mc_override, dict):
            mc_override = constraints.get("mc_cfg")
        try:
            cfg = _config_from_mapping(JointMCConfig(), mc_override)
        except Exception as exc:
            return {"success": False, "message": f"Configuration Monte-Carlo invalide: {exc}"}

        n_params = len(PARAM_ORDER)
        try:
            _bounds, pinned = resolve_bounds(constraints)
            n_free = max(0, n_params - len(pinned))
        except Exception:
            pinned = ()
            n_free = n_params

        default_max_nfev = int(CalibratorSettings().max_nfev)
        requested_raw = data.get("max_nfev")
        try:
            requested = default_max_nfev if requested_raw is None else int(requested_raw)
        except Exception:
            requested = default_max_nfev
        ambiguous = int(requested) == default_max_nfev
        max_nfev_eff = int(cfg.local_max_nfev(n_free)) if ambiguous else int(requested)

        n_starts = max(1, int(data.get("n_starts") or 1))
        stage1_runs = bool(int(cfg.n_design) > 0 and n_free > 0)
        n_starts_eff = min(n_starts, int(cfg.n_design) + 1) if stage1_runs else 1

        stage1_paths = int(cfg.stage1_paths)
        stage2_paths = int(cfg.stage2_paths)
        profile_paths = int(cfg.effective_profile_paths)
        final_paths = int(cfg.final_paths)

        # Evaluation counts, read off the calibrator's own documented costs.
        stage1_evals = (int(cfg.n_design) + 1) if stage1_runs else 0
        stage2_evals = n_starts_eff * (max_nfev_eff + 1)  # local search + re-score on its own draw
        selection_evals = n_starts_eff  # restarts re-scored on ONE common draw
        crn_evals = 2  # loss at the optimum and at the initial point
        matched_evals = 2  # loss_crn_matched / loss_fresh_matched
        profile_evals = n_free * int(cfg.profile_points)
        valley_evals = int(cfg.valley_points)
        noise_evals = int(cfg.noise_replicates) * (1 + n_free)
        refinement_evals = (2 + 4 * n_free) if bool(cfg.refinement_check) else 0
        final_evals = 1

        stages: list[Dict[str, Any]] = [
            {"stage": "stage1_design", "label_fr": "Étape 1 — plan d'expérience (hypercube latin)",
             "n_evaluations": int(stage1_evals), "n_paths_per_evaluation": stage1_paths},
            {"stage": "stage2_local", "label_fr": "Étape 2 — recherche locale (Nelder-Mead, CRN)",
             "n_evaluations": int(stage2_evals), "n_paths_per_evaluation": stage2_paths},
            {"stage": "matched_losses", "label_fr": "Coûts appariés (dans / hors échantillon)",
             "n_evaluations": int(matched_evals), "n_paths_per_evaluation": stage2_paths},
            {"stage": "restart_selection", "label_fr": "Sélection du meilleur redémarrage",
             "n_evaluations": int(selection_evals), "n_paths_per_evaluation": profile_paths},
            {"stage": "crn_losses", "label_fr": "Coûts à tirage commun (optimum, point initial)",
             "n_evaluations": int(crn_evals), "n_paths_per_evaluation": profile_paths},
            {"stage": "profiles", "label_fr": "Profils d'identifiabilité (4.11)",
             "n_evaluations": int(profile_evals), "n_paths_per_evaluation": profile_paths},
            {"stage": "valley", "label_fr": "Vallée (η, ρ) à produit constant",
             "n_evaluations": int(valley_evals), "n_paths_per_evaluation": profile_paths},
            {"stage": "noise_floor", "label_fr": "Plancher de bruit Monte-Carlo",
             "n_evaluations": int(noise_evals), "n_paths_per_evaluation": profile_paths},
            {"stage": "grid_bias", "label_fr": "Biais de discrétisation (grille raffinée)",
             "n_evaluations": int(refinement_evals), "n_paths_per_evaluation": profile_paths},
            {"stage": "final_repricing", "label_fr": "Repricing final (graine fraîche)",
             "n_evaluations": int(final_evals), "n_paths_per_evaluation": final_paths},
        ]
        for entry in stages:
            entry["n_paths_total"] = int(entry["n_evaluations"]) * int(entry["n_paths_per_evaluation"])

        n_evaluations = int(sum(int(e["n_evaluations"]) for e in stages))
        n_paths_total = int(sum(int(e["n_paths_total"]) for e in stages))
        local_only = int(max_nfev_eff) * int(n_starts_eff)
        ratio = (float(n_evaluations) / float(local_only)) if local_only > 0 else float("nan")
        # What `tab_advanced_calibration.py:499` literally multiplies:
        # `per_eval * max_nfev * n_starts`, with the values the CALLER passed —
        # not the budget the calibrator ends up applying. When `max_nfev` is the
        # ambiguous 80 the two differ, and the heuristic is wrong twice over.
        heuristic_evals = int(requested) * int(max(1, n_starts))
        heuristic_ratio = (
            (float(n_evaluations) / float(heuristic_evals))
            if heuristic_evals > 0
            else float("nan")
        )

        if ambiguous:
            nfev_note = (
                f"max_nfev = {requested} est exactement la valeur par défaut de "
                f"CalibratorSettings : « {requested} demandé explicitement » est "
                "indistinguable de « rien demandé », et le calibrateur applique son "
                "propre budget de "
                f"{max_nfev_eff} évaluations (local_nfev_per_param × nombre de paramètres "
                "libres). Choisir une autre valeur pour imposer un budget."
            )
        else:
            nfev_note = (
                f"Budget imposé par l'appelant : {max_nfev_eff} évaluations par recherche locale."
            )

        return self._json_safe(
            {
                "success": True,
                "model": ROUGH_VOL_MODEL_KEY,
                "method": "joint_h_mc",
                "expensive": True,
                "n_free_parameters": int(n_free),
                "pinned_parameters": [str(p) for p in pinned],
                "n_starts_requested": int(n_starts),
                "n_starts_effective": int(n_starts_eff),
                "max_nfev_requested": int(requested),
                "max_nfev_effective": int(max_nfev_eff),
                "max_nfev_source": "config" if ambiguous else "settings",
                "max_nfev_is_ambiguous": bool(ambiguous),
                "max_nfev_ambiguity_fr": nfev_note,
                "grid_n_max": int(cfg.grid_n_max),
                "stages": stages,
                "n_evaluations": n_evaluations,
                "n_paths_total": n_paths_total,
                "local_stage_only_evaluations": int(local_only),
                "ratio_vs_local_stage_only": float(ratio),
                "heuristic_evaluations": int(heuristic_evals),
                "ratio_vs_heuristic": float(heuristic_ratio),
                "message_fr": (
                    f"≈ {n_evaluations} évaluations Monte-Carlo de la fonction de coût, soit "
                    f"≈ {n_paths_total} trajectoires simulées au total. La recherche locale "
                    f"seule — la seule chose que compte l'heuristique « max_nfev × n_starts » "
                    f"de l'onglet Calibration avancée — n'en représente que {local_only}, "
                    f"soit un facteur ≈ {ratio:.1f}. Avec les valeurs demandées ici cette "
                    f"heuristique n'en compterait que {heuristic_evals}, "
                    f"facteur ≈ {heuristic_ratio:.1f}."
                ),
                "wall_time_fr": (
                    "Aucune durée n'est estimée ici : elle dépend de la machine. La constante "
                    "`per_eval` de l'onglet Calibration avancée (0,05 s) est calibrée sur des "
                    "modèles FFT, pas sur une évaluation Monte-Carlo de plusieurs milliers de "
                    "trajectoires : appliquée ici elle se tromperait d'un ordre de grandeur, en "
                    "plus du facteur de comptage ci-dessus. Après une première calibration, le "
                    "rapport fournit la durée moyenne par évaluation réellement mesurée "
                    "(mean_evaluation_seconds)."
                ),
            }
        )

    def _rough_vol_failure(
        self,
        *,
        step: str,
        message: str,
        stage: str,
        steps: list[Dict[str, Any]],
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """A pipeline stop, named by the step that refused. Never carries a parameter value."""
        out: Dict[str, Any] = {
            "success": False,
            "stage": str(stage),
            "failed_step": str(step),
            "message": str(message),
            "steps": steps,
            "params": {},
            "params_usable": False,
            "flags": [],
            "warnings_fr": [],
            "flag_details": [],
            "blocking_flags": [],
        }
        if extra:
            out.update(extra)
        return self._json_safe(out)

    @staticmethod
    def _median_row_spot(*frames: Any) -> float:
        """Median ``S0`` carried by the raw chain rows; NaN when they carry none."""
        values: list[float] = []
        for frame in frames:
            if frame is None:
                continue
            try:
                if isinstance(frame, pd.DataFrame):
                    if "S0" not in frame.columns:
                        continue
                    series = pd.to_numeric(frame["S0"], errors="coerce")
                else:
                    series = pd.to_numeric(
                        pd.Series([dict(row).get("S0") for row in frame], dtype="object"),
                        errors="coerce",
                    )
            except Exception:
                continue
            series = series[series > 0]
            if not series.empty:
                values.append(float(series.median()))
        if not values:
            return float("nan")
        return float(pd.Series(values).median())

    def run_rbergomi_hurst_pipeline(self, payload: Dict | None) -> Dict[str, Any]:
        """
        Run the rough-volatility pipeline of spec 4.1 -> 4.11 end to end.

        The controller **orchestrates only**: every number below is produced by
        ``app.model.calibration.rough_vol.*`` and
        ``app.model.volatility_models.rbergomi.*``.

        Payload
        -------
        ``ticker``
            Underlying whose Yahoo option chain is fetched (disk cache first).
            Optional when ``calls`` / ``puts`` are supplied directly.
        ``calls`` / ``puts``
            Raw chain rows (``DataFrame`` or list of mappings) in the
            ``fetch_options_details_yahoo`` schema. Supplying them skips the
            network entirely.
        ``stage``
            ``"prepare"`` (**the default**) runs 4.1 -> 4.9 only — closed form
            and quadrature, no Monte-Carlo — and returns the cost of the fit that
            *would* follow. ``"full"`` also runs the expensive 4.10/4.11 joint
            calibration. The cheap stage is the default on purpose: this
            calibrator must never start from a casual call.
        ``S0``, ``r``, ``q``, ``currency``, ``max_maturity_years``, ``max_expiries``, ``use_cache``
            Spot / rate overrides and fetch parameters. An explicit ``r`` pins the
            discounting of spec 4.2 at every quoted maturity (reproducible
            off-line) instead of resolving the repo yield curve; without it the
            curve is used and the reporting grid takes ``r`` / ``q`` from the
            forward point closest to the middle of the term structure. ``q``
            weights the reported grid metrics only — the fit itself runs on the
            market forwards.
        ``short_maturity_window``
            ``[T_min, T_max]`` in years for the spec-4.5 skew regression.
        ``constraints``
            The repo constraints protocol restricted to ``H`` / ``eta`` / ``rho``.
            ``xi0`` is data here, never a parameter: it is dropped before
            dispatch and the model layer refuses it loudly anyway.
        ``mc_cfg`` / ``weights_cfg``
            Mappings overriding ``JointMCConfig`` / ``WeightConfig``.
        ``max_nfev``, ``n_starts``, ``seed``, ``fit_to_observed_only``
            The shared ``CalibratorSettings`` fields. See
            :meth:`rbergomi_joint_cost_estimate` for the ``max_nfev == 80``
            ambiguity.

        Returns
        -------
        A ``_json_safe`` dict. ``success`` is a **verdict**: ``False`` means the
        surface does not identify ``H`` (or an earlier stage refused), and the
        French reason is in ``message`` with the per-flag detail in
        ``warnings_fr``. ``params_usable`` mirrors it — when it is ``False`` the
        numbers in ``params`` measure nothing and must not be presented as a
        calibration result.
        """
        import math as _math

        from app.model.calibration.rough_vol.chain_cleaning import (
            CleaningConfig,
            clean_option_chains,
            cleaning_report,
        )
        from app.model.calibration.rough_vol.forward_curve import (
            build_forward_curve,
            build_otm_surface,
            forward_curve_report,
        )
        from app.model.calibration.rough_vol.forward_variance import (
            build_forward_variance_curve,
            forward_variance_report,
        )
        from app.model.calibration.rough_vol.hurst_estimator import (
            estimate_hurst_from_skew,
            hurst_report,
        )
        from app.model.calibration.rough_vol.variance_swap import (
            build_variance_swap_curve,
            variance_swap_report,
        )
        from app.model.volatility_models.rbergomi.initializer import (
            initial_rbergomi_params,
            initializer_report,
        )

        data = payload or {}
        stage = str(data.get("stage") or ROUGH_VOL_STAGE_PREPARE).strip().lower()
        if stage not in (ROUGH_VOL_STAGE_PREPARE, ROUGH_VOL_STAGE_FULL):
            stage = ROUGH_VOL_STAGE_PREPARE

        labels = dict(ROUGH_VOL_STEPS)
        steps: list[Dict[str, Any]] = []
        ticker_out = str(data.get("ticker") or "").strip().upper() or None

        def _ok(step: str, message: str, **detail: Any) -> None:
            steps.append(
                {
                    "step": str(step),
                    "label_fr": labels.get(step, step),
                    "ok": True,
                    "message_fr": str(message),
                    **detail,
                }
            )

        def _ko(step: str, message: str) -> Dict[str, Any]:
            steps.append(
                {
                    "step": str(step),
                    "label_fr": labels.get(step, step),
                    "ok": False,
                    "message_fr": str(message),
                }
            )
            return self._rough_vol_failure(
                step=step,
                message=message,
                stage=stage,
                steps=steps,
                extra={"ticker": ticker_out},
            )

        currency = data.get("currency") or None

        # -- chain rows: supplied, or fetched from Yahoo (disk cache first) --
        calls = data.get("calls")
        puts = data.get("puts")
        spot_fetched: Any = None
        if calls is None and puts is None:
            if not ticker_out:
                return _ko("chain", "Ticker manquant : aucune chaîne d'options à traiter.")
            from app.model.market_data.market_data import fetch_options_details_yahoo

            try:
                calls, puts, spot_fetched, _rf, _div = fetch_options_details_yahoo(
                    ticker_out,
                    max_maturity_years=float(data.get("max_maturity_years") or 2.0),
                    max_expiries=int(data.get("max_expiries") or 12),
                    use_cache=bool(data.get("use_cache", True)),
                )
            except Exception as exc:
                return _ko("chain", f"Téléchargement de la chaîne {ticker_out} impossible: {exc}")

        S0_raw = data.get("S0")
        if S0_raw is None:
            S0_raw = spot_fetched
        if S0_raw is None:
            S0_raw = self._median_row_spot(calls, puts)
        try:
            S0 = float(S0_raw)
        except Exception:
            S0 = float("nan")
        if not (_math.isfinite(S0) and S0 > 0.0):
            return _ko(
                "chain",
                "Spot indisponible ou non exploitable : la surface ne peut pas être normalisée.",
            )

        # -- 4.1 cleaning ----------------------------------------------------
        cleaning_cfg = data.get("cleaning_cfg")
        try:
            config = CleaningConfig(**cleaning_cfg) if isinstance(cleaning_cfg, dict) else None
            chains = clean_option_chains(calls, puts, config=config, spot=S0)
        except Exception as exc:
            return _ko("chain", f"Nettoyage des chaînes impossible: {exc}")
        chains = [c for c in chains if _math.isfinite(float(c.T)) and float(c.T) > 0.0]
        if not chains:
            return _ko("chain", "Aucune échéance exploitable après nettoyage.")
        cleaning = cleaning_report(chains)
        _ok("chain", f"{len(chains)} échéance(s) nettoyée(s).", n_expiries=len(chains))

        # -- 4.2 forward curve -----------------------------------------------
        # An explicit `r` pins the discounting at every quoted maturity instead of
        # resolving the repo yield curve, which makes a run reproducible off-line.
        rates_pin: Dict[float, float] | None = None
        if data.get("r") is not None:
            try:
                rates_pin = {float(c.T): float(data["r"]) for c in chains}
            except Exception:
                rates_pin = None
        try:
            forward_points = build_forward_curve(
                chains, rates=rates_pin, currency=currency, S0=S0
            )
        except Exception as exc:
            return _ko("forward", f"Courbe forward impossible: {exc}")
        if not forward_points:
            return _ko(
                "forward",
                "Aucun point de courbe forward n'a pu être construit (parité call-put).",
            )
        forwards = forward_curve_report(forward_points)
        _ok("forward", f"{len(forward_points)} forward(s) par parité.", n_points=len(forward_points))

        # -- 4.2 OTM surface --------------------------------------------------
        by_T = {float(p.T): p for p in forward_points}
        pairs: list[tuple[Any, Any]] = []
        surfaces: list[list[Any]] = []
        n_rejected = 0
        for chain in chains:
            point = by_T.get(float(chain.T))
            if point is None:
                continue
            try:
                points, rejections = build_otm_surface(chain, point)
            except Exception as exc:
                return _ko("surface", f"Surface OTM impossible à T={float(chain.T):.6g}: {exc}")
            n_rejected += len(rejections)
            if points:
                pairs.append((chain, point))
                surfaces.append(list(points))
        flat_points = [p for group in surfaces for p in group]
        if not flat_points:
            return _ko("surface", "Aucune cotation hors de la monnaie exploitable sur la surface.")
        _ok(
            "surface",
            f"{len(flat_points)} cotation(s) OTM sur {len(surfaces)} échéance(s) "
            f"({n_rejected} rejetée(s)).",
            n_quotes=len(flat_points),
            n_maturities=len(surfaces),
            n_rejected=int(n_rejected),
        )

        # -- 4.3 variance-swap strikes ----------------------------------------
        try:
            variance_curve = build_variance_swap_curve(pairs, currency=currency)
        except Exception as exc:
            return _ko("variance_swap", f"Courbe de swaps de variance impossible: {exc}")
        variance = variance_swap_report(variance_curve)
        if not variance_curve.points:
            refusals = " ; ".join(str(f.message_fr) for f in variance_curve.failures[:3])
            return _ko(
                "variance_swap",
                "Aucun K_var exploitable." + (f" {refusals}" if refusals else ""),
            )
        _ok(
            "variance_swap",
            f"{len(variance_curve.points)} K_var retenu(s), "
            f"{len(variance_curve.failures)} échéance(s) refusée(s).",
            n_points=len(variance_curve.points),
            n_failures=len(variance_curve.failures),
        )

        # -- 4.4 forward-variance curve (xi0) ---------------------------------
        try:
            xi0_curve = build_forward_variance_curve(variance_curve)
        except Exception as exc:
            return _ko("forward_variance", f"Courbe de variance forward ξ₀ impossible: {exc}")
        forward_variance = forward_variance_report(xi0_curve)
        _ok(
            "forward_variance",
            f"ξ₀ construite sur {len(xi0_curve.T_knots)} nœud(s) ({xi0_curve.method}).",
            n_knots=len(xi0_curve.T_knots),
        )

        # -- 4.5 initial Hurst estimate ---------------------------------------
        window_raw = data.get("short_maturity_window")
        window: tuple[float, float] | None = None
        if isinstance(window_raw, (list, tuple)) and len(window_raw) == 2:
            try:
                window = (float(window_raw[0]), float(window_raw[1]))
            except Exception:
                window = None
        try:
            hurst = estimate_hurst_from_skew(
                surfaces,
                forward_points,
                window,
                clean_chains=chains,
                variance_curve=variance_curve,
            )
        except Exception as exc:
            return _ko("hurst", f"Estimation initiale de H impossible: {exc}")
        hurst_out = hurst_report(hurst)
        _ok("hurst", str(hurst.message_fr), unstable=bool(hurst.unstable))

        # -- 4.9 initial (H0, eta0, rho0) --------------------------------------
        try:
            params0, init_diag = initial_rbergomi_params(
                hurst,
                surfaces,
                xi0_curve=xi0_curve,
                forward_curve=forward_points,
                clean_chains=chains,
                variance_curve=variance_curve,
            )
        except Exception as exc:
            return _ko("initializer", f"Initialisation (H₀, η₀, ρ₀) impossible: {exc}")
        initializer = initializer_report(init_diag)
        _ok("initializer", str(initializer.get("message_fr") or "Point de départ construit."))

        # -- shared reporting grid ---------------------------------------------
        t_grid = sorted({float(chain.T) for chain, _point in pairs})
        m_grid = [float(m) for m in default_grid()[0]]
        anchor = forward_points[len(forward_points) // 2]
        r_val = float(anchor.r) if _math.isfinite(float(anchor.r)) else 0.0
        q_anchor = float(anchor.q_implied)
        q_val = q_anchor if _math.isfinite(q_anchor) else 0.0
        if data.get("r") is not None:
            r_val = float(data["r"])
        if data.get("q") is not None:
            q_val = float(data["q"])

        market_df = pd.DataFrame(
            [
                {"K": float(p.K), "T": float(p.T), "S0": S0, "iv": float(p.iv)}
                for p in flat_points
                if _math.isfinite(float(p.iv)) and float(p.iv) > 0.0
            ]
        )

        constraints_in = data.get("constraints")
        constraints: Dict[str, Any] = dict(constraints_in) if isinstance(constraints_in, dict) else {}
        # xi0 is DATA during the joint fit; the optimizer must be structurally unable
        # to move it. The model layer raises on this key — drop it here so the UI
        # cannot even try.
        constraints.pop("xi0", None)
        if isinstance(data.get("mc_cfg"), dict):
            constraints["mc_cfg"] = dict(data["mc_cfg"])
        if isinstance(data.get("weights_cfg"), dict):
            constraints["weights_cfg"] = dict(data["weights_cfg"])

        cost = self.rbergomi_joint_cost_estimate(
            {
                "constraints": constraints,
                "mc_cfg": constraints.get("mc_cfg"),
                "max_nfev": data.get("max_nfev"),
                "n_starts": data.get("n_starts"),
            }
        )

        base: Dict[str, Any] = {
            "stage": stage,
            "failed_step": None,
            "steps": steps,
            "ticker": ticker_out,
            "S0": float(S0),
            "r": float(r_val),
            "q": float(q_val),
            "m_grid": m_grid,
            "t_grid": t_grid,
            "n_quotes": int(len(flat_points)),
            "n_maturities": int(len(surfaces)),
            "cleaning": cleaning,
            "forward_curve": forwards,
            "variance_swap": variance,
            "forward_variance": forward_variance,
            "hurst": hurst_out,
            "initializer": initializer,
            "initial_params": params0.to_dict(),
            "cost": cost,
        }

        if stage == ROUGH_VOL_STAGE_PREPARE:
            base.update(
                {
                    "success": True,
                    "params": {},
                    "params_usable": False,
                    "message": (
                        "Préparation terminée (4.1 → 4.9). "
                        f"{initializer.get('message_fr') or ''} Aucun (H, η, ρ) n'est calibré "
                        "à ce stade : les valeurs affichées sont des points de départ."
                    ),
                    "flags": [],
                    "warnings_fr": [],
                    "flag_details": [],
                    "blocking_flags": [],
                }
            )
            return self._json_safe(base)

        # -- 4.10 / 4.11 joint calibration (EXPENSIVE) --------------------------
        constraints["xi0_curve"] = xi0_curve
        constraints["option_surface"] = flat_points
        constraints["clean_chains"] = chains
        constraints["initial_params"] = (params0, init_diag)

        calib = self.run_advanced_surface_calibration(
            {
                "model": ROUGH_VOL_MODEL_KEY,
                "df": market_df,
                "S0": float(S0),
                "r": float(r_val),
                "q": float(q_val),
                "m_grid": m_grid,
                "t_grid": t_grid,
                "constraints": constraints,
                "fit_to_observed_only": bool(data.get("fit_to_observed_only", True)),
                "max_nfev": data.get("max_nfev"),
                "n_starts": data.get("n_starts"),
                "seed": data.get("seed"),
            }
        )
        if not isinstance(calib, dict):
            return _ko("calibration", "Le calibrateur joint n'a rien retourné.")

        details = calib.get("details") if isinstance(calib.get("details"), dict) else {}
        report = details.get("report") if isinstance(details.get("report"), dict) else {}
        flags = [str(f) for f in (report.get("flags") or details.get("flags") or [])]
        warnings_fr = [str(w) for w in (report.get("warnings_fr") or details.get("warnings_fr") or [])]
        # The model builds ``warnings_fr`` as ``[LABELS[f] for f in flags]``, so the
        # two lists are positionally paired; keep the pairing explicit so the view
        # can render a flag next to its French sentence, and mark which of them are
        # on their own enough to make the verdict False.
        try:
            labels_fr = self.get_rough_vol_flag_labels()
        except Exception:  # pragma: no cover - model import failure is reported elsewhere
            labels_fr = {}
        flag_details = [
            {
                "flag": flag,
                "label_fr": str(
                    (labels_fr.get(flag) or {}).get("label_fr")
                    or (warnings_fr[i] if i < len(warnings_fr) else flag)
                ),
                "blocking": bool((labels_fr.get(flag) or {}).get("blocking", False)),
            }
            for i, flag in enumerate(flags)
        ]
        success = bool(calib.get("success"))
        message = str(calib.get("message") or "")

        steps.append(
            {
                "step": "calibration",
                "label_fr": labels["calibration"],
                "ok": success,
                "message_fr": message,
            }
        )

        base.update(
            {
                "success": success,
                "message": message,
                "failed_step": None if success else "calibration",
                "model": calib.get("model"),
                "method": calib.get("method"),
                "params": calib.get("params") or {},
                # A False verdict means the run carries no information about H: the
                # triple above then measures nothing and must NOT reach the screen as
                # a calibration result.
                "params_usable": success,
                "metrics": calib.get("metrics") or {},
                "metrics_vw": calib.get("metrics_vw") or {},
                "iv_market": calib.get("iv_market"),
                "iv_model": calib.get("iv_model"),
                "iv_error": calib.get("iv_error"),
                "vega_weights": calib.get("vega_weights"),
                "mask": calib.get("mask"),
                "m_grid": calib.get("m_grid") or m_grid,
                "t_grid": calib.get("t_grid") or t_grid,
                "flags": flags,
                "warnings_fr": warnings_fr,
                "flag_details": flag_details,
                "blocking_flags": [d["flag"] for d in flag_details if d["blocking"]],
                "calibration": report,
                "details": details,
            }
        )
        return self._json_safe(base)

    def compute_diagnostics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute per-maturity/moneyness/smile diagnostics from a calibration result dict.
        Kept in controller so the view layer never imports from app.model directly.
        """
        from app.model.calibration.diagnostics import compute_all_diagnostics
        try:
            return compute_all_diagnostics(result)
        except Exception:
            return {}

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
