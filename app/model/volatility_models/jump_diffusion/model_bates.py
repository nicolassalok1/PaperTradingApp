from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from app.model.calibration.base_calibrator import SurfaceGrid
from app.model.calibration.implied_vol import implied_vol_call
from app.model.volatility_models.base import VolSurfaceModel
from app.model.volatility_models.common.fft import FFTConfig, carr_madan_fft_call_prices, interp_prices
from app.model.volatility_models.jump_diffusion.cf import bates_log_return_cf


class BatesModel(VolSurfaceModel):
    model_key = "bates"
    label = "Bates (Heston + Jumps) via FFT"

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "kappa": (0.1, 10.0),
            "theta": (1e-4, 1.5),
            "sigma": (1e-3, 5.0),
            "rho": (-0.999, 0.999),
            "v0": (1e-4, 1.5),
            "lam": (0.0, 5.0),
            "muj": (-1.0, 1.0),
            "sigj": (1e-4, 2.0),
        }

    def implied_vol_surface(self, surface: SurfaceGrid, params: Dict[str, Any], **kwargs: Any) -> np.ndarray:
        S0 = float(surface.S0)
        r = float(surface.r)
        q = float(surface.q)
        m_grid = np.asarray(surface.m_grid, dtype=float)
        t_grid = np.asarray(surface.t_grid, dtype=float)

        kappa = float(params.get("kappa", 2.0))
        theta = float(params.get("theta", 0.04))
        sigma = float(params.get("sigma", 0.5))
        rho = float(params.get("rho", -0.5))
        v0 = float(params.get("v0", 0.04))
        lam = float(params.get("lam", 0.2))
        muj = float(params.get("muj", -0.05))
        sigj = float(params.get("sigj", 0.2))

        cfg = kwargs.get("fft_cfg") if isinstance(kwargs.get("fft_cfg"), FFTConfig) else FFTConfig()

        iv = np.full((len(t_grid), len(m_grid)), np.nan, dtype=float)
        for i_t, T in enumerate(t_grid):
            if float(T) <= 0:
                continue

            def _cf(u: np.ndarray, tt: float) -> np.ndarray:
                return bates_log_return_cf(
                    u,
                    tt,
                    r=r,
                    q=q,
                    kappa=kappa,
                    theta=theta,
                    sigma=sigma,
                    rho=rho,
                    v0=v0,
                    lam=lam,
                    muj=muj,
                    sigj=sigj,
                )

            K_grid, C_grid = carr_madan_fft_call_prices(S0=S0, r=r, q=q, T=float(T), cf_log_return=_cf, cfg=cfg)
            K = (S0 * m_grid).astype(float)
            C = interp_prices(K_grid=K_grid, C_grid=C_grid, K=K)
            for j_m in range(len(m_grid)):
                iv[i_t, j_m] = implied_vol_call(float(C[j_m]), S0, float(K[j_m]), float(T), r, q)
        return iv


__all__ = ["BatesModel"]

