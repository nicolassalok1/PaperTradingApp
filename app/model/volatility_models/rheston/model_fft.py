from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from app.model.calibration.base_calibrator import SurfaceGrid
from app.model.calibration.implied_vol import implied_vol_call
from app.model.volatility_models.base import VolSurfaceModel
from app.model.volatility_models.common.fft import FFTConfig, carr_madan_fft_call_prices, interp_prices
from app.model.volatility_models.rheston.cf_markovian import RHestonMarkovianConfig, rheston_log_return_cf_markovian


class RHestonFFTMarkovianModel(VolSurfaceModel):
    model_key = "rheston"
    label = "rHeston (Markovian approx) via FFT"

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            "H": (0.02, 0.49),
            "kappa": (0.1, 10.0),
            "theta": (1e-4, 1.5),
            "xi": (1e-3, 5.0),
            "rho": (-0.999, 0.999),
            "v0": (1e-4, 1.5),
        }

    def implied_vol_surface(self, surface: SurfaceGrid, params: Dict[str, Any], **kwargs: Any) -> np.ndarray:
        S0 = float(surface.S0)
        r = float(surface.r)
        q = float(surface.q)
        m_grid = np.asarray(surface.m_grid, dtype=float)
        t_grid = np.asarray(surface.t_grid, dtype=float)

        H = float(params.get("H", 0.1))
        kappa = float(params.get("kappa", 2.0))
        theta = float(params.get("theta", 0.04))
        xi = float(params.get("xi", 0.6))
        rho = float(params.get("rho", -0.5))
        v0 = float(params.get("v0", 0.04))

        fft_cfg = kwargs.get("fft_cfg") if isinstance(kwargs.get("fft_cfg"), FFTConfig) else FFTConfig()
        mk_cfg = kwargs.get("markovian_cfg") if isinstance(kwargs.get("markovian_cfg"), RHestonMarkovianConfig) else RHestonMarkovianConfig()

        iv = np.full((len(t_grid), len(m_grid)), np.nan, dtype=float)
        for i_t, T in enumerate(t_grid):
            if float(T) <= 0:
                continue

            def _cf(u: np.ndarray, tt: float) -> np.ndarray:
                d = rheston_log_return_cf_markovian(
                    u,
                    [float(tt)],
                    r=r,
                    q=q,
                    kappa=kappa,
                    theta=theta,
                    xi=xi,
                    rho=rho,
                    v0=v0,
                    H=H,
                    cfg=mk_cfg,
                )
                return d[float(tt)]

            K_grid, C_grid = carr_madan_fft_call_prices(S0=S0, r=r, q=q, T=float(T), cf_log_return=_cf, cfg=fft_cfg)
            K = (S0 * m_grid).astype(float)
            C = interp_prices(K_grid=K_grid, C_grid=C_grid, K=K)
            for j_m in range(len(m_grid)):
                iv[i_t, j_m] = implied_vol_call(float(C[j_m]), S0, float(K[j_m]), float(T), r, q)
        return iv


__all__ = ["RHestonFFTMarkovianModel"]

