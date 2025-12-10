"""
Yield Curve Engine (PRO Version)
- Continuous-compounding zero-coupon curve with interpolation/extrapolation
- Forward rates, discount factors, instantaneous forwards
- Nelson–Siegel calibration (grid search on tau + weighted LS on betas)
- CSV loader for FRED/Treasury curves
- Minimal plotting utility
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Utilities
# =============================================================================


def _maturity_years(label: str) -> float:
    """Convert a tenor label to year fraction."""
    label = label.strip().upper()
    if label.endswith("M"):
        return float(label[:-1]) / 12.0
    if label.endswith("Y"):
        return float(label[:-1])
    return float(label)


def nelson_siegel_yield(
    t: np.ndarray, beta0: float, beta1: float, beta2: float, tau: float
) -> np.ndarray:
    """Nelson–Siegel zero-coupon yield."""
    t = np.asarray(t, dtype=float)
    x = t / tau
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(t == 0, 1.0, (1.0 - np.exp(-x)) / x)
    return beta0 + beta1 * factor + beta2 * (factor - np.exp(-x))


def _weighted_ls(
    t: np.ndarray, y: np.ndarray, tau: float, weights: Optional[np.ndarray]
) -> Tuple[float, float, float]:
    """Solve for beta0, beta1, beta2 given tau using weighted least squares."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    x = t / tau
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(t == 0, 1.0, (1.0 - np.exp(-x)) / x)
    f2 = f1 - np.exp(-x)
    X = np.column_stack([np.ones_like(t), f1, f2])
    if weights is not None:
        w = np.asarray(weights, float)
        W = np.diag(w)
        beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
    else:
        beta = np.linalg.pinv(X) @ y
    return float(beta[0]), float(beta[1]), float(beta[2])


def fit_nelson_siegel_grid_search(
    t: Iterable[float],
    y: Iterable[float],
    tau_grid: Optional[Iterable[float]] = None,
    weights: Optional[Iterable[float]] = None,
) -> Tuple[float, float, float, float]:
    """
    Grid search on tau in [0.1, 10] (default), LS on betas for each tau.
    Returns (beta0, beta1, beta2, tau_best).
    """
    t = np.asarray(list(t), float)
    y = np.asarray(list(y), float)
    if tau_grid is None:
        tau_grid = np.linspace(0.1, 10.0, 50)
    w = np.asarray(weights, float) if weights is not None else None

    best_sse = math.inf
    best_params = (math.nan, math.nan, math.nan, math.nan)

    for tau in tau_grid:
        beta0, beta1, beta2 = _weighted_ls(t, y, tau, w)
        y_hat = nelson_siegel_yield(t, beta0, beta1, beta2, tau)
        resid = y - y_hat
        sse = np.sum((resid if w is None else resid * w) ** 2)
        if sse < best_sse:
            best_sse = sse
            best_params = (beta0, beta1, beta2, tau)

    return best_params


# =============================================================================
# Engine
# =============================================================================


@dataclass
class YieldCurveEngine:
    maturities: np.ndarray  # in years
    zero_rates: np.ndarray  # continuous rates
    interp: str = "linear"  # "linear" or "flat"
    extrap: str = "flat"  # "flat" or "linear"
    ns_params: Optional[Tuple[float, float, float, float]] = None

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yield_curve_csv(
        cls,
        path: str | Path,
        date: Optional[pd.Timestamp] = None,
        interp: str = "linear",
        extrap: str = "flat",
    ) -> "YieldCurveEngine":
        """
        Load a Treasury curve CSV (percent yields), pick last or nearest <= date.
        Columns: date, 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y.
        """
        path = Path(path)
        df = pd.read_csv(path, parse_dates=["date"])
        if date is None:
            row = df.iloc[-1]
        else:
            candidates = df[df["date"] <= pd.to_datetime(date)]
            row = candidates.iloc[-1] if not candidates.empty else df.iloc[-1]

        maturities = []
        rates = []
        for col in df.columns:
            if col.lower() == "date":
                continue
            maturities.append(_maturity_years(col))
            rates.append(float(row[col]) / 100.0)  # percent -> decimal

        maturities = np.array(maturities, float)
        zero_rates = np.array(rates, float)
        idx = np.argsort(maturities)
        return cls(maturities[idx], zero_rates[idx], interp=interp, extrap=extrap)

    # ------------------------------------------------------------------ #
    # Core evaluations
    # ------------------------------------------------------------------ #
    def _interp_rate(self, t: float) -> float:
        if t <= self.maturities[0]:
            if self.extrap == "flat":
                return float(self.zero_rates[0])
            return float(np.interp(t, self.maturities[:2], self.zero_rates[:2]))
        if t >= self.maturities[-1]:
            if self.extrap == "flat":
                return float(self.zero_rates[-1])
            return float(np.interp(t, self.maturities[-2:], self.zero_rates[-2:]))
        if self.interp == "flat":
            idx = np.searchsorted(self.maturities, t, side="right") - 1
            return float(self.zero_rates[idx])
        return float(np.interp(t, self.maturities, self.zero_rates))

    def zero_rate(self, t: float) -> float:
        t = float(t)
        if t <= 0:
            return float(self.zero_rates[0])
        return self._interp_rate(t)

    def discount_factor(self, t: float) -> float:
        r = self.zero_rate(t)
        return math.exp(-r * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        t1 = float(t1)
        t2 = float(t2)
        if t2 <= t1 or t2 <= 0:
            raise ValueError("Require t2 > t1 >= 0")
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return -(math.log(df2) - math.log(df1)) / (t2 - t1)

    def instantaneous_forward(self, t: float, h: float = 1e-4) -> float:
        t = float(t)
        h = float(h)
        if t < 0:
            raise ValueError("t must be >= 0")
        df_plus = self.discount_factor(t + h)
        df_minus = self.discount_factor(max(t - h, 0.0))
        return -(math.log(df_plus) - math.log(df_minus)) / (2 * h)

    # ------------------------------------------------------------------ #
    # Nelson–Siegel calibration
    # ------------------------------------------------------------------ #
    def calibrate_nelson_siegel(
        self, tau_grid: Optional[Iterable[float]] = None, weights: Optional[Iterable[float]] = None
    ):
        beta0, beta1, beta2, tau = fit_nelson_siegel_grid_search(
            self.maturities, self.zero_rates, tau_grid=tau_grid, weights=weights
        )
        self.ns_params = (beta0, beta1, beta2, tau)
        return self.ns_params

    def nelson_siegel_curve(self, t: np.ndarray) -> np.ndarray:
        if not self.ns_params:
            raise ValueError("Nelson–Siegel not calibrated.")
        beta0, beta1, beta2, tau = self.ns_params
        return nelson_siegel_yield(t, beta0, beta1, beta2, tau)

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def plot(self, show_ns: bool = True, title: Optional[str] = None):
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(self.maturities, self.zero_rates * 100, color="C0", label="Market zeros")
        t_smooth = np.linspace(max(1e-6, self.maturities.min()), self.maturities.max(), 200)
        interp_vals = np.array([self.zero_rate(t) for t in t_smooth]) * 100
        ax.plot(t_smooth, interp_vals, color="C1", label="Interpolated zero curve")
        if show_ns and self.ns_params:
            ns_vals = self.nelson_siegel_curve(t_smooth) * 100
            ax.plot(t_smooth, ns_vals, "r--", label="Nelson–Siegel")
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Zero rate (%)")
        ax.set_title(title or "Yield Curve")
        ax.legend()
        plt.tight_layout()
        plt.show()


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    csv_path = Path("data/yield_curve.csv")
    if csv_path.exists():
        yc = YieldCurveEngine.from_yield_curve_csv(csv_path)
        try:
            yc.calibrate_nelson_siegel()
        except Exception:
            pass
        yc.plot()
    else:
        try:
            from app.model.yieldcurve import builder
        except Exception:
            builder = None

        if builder:
            try:
                builder.main()
            except Exception as exc:
                print(f"Failed to build yield_curve.csv automatically: {exc}")

        if csv_path.exists():
            yc = YieldCurveEngine.from_yield_curve_csv(csv_path)
            try:
                yc.calibrate_nelson_siegel()
            except Exception:
                pass
            yc.plot()
        else:
            print(
                "yield_curve.csv not found and auto-build failed. Run `python -m app.model.yieldcurve.builder` manually."
            )

##TODO : else : app.model.yieldcurve.builder
