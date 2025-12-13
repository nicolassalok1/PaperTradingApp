from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - optional dependency
    least_squares = None


ResidualFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LeastSquaresRun:
    ok: bool
    converged: bool
    message: str
    x: np.ndarray
    nfev: int
    cost: float
    optimality: float
    x0: np.ndarray | None = None
    idx: int | None = None
    error: str | None = None


def _as_array(x: Iterable[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=float)


def multi_start_least_squares(
    residuals: ResidualFn,
    *,
    x0_candidates: Iterable[np.ndarray],
    bounds: Tuple[np.ndarray, np.ndarray],
    max_nfev: int,
    method: str = "trf",
) -> tuple[LeastSquaresRun | None, Dict[str, object]]:
    """
    Runs SciPy least_squares across multiple x0 candidates and returns the best run.
    """
    if least_squares is None:
        return None, {"success": False, "message": "SciPy indisponible: least_squares non disponible.", "runs": []}

    lb, ub = bounds
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    runs: list[LeastSquaresRun] = []
    best: LeastSquaresRun | None = None
    best_cost = float("inf")

    for idx, x0 in enumerate(x0_candidates):
        try:
            opt = least_squares(
                residuals,
                x0=np.asarray(x0, dtype=float),
                bounds=(lb, ub),
                max_nfev=int(max_nfev),
                method=str(method),
            )
            run = LeastSquaresRun(
                ok=True,
                converged=bool(getattr(opt, "success", False)),
                message=str(getattr(opt, "message", "")),
                x=np.asarray(opt.x, dtype=float),
                nfev=int(getattr(opt, "nfev", 0) or 0),
                cost=float(getattr(opt, "cost", np.nan)),
                optimality=float(getattr(opt, "optimality", np.nan)),
                x0=np.asarray(x0, dtype=float),
                idx=int(idx),
            )
        except Exception as exc:
            run = LeastSquaresRun(
                ok=False,
                converged=False,
                message="",
                x=_as_array(x0),
                nfev=0,
                cost=float("nan"),
                optimality=float("nan"),
                x0=np.asarray(x0, dtype=float),
                idx=int(idx),
                error=str(exc),
            )

        runs.append(run)
        if not run.ok:
            continue
        if not np.isfinite(run.cost):
            continue
        if run.cost < best_cost:
            best_cost = float(run.cost)
            best = run

    summary = {
        "success": best is not None,
        "best_run": None if best is None else int(best.idx or 0),
        "n_runs": int(len(runs)),
        "runs": [
            {
                "idx": r.idx,
                "ok": r.ok,
                "converged": r.converged,
                "cost": r.cost,
                "nfev": r.nfev,
                "optimality": r.optimality,
                "message": r.message,
                "error": r.error,
            }
            for r in runs
        ],
    }
    return best, summary


def latin_hypercube_samples(
    *,
    n: int,
    bounds: list[tuple[float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simple LHS sampler for box bounds.
    Returns array shape (n, d).
    """
    d = len(bounds)
    if n <= 0 or d == 0:
        return np.zeros((0, d), dtype=float)
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.uniform(size=(n, d))
    a = cut[:n]
    b = cut[1 : n + 1]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    H = np.zeros_like(rdpoints)
    for j in range(d):
        order = rng.permutation(n)
        H[:, j] = rdpoints[order, j]
    out = np.empty_like(H)
    for j, (lo, hi) in enumerate(bounds):
        out[:, j] = lo + H[:, j] * (hi - lo)
    return out.astype(float)


__all__ = ["LeastSquaresRun", "multi_start_least_squares", "latin_hypercube_samples"]

