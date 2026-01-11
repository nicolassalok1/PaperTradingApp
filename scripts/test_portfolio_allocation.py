import math

import numpy as np

from app.model.portfolio_allocation.engine import eigen_portfolio_optimize


def test_eigen_portfolio_equals_first_pc_for_collinear_assets():
    # Perfectly collinear assets should yield equal long-only weights on the first PC
    returns = np.array(
        [
            [0.01, 0.02],
            [0.02, 0.04],
            [0.015, 0.03],
            [0.005, 0.01],
        ]
    )
    w = eigen_portfolio_optimize(returns)
    assert len(w) == 2
    assert all(x >= 0 for x in w)
    assert math.isclose(sum(w), 1.0, rel_tol=0.0, abs_tol=1e-12)
    np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-6, rtol=0.0)


def test_eigen_portfolio_handles_nans_and_returns_long_only_weights():
    returns = np.array(
        [
            [0.01, -0.005, 0.002],
            [0.0, 0.001, -0.001],
            [np.nan, 0.002, 0.0],  # dropped row
            [0.005, -0.002, 0.001],
        ]
    )
    w = eigen_portfolio_optimize(returns)
    assert len(w) == 3
    assert all(x >= 0 for x in w)
    assert math.isclose(sum(w), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert np.all(np.isfinite(w))
