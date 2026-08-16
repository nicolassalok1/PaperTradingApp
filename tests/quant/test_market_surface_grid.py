"""`make_fixed_grid` puts a scattered market surface onto the fixed (T, moneyness) grid.

Real Yahoo surfaces never sit on the grid nodes: expiries fall at 0.86y / 1.1y / 1.9y,
strikes run from 0.4 to 1.6 in moneyness. The grid stops at T=1.0 and m∈[0.8, 1.2].
A node's value must be the *interpolation* of the observations around it — never a
raw observation snapped from far away (a 1.9-year put filling the 1-year slot, a
0.4-moneyness wing filling the 0.8 edge). Those snapped nodes are exactly the ones the
mask marks as observed, i.e. the ones every calibrator fits to.

Oracles: hand-computed linear interpolation between the bracketing slices / strikes,
plus invariants (a value on-node round-trips; nothing outside the observed IV range).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.model.calibration.market_surface import default_grid, make_fixed_grid

pytestmark = pytest.mark.unit


def _yahoo_shaped(expiries=(0.86, 1.1, 1.9), moneyness=(0.4, 0.78, 0.82, 1.0, 1.18, 1.22, 1.6)):
    """Term structure 0.20 -> 0.22 -> 0.30, symmetric smile 0.5*(m-1)^2, calls listed then puts."""
    base = {0.86: 0.20, 1.1: 0.22, 1.9: 0.30}
    rows = []
    for typ in ("call", "put"):
        for T in expiries:
            for m in moneyness:
                rows.append((m, T, base.get(T, 0.20) + 0.5 * (m - 1.0) ** 2))
    return pd.DataFrame(rows, columns=["moneyness", "ttm", "iv"])


def _node(grid, t_grid, m_grid, t, m):
    return float(grid[list(t_grid).index(t), list(m_grid).index(m)])


def test_one_year_node_is_interpolated_between_bracketing_expiries_not_the_two_year_put():
    m_grid, t_grid = default_grid()
    iv_grid, mask = make_fixed_grid(_yahoo_shaped(), m_grid, t_grid)

    # ATM at T=1.0 sits between the 0.86y (0.20) and 1.1y (0.22) slices.
    expected = 0.20 + (1.0 - 0.86) / (1.1 - 0.86) * (0.22 - 0.20)
    assert _node(iv_grid, t_grid, m_grid, 1.0, 1.0) == pytest.approx(expected, abs=2e-3)
    # ... and is still counted as observed (the 1y slice is well covered by data).
    assert mask[list(t_grid).index(1.0), list(m_grid).index(1.0)]


def test_edge_nodes_are_not_overwritten_by_far_wings():
    m_grid, t_grid = default_grid()
    iv_grid, _ = make_fixed_grid(_yahoo_shaped(), m_grid, t_grid)

    # m=0.8 at T=1.0: bracketed by 0.78 / 0.82 strikes -> smile value 0.5*0.2^2 = 0.02 above ATM.
    atm = _node(iv_grid, t_grid, m_grid, 1.0, 1.0)
    for m_edge in (0.8, 1.2):
        v = _node(iv_grid, t_grid, m_grid, 1.0, m_edge)
        assert v == pytest.approx(atm + 0.5 * (m_edge - 1.0) ** 2, abs=3e-3), (m_edge, v, atm)
    # The far wings (m=0.4 -> +0.18, m=1.6 -> +0.18) must not leak onto the grid edge.
    assert _node(iv_grid, t_grid, m_grid, 1.0, 1.2) < atm + 0.05


def test_grid_values_stay_inside_observed_range():
    m_grid, t_grid = default_grid()
    df = _yahoo_shaped()
    iv_grid, _ = make_fixed_grid(df, m_grid, t_grid)
    assert np.isfinite(iv_grid).all()
    assert iv_grid.min() >= df["iv"].min() - 1e-12
    assert iv_grid.max() <= df["iv"].max() + 1e-12


def test_on_node_observations_round_trip_exactly():
    m_grid, t_grid = default_grid()
    rng = np.random.default_rng(0)
    vals = 0.15 + 0.1 * rng.random((len(t_grid), len(m_grid)))
    rows = [(m, t, vals[i, j]) for i, t in enumerate(t_grid) for j, m in enumerate(m_grid)]
    df = pd.DataFrame(rows, columns=["moneyness", "ttm", "iv"])
    iv_grid, mask = make_fixed_grid(df, m_grid, t_grid)
    assert mask.all()
    np.testing.assert_allclose(iv_grid, vals, rtol=0, atol=1e-9)


def test_single_expiry_smile_lands_on_its_nearest_slice_only():
    m_grid, t_grid = default_grid()
    df = _yahoo_shaped(expiries=(0.29,))  # nearest node 0.25
    iv_grid, mask = make_fixed_grid(df, m_grid, t_grid)
    i = list(t_grid).index(0.25)
    assert mask[i].any()
    assert not mask[[k for k in range(len(t_grid)) if k != i]].any()
    assert np.isfinite(iv_grid).all()
