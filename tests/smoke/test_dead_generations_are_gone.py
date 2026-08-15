"""E4 — stale generations are deleted, and what only a CLI entry point uses is not.

"Unreachable by static analysis" is NOT proof of death in this codebase: Streamlit
reaches `app/vue/tabs/tab_*.py` through `main_app.autodiscover_tabs`, and
`scripts/test_options_model_integrity.py` walks the whole options package. Every
module removed here cleared three independent bars:

1. absent from `sys.modules` after `AppTest.from_file("streamlit_app.py").run()`
   booted the real app (28 tabs rendered, no tab-render error);
2. absent from `sys.modules` after a full pytest session (628 tests);
3. unreachable in the AST import graph from EVERY root — `streamlit_app.py`,
   `conftest.py`, `tests/**`, `scripts/**`, and the `__main__`-guarded entry
   points inside `app/` itself.

Bar 3 is why the second half of this module exists. The entry points under
`app/model/market_data/scripts/` are launched by hand and imported by nobody, so
a naive sweep marks them dead — and then marks everything they import dead too.
`update_balance` imports `portfolio.settlement`, `update_portfolio_value` imports
`portfolio.valuation`, `update_spots` imports `market_data.cache_refresh`: 17
modules were only ever reached that way. Deleting a CLI script's dependency
breaks nothing any test imports, which is exactly what makes it dangerous.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.smoke


# Stale generations, superseded and wired to nothing.
REMOVED_MODULES = [
    # Constant 0.02 placeholder; the live rate comes from yieldcurve.rates_utils.
    "app.model.market_data.rates",
    # Superseded portfolio layer (README already listed `repository` as legacy).
    "app.model.portfolio.repository",
    "app.model.portfolio.service",
    "app.model.portfolio.stats",
    # Superseded trading layer; execution/systems/logs/buy_sell stay, they are
    # reached from the market_data CLI scripts.
    "app.model.trading.hedging",
    "app.model.trading.service",
    # Bates calibration was never wired: the controller exposes no "bates" spec
    # and tab_advanced_calibration filters the key out. `jump_diffusion.cf`
    # keeps `bates_log_return_cf`, which IS live.
    "app.model.volatility_models.jump_diffusion.calibrator_bates",
    "app.model.volatility_models.jump_diffusion.model_bates",
    # Duplicate of the live app/vue/components/options/ui_helpers.py.
    "app.vue.components.ui_helpers",
    # E5 — the JSON-portfolio generation, superseded by the Alpaca path in
    # app/controller/trading_controller.py. It looked live only because
    # `dashboard/__init__.py` re-exported `service`, so importing `dashboard.cache`
    # pulled the whole subtree into sys.modules. With that re-export gone, the two
    # CLI roots load 33 app modules and none of these is among them.
    "app.model.dashboard.service",
    "app.model.dashboard.utils",
    "app.model.portfolio.positions",
    "app.model.trading.execution",
    "app.model.trading.systems",
    "app.model.trading.buy_sell",
    "app.model.backtesting.engine",
    "app.model.backtesting.signals",
]


# Reached ONLY through a `__main__`-guarded script under app/. No test imports
# them, so nothing here would have caught their removal.
CLI_ONLY_DEPENDENCIES = [
    "app.model.portfolio.settlement",      # app/model/market_data/scripts/update_balance.py
    "app.model.portfolio.valuation",       # .../update_portfolio_value.py
    "app.model.market_data.cache_refresh",  # .../update_spots.py
    # Survivors of E5: everything the two CLI roots still reach inside the
    # JSON-portfolio cluster. `trading.logs` also absorbed `append_trade_log`, the
    # single function `trading.buy_sell` still had a caller for.
    "app.model.portfolio.forwards",        # via portfolio.settlement
    "app.model.trading.logs",              # via portfolio.settlement
    "app.model.dashboard.cache",           # via portfolio.settlement
]


# Live counterparts of the modules removed above — over-deletion tripwire.
SURVIVING_COUNTERPARTS = [
    "app.model.yieldcurve.rates_utils",
    "app.vue.components.options.ui_helpers",
    "app.model.volatility_models.jump_diffusion.calibrator",
    "app.model.volatility_models.jump_diffusion.cf",
]


def _spec_or_none(modname):
    """`find_spec` returns None for a missing module, but raises once the parent
    package is gone too — which is what E5 did to `app.model.backtesting`. Both
    mean the same thing here, and the raising case means it more strongly."""
    try:
        return importlib.util.find_spec(modname)
    except ModuleNotFoundError:
        return None


@pytest.mark.parametrize("modname", REMOVED_MODULES)
def test_stale_generation_is_gone(modname):
    """Pins the deletion: a superseded module must not silently reappear."""
    assert _spec_or_none(modname) is None, f"{modname} is back"


@pytest.mark.parametrize("modname", CLI_ONLY_DEPENDENCIES + SURVIVING_COUNTERPARTS)
def test_module_reached_only_by_a_cli_entry_point_survives(modname):
    assert importlib.util.find_spec(modname) is not None, f"{modname} was over-deleted"


def test_bates_characteristic_function_is_still_available():
    """The Bates CALIBRATOR goes; its characteristic function is live code."""
    from app.model.volatility_models.jump_diffusion.cf import bates_log_return_cf

    assert callable(bates_log_return_cf)
