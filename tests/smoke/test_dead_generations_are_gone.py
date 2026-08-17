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
    # G2 — a second, "compute button" generation of the option panels
    # (components/options/{spreads,vanilla,path,exotic,barrier,calendars}/,
    # layout/pricing/shared, selector/options_text/payoff_viewer, and the 7-line
    # panels/tab_* shims). Imported by nothing: the router reaches panels/** only,
    # and importing every tab module loads none of these. HANDOFF (session 2)
    # called them live; they were only ever loaded by the smoke walker that
    # imports every file under components/options.
    "app.vue.components.options.layout",
    "app.vue.components.options.pricing",
    "app.vue.components.options.shared",
    "app.vue.components.options.spreads.straddle",
    "app.vue.components.options.vanilla.american",
    "app.vue.components.options.path.asian",
    "app.vue.components.options.exotic.quanto",
    "app.vue.components.options.barrier.vanilla_barrier",
    "app.vue.components.options.calendars.calendar",
    "app.vue.components.options.panels.tab_vertical_spread",
    "app.vue.components.options.panels.tab_straddle",
    "app.vue.components.selector",
    "app.vue.components.options_text",
    "app.vue.components.payoff_viewer",
    # H1 — the "🤖 Bots" and "🧪 Exercices" tabs, retired on Nicolas' call
    # (2026-08-17): the whole vertical of each, view -> controller -> model. Both
    # were reached only from their own tab module, so removing the two tabs left
    # every module below with zero importers.
    "app.vue.tabs.tab_bots",
    "app.vue.tabs.tab_exercices",
    "app.vue.components.exercises.portfolio_allocation",
    "app.controller.bots_controller",
    "app.controller.exercises_controller",
    "app.model.bots.assistant",
    "app.model.bots.grid_bot",
    "app.model.bots.storage",
    "app.model.bots.volatility",
    "app.model.exercises.portfolio_allocation.engine",
    "app.model.exercises.portfolio_allocation.yahoo_data",
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
    # G2 — the live option panels sit under panels/**, one level down.
    "app.vue.components.options.router",
    "app.vue.components.options.panels.tab_calendar",
    "app.vue.components.options.panels.spreads.tab_straddle",
    "app.vue.components.options.panels.vanilla.tab_american",
    "app.vue.components.options.panels.exotics.tab_quanto",
    "app.model.volatility_models.jump_diffusion.calibrator",
    "app.model.volatility_models.jump_diffusion.cf",
    # H1 — `app.model.portfolio_allocation` is a DIFFERENT package from the deleted
    # `app.model.exercises.portfolio_allocation`: it powers "🧭 Portefeuille & Risque"
    # (eigen_portfolio_optimize, AlpacaPortfolioClient) and its paper guard is pinned
    # in tests/quant/test_paper_flag_consistency.py. Same name, opposite fate.
    "app.model.portfolio_allocation.engine",
    # The ChatGPT wrapper the bots assistant called. Kept: generic infrastructure,
    # still covered by tests/test_bots_hygiene.py (redaction + cost guard).
    "app.model.ai.chatgpt",
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


RETIRED_TAB_LABELS = ["🤖 Bots", "🧪 Exercices"]


@pytest.mark.parametrize("label", RETIRED_TAB_LABELS)
def test_retired_tab_is_not_wired_anywhere(label):
    """H1 — a tab survives three separate wirings in main_app: the TAB_GROUPS
    ordering, the module->label override table, and (for Bots) an explicit
    fallback registration. Missing one of them puts the tab back on screen or
    crashes the boot."""
    from app.vue import main_app

    assert label not in [lbl for labels in main_app.TAB_GROUPS.values() for lbl in labels]
    assert label not in main_app.DEFAULT_LABEL_OVERRIDES.values()
    assert label not in main_app.autodiscover_tabs()


def test_bates_characteristic_function_is_still_available():
    """The Bates CALIBRATOR goes; its characteristic function is live code."""
    from app.model.volatility_models.jump_diffusion.cf import bates_log_return_cf

    assert callable(bates_log_return_cf)
