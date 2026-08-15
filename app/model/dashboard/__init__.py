"""Dashboard model package.

Deliberately empty of re-exports. This module used to pull `dashboard.service`
into every import of `dashboard.cache`, which had two costs:

  - a circular import that broke `market_data/scripts/update_balance.py` outright
    (`settlement -> buy_sell -> dashboard.cache -> here -> service -> settlement`,
    the last hop landing on a partially initialised module);
  - a whole superseded generation kept alive on paper, since anything importing
    `dashboard.cache` made ~900 lines of unreachable code appear loaded at runtime
    and therefore un-provable as dead.

Import the submodule you need — `from app.model.dashboard.cache import ...`.
"""
