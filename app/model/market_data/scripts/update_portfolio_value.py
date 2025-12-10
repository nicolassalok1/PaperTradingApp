"""
Recompute the global portfolio_value and persist it to data/dashboard_vars.json.

The calculation uses:
  - spot_portfolio.json: sum of (side-sign * quantity * latest price from dashboard_vars.prices)
  - forwards_portfolio.json: sum of (side-sign * quantity * forward_price)

If no price is available for a spot symbol, it contributes 0.
Other keys in dashboard_vars.json are preserved.
"""

from app.model.portfolio.valuation import recompute_portfolio_value


def main() -> None:
    recompute_portfolio_value()


if __name__ == "__main__":
    main()
