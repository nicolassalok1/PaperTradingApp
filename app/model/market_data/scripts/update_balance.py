"""
Settle matured forwards by updating portfolio and balance via core buy/sell flows.

For each forward in forwards_portfolio.json with maturity <= today:
  - If side == "long": buy the quantity at the forward price.
  - If side == "short": sell the quantity at the forward price.
The balance is adjusted by the payoff using the asset price at maturity (hist close fallback).
Processed forwards are removed from the forwards portfolio.

All trades are logged with source="forwards".
"""

from app.model.portfolio.settlement import check_and_settle_forward, process_matured_forwards


def main() -> None:
    process_matured_forwards()


if __name__ == "__main__":
    main()
