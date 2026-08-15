"""
Settle matured forwards against the cash balance.

For each forward in forwards_portfolio.json with maturity <= today, the balance is
adjusted by the payoff — (settlement price - forward price) x quantity for a long,
the opposite for a short — using the last close on or before maturity, falling back
to the live quote. Processed forwards are removed from the forwards portfolio.

All settlements are logged with source="forwards". No position is opened or closed:
the earlier description of this script driving buy/sell flows was never accurate.
"""

from app.model.portfolio.settlement import check_and_settle_forward, process_matured_forwards


def main() -> None:
    process_matured_forwards()


if __name__ == "__main__":
    main()
