"""
Portfolio controller.
Thin wrappers for portfolio services used by the view.
"""

from __future__ import annotations

import pandas as pd

from app.model.options.core.book import load_options_book
from app.model.portfolio.positions import load_portfolio_default, save_portfolio_default
from app.model.portfolio.service import apply_orders
from app.model.portfolio.stats import compute_eigen_orders
from app.model.portfolio.valuation import load_portfolio_price_panel, portfolio_dict_to_df


def load_portfolio() -> dict:
    return load_portfolio_default()


def save_portfolio(portfolio: dict) -> None:
    save_portfolio_default(portfolio)


def load_options() -> dict:
    return load_options_book()


def compute_eigen(pf_df: pd.DataFrame, price_panel: pd.DataFrame):
    return compute_eigen_orders(pf_df, price_panel)


def load_price_panel(tickers: list[str], loader, *, period: str = "1y", interval: str = "1d"):
    return load_portfolio_price_panel(tickers, loader, period=period, interval=interval)


def portfolio_to_df(portfolio: dict) -> pd.DataFrame:
    return portfolio_dict_to_df(portfolio)


__all__ = [
    "apply_orders",
    "compute_eigen",
    "load_options",
    "load_price_panel",
    "load_portfolio",
    "portfolio_to_df",
    "save_portfolio",
]
