# placeholder
# Portfolio statistics helpers (eigen-portfolio weights and orders).
import numpy as np
import pandas as pd
from app.utils.math_utils import floor_n


def compute_eigen_orders(
    portfolio_df: pd.DataFrame,
    price_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Calcule les poids de l'eigen-portfolio (vecteur propre principal de la covariance des rendements)
    puis les ordres nécessaires pour s'y conformer.
    """
    if portfolio_df.empty:
        raise ValueError("Portfolio vide.")

    price_panel = price_panel.copy()
    price_panel = price_panel.ffill().dropna(how="all")
    if price_panel.empty or price_panel.shape[0] < 5:
        raise ValueError("Pas assez d'historique pour calculer les rendements.")

    log_returns = np.log(price_panel / price_panel.shift(1)).dropna(how="any")
    if log_returns.empty:
        raise ValueError("Impossible de calculer les rendements log.")

    cov = log_returns.cov()
    vals, vecs = np.linalg.eigh(cov)
    idx_max = int(np.argmax(vals))
    principal = vecs[:, idx_max]
    eigen_weights_signed = pd.Series(
        principal / np.sum(np.abs(principal)), index=log_returns.columns
    )
    eigen_weights = eigen_weights_signed.abs()
    eigen_weights = eigen_weights / eigen_weights.sum()

    latest_prices = price_panel.ffill().iloc[-1]
    pf = portfolio_df.copy()
    pf["spot"] = pf["ticker"].map(latest_prices)
    pf = pf.dropna(subset=["spot"])
    if pf.empty:
        raise ValueError("Prix introuvables pour les tickers du portfolio.")

    pf["side_sign"] = pf["side"].apply(lambda s: 1.0 if str(s).lower() == "long" else -1.0)
    pf["current_value"] = pf["quantity"] * pf["spot"] * pf["side_sign"]
    gross = pf["current_value"].abs().sum()
    if gross <= 0:
        raise ValueError("Exposition actuelle nulle; ordres impossibles.")

    target_weights = eigen_weights.reindex(pf["ticker"]).fillna(0.0)
    pf["target_value"] = target_weights.values * gross
    pf["order_value"] = pf["target_value"] - pf["current_value"]
    pf["order_qty"] = pf["order_value"] / pf["spot"]
    pf["order_side"] = pf["order_qty"].apply(
        lambda q: "buy" if q > 0 else ("sell" if q < 0 else "flat")
    )
    pf["target_side"] = pf["target_value"].apply(lambda v: "long" if v >= 0 else "short")
    pf["current_side"] = pf["side"]

    pf["order_qty"] = pf["order_qty"].apply(lambda v: floor_n(v, 4))
    pf["spot"] = pf["spot"].apply(lambda v: floor_n(v, 4))
    pf["current_value"] = pf["current_value"].apply(lambda v: floor_n(v, 4))
    pf["target_value"] = pf["target_value"].apply(lambda v: floor_n(v, 4))
    pf["cost_profit"] = (pf["spot"] * pf["order_qty"]).apply(lambda v: floor_n(v, 4))

    orders = pf[
        [
            "ticker",
            "spot",
            "target_value",
            "order_qty",
            "order_side",
            "current_side",
            "target_side",
            "cost_profit",
        ]
    ]
    return orders, eigen_weights, latest_prices
