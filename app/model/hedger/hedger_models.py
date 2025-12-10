from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from app.utils.paths import JSON_DIR


@dataclass
class OptionSpec:
    id: str
    symbol: str
    option_type: str  # "call" / "put"
    strike: float
    maturity_years: float
    side: int  # +1 long, -1 short
    quantity: float
    S0: float
    r: float = 0.0
    q: float = 0.0
    heston_params: Dict | None = None

    @staticmethod
    def from_json(id_: str, data: dict) -> "OptionSpec":
        opt_type = str(data.get("option_type", "call")).lower()
        strike = float(data.get("strike", 0.0) or 0.0)
        T = float(data.get("maturity_years", data.get("T", 1.0)) or 1.0)
        side = 1 if str(data.get("side", "long")).lower() == "long" else -1
        qty = float(data.get("quantity", 1.0) or 1.0)
        misc = data.get("misc", {}) if isinstance(data.get("misc", {}), dict) else {}
        hp = (
            (data.get("pricing_params") or {}).get("heston_params")
            or data.get("heston_params")
            or misc.get("heston_params")
            or {}
        )
        S0 = float(
            data.get("S0")
            or data.get("S_0")
            or misc.get("spot_at_pricing")
            or (data.get("pricing_params") or {}).get("S0")
            or strike
        )
        r = float(
            (data.get("pricing_params") or {}).get("rf_rate")
            or (data.get("pricing_params") or {}).get("r")
            or data.get("r", hp.get("rf_rate", 0.0))
            or hp.get("rf_rate", 0.0)
            or 0.0
        )
        q = float(
            (data.get("pricing_params") or {}).get("dividend_yield")
            or (data.get("pricing_params") or {}).get("q")
            or data.get("q", hp.get("dividend_yield", 0.0))
            or hp.get("dividend_yield", 0.0)
            or 0.0
        )
        return OptionSpec(
            id=id_,
            symbol=data.get("underlying", ""),
            option_type=opt_type,
            strike=strike,
            maturity_years=T,
            side=side,
            quantity=qty,
            S0=S0,
            r=r,
            q=q,
            heston_params=hp if hp else None,
        )


def compute_terminal_reward(
    option: OptionSpec, position: float, cash: float, price_path: np.ndarray
) -> float:
    S_T = float(price_path[-1])
    payoff = (
        max(S_T - option.strike, 0.0)
        if option.option_type == "call"
        else max(option.strike - S_T, 0.0)
    )
    deriv_pnl = option.side * option.quantity * payoff
    underlying_pnl = position * S_T
    pnl = cash + underlying_pnl + deriv_pnl
    return -float(pnl**2)


def build_state(option: OptionSpec, price_path: np.ndarray, t: int, position: float) -> np.ndarray:
    S = float(price_path[t])
    S0 = option.S0
    K = option.strike
    t_norm = t / max(1, len(price_path) - 1)
    S_norm = S / S0 if S0 > 0 else 1.0
    moneyness = S / K if K > 0 else 1.0
    pos_norm = position / max(1.0, option.quantity)
    return np.array([S_norm, t_norm, moneyness, pos_norm, float(option.side)], dtype=np.float32)
