from typing import Dict

from app.utils.io import load_json_file, save_json_file
from app.utils.math_utils import floor_2
from app.utils.paths import JSON_DIR

# Public API surface (helps avoid partial imports during reloads)
__all__ = [
    "compute_option_payoff",
    "compute_option_pnl",
    "load_expired",
    "save_expired",
    # Backward-compat aliases
    "load_expire",
    "save_expire",
]

# Legacy cache location shared with the Options UI and dashboard.
EXPIRED_OPTIONS_FILE = JSON_DIR / "options_expired.json"


def compute_option_payoff(option: dict, spot: float) -> float:
    """Compute intrinsic payoff of a vanilla option (call/put) at spot."""
    if not option:
        return 0.0
    opt_type = str(option.get("type") or option.get("option_type", "call")).lower()
    strike = float(option.get("strike", option.get("strike_price", 0.0)) or 0.0)
    qty = float(option.get("quantity", 0.0) or 0.0)
    mult = 1.0 if str(option.get("side", "long")).lower() == "long" else -1.0
    intrinsic = 0.0
    if opt_type.startswith("c"):
        intrinsic = max(spot - strike, 0.0)
    elif opt_type.startswith("p"):
        intrinsic = max(strike - spot, 0.0)
    return mult * intrinsic * qty


def compute_option_pnl(option: dict, spot_at_event: float, mark_price: float | None = None) -> Dict:
    """Basic PnL computation: payoff minus premium."""
    payoff = compute_option_payoff(option, spot_at_event)
    premium = float(option.get("premium", option.get("price", 0.0)) or 0.0) * float(
        option.get("quantity", 0.0) or 0.0
    )
    pnl_total = payoff - premium
    return {
        "payoff_per_unit": floor_2(payoff / (option.get("quantity") or 1)),
        "pnl_total": floor_2(pnl_total),
    }


def load_expired() -> dict:
    """Load expired options from the shared cache (empty dict on failure)."""
    return load_json_file(EXPIRED_OPTIONS_FILE, {})


def save_expired(data: dict) -> None:
    """Persist expired options to the shared cache."""
    payload = data if isinstance(data, dict) else {}
    save_json_file(EXPIRED_OPTIONS_FILE, payload)


# ---------------------------------------------------------------------------
# Backward compatibility (some older panels used misnamed helpers)
# ---------------------------------------------------------------------------
def load_expire() -> dict:
    """Alias for load_expired."""
    return load_expired()


def save_expire(data: dict) -> None:
    """Alias for save_expired."""
    save_expired(data)
