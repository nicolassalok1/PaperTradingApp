from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.model.alpaca_orders.service import AlpacaOrdersService
from app.model.market_data.market_data import fetch_spot_price
from app.utils.secrets import get_secret

from .storage import GridBotConfig


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _is_paper_base_url(base_url: str | None) -> bool:
    return "paper" in (base_url or "").lower()


def _try_orders_service() -> tuple[AlpacaOrdersService | None, str]:
    try:
        return AlpacaOrdersService(), ""
    except Exception as exc:  # noqa: BLE001 - surface in UI
        return None, str(exc)


def _extract_open_limit_buy_prices(orders: list[dict[str, Any]], symbol: str) -> set[float]:
    sym = (symbol or "").strip().upper()
    out: set[float] = set()
    for o in orders or []:
        if not isinstance(o, dict):
            continue
        if str(o.get("symbol") or "").strip().upper() != sym:
            continue
        side = str(o.get("side") or "").strip().lower()
        if side != "buy":
            continue
        lp = (
            o.get("limit_price")
            or o.get("limit_price")
            or o.get("limitPrice")
            or o.get("limit")
        )
        px = _safe_float(lp)
        if px is None or px <= 0:
            continue
        out.add(round(px, 2))
    return out


def _build_grid_limit_prices(reference_price: float, n_levels: int, step_pct: float) -> list[float]:
    ref = float(reference_price or 0.0)
    if ref <= 0:
        return []
    n = max(1, min(int(n_levels or 0), 50))
    step = max(0.0001, min(float(step_pct or 0.0), 0.5))
    prices: list[float] = []
    for i in range(1, n + 1):
        px = ref * (1.0 - step * i)
        if px <= 0:
            continue
        prices.append(round(px, 2))
    # de-dupe and keep descending (closest first)
    return sorted(set(prices), reverse=True)


@dataclass
class GridBotRunReport:
    symbol: str
    reference_price: float | None
    desired_prices: list[float]
    existing_open_limit_buy_prices: list[float]
    to_submit_prices: list[float]
    submitted_orders: list[dict[str, Any]]
    simulated_orders: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]


def run_grid_bot_once(
    config: GridBotConfig,
    *,
    allow_submit: bool = False,
    allow_live: bool = False,
) -> GridBotRunReport:
    """
    Compute (and optionally submit) a simple DCA/grid of BUY limit orders below a reference price.

    Safety defaults:
    - No submission unless allow_submit=True AND config.enabled=True AND config.dry_run=False
    - Refuses live trading unless allow_live=True (base_url does not contain 'paper')
    """
    cfg = config.normalized()
    errors: list[str] = []
    warnings: list[str] = []

    if not cfg.symbol:
        return GridBotRunReport(
            symbol="",
            reference_price=None,
            desired_prices=[],
            existing_open_limit_buy_prices=[],
            to_submit_prices=[],
            submitted_orders=[],
            simulated_orders=[],
            errors=["Missing symbol"],
            warnings=[],
        )

    spot = fetch_spot_price(cfg.symbol)
    ref_price = _safe_float(spot)
    if ref_price is None or ref_price <= 0:
        errors.append(f"Cannot resolve spot price for {cfg.symbol}")

    desired_prices = _build_grid_limit_prices(ref_price or 0.0, cfg.n_levels, cfg.step_pct)

    orders_service, svc_err = _try_orders_service()
    open_orders: list[dict[str, Any]] = []
    if orders_service is None:
        warnings.append(f"Alpaca orders unavailable: {svc_err}")
    else:
        try:
            open_orders = orders_service.get_open_orders()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to fetch open orders: {exc}")

    existing_prices = _extract_open_limit_buy_prices(open_orders, cfg.symbol)
    to_submit = [p for p in desired_prices if round(p, 2) not in existing_prices]

    submitted: list[dict[str, Any]] = []
    simulated: list[dict[str, Any]] = []

    base_url = get_secret("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
    is_paper = _is_paper_base_url(base_url)
    if not is_paper and not allow_live:
        if allow_submit and cfg.enabled and not cfg.dry_run and to_submit:
            errors.append("Refusing to submit orders: APCA_API_BASE_URL does not look like paper trading.")

    will_submit = bool(allow_submit and cfg.enabled and not cfg.dry_run and not errors)

    for px in to_submit:
        payload = {
            "symbol": cfg.symbol,
            "qty": float(cfg.qty),
            "side": "buy",
            "type": "limit",
            "time_in_force": cfg.time_in_force,
            "limit_price": float(px),
        }
        if will_submit and orders_service is not None:
            try:
                submitted.append(
                    orders_service.submit_limit_order(cfg.symbol, float(cfg.qty), float(px), "buy")
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Submit failed for {cfg.symbol} @ {px}: {exc}")
        else:
            simulated.append(payload)

    return GridBotRunReport(
        symbol=cfg.symbol,
        reference_price=ref_price,
        desired_prices=desired_prices,
        existing_open_limit_buy_prices=sorted(existing_prices, reverse=True),
        to_submit_prices=to_submit,
        submitted_orders=submitted,
        simulated_orders=simulated,
        errors=errors,
        warnings=warnings,
    )


__all__ = ["GridBotRunReport", "run_grid_bot_once"]

