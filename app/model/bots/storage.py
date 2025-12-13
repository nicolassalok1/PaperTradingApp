from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.utils.paths import JSON_DIR


@dataclass(frozen=True)
class GridBotConfig:
    symbol: str
    enabled: bool = False
    qty: float = 1.0
    n_levels: int = 5
    step_pct: float = 0.05
    reference: str = "spot"  # future: position_avg_entry
    time_in_force: str = "gtc"
    dry_run: bool = True

    def normalized(self) -> "GridBotConfig":
        sym = (self.symbol or "").strip().upper()
        n_levels = int(self.n_levels or 0)
        n_levels = max(1, min(n_levels, 50))

        step = float(self.step_pct or 0.0)
        step = max(0.0001, min(step, 0.5))

        qty = float(self.qty or 0.0)
        qty = max(qty, 0.0)

        ref = (self.reference or "spot").strip().lower()
        if ref not in {"spot", "position_avg_entry"}:
            ref = "spot"

        tif = (self.time_in_force or "gtc").strip().lower()
        if tif not in {"day", "gtc"}:
            tif = "gtc"

        return GridBotConfig(
            symbol=sym,
            enabled=bool(self.enabled),
            qty=qty,
            n_levels=n_levels,
            step_pct=step,
            reference=ref,
            time_in_force=tif,
            dry_run=bool(self.dry_run),
        )


_GRID_CONFIG_PATH = JSON_DIR / "bots_grid_configs.json"


def _read_json(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


def load_grid_configs() -> dict[str, GridBotConfig]:
    """
    Load grid bot configs persisted under `data/`.
    Returns a mapping keyed by symbol.
    """
    raw = _read_json(_GRID_CONFIG_PATH)
    if not isinstance(raw, dict):
        return {}

    out: dict[str, GridBotConfig] = {}
    for sym, cfg in raw.items():
        if not isinstance(sym, str):
            continue
        if not isinstance(cfg, dict):
            continue
        try:
            obj = GridBotConfig(
                symbol=str(cfg.get("symbol") or sym),
                enabled=bool(cfg.get("enabled", False)),
                qty=float(cfg.get("qty", 1.0) or 1.0),
                n_levels=int(cfg.get("n_levels", 5) or 5),
                step_pct=float(cfg.get("step_pct", 0.05) or 0.05),
                reference=str(cfg.get("reference", "spot") or "spot"),
                time_in_force=str(cfg.get("time_in_force", "gtc") or "gtc"),
                dry_run=bool(cfg.get("dry_run", True)),
            ).normalized()
        except Exception:
            continue

        if obj.symbol:
            out[obj.symbol] = obj

    return out


def upsert_grid_config(config: GridBotConfig) -> GridBotConfig:
    cfg = config.normalized()
    if not cfg.symbol:
        raise ValueError("symbol is required")

    configs = load_grid_configs()
    configs[cfg.symbol] = cfg
    _write_json(_GRID_CONFIG_PATH, {k: asdict(v) for k, v in sorted(configs.items())})
    return cfg


def delete_grid_config(symbol: str) -> None:
    sym = (symbol or "").strip().upper()
    if not sym:
        return
    configs = load_grid_configs()
    if sym in configs:
        del configs[sym]
        _write_json(_GRID_CONFIG_PATH, {k: asdict(v) for k, v in sorted(configs.items())})


__all__ = [
    "GridBotConfig",
    "load_grid_configs",
    "upsert_grid_config",
    "delete_grid_config",
]

