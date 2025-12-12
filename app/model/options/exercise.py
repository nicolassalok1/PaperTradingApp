from __future__ import annotations

import numpy as np


def build_exercise_dates(
    option_type: str,
    T: float,
    *,
    frequency: int | None = None,
    custom_dates: list[float] | None = None,
) -> list[float]:
    """
    Build exercise dates (in years) for European/Bermudan/American styles.
    - European: [T]
    - Bermudan: evenly spaced frequency points between (0, T]
    - American: dense grid (frequency per year, default daily 252)
    """
    opt = (option_type or "").lower()
    if custom_dates:
        return sorted([t for t in custom_dates if t > 0])

    if opt.startswith("eu"):
        return [float(T)]

    if opt.startswith("ber"):
        n = int(frequency or 5)
        grid = np.linspace(T / n, T, num=n, endpoint=True)
        return [float(t) for t in grid if t > 0]

    # American default: daily frequency
    freq = int(frequency or max(1, int(252 * T)))
    grid = np.linspace(T / freq, T, num=freq, endpoint=True)
    return [float(t) for t in grid if t > 0]


__all__ = ["build_exercise_dates"]
