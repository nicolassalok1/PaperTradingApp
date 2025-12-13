from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

from app.model.yieldcurve.base import YieldCurve
from app.model.yieldcurve.interpolation import linear_zero_interp, log_discount_interp


@dataclass
class CurveNode:
    tenor: str | None
    t_years: float
    zero_rate: float | None
    discount_factor: float | None


class NodeYieldCurve(YieldCurve):
    """
    Curve built from deterministic nodes (zc or df) with interpolation.
    """

    def __init__(self, nodes: Iterable[dict], default_rate: float = 0.02):
        sanitized: List[CurveNode] = []
        for raw in nodes or []:
            t = raw.get("t_years")
            zc = raw.get("zero_rate")
            df = raw.get("discount_factor")
            tenor = raw.get("tenor")
            if t is None:
                continue
            try:
                t_f = float(t)
            except Exception:
                continue
            if not math.isfinite(t_f) or t_f <= 0:
                continue
            zc_f = None
            df_f = None
            if zc is not None:
                try:
                    zc_f = float(zc)
                except Exception:
                    zc_f = None
                if zc_f is not None:
                    if not math.isfinite(zc_f):
                        zc_f = None
                    elif zc_f > 1.0:
                        zc_f = zc_f / 100.0
            if df is not None:
                try:
                    df_f = float(df)
                except Exception:
                    df_f = None
                if df_f is not None and (not math.isfinite(df_f) or df_f <= 0):
                    df_f = None
            if df_f is None and zc_f is not None:
                df_f = math.exp(-zc_f * t_f)
            if zc_f is None and df_f is not None and df_f > 0:
                zc_f = -math.log(df_f) / t_f if t_f > 0 else default_rate
            if df_f is None or zc_f is None or not math.isfinite(zc_f) or not math.isfinite(df_f):
                continue
            sanitized.append(CurveNode(tenor=tenor, t_years=t_f, zero_rate=zc_f, discount_factor=df_f))

        sanitized.sort(key=lambda n: n.t_years)
        self._nodes = sanitized
        self._default_rate = float(default_rate)

    @property
    def maturities(self) -> List[float]:
        return [n.t_years for n in self._nodes]

    @property
    def nodes(self) -> List[CurveNode]:
        return list(self._nodes)

    def zero_rate(self, T_years: float) -> float:
        t = float(T_years)
        if not self._nodes:
            return self._default_rate
        if t <= self._nodes[0].t_years:
            return self._nodes[0].zero_rate
        if t >= self._nodes[-1].t_years:
            return self._nodes[-1].zero_rate
        zs = [n.zero_rate for n in self._nodes]
        Ts = [n.t_years for n in self._nodes]
        val = linear_zero_interp(Ts, zs, t)
        if val is None or not math.isfinite(val):
            return self._default_rate
        return float(val)

    def discount_factor(self, T_years: float) -> float:
        t = float(T_years)
        if not self._nodes:
            return math.exp(-self._default_rate * t)
        dfs = [n.discount_factor for n in self._nodes]
        Ts = [n.t_years for n in self._nodes]
        val = log_discount_interp(Ts, dfs, t)
        if val is None or not math.isfinite(val) or val <= 0:
            return math.exp(-self._default_rate * t)
        return float(val)

    def forward_rate(self, start_years: float, end_years: float) -> float | None:
        if end_years <= start_years or end_years <= 0:
            return None
        try:
            df1 = self.discount_factor(max(start_years, 0.0))
            df2 = self.discount_factor(end_years)
            return -(math.log(df2) - math.log(df1)) / (float(end_years) - float(start_years))
        except Exception:
            return None


class FlatYieldCurve(YieldCurve):
    """
    Flat curve fallback using a constant rate.
    """

    def __init__(self, rate: float = 0.02):
        self.rate = float(rate)

    @property
    def maturities(self) -> List[float]:
        return []

    def zero_rate(self, T_years: float) -> float:
        return self.rate

    def discount_factor(self, T_years: float) -> float:
        return math.exp(-self.rate * float(T_years))

    def forward_rate(self, start_years: float, end_years: float) -> float | None:
        if end_years <= start_years or end_years <= 0:
            return None
        return self.rate
