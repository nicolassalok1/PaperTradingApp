"""
Model service for the 🌡️ Vol Implicite tab — Alpaca-backed.

Data strategy (Alpaca has no historical implied-vol endpoint, unlike IB/TWS
`OPTION_IMPLIED_VOLATILITY`):
- Historical vol series : realized volatility computed from Alpaca daily stock
  bars (fallback: the app's Stooq/Yahoo OHLC history).
- Current implied vol   : ATM ~30 DTE median IV from Alpaca option snapshots
  (`greeks`/`impliedVolatility` when the feed provides them, otherwise a
  Black-Scholes inversion of the quote mid).
- Every current-IV observation is upserted into `cache/IVHistory/iv_daily_*.csv`
  so a true daily IV history accumulates locally over time.

MVC: model layer — may call external APIs and read/write the local cache,
must not import Streamlit.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from app.model.calibration.implied_vol import implied_vol_call
from app.model.iv_dashboard import analytics
from app.model.market_data.market_data import fetch_ohlc_history, fetch_spot_price
from app.utils.paths import CACHE_IV_HISTORY_DIR
from app.utils.secrets import get_secret

ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
DEFAULT_OPTION_FEED = "indicative"
_SNAPSHOT_MAX_PAGES = 3


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def _alpaca_keys() -> Tuple[Optional[str], Optional[str]]:
    return get_secret("APCA_API_KEY_ID"), get_secret("APCA_API_SECRET_KEY")


def _alpaca_data_headers() -> Optional[Dict[str, str]]:
    key, secret = _alpaca_keys()
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


# --------------------------------------------------------------------------- #
# Daily closes (Alpaca stock bars, with graceful fallback)
# --------------------------------------------------------------------------- #
def _fetch_closes_alpaca(symbol: str, start: dt.datetime, *, feed: str | None = None) -> pd.DataFrame:
    """
    Daily closes via alpaca-py StockHistoricalDataClient.
    Returns a DataFrame with columns [Date, Close]; raises on any failure.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, secret = _alpaca_keys()
    if not key or not secret:
        raise EnvironmentError("Clés Alpaca absentes (APCA_API_KEY_ID / APCA_API_SECRET_KEY).")

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    kwargs: Dict[str, Any] = {
        "symbol_or_symbols": symbol,
        "timeframe": TimeFrame.Day,
        "start": start,
        # Stay clear of the free-plan "recent SIP data" restriction.
        "end": dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=16),
    }
    if feed:
        try:
            from alpaca.data.enums import DataFeed

            kwargs["feed"] = DataFeed(feed)
        except Exception:
            kwargs["feed"] = feed
    try:
        from alpaca.data.enums import Adjustment

        kwargs["adjustment"] = Adjustment.SPLIT
    except Exception:
        pass

    bars = client.get_stock_bars(StockBarsRequest(**kwargs))
    df = getattr(bars, "df", None)
    if df is None or df.empty:
        raise RuntimeError(f"Alpaca n'a retourné aucune barre daily pour {symbol}.")

    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(symbol, level="symbol")
        except Exception:
            df = df.reset_index().set_index("timestamp")
    df = df.reset_index()

    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    close_col = next((c for c in df.columns if str(c).lower() == "close"), None)
    if close_col is None:
        raise RuntimeError("Réponse Alpaca sans colonne close.")

    dates = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    out = pd.DataFrame(
        {
            "Date": dates.dt.tz_convert(None).dt.normalize(),
            "Close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"Barres Alpaca inexploitables pour {symbol}.")
    return out


def fetch_daily_closes(
    symbol: str,
    *,
    years: float = 2.0,
    extra_days: int = 60,
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Daily close history covering `years` (+ warm-up buffer `extra_days`).

    Order: Alpaca (default feed) -> Alpaca (IEX) -> app fallback (Stooq/Yahoo).
    Returns (df[Date, Close], source_tag, log_messages).
    """
    sym = (symbol or "").strip().upper()
    log: List[str] = []
    if not sym:
        return pd.DataFrame(), "none", ["Symbole vide."]

    lookback_days = int(float(years) * 365.25) + max(0, int(extra_days))
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=lookback_days)

    for feed, tag in ((None, "alpaca"), ("iex", "alpaca (iex)")):
        try:
            df = _fetch_closes_alpaca(sym, start, feed=feed)
            log.append(f"{len(df)} barres daily reçues via {tag}.")
            return df, tag, log
        except Exception as exc:  # noqa: BLE001 — defensive, fallback chain
            log.append(f"Barres {tag} indisponibles : {exc}")

    try:
        period = f"{max(1, int(math.ceil(float(years))) + 1)}y"
        df_fb = fetch_ohlc_history(sym, period=period, interval="1d")
        if df_fb is not None and not df_fb.empty and "Close" in df_fb.columns:
            out = df_fb[["Date", "Close"]].copy()
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
            out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
            out = out[out["Date"] >= pd.Timestamp(start.date())]
            if not out.empty:
                log.append(f"{len(out)} barres daily via fallback Stooq/Yahoo.")
                return out, "fallback (Stooq/Yahoo)", log
    except Exception as exc:  # noqa: BLE001
        log.append(f"Fallback Stooq/Yahoo en échec : {exc}")

    return pd.DataFrame(), "none", log


# --------------------------------------------------------------------------- #
# Current ATM implied vol (Alpaca option snapshots)
# --------------------------------------------------------------------------- #
def _decode_opra(opra: str) -> Tuple[Optional[float], Optional[dt.date], Optional[str]]:
    """SPY260918C00450000 -> (450.0, 2026-09-18, 'call')."""
    try:
        s = str(opra)
        expiry = dt.datetime.strptime(s[-15:-9], "%y%m%d").date()
        opt_type = "call" if s[-9].upper() == "C" else "put"
        strike = int(s[-8:]) / 1000.0
        return strike, expiry, opt_type
    except Exception:
        return None, None, None


def _first_not_none(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def _snapshot_iv(snap: Dict[str, Any]) -> float:
    greeks = _first_not_none(snap.get("greeks"), snap.get("latestGreeks")) or {}
    if not isinstance(greeks, dict):
        greeks = {}
    raw = _first_not_none(
        snap.get("impliedVolatility"),
        snap.get("implied_volatility"),
        snap.get("iv"),
        greeks.get("iv"),
        greeks.get("impliedVolatility"),
        greeks.get("implied_volatility"),
    )
    try:
        return float(raw) if raw is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _snapshot_mid(snap: Dict[str, Any]) -> Optional[float]:
    quote = snap.get("latestQuote") or snap.get("latest_quote") or {}
    if isinstance(quote, dict):
        try:
            bid = quote.get("bp")
            ask = quote.get("ap")
            bid_f = float(bid) if bid is not None else None
            ask_f = float(ask) if ask is not None else None
            if bid_f is not None and ask_f is not None and bid_f > 0 and ask_f > 0:
                return 0.5 * (bid_f + ask_f)
            if ask_f is not None and ask_f > 0:
                return ask_f
        except (TypeError, ValueError):
            pass
    trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
    if isinstance(trade, dict):
        try:
            px = trade.get("p") or trade.get("price")
            px_f = float(px) if px is not None else None
            if px_f is not None and px_f > 0:
                return px_f
        except (TypeError, ValueError):
            pass
    return None


def _fetch_atm_snapshots(
    symbol: str,
    *,
    feed: str,
    spot: Optional[float],
    dte_min: int,
    dte_max: int,
    strike_band: float = 0.10,
    timeout_sec: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Targeted Alpaca option-snapshot fetch using server-side expiry/strike filters.
    Returns {opra_symbol: snapshot}; raises on HTTP/credential failure.
    """
    headers = _alpaca_data_headers()
    if headers is None:
        raise EnvironmentError("Clés Alpaca absentes (APCA_API_KEY_ID / APCA_API_SECRET_KEY).")

    today = dt.date.today()
    params: Dict[str, Any] = {
        "feed": feed,
        "limit": 1000,
        "expiration_date_gte": (today + dt.timedelta(days=int(dte_min))).isoformat(),
        "expiration_date_lte": (today + dt.timedelta(days=int(dte_max))).isoformat(),
    }
    if spot is not None and np.isfinite(spot) and spot > 0:
        params["strike_price_gte"] = round(float(spot) * (1.0 - strike_band), 2)
        params["strike_price_lte"] = round(float(spot) * (1.0 + strike_band), 2)

    url = f"{ALPACA_DATA_BASE_URL}/v1beta1/options/snapshots/{sym_url(symbol)}"
    snapshots: Dict[str, Dict[str, Any]] = {}
    page_token: Optional[str] = None
    for _ in range(_SNAPSHOT_MAX_PAGES):
        if page_token:
            params["page_token"] = page_token
        resp = requests.get(url, headers=headers, params=params, timeout=timeout_sec)
        resp.raise_for_status()
        payload = resp.json() or {}
        page = payload.get("snapshots") or {}
        if isinstance(page, dict):
            for opra, snap in page.items():
                if isinstance(snap, dict):
                    snapshots[str(opra)] = snap
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return snapshots


def sym_url(symbol: str) -> str:
    from urllib.parse import quote_plus

    return quote_plus((symbol or "").strip().upper())


def _contracts_from_snapshots(snapshots: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for opra, snap in snapshots.items():
        strike, expiry, opt_type = _decode_opra(opra)
        if strike is None or expiry is None or opt_type is None:
            continue
        rows.append(
            {
                "opra": opra,
                "K": float(strike),
                "expiry": expiry,
                "type": opt_type,
                "iv": _snapshot_iv(snap),
                "mid": _snapshot_mid(snap),
            }
        )
    return rows


def _contracts_from_chain_df(df: pd.DataFrame, today: dt.date) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return rows
    for _, r in df.iterrows():
        try:
            expiry = today + dt.timedelta(days=int(round(float(r["T"]) * 365.0)))
            rows.append(
                {
                    "opra": str(r.get("opra") or ""),
                    "K": float(r["K"]),
                    "expiry": expiry,
                    "type": str(r["type"]).lower(),
                    "iv": float(r["iv"]) if pd.notna(r["iv"]) else float("nan"),
                    "mid": None,  # chain cache has no quotes -> no BS fallback
                }
            )
        except Exception:
            continue
    return rows


def fetch_current_atm_iv(
    symbol: str,
    *,
    target_dte: int = 30,
    dte_min: int = 15,
    dte_max: int = 60,
    moneyness_band: float = 0.05,
    feed: str | None = None,
    r_annual: float = 0.0,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Median ATM implied vol near `target_dte` from Alpaca option snapshots.

    Per contract: direct Alpaca IV when present, otherwise Black-Scholes
    inversion of the quote mid (puts converted to synthetic calls via
    put-call parity with q=0). Returns (info dict | None, log messages).
    """
    sym = (symbol or "").strip().upper()
    log: List[str] = []
    if not sym:
        return None, ["Symbole vide."]

    feed_val = (feed or os.getenv("ALPACA_OPTION_DATA_FEED") or DEFAULT_OPTION_FEED).strip()
    spot_raw = fetch_spot_price(sym)
    spot = float(spot_raw) if spot_raw is not None else None
    if spot is None or not np.isfinite(spot) or spot <= 0:
        log.append("Spot indisponible : impossible de sélectionner les strikes ATM.")
        return None, log

    today = dt.date.today()
    contracts: List[Dict[str, Any]] = []
    try:
        snaps = _fetch_atm_snapshots(
            sym, feed=feed_val, spot=spot, dte_min=dte_min, dte_max=dte_max
        )
        contracts = _contracts_from_snapshots(snaps)
        log.append(f"{len(contracts)} contrats candidats via snapshots filtrés (feed={feed_val}).")
    except Exception as exc:  # noqa: BLE001 — fallback to the cached full chain
        log.append(f"Snapshots filtrés indisponibles ({exc}) ; fallback chaîne complète.")
        try:
            from app.model.options.logic import download_options_alpaca

            chain = download_options_alpaca(sym, feed=feed_val, max_pages=_SNAPSHOT_MAX_PAGES)
            contracts = _contracts_from_chain_df(chain, today)
            log.append(f"{len(contracts)} contrats via chaîne Alpaca (cache).")
        except Exception as exc2:  # noqa: BLE001
            log.append(f"Chaîne d'options Alpaca en échec : {exc2}")
            return None, log

    usable = [
        c
        for c in contracts
        if c["expiry"] is not None and dte_min <= (c["expiry"] - today).days <= dte_max
    ]
    if not usable:
        log.append(f"Aucun contrat entre {dte_min} et {dte_max} jours d'échéance.")
        return None, log

    expiries = sorted({c["expiry"] for c in usable})
    best_expiry = min(expiries, key=lambda e: abs((e - today).days - int(target_dte)))
    dte = (best_expiry - today).days
    at_expiry = [c for c in usable if c["expiry"] == best_expiry]

    for band in (moneyness_band, max(moneyness_band, 0.10)):
        atm = [c for c in at_expiry if abs(c["K"] / spot - 1.0) <= band]
        if atm:
            break
    if not atm:
        atm = sorted(at_expiry, key=lambda c: abs(c["K"] - spot))[:4]

    T = max(dte, 1) / 365.0
    ivs: List[float] = []
    n_direct = 0
    n_inverted = 0
    for c in atm:
        iv = c.get("iv")
        if iv is not None and np.isfinite(iv) and 0.0 < iv < 5.0:
            ivs.append(float(iv))
            n_direct += 1
            continue
        mid = c.get("mid")
        if mid is None or not np.isfinite(mid) or mid <= 0:
            continue
        if c["type"] == "call":
            call_price = float(mid)
        else:  # put -> synthetic call via parity (q = 0)
            call_price = float(mid) + spot - c["K"] * math.exp(-r_annual * T)
        iv_bs = implied_vol_call(call_price, spot, c["K"], T, r_annual, 0.0)
        if iv_bs is not None and np.isfinite(iv_bs) and 0.0 < iv_bs < 5.0:
            ivs.append(float(iv_bs))
            n_inverted += 1

    if not ivs:
        log.append(
            "Aucune IV exploitable sur la tranche ATM "
            f"(feed={feed_val} : greeks absents et quotes inexploitables)."
        )
        return None, log

    if n_inverted == 0:
        method = "greeks Alpaca"
    elif n_direct == 0:
        method = "inversion Black-Scholes (mid)"
    else:
        method = "mixte (greeks + inversion BS)"

    info = {
        "iv": float(np.median(ivs)),
        "spot": spot,
        "expiry": best_expiry,
        "dte": int(dte),
        "n_contracts": int(len(ivs)),
        "method": method,
        "feed": feed_val,
    }
    log.append(
        f"IV ATM {info['iv']:.4f} ({info['iv'] * 100:.2f}%) — échéance {best_expiry} "
        f"({dte} j), {len(ivs)} contrats, méthode : {method}."
    )
    return info, log


# --------------------------------------------------------------------------- #
# Local daily IV history cache
# --------------------------------------------------------------------------- #
def _iv_history_path(symbol: str):
    sym = (symbol or "").strip().upper()
    return CACHE_IV_HISTORY_DIR / f"iv_daily_{sym}.csv"


def record_iv_observation(symbol: str, info: Dict[str, Any]) -> None:
    """Upsert today's ATM IV observation into cache/IVHistory/iv_daily_{SYM}.csv."""
    try:
        path = _iv_history_path(symbol)
        today = dt.date.today().isoformat()
        row = {
            "date": today,
            "iv": float(info.get("iv")),
            "dte": info.get("dte"),
            "n_contracts": info.get("n_contracts"),
            "method": info.get("method"),
            "spot": info.get("spot"),
        }
        if path.exists():
            df = pd.read_csv(path)
            df = df[df.get("date", pd.Series(dtype=str)).astype(str) != today]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df = df.sort_values("date")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as exc:  # noqa: BLE001 — cache is best-effort
        logging.warning(f"[iv-dashboard] écriture du cache IV impossible pour {symbol}: {exc}")


def load_iv_history(symbol: str) -> pd.DataFrame:
    """Locally accumulated daily IV observations (may be empty)."""
    path = _iv_history_path(symbol)
    if not path.exists():
        return pd.DataFrame(columns=["date", "iv"])
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
        return df.dropna(subset=["date", "iv"]).sort_values("date").reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["date", "iv"])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def get_iv_dashboard_data(
    symbol: str,
    *,
    years: float = 2.0,
    rv_window: int = analytics.DEFAULT_RV_WINDOW,
    forward_window: int = analytics.DEFAULT_FORWARD_WINDOW,
    percentile_window: int = analytics.DEFAULT_PERCENTILE_WINDOW,
    include_current_iv: bool = True,
) -> Dict[str, Any]:
    """
    Build the full payload for the 🌡️ Vol Implicite tab. Raises RuntimeError when
    no price history is available at all; other failures degrade into the payload.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("Symbole requis.")

    log: List[str] = []
    extra_days = int(int(rv_window) * 1.6) + 15
    closes_df, source, fetch_log = fetch_daily_closes(sym, years=years, extra_days=extra_days)
    log.extend(fetch_log)
    if closes_df is None or closes_df.empty:
        raise RuntimeError(
            f"Aucune donnée de prix disponible pour {sym} (Alpaca et fallback en échec)."
        )

    closes = closes_df.set_index("Date")["Close"]
    rv = analytics.compute_realized_vol(closes, int(rv_window))
    pct = analytics.compute_percentile_series(rv, int(percentile_window))

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(float(years) * 365.25))
    series_df = pd.DataFrame({"close": closes, "vol": rv, "vol_percentile": pct})
    series_df = series_df.dropna(subset=["vol"])
    series_df = series_df[series_df.index >= cutoff]
    if series_df.empty:
        raise RuntimeError(
            f"Série de volatilité vide pour {sym} — historique trop court "
            f"pour une fenêtre RV de {rv_window} jours."
        )

    current_vol = float(series_df["vol"].iloc[-1])
    current_pct = float(series_df["vol_percentile"].iloc[-1]) if pd.notna(
        series_df["vol_percentile"].iloc[-1]
    ) else float("nan")
    regime = analytics.classify_regime(current_pct)
    log.append(
        f"Vol réalisée courante ({rv_window} j, annualisée) : {current_vol:.4f} "
        f"({current_vol * 100:.2f}%)."
    )

    vol_stats = {
        "min": float(series_df["vol"].min()),
        "mean": float(series_df["vol"].mean()),
        "max": float(series_df["vol"].max()),
    }

    current_iv: Optional[Dict[str, Any]] = None
    iv_error: Optional[str] = None
    iv_vs_series_percentile = float("nan")
    iv_regime: Optional[Dict[str, str]] = None
    iv_minus_rv: Optional[float] = None
    if include_current_iv:
        current_iv, iv_log = fetch_current_atm_iv(sym)
        log.extend(iv_log)
        if current_iv is not None:
            record_iv_observation(sym, current_iv)
            trailing = series_df["vol"].tail(int(percentile_window))
            iv_vs_series_percentile = analytics.percentile_within(trailing, current_iv["iv"])
            iv_regime = analytics.classify_regime(iv_vs_series_percentile)
            iv_minus_rv = float(current_iv["iv"]) - current_vol
        else:
            iv_error = iv_log[-1] if iv_log else "IV indisponible."

    iv_history = load_iv_history(sym)

    analysis: Optional[Dict[str, Any]] = None
    analysis_error: Optional[str] = None
    try:
        analysis = analytics.analyze_forward_vol(
            series_df["vol"],
            forward_window=int(forward_window),
            percentile=series_df["vol_percentile"],
        )
    except ValueError as exc:
        analysis_error = str(exc)
        log.append(f"Analyse forward vol impossible : {exc}")

    return {
        "symbol": sym,
        "source": source,
        "years": float(years),
        "rv_window": int(rv_window),
        "forward_window": int(forward_window),
        "percentile_window": int(percentile_window),
        "series": series_df,
        "current_vol": current_vol,
        "current_percentile": current_pct,
        "regime": regime,
        "vol_stats": vol_stats,
        "current_iv": current_iv,
        "iv_error": iv_error,
        "iv_vs_series_percentile": iv_vs_series_percentile,
        "iv_regime": iv_regime,
        "iv_minus_rv": iv_minus_rv,
        "iv_history": iv_history,
        "analysis": analysis,
        "analysis_error": analysis_error,
        "log": log,
    }


__all__ = [
    "fetch_daily_closes",
    "fetch_current_atm_iv",
    "record_iv_observation",
    "load_iv_history",
    "get_iv_dashboard_data",
]
