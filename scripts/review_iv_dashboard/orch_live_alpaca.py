"""
Orchestrator probe — §4.6 live Alpaca validation (READ-ONLY data calls, never an order).

Loads APCA_* paper keys from an external .env path passed as argv[1] into the
process environment only (never copied, never printed). Measures:
  (a) filtered snapshot fetch for SPY: are expiry/strike filters honored? pages/latency
  (b) greeks/IV presence on the `indicative` feed today
  (c) daily bars via default feed vs IEX with end = now-16min
  (d) end-to-end get_iv_dashboard_data("SPY") vs an independent ATM IV estimate
Run: .venv/Scripts/python.exe scripts/review_iv_dashboard/orch_live_alpaca.py <path/to/.env>
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_keys(env_path: str) -> None:
    for line in Path(env_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.startswith("APCA_") and k not in os.environ:
            os.environ[k] = v


def redact(s: str) -> str:
    for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        v = os.environ.get(k)
        if v:
            s = s.replace(v, "***")
    return s


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: orch_live_alpaca.py <path/to/.env>")
        sys.exit(2)
    load_keys(sys.argv[1])
    if not os.environ.get("APCA_API_KEY_ID"):
        print("NO KEYS -> skip")
        sys.exit(0)
    print("keys loaded: yes (redacted) | base url is paper:", "paper" in os.environ.get("APCA_API_BASE_URL", ""))

    import numpy as np
    import requests
    from scipy.stats import norm

    from app.model.iv_dashboard import service as svc

    headers = svc._alpaca_data_headers()
    sym = "SPY"
    today = dt.date.today()
    print("today (local):", today, "| utc now:", dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"))

    # ------------------------------------------------------------------ (a)
    print("\n=== (a) filtered snapshot fetch: are server-side filters honored? ===")
    spot = svc.fetch_spot_price(sym)
    print("spot via fetch_spot_price:", spot)
    url = f"{svc.ALPACA_DATA_BASE_URL}/v1beta1/options/snapshots/{sym}"
    exp_gte = (today + dt.timedelta(days=15)).isoformat()
    exp_lte = (today + dt.timedelta(days=60)).isoformat()
    params = {"feed": "indicative", "limit": 1000, "expiration_date_gte": exp_gte, "expiration_date_lte": exp_lte}
    if spot:
        params["strike_price_gte"] = round(spot * 0.9, 2)
        params["strike_price_lte"] = round(spot * 1.1, 2)
    pages, n_total, t0 = 0, 0, time.perf_counter()
    token = None
    keys_all = []
    status = None
    while pages < 10:
        p = dict(params)
        if token:
            p["page_token"] = token
        r = requests.get(url, headers=headers, params=p, timeout=20)
        status = r.status_code
        if r.status_code != 200:
            print("HTTP", r.status_code, redact(r.text[:200]))
            break
        payload = r.json() or {}
        snaps = payload.get("snapshots") or {}
        pages += 1
        n_total += len(snaps)
        keys_all.extend(snaps.keys())
        token = payload.get("next_page_token")
        if not token:
            break
    lat = time.perf_counter() - t0
    print(f"filtered: status={status} pages={pages} contracts={n_total} latency={lat:.2f}s")
    if keys_all:
        dec = [svc._decode_opra(k) for k in keys_all]
        exps = sorted({d[1] for d in dec if d[1]})
        strikes = sorted({d[0] for d in dec if d[0]})
        print(f"  expiries in result: {exps[0]} .. {exps[-1]} (n={len(exps)}) | requested {exp_gte}..{exp_lte}")
        print(f"  strikes in result: {strikes[0]} .. {strikes[-1]} (n={len(strikes)}) | requested {params.get('strike_price_gte')}..{params.get('strike_price_lte')}")
        out_exp = [e for e in exps if not (dt.date.fromisoformat(exp_gte) <= e <= dt.date.fromisoformat(exp_lte))]
        out_k = [k for k in strikes if spot and not (params['strike_price_gte'] <= k <= params['strike_price_lte'])]
        print(f"  VIOLATIONS: expiries outside window={len(out_exp)} strikes outside band={len(out_k)}")
        print("  -> filters honored:", not out_exp and not out_k)
    # Unfiltered single page for contrast
    t0 = time.perf_counter()
    r = requests.get(url, headers=headers, params={"feed": "indicative", "limit": 1000}, timeout=20)
    lat = time.perf_counter() - t0
    if r.status_code == 200:
        snaps = (r.json() or {}).get("snapshots") or {}
        dec = [svc._decode_opra(k) for k in snaps]
        exps = sorted({d[1] for d in dec if d[1]})
        print(f"unfiltered page 1: contracts={len(snaps)} next_token={'yes' if (r.json() or {}).get('next_page_token') else 'no'} latency={lat:.2f}s expiries {exps[0] if exps else None}..{exps[-1] if exps else None}")
    else:
        print("unfiltered: HTTP", r.status_code, redact(r.text[:200]))
    # Sanity: a deliberately bogus filter name — is it rejected (400) or silently ignored?
    r = requests.get(url, headers=headers, params={"feed": "indicative", "limit": 5, "bogus_filter_xyz": "1"}, timeout=20)
    print(f"bogus param probe: HTTP {r.status_code} (400 => unknown params rejected; 200 => unknown params silently ignored)")
    # OPRA feed probe (403 expected on free plan)
    r = requests.get(url, headers=headers, params={"feed": "opra", "limit": 5}, timeout=20)
    print(f"feed=opra probe: HTTP {r.status_code} {redact(r.text[:120]) if r.status_code != 200 else ''}")

    # ------------------------------------------------------------------ (b)
    print("\n=== (b) greeks / IV presence on the indicative feed today ===")
    try:
        snaps = svc._fetch_atm_snapshots(sym, feed="indicative", spot=spot, dte_min=15, dte_max=60)
        rows = svc._contracts_from_snapshots(snaps)
        n = len(rows)
        n_iv = sum(1 for c in rows if np.isfinite(c["iv"]) and c["iv"] > 0)
        n_mid = sum(1 for c in rows if c["mid"] is not None)
        sample = next(iter(snaps.values())) if snaps else {}
        print(f"contracts={n} with_direct_iv={n_iv} with_mid={n_mid} | snapshot keys: {sorted(sample.keys())}")
        g = sample.get("greeks") or {}
        print("  sample greeks keys:", sorted(g.keys()) if isinstance(g, dict) else g, "| impliedVolatility:", sample.get("impliedVolatility"))
        q = sample.get("latestQuote") or {}
        print("  sample quote:", {k: q.get(k) for k in ("bp", "ap", "bs", "as", "t")})
        # crossed / zero-bid stats
        crossed = sum(1 for s in snaps.values() if (s.get("latestQuote") or {}).get("bp", 0) and (s.get("latestQuote") or {}).get("ap", 0) and s["latestQuote"]["bp"] > s["latestQuote"]["ap"])
        zero_bid = sum(1 for s in snaps.values() if not (s.get("latestQuote") or {}).get("bp"))
        print(f"  crossed quotes={crossed} zero/missing bid={zero_bid} of {len(snaps)}")
    except Exception as exc:  # noqa: BLE001
        print("snapshot fetch failed:", redact(str(exc))[:300])

    # ------------------------------------------------------------------ (c)
    print("\n=== (c) daily bars: default feed vs IEX, end = now - 16 min ===")
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=40)
    for feed in (None, "iex", "sip"):
        try:
            t0 = time.perf_counter()
            df = svc._fetch_closes_alpaca(sym, start, feed=feed)
            print(f"feed={feed!s:5}: OK rows={len(df)} last date={df['Date'].iloc[-1].date()} last close={df['Close'].iloc[-1]:.2f} ({time.perf_counter()-t0:.2f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"feed={feed!s:5}: FAIL {redact(str(exc))[:160]}")
    # without the -16min dodge
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        k, s = svc._alpaca_keys()
        cl = StockHistoricalDataClient(api_key=k, secret_key=s)
        bars = cl.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day, start=start))
        df = bars.df
        print(f"no-end param (default feed): rows={len(df)} last ts={df.index.get_level_values('timestamp')[-1]}")
    except Exception as exc:  # noqa: BLE001
        print("no-end param: FAIL", redact(str(exc))[:160])

    # ------------------------------------------------------------------ (d)
    print("\n=== (d) end-to-end get_iv_dashboard_data('SPY') vs independent ATM IV ===")
    t0 = time.perf_counter()
    res = svc.get_iv_dashboard_data(sym, years=2.0)
    print(f"e2e {time.perf_counter()-t0:.1f}s | source={res['source']} points={len(res['series'])} current RV={res['current_vol']:.4f} pct={res['current_percentile']:.3f} regime={res['regime']['label']}")
    civ = res["current_iv"]
    print("current_iv:", None if civ is None else {k: (str(v) if k == 'expiry' else v) for k, v in civ.items()})
    print("iv_vs_series_percentile:", res["iv_vs_series_percentile"], "| iv_regime:", res["iv_regime"] and res["iv_regime"]["label"], "| iv_minus_rv:", res["iv_minus_rv"])
    print("log:")
    for ln in res["log"]:
        print("   -", redact(ln))
    print("analysis_error:", res["analysis_error"])
    if res["analysis"]:
        a = res["analysis"]
        print(f"reg_forward slope={a['reg_forward']['slope']:.3f} r2={a['reg_forward']['r2']:.3f} n={a['reg_forward']['n']} intersection={a['intersection']:.4f} n_high={a['n_high']} n_low={a['n_low']}")

    # Independent estimate: invert call AND put at the nearest-ATM strike of the chosen expiry with r=4%, q=1.3%
    if civ is not None:
        try:
            snaps = svc._fetch_atm_snapshots(sym, feed="indicative", spot=civ["spot"], dte_min=15, dte_max=60)
            rows = [c for c in svc._contracts_from_snapshots(snaps) if c["expiry"] == civ["expiry"]]
            S = float(civ["spot"])
            T = civ["dte"] / 365.0
            r, q = 0.04, 0.013

            def bs(K, sig, kind):
                d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
                d2 = d1 - sig * math.sqrt(T)
                if kind == "call":
                    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

            def inv(price, K, kind):
                lo, hi = 0.01, 3.0
                if bs(K, lo, kind) > price or bs(K, hi, kind) < price:
                    return float("nan")
                for _ in range(80):
                    m = 0.5 * (lo + hi)
                    if bs(K, m, kind) > price:
                        hi = m
                    else:
                        lo = m
                return 0.5 * (lo + hi)

            Ks = sorted({c["K"] for c in rows}, key=lambda k: abs(k - S))[:2]
            est = []
            for K in Ks:
                for c in rows:
                    if c["K"] == K and c["mid"]:
                        est.append((K, c["type"], c["mid"], c["iv"], inv(c["mid"], K, c["type"])))
            print("independent inversion (r=4%, q=1.3%) at nearest strikes:")
            for K, kind, mid, iv_feed, iv_ind in est:
                print(f"   K={K} {kind:4} mid={mid:.2f} feed_iv={iv_feed:.4f} indep_iv={iv_ind:.4f}")
            vals = [e[4] for e in est if np.isfinite(e[4])]
            if vals:
                ref = float(np.median(vals))
                print(f"   -> independent ATM IV median = {ref:.4f} vs tab headline {civ['iv']:.4f} : diff = {(civ['iv']-ref)*100:+.2f} vol pts (gate ±3)")
        except Exception as exc:  # noqa: BLE001
            print("independent estimate failed:", redact(str(exc))[:200])

    # cache file written?
    p = svc._iv_history_path(sym)
    print("\ncache file:", p, "exists:", p.exists())
    if p.exists():
        print(p.read_text(encoding="utf-8")[-400:])


if __name__ == "__main__":
    main()
