"""p4 skeptic probe — snapshot-page-cap-silent: does the 3-page cap actually reach
the DISPLAYED number on a SPY-like chain?

Runs the real fetch_current_atm_iv with:
  - requests.get mocked: paginated (limit=1000), ascending OPRA order, honours the
    server-side expiry/strike filters, no greeks (inversion path), two-sided quotes
    priced with BS (r=4%, q=1.3%, sigma=16%).
  - fetch_spot_price mocked -> 640.
  - dt.date.today() patched so we can sweep 'today' over several weeks.
SPY-like calendar: daily expiries (Mon-Fri) for N_DAILY_WEEKS weeks, then Mon/Wed/Fri,
plus monthly 3rd Fridays. $1 strikes in the +-10% band, calls + puts.
Reports: contracts in filter, cap hit?, best expiry, whether that expiry was cut,
calls/puts in ATM set, displayed IV bias vs 16%, and whether the log mentions it.
"""
import datetime as dt
import math
import sys
import types
from typing import Dict, List

sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
from app.model.calibration.implied_vol import bs_call_price  # noqa: E402
from app.model.iv_dashboard import service  # noqa: E402

S, R, Q, SIG = 640.0, 0.04, 0.013, 0.16
N_DAILY_WEEKS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def calendar(today: dt.date) -> List[dt.date]:
    out = set()
    for d in range(1, 80):
        day = today + dt.timedelta(days=d)
        wd = day.weekday()
        if wd >= 5:
            continue
        if d <= 7 * N_DAILY_WEEKS:
            out.add(day)
        elif wd in (0, 2, 4):
            out.add(day)
        if wd == 4 and 15 <= day.day <= 21:
            out.add(day)
    return sorted(out)


def build_snapshots(today: dt.date, params: Dict) -> Dict[str, Dict]:
    snaps = {}
    e_gte = dt.date.fromisoformat(params["expiration_date_gte"])
    e_lte = dt.date.fromisoformat(params["expiration_date_lte"])
    k_gte, k_lte = params["strike_price_gte"], params["strike_price_lte"]
    for exp in calendar(today):
        if not (e_gte <= exp <= e_lte):
            continue
        T = (exp - today).days / 365.0
        for K in range(int(math.ceil(k_gte)), int(math.floor(k_lte)) + 1):
            c = bs_call_price(S, K, T, R, Q, SIG)
            p = c - S * math.exp(-Q * T) + K * math.exp(-R * T)
            for typ, px in (("C", c), ("P", p)):
                opra = f"SPY{exp:%y%m%d}{typ}{int(K * 1000):08d}"
                snaps[opra] = {"latestQuote": {"bp": max(px - 0.05, 0.01), "ap": px + 0.05}}
    return dict(sorted(snaps.items()))  # ascending OPRA order (assumed Alpaca order)


class FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def run_for(today: dt.date):
    full = {}
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        nonlocal full
        if not full:
            full = build_snapshots(today, params)
        keys = list(full.keys())
        start = int(params.get("page_token") or 0)
        lim = int(params["limit"])
        page = {k: full[k] for k in keys[start:start + lim]}
        nxt = str(start + lim) if start + lim < len(keys) else None
        calls.append(len(page))
        return FakeResp({"snapshots": page, "next_page_token": nxt})

    class FakeDate(dt.date):
        @classmethod
        def today(cls):
            return cls(today.year, today.month, today.day)

    shim = types.SimpleNamespace(date=FakeDate, timedelta=dt.timedelta, datetime=dt.datetime, timezone=dt.timezone)
    saved = (service.requests.get, service.fetch_spot_price, service.dt, service._alpaca_data_headers)
    service.requests.get = fake_get
    service.fetch_spot_price = lambda sym: S
    service.dt = shim
    service._alpaca_data_headers = lambda: {"APCA-API-KEY-ID": "x", "APCA-API-SECRET-KEY": "y"}
    try:
        info, log = service.fetch_current_atm_iv("SPY")
    finally:
        service.requests.get, service.fetch_spot_price, service.dt, service._alpaca_data_headers = saved
    if not info:
        print("   log:", log)

    fetched = sum(calls)
    n_total = len(full)
    cap_hit = fetched < n_total
    # composition of what the service actually saw for best expiry
    best = info["expiry"] if info else None
    seen_keys = list(full.keys())[:fetched]
    seen_best = [k for k in seen_keys if best and k[3:9] == f"{best:%y%m%d}"]
    all_best = [k for k in full if best and k[3:9] == f"{best:%y%m%d}"]
    n_c = sum(1 for k in seen_best if k[9] == "C")
    n_p = sum(1 for k in seen_best if k[9] == "P")
    atm_seen_c = sum(1 for k in seen_best if k[9] == "C" and abs(int(k[10:]) / 1000 / S - 1) <= 0.05)
    atm_seen_p = sum(1 for k in seen_best if k[9] == "P" and abs(int(k[10:]) / 1000 / S - 1) <= 0.05)
    trunc_word = any(("tronqu" in m.lower()) or ("pagination" in m.lower()) for m in log)
    bias_bp = (info["iv"] - SIG) * 1e4 if info else float("nan")
    print(
        f"today={today} in_filter={n_total:5d} fetched={fetched:4d} cap_hit={str(cap_hit):5s} "
        f"best={best} ({info['dte'] if info else '-'}d) best_cut={len(seen_best) < len(all_best)!s:5s} "
        f"atm C/P={atm_seen_c:3d}/{atm_seen_p:3d} n_used={info['n_contracts'] if info else 0:3d} "
        f"IV bias={bias_bp:+6.0f} bp log_mentions_trunc={trunc_word}"
    )
    return cap_hit, bias_bp


if __name__ == "__main__":
    print(f"N_DAILY_WEEKS={N_DAILY_WEEKS}  (_SNAPSHOT_MAX_PAGES={service._SNAPSHOT_MAX_PAGES} x limit 1000)")
    base = dt.date(2026, 8, 21)
    worst = 0.0
    hits = 0
    n = 0
    for k in range(0, 42):
        d = base + dt.timedelta(days=k)
        if d.weekday() >= 5:
            continue
        cap_hit, b = run_for(d)
        n += 1
        hits += cap_hit
        if abs(b) > abs(worst):
            worst = b
    print(f"summary: cap hit on {hits}/{n} trading days; worst displayed-IV bias {worst:+.0f} bp")
