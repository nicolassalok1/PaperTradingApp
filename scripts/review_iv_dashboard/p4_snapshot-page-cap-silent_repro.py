"""p4 repro — snapshot-page-cap-silent.

Mock requests.get inside service with a paginated synthetic SPY-like chain that honours the
server-side filters sent by _fetch_atm_snapshots, then run the REAL fetch_current_atm_iv and
observe: number of HTTP calls, whether the log mentions truncation, and which expiry / how
many contracts survive under ascending, descending and shuffled server ordering.
"""
from __future__ import annotations
import datetime as dt, random, sys
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
from app.model.iv_dashboard import service as svc

S = 640.0
TODAY = dt.date.today()

# SPY-like listing: Mon/Wed/Fri expiries for the next 90 days, $1 strikes from -12% to +12%
expiries = [TODAY + dt.timedelta(days=d) for d in range(1, 91) if (TODAY + dt.timedelta(days=d)).weekday() in (0, 2, 4)]
strikes = list(range(int(S * 0.88), int(S * 1.12) + 1))
listing = []
for e in expiries:
    for K in strikes:
        for t in "CP":
            listing.append((f"SPY{e:%y%m%d}{t}{K*1000:08d}", e, K))
print(f"full listing: {len(listing)} contracts, {len(expiries)} expiries")

class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p

def make_server(order: str):
    calls = []
    def fake_get(url, headers=None, params=None, timeout=None):
        p = params or {}
        gte = dt.date.fromisoformat(p["expiration_date_gte"]); lte = dt.date.fromisoformat(p["expiration_date_lte"])
        kmin = p.get("strike_price_gte", -1e9); kmax = p.get("strike_price_lte", 1e9)
        rows = [r for r in listing if gte <= r[1] <= lte and kmin <= r[2] <= kmax]
        if order == "asc": rows.sort(key=lambda r: r[0])
        elif order == "desc": rows.sort(key=lambda r: r[0], reverse=True)
        else: random.Random(42).shuffle(rows)
        limit = int(p.get("limit", 100))
        start = int(p.get("page_token") or 0)
        page = rows[start:start + limit]
        calls.append((len(rows), limit, start))
        snaps = {}
        for opra, e, K in page:
            # two-sided quote around a plausible price (only needs to invert to *some* IV)
            dte = (e - TODAY).days
            px = max(0.05, 0.4 * S * 0.16 * (dte / 365.0) ** 0.5 * 0.8 - 0.5 * abs(K - S) + 5.0)
            snaps[opra] = {"latestQuote": {"bp": px - 0.05, "ap": px + 0.05}}
        nxt = str(start + limit) if start + limit < len(rows) else None
        return _Resp({"snapshots": snaps, "next_page_token": nxt})
    return fake_get, calls

svc._alpaca_data_headers = lambda: {"APCA-API-KEY-ID": "x", "APCA-API-SECRET-KEY": "y"}
svc.fetch_spot_price = lambda sym: S

for order in ("asc", "desc", "shuffled"):
    fake_get, calls = make_server(order)
    svc.requests.get = fake_get
    info, log = svc.fetch_current_atm_iv("SPY")
    n_filtered = calls[0][0]
    fetched = sum(min(c[1], c[0] - c[2]) for c in calls)
    print(f"\n-- server order: {order} --")
    print(f"  contracts matching server filters: {n_filtered}; HTTP calls: {len(calls)}; fetched: {fetched}; cap hit: {fetched < n_filtered}")
    print(f"  log mentions truncation/pagination: {any(w in ' '.join(log).lower() for w in ('tronqu', 'page', 'pagination', 'cap', 'limite'))}")
    for line in log: print("   log:", line)
    if info:
        # which expiries were actually available among fetched contracts?
        print(f"  result: expiry={info['expiry']} dte={info['dte']} n_contracts={info['n_contracts']}")

# how many contracts would the finder's "tightened" filters return? (+-5% band, same 15-60 window)
n5 = sum(1 for r in listing if 15 <= (r[1] - TODAY).days <= 60 and abs(r[2] / S - 1) <= 0.05)
n5_20 = sum(1 for r in listing if 20 <= (r[1] - TODAY).days <= 40 and abs(r[2] / S - 1) <= 0.05)
print(f"\n+-5% band, 15-60 DTE: {n5} contracts (one page of 1000 suffices: {n5 <= 1000})")
print(f"+-5% band, 20-40 DTE: {n5_20} contracts (one page suffices: {n5_20 <= 1000})")
