"""p4 skeptic repro — local-date-vs-exchange-date.

Offline, deterministic. Drives record_iv_observation() with a controlled clock
(a FakeDate whose today() is the *local* date of a fixed UTC instant in a given
IANA zone) and measures how many cache rows two distinct US sessions produce.
Then reproduces the DTE / T arithmetic of fetch_current_atm_iv (service.py
L389-L398) to measure the IV bias of the one-day-short T on the BS inversion path.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sys
import tempfile
import types
from zoneinfo import ZoneInfo

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model.calibration.implied_vol import bs_call_price, implied_vol_call  # noqa: E402
from app.model.iv_dashboard import service  # noqa: E402

NY = ZoneInfo("America/New_York")
INFO = {"iv": 0.16, "dte": 30, "n_contracts": 8, "method": "greeks Alpaca", "spot": 645.0}

# Two distinct US sessions: Aug-20 15:30 ET (live) and Aug-21 10:00 ET (live).
SESSIONS = [
    dt.datetime(2026, 8, 20, 15, 30, tzinfo=NY),
    dt.datetime(2026, 8, 21, 10, 0, tzinfo=NY),
]


def make_fake_dt(now_utc: dt.datetime, local_zone: ZoneInfo):
    """Module shim for `service.dt` whose date.today() follows a fake local clock."""
    local_date = now_utc.astimezone(local_zone).date()

    class FakeDate(dt.date):
        @classmethod
        def today(cls):
            return cls(local_date.year, local_date.month, local_date.day)

    shim = types.ModuleType("dt_shim")
    shim.date = FakeDate
    shim.datetime = dt.datetime
    shim.timedelta = dt.timedelta
    return shim


def run_zone(zone_name: str):
    zone = ZoneInfo(zone_name)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="p4_tz_"))
    service.CACHE_IV_HISTORY_DIR = tmp
    real_dt = service.dt
    keys = []
    try:
        for s in SESSIONS:
            service.dt = make_fake_dt(s.astimezone(dt.timezone.utc), zone)
            service.record_iv_observation("SPY", {**INFO, "iv": 0.16 + 0.01 * len(keys)})
            keys.append((s.isoformat(), s.astimezone(zone).strftime("%Y-%m-%d %H:%M"), service.dt.date.today().isoformat()))
    finally:
        service.dt = real_dt
    hist = service.load_iv_history("SPY")
    print(f"[{zone_name}]")
    for et, loc, key in keys:
        print(f"  session {et} | local {loc} | cache key = {key}")
    print(f"  rows on disk = {len(hist)}  (distinct US sessions = {len(SESSIONS)})  ivs kept = {hist['iv'].tolist()}")
    return len(hist)


def dte_bias(local_zone_name: str):
    """Replicates L389-L398: dte = (best_expiry - today).days ; T = max(dte,1)/365."""
    zone = ZoneInfo(local_zone_name)
    instant = SESSIONS[0]                                  # Aug-20 15:30 ET
    today_local = instant.astimezone(zone).date()
    today_exch = instant.astimezone(NY).date()
    expiry = dt.date(2026, 9, 18)
    dte_local = (expiry - today_local).days
    dte_exch = (expiry - today_exch).days
    T_local = max(dte_local, 1) / 365.0
    T_exch = max(dte_exch, 1) / 365.0
    S, K, r, sigma = 645.0, 645.0, 0.045, 0.16
    price = bs_call_price(S, K, T_exch, r, 0.0, sigma)    # market price consistent with exchange DTE
    iv_local = implied_vol_call(price, S, K, T_local, r, 0.0)
    print(f"[DTE/T bias, local zone {local_zone_name}] exchange date {today_exch} dte={dte_exch} | "
          f"local date {today_local} dte={dte_local}")
    print(f"  true sigma 16.000% -> inverted with local T: {iv_local*100:.3f}%  (bias {(iv_local-sigma)*1e4:+.1f} bp)")


if __name__ == "__main__":
    import time
    print(f"host local tz = {time.tzname} (utcoffset {-time.timezone/3600:+.0f}h); "
          f"host dt.date.today()={dt.date.today()} ; NY date now={dt.datetime.now(NY).date()}")
    # Hours per day where the host-local date != New-York date (UTC+8 vs EDT = 12h)
    off_local = dt.datetime.now().astimezone().utcoffset()
    off_ny = dt.datetime.now(NY).utcoffset()
    print(f"host/NY offset difference = {(off_local - off_ny).total_seconds()/3600:+.0f} h "
          f"-> local date differs from NY date that many hours per day")
    for z in ("Asia/Singapore", "Asia/Makassar", "Europe/Paris", "America/New_York"):
        run_zone(z)
    dte_bias("Asia/Singapore")
    dte_bias("Europe/Paris")
