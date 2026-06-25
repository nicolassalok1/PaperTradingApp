#!/usr/bin/env python3
"""
SPX-VIX Portfolio Allocation — Python CLI engine (fallback / reference).

Numerically identical to the TypeScript engine in ../engine. Use this if your
app has a Python backend, or call it from a Node route via child_process.

Examples
--------
  # JSON metrics from the supplied CSV (default), printed to stdout
  python engine.py --source csv --csv spx_vix_daily.csv

  # JSON incl. weekly-sampled series for charting
  python engine.py --csv spx_vix_daily.csv --series --sample 5

  # Pull from Yahoo and also save the figure
  python engine.py --source yahoo --fig nav_and_weights.png

  # Human-readable table
  python engine.py --csv spx_vix_daily.csv --format table

Output JSON shape: {"metrics": {...}, "series": [{date,nav,wSpx,wVix,gross}, ...]}
(series is [] unless --series is passed).
"""
import argparse, json, sys
import numpy as np
import pandas as pd

CFG = dict(nav0=10_000_000.0, lookback=252, vol_target=0.10, var_conf=0.95,
           var_limit=0.025, gross_cap=1.50, name_cap=1.00, tc=1e-4, ann=252)


# ------------------------------- data --------------------------------------
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    dcol = next(c for c in df.columns if c.lower().startswith("date"))
    df[dcol] = pd.to_datetime(df[dcol])
    df = df.set_index(dcol).rename(columns={c: c.upper() for c in df.columns})
    return df[["SPX", "VIX"]].astype(float).sort_index().dropna()


def load_yahoo(start="1990-01-01"):
    import yfinance as yf
    out = {}
    for tk, name in [("^GSPC", "SPX"), ("^VIX", "VIX")]:
        d = yf.download(tk, start=start, auto_adjust=False, progress=False, threads=False)
        s = d["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        out[name] = s.astype(float)
    return pd.DataFrame(out).sort_index().dropna()


# ------------------------------- sizing ------------------------------------
def decide(window, c):
    cov = np.cov(window, rowvar=False)
    vol = np.sqrt(np.clip(np.diag(cov), 0, None))
    inv = np.where(vol > 1e-12, 1.0 / vol, 0.0)
    b = inv / inv.sum()
    sig = float(np.sqrt(max(b @ cov @ b, 1e-18)) * np.sqrt(c["ann"]))
    k_vol = c["vol_target"] / sig if sig > 0 else np.inf
    pnl = window @ b
    var_b = max(-np.quantile(pnl, 1 - c["var_conf"]), 0.0)
    k_var = c["var_limit"] / var_b if var_b > 1e-12 else np.inf
    k_gross = c["gross_cap"] / np.abs(b).sum()
    k_name = c["name_cap"] / np.abs(b).max()
    ceil = {"vol_target": k_vol, "gross": k_gross, "per_name": k_name, "VaR": k_var}
    bind = min(ceil, key=ceil.get)
    k = ceil[bind]
    w = k * b
    return w, bind, float(np.abs(w).sum())


def spx_only(R, c):
    L = c["lookback"]; wp = 0.0; nav = c["nav0"]; r = []
    for t in range(L - 1, len(R)):
        if t > L - 1:
            g = wp * R[t, 0]; nav *= 1 + g; r.append(g)
        win = R[t - L + 1:t + 1, 0]
        sig = win.std(ddof=1) * np.sqrt(c["ann"])
        wp = min(c["vol_target"] / sig, c["name_cap"], c["gross_cap"])
    r = np.array(r); vol = r.std(ddof=1) * np.sqrt(c["ann"])
    return dict(cagr=float((nav / c["nav0"]) ** (c["ann"] / len(r)) - 1),
                vol=float(vol), sharpe=float(r.mean() * c["ann"] / vol))


# ------------------------------ backtest -----------------------------------
def backtest(prices, c, sample=1):
    rets = prices.pct_change().dropna()
    R = rets.values; dates = rets.index; T = len(R); L = c["lookback"]

    vols = rets.std() * np.sqrt(c["ann"])
    corr = float(rets["SPX"].corr(rets["VIX"]))

    wp = np.zeros(2); nav = c["nav0"]; nav_nc = c["nav0"]
    rec = []
    for t in range(L - 1, T):
        gret = 0.0; wd = np.zeros(2)
        if t > L - 1:
            gret = float(wp @ R[t]); nav *= 1 + gret; nav_nc *= 1 + gret
            wd = wp * (1 + R[t]) / (1 + gret)
        w, bind, gross = decide(R[t - L + 1:t + 1], c)
        turn = float(np.abs(w - wd).sum()); cost = c["tc"] * turn
        nav *= 1 - cost
        net = (1 + gret) * (1 - cost) - 1
        rec.append((dates[t], nav, net, w[0], w[1], gross, turn, bind))
        wp = w

    df = pd.DataFrame(rec, columns=["date", "nav", "net", "wSpx", "wVix",
                                    "gross", "turn", "bind"]).iloc[1:]
    r = df["net"].values; navv = df["nav"].values; n = len(r)
    dd = navv / np.maximum.accumulate(navv) - 1
    metrics = dict(
        sampleStart=str(df["date"].iloc[0].date()),
        sampleEnd=str(df["date"].iloc[-1].date()),
        tradingDays=int(n),
        annReturnCagr=float((navv[-1] / c["nav0"]) ** (c["ann"] / n) - 1),
        realisedVol=float(r.std(ddof=1) * np.sqrt(c["ann"])),
        volTarget=c["vol_target"],
        sharpeRf0=float(r.mean() * c["ann"] / (r.std(ddof=1) * np.sqrt(c["ann"]))),
        maxDrawdown=float(dd.min()),
        realisedVar95_1d=float(-np.quantile(r, 1 - c["var_conf"])),
        varLimit=c["var_limit"],
        avgGrossExposure=float(df["gross"].mean()),
        maxNameWeight=float(np.maximum(df["wSpx"].abs(), df["wVix"].abs()).max()),
        avgDailyTurnover=float(df["turn"].mean()),
        annualisedTurnover=float(df["turn"].mean() * c["ann"]),
        annualisedCostDrag=float((nav_nc / c["nav0"]) ** (c["ann"] / n) - 1
                                 - (navv[-1] / c["nav0"]) ** (c["ann"] / n) + 1),
        totalCostDragUsd=float(nav_nc - navv[-1]),
        finalNavUsd=float(navv[-1]),
        corrSpxVixDaily=corr,
        annVolSpx=float(vols["SPX"]),
        annVolVix=float(vols["VIX"]),
        bindingShare={k: round(v, 4) for k, v in
                      df["bind"].value_counts(normalize=True).items()},
        benchmarkSpxOnly=spx_only(R, c),
    )

    step = max(1, sample)
    sdf = df.iloc[::step]
    series = [dict(date=str(d.date()), nav=float(nv), wSpx=float(a),
                   wVix=float(b), gross=float(g))
             for d, nv, a, b, g in zip(sdf["date"], sdf["nav"], sdf["wSpx"],
                                       sdf["wVix"], sdf["gross"])]
    return metrics, series, df


# ------------------------------- figure ------------------------------------
def save_fig(df, c, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    a1.plot(df["date"], df["nav"] / 1e6, color="#0b3d2e", lw=1.2)
    a1.set_yscale("log"); a1.set_ylabel("NAV ($mm, log)"); a1.grid(alpha=0.25)
    a1.set_title("SPX-VIX risk-targeted portfolio — NAV & weights", loc="left", weight="bold")
    a2.plot(df["date"], df["wSpx"], color="#1f6f54", lw=1, label="w SPX")
    a2.plot(df["date"], df["wVix"], color="#b5651d", lw=1, label="w VIX")
    a2.plot(df["date"], df["gross"], color="#555", lw=0.8, label="gross |w|")
    a2.axhline(c["name_cap"], color="#c0392b", ls="--", lw=0.8)
    a2.axhline(c["gross_cap"], color="#7d3c98", ls="--", lw=0.8)
    a2.set_ylabel("weight"); a2.legend(frameon=False, ncol=3); a2.grid(alpha=0.25)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# -------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SPX-VIX portfolio backtest engine")
    ap.add_argument("--source", choices=["csv", "yahoo"], default="csv")
    ap.add_argument("--csv", default="spx_vix_daily.csv")
    ap.add_argument("--format", choices=["json", "table"], default="json")
    ap.add_argument("--series", action="store_true", help="include chart series in JSON")
    ap.add_argument("--sample", type=int, default=5, help="series thinning step")
    ap.add_argument("--fig", default=None, help="save NAV/weights PNG to this path")
    # mandate overrides (defaults reproduce the take-home)
    for key in ("vol_target", "var_limit", "gross_cap", "name_cap", "tc"):
        ap.add_argument(f"--{key.replace('_','-')}", type=float, default=CFG[key])
    ap.add_argument("--lookback", type=int, default=CFG["lookback"])
    args = ap.parse_args()

    c = dict(CFG)
    for key in ("vol_target", "var_limit", "gross_cap", "name_cap", "tc", "lookback"):
        c[key] = getattr(args, key)

    prices = load_csv(args.csv) if args.source == "csv" else load_yahoo()
    metrics, series, df = backtest(prices, c, sample=args.sample)
    if args.fig:
        save_fig(df, c, args.fig)
        print(f"[engine] wrote {args.fig}", file=sys.stderr)

    if args.format == "table":
        print(f"{prices.index.min().date()} -> {prices.index.max().date()}  ({len(prices)} days)")
        for k, v in metrics.items():
            if k in ("bindingShare", "benchmarkSpxOnly"):
                print(f"{k:>22} : {json.dumps(v)}")
            elif isinstance(v, float):
                print(f"{k:>22} : {v:.4f}")
            else:
                print(f"{k:>22} : {v}")
    else:
        print(json.dumps({"metrics": metrics,
                          "series": series if args.series else []}))


if __name__ == "__main__":
    main()
