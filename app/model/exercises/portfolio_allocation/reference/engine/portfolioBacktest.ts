// ============================================================================
// SPX–VIX risk-targeted portfolio — pure TypeScript backtest engine.
// No dependencies. Runs identically in the browser and in Node.
//
// Point-in-time, daily rebalanced, LONG BOTH legs, risk-sized:
//   1. trailing-`lookback` daily simple returns -> covariance & vols
//   2. base book = inverse-volatility, both long, gross-normalised to 1
//   3. scale to the ex-ante annualised vol target
//   4. single uniform scalar k = min(k_volTarget, k_gross, k_perName, k_VaR);
//      hard caps are inviolable ceilings, the vol target yields when one binds
//   5. daily rebalance at the close, `tc` per side per instrument on
//      drift-adjusted turnover. No look-ahead: weights set at close t-1 earn
//      the day-t return.
//
// Numerical conventions are matched to NumPy/pandas so results reproduce the
// Python reference to the last basis point:
//   - covariance uses ddof = 1 (sample, divide by N-1)
//   - quantiles use linear interpolation (NumPy default)
//   - simple returns r_t = P_t / P_{t-1} - 1
// ============================================================================

import {
  PricePoint, BacktestConfig, DEFAULT_CONFIG, BacktestResult,
  SeriesPoint, Metrics, BindLabel,
} from "./types";

// --------------------------- small math helpers ---------------------------
function mean(a: number[]): number {
  let s = 0;
  for (const x of a) s += x;
  return s / a.length;
}

/** Sample standard deviation (ddof = 1), matching numpy std(ddof=1). */
function stdSample(a: number[]): number {
  const n = a.length;
  if (n < 2) return 0;
  const m = mean(a);
  let s = 0;
  for (const x of a) {
    const d = x - m;
    s += d * d;
  }
  return Math.sqrt(s / (n - 1));
}

/** 2x2 covariance of two equal-length series, ddof = 1 (matches np.cov). */
function cov2(x: number[], y: number[]): [number, number, number] {
  const n = x.length;
  const mx = mean(x);
  const my = mean(y);
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  const d = n - 1;
  return [sxx / d, syy / d, sxy / d]; // [var_x, var_y, cov_xy]
}

/** Quantile with linear interpolation on UNSORTED data (matches np.quantile). */
function quantileLinear(data: number[], q: number): number {
  const a = data.slice().sort((p, r) => p - r);
  const n = a.length;
  if (n === 1) return a[0];
  const h = (n - 1) * q;
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  const g = h - lo;
  return a[lo] * (1 - g) + a[hi] * g;
}

// --------------------------- sizing (one close) ----------------------------
interface DecideOut {
  w: [number, number];
  gross: number;
  bind: BindLabel;
  sigAnte: number;
  varAnte: number;
}

/**
 * Decide target weights from a window of daily returns ending at the decision
 * day (inclusive). `win` is an array of [rSpx, rVix] rows. Uses only data
 * inside `win` (point-in-time).
 */
function decideWeights(win: number[][], cfg: BacktestConfig): DecideOut {
  const xs = win.map((r) => r[0]);
  const ys = win.map((r) => r[1]);
  const [vx, vy, cxy] = cov2(xs, ys);
  const volX = Math.sqrt(Math.max(vx, 0));
  const volY = Math.sqrt(Math.max(vy, 0));

  // base book: inverse-vol, both long, gross-normalised to 1
  const ix = volX > 1e-12 ? 1 / volX : 0;
  const iy = volY > 1e-12 ? 1 / volY : 0;
  const sInv = ix + iy;
  const b0 = ix / sInv;
  const b1 = iy / sInv;

  // ex-ante annualised vol of the unit book
  const varDaily = b0 * b0 * vx + b1 * b1 * vy + 2 * b0 * b1 * cxy;
  const sigAnnB = Math.sqrt(Math.max(varDaily, 1e-18)) * Math.sqrt(cfg.ann);
  const kVol = sigAnnB > 0 ? cfg.volTarget / sigAnnB : Infinity;

  // historical 1-day VaR of the unit book (loss as a positive number)
  const pnlB = win.map((r) => b0 * r[0] + b1 * r[1]);
  const qLoss = quantileLinear(pnlB, 1 - cfg.varConf); // 5th pct (negative)
  const varB = Math.max(-qLoss, 0);
  const kVar = varB > 1e-12 ? cfg.varLimit / varB : Infinity;

  // hard-limit ceilings on the scalar
  const grossUnit = Math.abs(b0) + Math.abs(b1); // = 1
  const kGross = cfg.grossCap / grossUnit;
  const kName = cfg.nameCap / Math.max(Math.abs(b0), Math.abs(b1));

  const ceilings: Record<BindLabel, number> = {
    vol_target: kVol, gross: kGross, per_name: kName, VaR: kVar,
  };
  let bind: BindLabel = "vol_target";
  let k = Infinity;
  (Object.keys(ceilings) as BindLabel[]).forEach((key) => {
    if (ceilings[key] < k) { k = ceilings[key]; bind = key; }
  });

  const w: [number, number] = [k * b0, k * b1];
  return {
    w, bind,
    gross: Math.abs(w[0]) + Math.abs(w[1]),
    sigAnte: sigAnnB * k,
    varAnte: varB * k,
  };
}

// --------------------------- prices -> returns -----------------------------
function toReturns(prices: PricePoint[]): { R: number[][]; rdates: string[] } {
  const R: number[][] = [];
  const rdates: string[] = [];
  for (let i = 1; i < prices.length; i++) {
    const p0 = prices[i - 1], p1 = prices[i];
    R.push([p1.spx / p0.spx - 1, p1.vix / p0.vix - 1]);
    rdates.push(p1.date); // return at index i-1 belongs to the later date
  }
  return { R, rdates };
}

// ------------------------------ benchmark ----------------------------------
/** SPX-only book vol-targeted to the same target, same caps. Realistic floor. */
function spxOnly(R: number[][], cfg: BacktestConfig) {
  const L = cfg.lookback;
  let wPrev = 0;
  const rets: number[] = [];
  let nav = cfg.nav0;
  for (let t = L - 1; t < R.length; t++) {
    if (t > L - 1) {
      const g = wPrev * R[t][0];
      nav *= 1 + g;
      rets.push(g);
    }
    const win = R.slice(t - L + 1, t + 1).map((r) => r[0]);
    const sig = stdSample(win) * Math.sqrt(cfg.ann);
    const k = Math.min(cfg.volTarget / sig, cfg.nameCap, cfg.grossCap);
    wPrev = k;
  }
  const vol = stdSample(rets) * Math.sqrt(cfg.ann);
  const cagr = Math.pow(nav / cfg.nav0, cfg.ann / rets.length) - 1;
  return { cagr, vol, sharpe: (mean(rets) * cfg.ann) / vol };
}

// ------------------------------ backtest -----------------------------------
export function runBacktest(
  prices: PricePoint[],
  config: Partial<BacktestConfig> = {},
  options: { sample?: number } = {},
): BacktestResult {
  const cfg: BacktestConfig = { ...DEFAULT_CONFIG, ...config };
  const clean = prices
    .filter((p) => Number.isFinite(p.spx) && Number.isFinite(p.vix))
    .slice()
    .sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  if (clean.length < cfg.lookback + 50) {
    throw new Error(
      `Need > ${cfg.lookback + 50} rows; got ${clean.length}.`,
    );
  }

  const { R, rdates } = toReturns(clean);
  const T = R.length;
  const L = cfg.lookback;

  // full-sample descriptive stats
  const allSpx = R.map((r) => r[0]);
  const allVix = R.map((r) => r[1]);
  const [vsx, vvx, csv] = cov2(allSpx, allVix);
  const annVolSpx = Math.sqrt(vsx) * Math.sqrt(cfg.ann);
  const annVolVix = Math.sqrt(vvx) * Math.sqrt(cfg.ann);
  const corr = csv / (Math.sqrt(vsx) * Math.sqrt(vvx));

  // single loop, tracking cost and no-cost NAV in parallel (weights identical)
  let wPrev: [number, number] = [0, 0];
  let nav = cfg.nav0;
  let navNc = cfg.nav0;

  const dates: string[] = [];
  const navArr: number[] = [];
  const netRet: number[] = [];
  const wSpxArr: number[] = [];
  const wVixArr: number[] = [];
  const grossArr: number[] = [];
  const turnArr: number[] = [];
  const bindArr: BindLabel[] = [];

  for (let t = L - 1; t < T; t++) {
    let grossRet = 0;
    let wDrift: [number, number] = [0, 0];
    if (t > L - 1) {
      grossRet = wPrev[0] * R[t][0] + wPrev[1] * R[t][1];
      nav *= 1 + grossRet;
      navNc *= 1 + grossRet;
      const den = 1 + grossRet;
      wDrift = [wPrev[0] * (1 + R[t][0]) / den, wPrev[1] * (1 + R[t][1]) / den];
    }

    const win = R.slice(t - L + 1, t + 1);
    const dec = decideWeights(win, cfg);

    const turnover = Math.abs(dec.w[0] - wDrift[0]) + Math.abs(dec.w[1] - wDrift[1]);
    const cost = cfg.tc * turnover;
    nav *= 1 - cost;
    const net = (1 + grossRet) * (1 - cost) - 1;

    dates.push(rdates[t]);
    navArr.push(nav);
    netRet.push(net);
    wSpxArr.push(dec.w[0]);
    wVixArr.push(dec.w[1]);
    grossArr.push(dec.gross);
    turnArr.push(turnover);
    bindArr.push(dec.bind);

    wPrev = dec.w;
  }

  // metrics computed on the "live" window (drop the day-0 setup row)
  const liveFrom = 1;
  const r = netRet.slice(liveFrom);
  const navLive = navArr.slice(liveFrom);
  const n = r.length;

  const cagr = Math.pow(navLive[n - 1] / cfg.nav0, cfg.ann / n) - 1;
  const vol = stdSample(r) * Math.sqrt(cfg.ann);
  const sharpe = (mean(r) * cfg.ann) / vol;

  let peak = -Infinity, mdd = 0;
  for (const v of navLive) {
    if (v > peak) peak = v;
    const dd = v / peak - 1;
    if (dd < mdd) mdd = dd;
  }
  const var95 = -quantileLinear(r, 1 - cfg.varConf);
  const avgGross = mean(grossArr.slice(liveFrom));
  const maxName = Math.max(
    ...wSpxArr.slice(liveFrom).map(Math.abs),
    ...wVixArr.slice(liveFrom).map(Math.abs),
  );
  const turnDaily = mean(turnArr.slice(liveFrom));

  const cagrNc = Math.pow(navNc / cfg.nav0, cfg.ann / n) - 1;
  const costDragAnn = cagrNc - cagr;
  const totalCostUsd = navNc - navArr[navArr.length - 1];

  const counts: Record<string, number> = {};
  for (const b of bindArr.slice(liveFrom)) counts[b] = (counts[b] || 0) + 1;
  const bindingShare: Record<string, number> = {};
  for (const k of Object.keys(counts)) bindingShare[k] = counts[k] / n;

  const metrics: Metrics = {
    sampleStart: dates[liveFrom],
    sampleEnd: dates[dates.length - 1],
    tradingDays: n,
    annReturnCagr: cagr,
    realisedVol: vol,
    volTarget: cfg.volTarget,
    sharpeRf0: sharpe,
    maxDrawdown: mdd,
    realisedVar95_1d: var95,
    varLimit: cfg.varLimit,
    avgGrossExposure: avgGross,
    maxNameWeight: maxName,
    avgDailyTurnover: turnDaily,
    annualisedTurnover: turnDaily * cfg.ann,
    annualisedCostDrag: costDragAnn,
    totalCostDragUsd: totalCostUsd,
    finalNavUsd: navArr[navArr.length - 1],
    corrSpxVixDaily: corr,
    annVolSpx,
    annVolVix,
    bindingShare,
    benchmarkSpxOnly: spxOnly(R, cfg),
  };

  // chart series (computation is daily; output may be thinned via `sample`)
  const step = Math.max(1, options.sample ?? 1);
  const series: SeriesPoint[] = [];
  for (let i = liveFrom; i < navArr.length; i++) {
    if ((i - liveFrom) % step !== 0 && i !== navArr.length - 1) continue;
    series.push({
      date: dates[i], nav: navArr[i],
      wSpx: wSpxArr[i], wVix: wVixArr[i], gross: grossArr[i],
    });
  }

  return { metrics, series };
}
