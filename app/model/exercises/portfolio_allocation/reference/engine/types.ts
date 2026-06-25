// Shared types for the SPX/VIX Portfolio Allocation exercise engine.

/** One trading day of close prices. `date` is ISO `YYYY-MM-DD`. */
export interface PricePoint {
  date: string;
  spx: number;
  vix: number;
}

/** Mandate / model parameters. Defaults reproduce the take-home exactly. */
export interface BacktestConfig {
  nav0: number;        // static AUM in USD
  lookback: number;    // trailing window (days) for vol / cov / VaR
  volTarget: number;   // ex-ante annualised vol target (e.g. 0.10)
  varConf: number;     // VaR confidence (e.g. 0.95)
  varLimit: number;    // 1-day historical VaR cap, fraction of NAV (e.g. 0.025)
  grossCap: number;    // sum|w_i| cap (e.g. 1.50)
  nameCap: number;     // per-instrument |w_i| cap (e.g. 1.00)
  tc: number;          // transaction cost per side per instrument (e.g. 1e-4)
  ann: number;         // trading days / year (252)
}

export const DEFAULT_CONFIG: BacktestConfig = {
  nav0: 10_000_000,
  lookback: 252,
  volTarget: 0.10,
  varConf: 0.95,
  varLimit: 0.025,
  grossCap: 1.50,
  nameCap: 1.00,
  tc: 1e-4,
  ann: 252,
};

export type BindLabel = "vol_target" | "gross" | "per_name" | "VaR";

/** Per-day point for charting (NAV curve + weight path). */
export interface SeriesPoint {
  date: string;
  nav: number;   // net-of-cost NAV in USD
  wSpx: number;  // weight (fraction of NAV)
  wVix: number;
  gross: number; // sum of |weights|
}

export interface Metrics {
  sampleStart: string;
  sampleEnd: string;
  tradingDays: number;
  annReturnCagr: number;     // geometric, net of cost
  realisedVol: number;       // annualised
  volTarget: number;
  sharpeRf0: number;         // rf = 0
  maxDrawdown: number;       // negative
  realisedVar95_1d: number;  // positive loss fraction
  varLimit: number;
  avgGrossExposure: number;
  maxNameWeight: number;
  avgDailyTurnover: number;
  annualisedTurnover: number;
  annualisedCostDrag: number;
  totalCostDragUsd: number;
  finalNavUsd: number;
  corrSpxVixDaily: number;
  annVolSpx: number;
  annVolVix: number;
  bindingShare: Record<string, number>;
  /** SPX-only, vol-targeted to the same target — a realistic comparison floor. */
  benchmarkSpxOnly: { cagr: number; vol: number; sharpe: number };
}

export interface BacktestResult {
  metrics: Metrics;
  series: SeriesPoint[];
}
