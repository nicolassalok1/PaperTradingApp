// Server-side fetch of SPX (^GSPC) and VIX (^VIX) daily closes from Yahoo
// Finance. MUST run on the server (Node / a Next.js route handler): the Yahoo
// chart endpoint blocks browser CORS. Returns aligned PricePoint[].
import { PricePoint } from "./types";

const SYMBOLS = { spx: "%5EGSPC", vix: "%5EVIX" }; // ^GSPC, ^VIX
const HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"];

interface YahooChart {
  chart: {
    result?: Array<{
      timestamp?: number[];
      indicators: { quote: Array<{ close?: (number | null)[] }> };
    }>;
    error?: unknown;
  };
}

async function fetchSeries(
  symbol: string,
  period1: number,
  period2: number,
): Promise<Map<string, number>> {
  let lastErr: unknown;
  for (const host of HOSTS) {
    const url =
      `https://${host}/v8/finance/chart/${symbol}` +
      `?period1=${period1}&period2=${period2}&interval=1d`;
    try {
      const res = await fetch(url, {
        headers: { "User-Agent": "Mozilla/5.0", Accept: "application/json" },
      });
      if (!res.ok) throw new Error(`Yahoo ${symbol} HTTP ${res.status}`);
      const json = (await res.json()) as YahooChart;
      const r = json.chart.result?.[0];
      const ts = r?.timestamp;
      const close = r?.indicators?.quote?.[0]?.close;
      if (!ts || !close) throw new Error(`Yahoo ${symbol}: empty payload`);
      const m = new Map<string, number>();
      for (let i = 0; i < ts.length; i++) {
        const c = close[i];
        if (c == null || !Number.isFinite(c)) continue;
        const date = new Date(ts[i] * 1000).toISOString().slice(0, 10);
        m.set(date, c); // last value for a date wins (daily bars are unique)
      }
      return m;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr ?? new Error(`Yahoo fetch failed for ${symbol}`);
}

export interface FetchYahooOptions {
  /** ISO date (YYYY-MM-DD). Default 1990-01-01. */
  start?: string;
  /** ISO date (YYYY-MM-DD). Default today. */
  end?: string;
}

/** Fetch and inner-join SPX & VIX daily closes into a sorted PricePoint[]. */
export async function fetchYahooPrices(
  opts: FetchYahooOptions = {},
): Promise<PricePoint[]> {
  const period1 = Math.floor(
    new Date(opts.start ?? "1990-01-01").getTime() / 1000,
  );
  const period2 = Math.floor(
    (opts.end ? new Date(opts.end).getTime() : Date.now()) / 1000,
  );

  const [spx, vix] = await Promise.all([
    fetchSeries(SYMBOLS.spx, period1, period2),
    fetchSeries(SYMBOLS.vix, period1, period2),
  ]);

  const out: PricePoint[] = [];
  for (const [date, s] of spx) {
    const v = vix.get(date);
    if (v != null) out.push({ date, spx: s, vix: v });
  }
  out.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  if (out.length < 500) {
    throw new Error(`Only ${out.length} aligned SPX/VIX rows from Yahoo.`);
  }
  return out;
}
