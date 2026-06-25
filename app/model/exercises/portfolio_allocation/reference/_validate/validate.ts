// Validate the TS engine against the Python reference on the supplied CSV.
//   tsx _validate/validate.ts
import { readFileSync } from "node:fs";
import { parseCsvText } from "../engine/csv";
import { runBacktest } from "../engine/portfolioBacktest";

const csv = readFileSync(new URL("../spx_vix_daily.csv", import.meta.url), "utf8");
const prices = parseCsvText(csv);
const { metrics: m } = runBacktest(prices);

// Python reference (from metrics.json on the same CSV)
const ref: Record<string, number> = {
  annReturnCagr: 0.1446,
  realisedVol: 0.0958,
  sharpeRf0: 1.457,
  maxDrawdown: -0.2033,
  realisedVar95_1d: 0.008,
  avgGrossExposure: 0.9887,
  annualisedTurnover: 3.09,
  annualisedCostDrag: 0.00036,
  corrSpxVixDaily: -0.7,
  annVolSpx: 0.1817,
  annVolVix: 1.1094,
  tradingDays: 8312,
  benchSharpe: 0.61,
};

const got: Record<string, number> = {
  annReturnCagr: m.annReturnCagr,
  realisedVol: m.realisedVol,
  sharpeRf0: m.sharpeRf0,
  maxDrawdown: m.maxDrawdown,
  realisedVar95_1d: m.realisedVar95_1d,
  avgGrossExposure: m.avgGrossExposure,
  annualisedTurnover: m.annualisedTurnover,
  annualisedCostDrag: m.annualisedCostDrag,
  corrSpxVixDaily: m.corrSpxVixDaily,
  annVolSpx: m.annVolSpx,
  annVolVix: m.annVolVix,
  tradingDays: m.tradingDays,
  benchSharpe: m.benchmarkSpxOnly.sharpe,
};

console.log("metric                     TS            Python        |diff|");
console.log("-".repeat(64));
let worst = 0;
for (const k of Object.keys(ref)) {
  const d = Math.abs(got[k] - ref[k]);
  // tolerance scales with the magnitude of the (rounded) reference value
  const tol = Math.max(Math.abs(ref[k]) * 0.01, 0.0005);
  worst = Math.max(worst, d / (Math.abs(ref[k]) || 1));
  const flag = d <= tol ? "ok" : "DIFF";
  console.log(
    `${k.padEnd(24)} ${got[k].toFixed(5).padStart(12)} ${ref[k]
      .toFixed(5)
      .padStart(12)} ${d.toFixed(5).padStart(10)}  ${flag}`,
  );
}
console.log("-".repeat(64));
console.log(`max relative diff: ${(worst * 100).toFixed(3)}%`);
console.log(
  `sample ${m.sampleStart} -> ${m.sampleEnd}, ${m.tradingDays} days, final NAV $${(
    m.finalNavUsd / 1e6
  ).toFixed(1)}mm`,
);
