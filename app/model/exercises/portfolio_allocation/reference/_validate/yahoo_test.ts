import { fetchYahooPrices } from "../engine/yahoo";
import { runBacktest } from "../engine/portfolioBacktest";

async function main() {
  const prices = await fetchYahooPrices({ start: "1990-01-01" });
  console.log("rows:", prices.length, "|", prices[0].date, "->", prices[prices.length - 1].date);
  const { metrics: m } = runBacktest(prices);
  console.log(
    `CAGR ${(m.annReturnCagr * 100).toFixed(2)}% | vol ${(m.realisedVol * 100).toFixed(2)}% | ` +
    `Sharpe ${m.sharpeRf0.toFixed(2)} | DD ${(m.maxDrawdown * 100).toFixed(1)}% | ` +
    `VaR ${(m.realisedVar95_1d * 100).toFixed(2)}% | gross ${(m.avgGrossExposure * 100).toFixed(0)}% | days ${m.tradingDays}`,
  );
}
main().catch((e) => { console.error("ERR:", e?.message || e); process.exit(1); });
