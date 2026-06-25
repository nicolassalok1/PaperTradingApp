"use client";
// ============================================================================
// Exercices > Portfolio Allocation — SPX/VIX risk-targeted book.
// Reference component. The COMPUTATION (engine/) is validated and final; this
// UI is a starting point — restyle it with your design system / component lib.
//
// Requires `recharts` (npm i recharts). If your app uses another chart lib,
// swap the <Chart*> blocks; the engine output shape is lib-agnostic.
//
// Data flow:
//   - "CSV"   : user uploads Date,SPX,VIX -> parsed & backtested in-browser.
//   - "Yahoo" : a server route fetches ^GSPC/^VIX (no CORS), then the SAME
//               engine runs in-browser, so tweaking parameters is instant.
// ============================================================================
import { useCallback, useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { runBacktest } from "../engine/portfolioBacktest";
import { parseCsvText } from "../engine/csv";
import {
  DEFAULT_CONFIG, BacktestConfig, BacktestResult, PricePoint,
} from "../engine/types";

// Adapt to your router. Must match where you mount the Yahoo route handler.
const YAHOO_ROUTE = "/api/exercises/portfolio-allocation/yahoo";

type Source = "csv" | "yahoo";

const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
const money = (x: number) => `$${(x / 1e6).toFixed(1)}mm`;

export default function PortfolioAllocation() {
  const [source, setSource] = useState<Source>("csv");
  const [cfg, setCfg] = useState<BacktestConfig>({ ...DEFAULT_CONFIG });
  const [prices, setPrices] = useState<PricePoint[] | null>(null);
  const [priceLabel, setPriceLabel] = useState<string>("");
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>("");

  const onCsv = useCallback(async (file: File) => {
    setError(""); setResult(null);
    try {
      const pts = parseCsvText(await file.text());
      setPrices(pts);
      setPriceLabel(`${file.name} — ${pts.length} rows (${pts[0].date} → ${pts[pts.length - 1].date})`);
    } catch (e: any) {
      setPrices(null); setError(e?.message ?? "Failed to parse CSV.");
    }
  }, []);

  const loadYahoo = useCallback(async () => {
    setError(""); setResult(null); setBusy(true);
    try {
      const res = await fetch(YAHOO_ROUTE, { cache: "no-store" });
      if (!res.ok) throw new Error(`Yahoo route HTTP ${res.status}`);
      const pts = (await res.json()) as PricePoint[];
      setPrices(pts);
      setPriceLabel(`Yahoo Finance — ${pts.length} rows (${pts[0].date} → ${pts[pts.length - 1].date})`);
    } catch (e: any) {
      setPrices(null); setError(e?.message ?? "Failed to fetch from Yahoo.");
    } finally {
      setBusy(false);
    }
  }, []);

  const run = useCallback(() => {
    if (!prices) { setError("Load a dataset first."); return; }
    setError(""); setBusy(true);
    // defer so the spinner can paint before the (fast) synchronous compute
    setTimeout(() => {
      try {
        setResult(runBacktest(prices, cfg, { sample: 5 })); // weekly chart points
      } catch (e: any) {
        setError(e?.message ?? "Backtest failed.");
      } finally {
        setBusy(false);
      }
    }, 0);
  }, [prices, cfg]);

  const m = result?.metrics;
  const chart = useMemo(
    () => (result?.series ?? []).map((p) => ({
      date: p.date,
      navM: p.nav / 1e6,
      wSpx: p.wSpx,
      wVix: p.wVix,
      gross: p.gross,
    })),
    [result],
  );

  // constraint compliance checks
  const volOk = m ? Math.abs(m.realisedVol - cfg.volTarget) <= 0.01 + 1e-9 : false;
  const varOk = m ? m.realisedVar95_1d <= cfg.varLimit + 1e-9 : false;
  const grossOk = m ? m.avgGrossExposure <= cfg.grossCap + 1e-9 : false;
  const nameOk = m ? m.maxNameWeight <= cfg.nameCap + 1e-9 : false;

  const setNum = (k: keyof BacktestConfig, v: number) =>
    setCfg((c) => ({ ...c, [k]: v }));

  return (
    <div className="pa-root" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <header>
        <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>Portfolio Allocation — SPX / VIX</h2>
        <p style={{ margin: "4px 0 0", fontSize: 13, opacity: 0.7 }}>
          Risk-targeted two-asset book: long SPX (return engine) + long VIX
          (convex hedge), inverse-vol weighted, scaled to a {pct(cfg.volTarget, 0)} ex-ante
          vol target, projected onto the gross / per-name / VaR caps. Point-in-time, daily.
        </p>
      </header>

      {/* ---- controls ---- */}
      <section style={panel}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ fontSize: 13, fontWeight: 600 }}>Données :</span>
          <Seg active={source === "csv"} onClick={() => { setSource("csv"); setResult(null); }}>Fichier CSV</Seg>
          <Seg active={source === "yahoo"} onClick={() => { setSource("yahoo"); setResult(null); }}>Yahoo Finance</Seg>

          {source === "csv" ? (
            <label style={{ ...btn, cursor: "pointer" }}>
              Choisir un CSV (Date,SPX,VIX)
              <input
                type="file" accept=".csv,text/csv" style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && onCsv(e.target.files[0])}
              />
            </label>
          ) : (
            <button style={btn} onClick={loadYahoo} disabled={busy}>
              {busy ? "Chargement…" : "Charger depuis Yahoo (^GSPC, ^VIX)"}
            </button>
          )}
        </div>
        {priceLabel && <div style={{ fontSize: 12, opacity: 0.7, marginTop: 8 }}>{priceLabel}</div>}

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 12 }}>
          <NumField label="Lookback (j)" value={cfg.lookback} step={1} onChange={(v) => setNum("lookback", Math.max(20, Math.round(v)))} />
          <NumField label="Cible vol (%)" value={cfg.volTarget * 100} step={0.5} onChange={(v) => setNum("volTarget", v / 100)} />
          <NumField label="Cap gross (%)" value={cfg.grossCap * 100} step={5} onChange={(v) => setNum("grossCap", v / 100)} />
          <NumField label="Cap / instrument (%)" value={cfg.nameCap * 100} step={5} onChange={(v) => setNum("nameCap", v / 100)} />
          <NumField label="Cap VaR 95% (%)" value={cfg.varLimit * 100} step={0.25} onChange={(v) => setNum("varLimit", v / 100)} />
          <div style={{ alignSelf: "flex-end" }}>
            <button style={{ ...btn, background: "#0b3d2e", color: "#fff", borderColor: "#0b3d2e" }} onClick={run} disabled={busy || !prices}>
              {busy ? "Calcul…" : "Lancer le backtest"}
            </button>
          </div>
        </div>
        {error && <div style={{ color: "#c0392b", fontSize: 13, marginTop: 8 }}>{error}</div>}
      </section>

      {/* ---- results ---- */}
      {m && (
        <>
          <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10 }}>
            <Stat label="Rendement annualisé (CAGR)" value={pct(m.annReturnCagr)} />
            <Stat label="Vol réalisée" value={pct(m.realisedVol)} sub={`cible ${pct(cfg.volTarget, 0)} ±1%`} ok={volOk} />
            <Stat label="Sharpe (rf=0)" value={m.sharpeRf0.toFixed(2)} sub={`SPX seul : ${m.benchmarkSpxOnly.sharpe.toFixed(2)}`} />
            <Stat label="Drawdown max" value={pct(m.maxDrawdown)} />
            <Stat label="VaR 95% 1j réalisée" value={pct(m.realisedVar95_1d, 2)} sub={`limite ${pct(cfg.varLimit, 1)}`} ok={varOk} />
            <Stat label="Gross moyen" value={pct(m.avgGrossExposure, 0)} sub={`cap ${pct(cfg.grossCap, 0)}`} ok={grossOk} />
            <Stat label="Poids max / instrument" value={pct(m.maxNameWeight, 0)} sub={`cap ${pct(cfg.nameCap, 0)}`} ok={nameOk} />
            <Stat label="Turnover (×/an)" value={m.annualisedTurnover.toFixed(2)} sub={`coût ${(m.annualisedCostDrag * 1e4).toFixed(1)} bp/an`} />
          </section>

          <div style={{ fontSize: 12, opacity: 0.7 }}>
            {m.sampleStart} → {m.sampleEnd} · {m.tradingDays.toLocaleString()} jours · NAV finale {money(m.finalNavUsd)} ·
            corr(SPX,VIX) {m.corrSpxVixDaily.toFixed(2)} · vol VIX {pct(m.annVolVix, 0)}
          </div>

          {/* NAV (log) */}
          <section style={panel}>
            <h3 style={chartTitle}>NAV cumulée (échelle log, nette de coûts)</h3>
            <div style={{ width: "100%", height: 260 }}>
              <ResponsiveContainer>
                <LineChart data={chart} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
                  <CartesianGrid strokeOpacity={0.15} />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={48} />
                  <YAxis scale="log" domain={["auto", "auto"]} tick={{ fontSize: 11 }} width={48}
                    tickFormatter={(v) => `${v}M`} />
                  <Tooltip formatter={(v: number) => [`$${v.toFixed(1)}mm`, "NAV"]} />
                  <Line type="monotone" dataKey="navM" stroke="#0b3d2e" dot={false} strokeWidth={1.4} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* Weights */}
          <section style={panel}>
            <h3 style={chartTitle}>Trajectoire des poids (fraction de NAV)</h3>
            <div style={{ width: "100%", height: 240 }}>
              <ResponsiveContainer>
                <LineChart data={chart} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
                  <CartesianGrid strokeOpacity={0.15} />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={48} />
                  <YAxis domain={[-0.1, Math.max(1.6, cfg.grossCap + 0.1)]} tick={{ fontSize: 11 }} width={40} />
                  <Tooltip formatter={(v: number, n) => [v.toFixed(3), n]} />
                  <Legend />
                  <ReferenceLine y={cfg.nameCap} stroke="#c0392b" strokeDasharray="4 4" />
                  <ReferenceLine y={cfg.grossCap} stroke="#7d3c98" strokeDasharray="4 4" />
                  <ReferenceLine y={0} stroke="#aaa" />
                  <Line type="monotone" dataKey="wSpx" name="w SPX" stroke="#1f6f54" dot={false} strokeWidth={1.2} />
                  <Line type="monotone" dataKey="wVix" name="w VIX" stroke="#b5651d" dot={false} strokeWidth={1.2} />
                  <Line type="monotone" dataKey="gross" name="gross |w|" stroke="#555" dot={false} strokeWidth={1} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <p style={{ fontSize: 12, opacity: 0.65, lineHeight: 1.5 }}>
            <strong>Mise en garde.</strong> Le backtest capte le rendement de l'indice VIX sans friction.
            En réel (futures/VXX), le roll en contango détruit une grande part de la jambe VIX : lis le Sharpe
            comme une borne haute, pas une espérance. Le benchmark SPX-seul ({m.benchmarkSpxOnly.sharpe.toFixed(2)})
            est plus proche d'un plancher atteignable.
          </p>
        </>
      )}
    </div>
  );
}

// ------------------------------- small UI bits -------------------------------
const panel: React.CSSProperties = { border: "1px solid #e3e8e6", borderRadius: 10, padding: 14, background: "#fff" };
const btn: React.CSSProperties = { border: "1px solid #cfd8d4", borderRadius: 8, padding: "8px 12px", fontSize: 13, background: "#fff" };
const chartTitle: React.CSSProperties = { margin: "0 0 8px", fontSize: 13, fontWeight: 600 };

function Seg({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick} style={{
      ...btn,
      background: active ? "#0b3d2e" : "#fff",
      color: active ? "#fff" : "#222",
      borderColor: active ? "#0b3d2e" : "#cfd8d4",
    }}>{children}</button>
  );
}

function NumField({ label, value, step, onChange }: { label: string; value: number; step: number; onChange: (v: number) => void }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", fontSize: 11, gap: 4, opacity: 0.9 }}>
      {label}
      <input
        type="number" value={Number.isFinite(value) ? +value.toFixed(2) : 0} step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: 96, padding: "6px 8px", border: "1px solid #cfd8d4", borderRadius: 6, fontSize: 13 }}
      />
    </label>
  );
}

function Stat({ label, value, sub, ok }: { label: string; value: string; sub?: string; ok?: boolean }) {
  return (
    <div style={{ ...panel, padding: 12 }}>
      <div style={{ fontSize: 11, opacity: 0.7 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, marginTop: 2 }}>
        {value}
        {ok !== undefined && (
          <span style={{ fontSize: 12, marginLeft: 6, color: ok ? "#1f6f54" : "#c0392b" }}>{ok ? "✓" : "✗"}</span>
        )}
      </div>
      {sub && <div style={{ fontSize: 11, opacity: 0.6, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}
