// Parse the exercise CSV (header `Date,SPX,VIX`, ISO dates, comma-separated).
// Tolerant of CRLF, surrounding whitespace and any column order.
import { PricePoint } from "./types";

export function parseCsvText(text: string): PricePoint[] {
  const lines = text.replace(/\r/g, "").split("\n").filter((l) => l.trim() !== "");
  if (lines.length < 2) throw new Error("CSV has no data rows.");

  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const di = header.findIndex((h) => h.startsWith("date"));
  const si = header.indexOf("spx");
  const vi = header.indexOf("vix");
  if (di < 0 || si < 0 || vi < 0) {
    throw new Error("CSV must have columns: Date, SPX, VIX.");
  }

  const out: PricePoint[] = [];
  for (let i = 1; i < lines.length; i++) {
    const c = lines[i].split(",");
    const date = (c[di] ?? "").trim();
    const spx = Number((c[si] ?? "").trim());
    const vix = Number((c[vi] ?? "").trim());
    if (!date || !Number.isFinite(spx) || !Number.isFinite(vix)) continue;
    out.push({ date, spx, vix });
  }
  if (out.length === 0) throw new Error("No valid rows parsed from CSV.");
  return out;
}
