// Example Next.js App Router route handler.
// Copy to: app/api/exercises/portfolio-allocation/yahoo/route.ts
// (Pages Router equivalent: pages/api/exercises/portfolio-allocation/yahoo.ts
//  exporting a default handler that calls fetchYahooPrices and res.json()s it.)
//
// Runs server-side, so the Yahoo CORS restriction does not apply. Adjust the
// relative import path to wherever you placed engine/yahoo.ts.

import { NextResponse } from "next/server";
import { fetchYahooPrices } from "@/exercises/portfolio-allocation/engine/yahoo";

export const runtime = "nodejs";        // needs Node fetch (not edge)
export const dynamic = "force-dynamic"; // never cache the price pull

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const start = searchParams.get("start") ?? "1990-01-01";
  const end = searchParams.get("end") ?? undefined;
  try {
    const prices = await fetchYahooPrices({ start, end });
    return NextResponse.json(prices, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (e: any) {
    return NextResponse.json(
      { error: e?.message ?? "Yahoo fetch failed" },
      { status: 502 },
    );
  }
}
