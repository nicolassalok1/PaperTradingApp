"""
Market-data cleaning / validation for the rough-volatility pipeline (spec 4.1).

Purpose
-------
Turn a *raw* option chain (the row schema produced by
``app.model.market_data.market_data.fetch_options_details_yahoo``, or any fixture
with the same schema) into one :class:`CleanChain` per expiry, carrying:

* the surviving quotes (:class:`CleanQuote`) with mid, spread and quality flags,
* an exhaustive **removal log** (:class:`RemovalRecord`) — every dropped contract
  gets a machine-readable reason code so the Phase-5 report can enumerate them,
* a :class:`ViabilityReport` telling downstream stages whether the expiry carries
  enough OTM quotes to build a variance swap strike (``K_var``) and/or a skew.

Every threshold lives in the frozen :class:`CleaningConfig`; nothing is buried in
a function body.

Vendor implied vol
------------------
The vendor ``iv`` column is **never** an input to any decision taken here. It is
carried through untouched on :attr:`CleanQuote.vendor_iv` and flagged
(:data:`FLAG_VENDOR_IV_SENTINEL`) when it looks like a placeholder
(``iv <= CleaningConfig.vendor_iv_sentinel``, default ``1e-4``). It exists purely
as a *cross-check* against the IVs the pipeline recomputes itself from cleaned
mids (see ``forward_curve.build_otm_surface``).

Note on reachability: ``market_data._norm_contract`` already drops any contract
whose vendor ``iv`` is ``<= 0`` or non-castable, so a live Yahoo fetch cannot
produce a zero-IV sentinel row. Such rows are only reachable from committed
fixtures. They are handled anyway, deliberately.

Exercise-style caveat (NOT hidden — read this)
----------------------------------------------
Yahoo single-name and ETF option chains are **American-style**. The
log-contract / variance-swap replication used downstream, and the Black-76
inversion used to recompute IVs, both assume **European** exercise. This module
does not attempt an American de-Americanisation: it *states the assumption* and
stamps :data:`FLAG_AMERICAN_EXERCISE_ASSUMED_EUROPEAN` on every chain built with
``CleaningConfig.assume_european_exercise`` (default ``True``), so the assumption
propagates into the report instead of disappearing.

The assumption is: the early-exercise premium is negligible for *short-dated
OTM* options on *low-dividend* underlyings. It is **not** negligible for deep
ITM puts, for high-dividend names, or for long maturities — which is one more
reason the pipeline consumes OTM quotes only. Prefer index-proxy underlyings
(cash-settled, European: SPX-like) for demos; ETF proxies such as SPY are
American and only approximately compliant.

Layering: model layer only (no Streamlit, no controller, no view). Importing this
module has no side effects and touches no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Reason codes (removals) and flags (kept, but annotated)
# ---------------------------------------------------------------------------

# --- structural / schema ---------------------------------------------------
REASON_BAD_TYPE = "bad_option_type"
REASON_BAD_STRIKE = "bad_strike"
REASON_BAD_MATURITY = "bad_maturity"
# --- quote validity (spec 4.1, bullet 1) -----------------------------------
REASON_MISSING_QUOTE = "missing_quote"
REASON_NEGATIVE_BID = "negative_bid"
REASON_NONPOSITIVE_ASK = "nonpositive_ask"
REASON_CROSSED_QUOTE = "crossed_quote"
REASON_NONPOSITIVE_MID = "nonpositive_mid"
# --- staleness (spec 4.1, bullet 2) ----------------------------------------
REASON_STALE_QUOTE = "stale_quote"
# --- bookkeeping ------------------------------------------------------------
REASON_DUPLICATE_STRIKE = "duplicate_strike"
# --- spread filter (spec 4.1, bullet 3) ------------------------------------
REASON_WIDE_SPREAD = "wide_spread"
# --- zero-bid tail truncation (CBOE VIX rule; see CleaningConfig) -----------
REASON_ZERO_BID_TAIL = "zero_bid_tail"
# --- arbitrage repair (spec 4.1, bullet 4) ---------------------------------
REASON_VERTICAL_ARBITRAGE = "vertical_arbitrage"
REASON_BUTTERFLY_ARBITRAGE = "butterfly_arbitrage"

#: French labels for the Phase-5 report (identifiers stay English, UI is French).
REASON_LABELS_FR: dict[str, str] = {
    REASON_BAD_TYPE: "Type d'option invalide",
    REASON_BAD_STRIKE: "Strike invalide ou non positif",
    REASON_BAD_MATURITY: "Maturité invalide ou non positive",
    REASON_MISSING_QUOTE: "Cotation absente (bid/ask manquant)",
    REASON_NEGATIVE_BID: "Bid négatif",
    REASON_NONPOSITIVE_ASK: "Ask nul ou négatif",
    REASON_CROSSED_QUOTE: "Fourchette croisée (ask < bid)",
    REASON_NONPOSITIVE_MID: "Prix moyen nul ou négatif",
    REASON_STALE_QUOTE: "Contrat dormant (volume et open interest nuls)",
    REASON_DUPLICATE_STRIKE: "Strike dupliqué (cotation la moins bonne écartée)",
    REASON_WIDE_SPREAD: "Fourchette trop large",
    REASON_ZERO_BID_TAIL: "Queue tronquée (deux bids nuls consécutifs, règle CBOE)",
    REASON_VERTICAL_ARBITRAGE: "Monotonie des spreads verticaux violée",
    REASON_BUTTERFLY_ARBITRAGE: "Convexité (butterfly) violée",
}

#: Bid is exactly zero: the quote is one-sided. Usable for the far tails only,
#: never for an ATM anchor. Kept, never silently dropped.
FLAG_ONE_SIDED = "one_sided_quote"
#: Vendor IV is a placeholder (<= ``CleaningConfig.vendor_iv_sentinel``).
FLAG_VENDOR_IV_SENTINEL = "vendor_iv_sentinel"
#: Vendor IV missing / non-castable.
FLAG_VENDOR_IV_MISSING = "vendor_iv_missing"
#: Chain-level: American chain consumed under a European assumption (see module docstring).
FLAG_AMERICAN_EXERCISE_ASSUMED_EUROPEAN = "american_exercise_assumed_european"
#: Chain-level: no quote survived cleaning.
FLAG_EMPTY_CHAIN = "empty_chain"
#: Chain-level: spot price unusable (NaN / non positive) in the raw rows.
FLAG_NO_SPOT = "no_spot"

CALL = "call"
PUT = "put"

#: Verbatim caveat, re-exported so the Phase-5 report can print it without
#: paraphrasing it into something weaker.
AMERICAN_EXERCISE_CAVEAT = (
    "Les chaînes Yahoo mono-sous-jacent et ETF sont de style américain ; la "
    "réplication log-contract et l'inversion Black-76 supposent un exercice "
    "européen. La prime d'exercice anticipé est traitée comme négligeable pour "
    "les options OTM de courte maturité sur sous-jacents à faible dividende. "
    "Préférer un proxy indiciel pour les démonstrations."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleaningConfig:
    """
    Every filter threshold of spec 4.1, in one frozen place.

    Attributes
    ----------
    drop_stale:
        Drop contracts with ``volume == 0 and openInterest == 0``. A missing /
        non-castable activity field is read as ``0`` (Yahoo omits the field when
        there was no activity), so "unknown activity" is treated as "no
        activity". Set to ``False`` to keep dormant contracts.
    s_max_otm / s_max_atm:
        Maximum relative spread ``(ask - bid) / mid``. ``s_max_atm`` applies
        inside the ATM band, ``s_max_otm`` outside it (tails are allowed to be
        wider). Defaults: 0.5 for the tails, 0.25 near the money.
    atm_band_log:
        Half-width, in absolute log-moneyness ``|ln(K / F_ref)|``, of the band
        where ``s_max_atm`` applies.
    n_min_otm_per_side:
        Minimum number of OTM quotes required on *each* side of the forward for
        the expiry to be usable for ``K_var``.
    n_min_skew_strikes / skew_band_log:
        An expiry that fails the ``K_var`` test is still usable for skew if it
        carries at least ``n_min_skew_strikes`` distinct strikes inside
        ``|ln(K / F_ref)| <= skew_band_log``.
    vendor_iv_sentinel:
        Vendor IVs at or below this level are placeholders. Flag only — vendor
        IV never gates a quote.
    arb_tol_abs / arb_tol_rel:
        Tolerance for the no-arbitrage checks:
        ``tol = arb_tol_abs + arb_tol_rel * scale`` with ``scale`` the largest
        mid involved in the comparison. ``arb_tol_abs`` defaults to ``0.0`` on
        purpose (no invented market number); on penny-quoted US chains set it to
        half a tick (``0.005``) so quote rounding is not reported as arbitrage.
    apply_spread_filter_to_one_sided:
        A zero-bid quote has ``(ask - bid) / mid == 2`` *identically*, whatever
        the ask: the ratio test degenerates and would drop every one-sided quote,
        contradicting the rule that one-sided quotes are kept and flagged for the
        tails. They are therefore exempt from the ratio test by default and
        governed by :data:`FLAG_ONE_SIDED` instead, leaving the decision to the
        consumer (see ``forward_curve.SurfaceConfig.include_one_sided``). Set to
        ``True`` to apply the ratio test to them anyway, which drops them all.
    zero_bid_stop_count:
        Number of *consecutive* zero-bid strikes that terminates a tail, walking
        outward from the money. This is the CBOE VIX white-paper rule and it is
        what replaces the degenerate ratio test on one-sided quotes (see
        ``apply_spread_filter_to_one_sided``): exempting one-sided quotes from
        the spread filter without a second gate lets worthless contracts through
        at their half-tick ask, and because those contracts cluster at short
        maturities the resulting ``K_var`` bias is maturity-dependent — measured
        at +6.9 % on a 7-day expiry versus +0.0 % at 6 months — which tilts the
        ``log|psi(T)|`` regression of spec 4.5 and biases *H* itself.
        Isolated zero-bid quotes inside the wall are still kept and flagged, so
        spec 4.1's "one-sided, usable for tails" rule stays alive; only the run
        that reaches ``zero_bid_stop_count`` and everything beyond it is dropped.
        Set to ``0`` to disable truncation entirely.
    drop_duplicate_strikes:
        Yahoo occasionally returns two contracts on the same (type, strike).
        Duplicates break the strike-ordered arbitrage scans (zero-width
        butterflies), so the tighter-spread quote is kept and the other is
        logged as :data:`REASON_DUPLICATE_STRIKE`. Disable at your own risk.
    assume_european_exercise:
        Stamp :data:`FLAG_AMERICAN_EXERCISE_ASSUMED_EUROPEAN` on every chain.
    """

    drop_stale: bool = True
    s_max_otm: float = 0.5
    s_max_atm: float = 0.25
    atm_band_log: float = 0.05
    n_min_otm_per_side: int = 3
    n_min_skew_strikes: int = 5
    skew_band_log: float = 0.20
    vendor_iv_sentinel: float = 1e-4
    arb_tol_abs: float = 0.0
    arb_tol_rel: float = 1e-6
    apply_spread_filter_to_one_sided: bool = False
    zero_bid_stop_count: int = 2
    drop_duplicate_strikes: bool = True
    assume_european_exercise: bool = True


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemovalRecord:
    """One dropped contract with the reason code that dropped it."""

    reason: str
    option_type: str
    strike: float
    T: float
    contract_symbol: str | None = None
    detail: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": str(self.reason),
            "reason_fr": REASON_LABELS_FR.get(self.reason, self.reason),
            "option_type": str(self.option_type),
            "strike": float(self.strike),
            "T": float(self.T),
            "contract_symbol": None if self.contract_symbol is None else str(self.contract_symbol),
            "detail": {str(k): float(v) for k, v in dict(self.detail).items()},
        }


@dataclass(frozen=True)
class CleanQuote:
    """
    A single surviving contract.

    ``mid`` is ``(bid + ask) / 2``; ``spread_rel`` is ``(ask - bid) / mid``.
    ``one_sided`` is ``True`` when ``bid == 0`` (no genuine two-way market): the
    quote is kept, flagged, and should only feed the tails.
    ``vendor_iv`` is carried for cross-checking and is never used as an input.
    """

    option_type: str
    strike: float
    T: float
    bid: float
    ask: float
    mid: float
    spread_abs: float
    spread_rel: float
    volume: float
    open_interest: float
    vendor_iv: float
    one_sided: bool
    contract_symbol: str | None = None
    flags: tuple[str, ...] = ()

    @property
    def liquidity(self) -> float:
        """Crude activity score used only as a deterministic tie-break."""
        return float(self.volume) + float(self.open_interest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_type": str(self.option_type),
            "strike": float(self.strike),
            "T": float(self.T),
            "bid": float(self.bid),
            "ask": float(self.ask),
            "mid": float(self.mid),
            "spread_abs": float(self.spread_abs),
            "spread_rel": float(self.spread_rel),
            "volume": float(self.volume),
            "open_interest": float(self.open_interest),
            "vendor_iv": float(self.vendor_iv),
            "one_sided": bool(self.one_sided),
            "contract_symbol": None if self.contract_symbol is None else str(self.contract_symbol),
            "flags": [str(f) for f in self.flags],
        }


@dataclass(frozen=True)
class ViabilityReport:
    """Minimum-viability verdict for one expiry (spec 4.1, bullet 5)."""

    forward_ref: float
    n_otm_calls: int
    n_otm_puts: int
    n_strikes_near_atm: int
    usable_for_kvar: bool
    usable_for_skew: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_ref": float(self.forward_ref),
            "n_otm_calls": int(self.n_otm_calls),
            "n_otm_puts": int(self.n_otm_puts),
            "n_strikes_near_atm": int(self.n_strikes_near_atm),
            "usable_for_kvar": bool(self.usable_for_kvar),
            "usable_for_skew": bool(self.usable_for_skew),
            "reasons": [str(r) for r in self.reasons],
        }


@dataclass(frozen=True)
class CleanChain:
    """
    Cleaned quotes for one expiry, plus the full audit trail.

    See the module docstring for the American-exercise caveat that applies to
    every chain sourced from Yahoo single-name / ETF data.
    """

    T: float
    underlying: str | None
    expiry: Any
    S0: float
    forward_ref: float
    calls: tuple[CleanQuote, ...]
    puts: tuple[CleanQuote, ...]
    removals: tuple[RemovalRecord, ...]
    viability: ViabilityReport
    flags: tuple[str, ...] = ()

    @property
    def quotes(self) -> tuple[CleanQuote, ...]:
        return tuple(self.calls) + tuple(self.puts)

    @property
    def n_quotes(self) -> int:
        return len(self.calls) + len(self.puts)

    @property
    def is_empty(self) -> bool:
        return self.n_quotes == 0

    def removals_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self.removals:
            counts[rec.reason] = counts.get(rec.reason, 0) + 1
        return counts

    def diagnostics(self) -> dict[str, Any]:
        """Plain-python diagnostics dict (survives the controller ``_json_safe``)."""
        return {
            "T": float(self.T),
            "underlying": None if self.underlying is None else str(self.underlying),
            "expiry": _expiry_to_str(self.expiry),
            "S0": float(self.S0),
            "forward_ref": float(self.forward_ref),
            "n_calls": int(len(self.calls)),
            "n_puts": int(len(self.puts)),
            "n_removed": int(len(self.removals)),
            "removals_by_reason": {str(k): int(v) for k, v in self.removals_by_reason().items()},
            "n_one_sided": int(sum(1 for q in self.quotes if q.one_sided)),
            "n_vendor_iv_sentinel": int(
                sum(1 for q in self.quotes if FLAG_VENDOR_IV_SENTINEL in q.flags)
            ),
            "viability": self.viability.to_dict(),
            "flags": [str(f) for f in self.flags],
        }

    def to_dict(self) -> dict[str, Any]:
        out = self.diagnostics()
        out["calls"] = [q.to_dict() for q in self.calls]
        out["puts"] = [q.to_dict() for q in self.puts]
        out["removals"] = [rec.to_dict() for rec in self.removals]
        return out


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float:
    """Cast to float, mapping ``None`` / NA / non-castable / non-finite to NaN."""
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _as_activity(value: Any) -> float:
    """Volume / open interest: missing means 'no recorded activity' (0.0)."""
    out = _as_float(value)
    if math.isnan(out) or out < 0.0:
        return 0.0
    return out


def _expiry_to_str(expiry: Any) -> str | None:
    if expiry is None:
        return None
    isoformat = getattr(expiry, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            return str(expiry)
    return str(expiry)


def _records(rows: Any) -> list[dict[str, Any]]:
    """Normalise a DataFrame / iterable-of-mappings into a list of plain dicts."""
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return []
        return list(rows.to_dict("records"))
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            out.append(dict(row))
    return out


def _median(values: Sequence[float]) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return float("nan")
    n = len(finite)
    mid = n // 2
    if n % 2 == 1:
        return float(finite[mid])
    return float(0.5 * (finite[mid - 1] + finite[mid]))


def _quality_key(quote: CleanQuote) -> tuple[float, float, float]:
    """
    Deterministic 'worse quote' ordering: larger key == worse.

    Ranked by relative spread first (the only real quote-quality signal), then by
    *lack* of activity, then by strike so ties never depend on input order.
    """
    spread = quote.spread_rel if math.isfinite(quote.spread_rel) else float("inf")
    return (float(spread), -float(quote.liquidity), float(quote.strike))


def _arb_tol(config: CleaningConfig, scale: float) -> float:
    scale = abs(float(scale)) if math.isfinite(scale) else 0.0
    return float(config.arb_tol_abs) + float(config.arb_tol_rel) * scale


def _log_moneyness(strike: float, reference: float) -> float:
    if not (math.isfinite(strike) and math.isfinite(reference)) or strike <= 0 or reference <= 0:
        return float("nan")
    return math.log(strike / reference)


# ---------------------------------------------------------------------------
# Stage 1 — per-contract validity
# ---------------------------------------------------------------------------


def _parse_quote(
    record: Mapping[str, Any],
    *,
    config: CleaningConfig,
    T_fallback: float,
    removals: list[RemovalRecord],
) -> CleanQuote | None:
    """Apply the per-contract filters. Returns ``None`` (and logs) on removal."""
    opt_type = str(record.get("type") or "").strip().lower()
    strike = _as_float(record.get("strike"))
    T = _as_float(record.get("T"))
    if math.isnan(T):
        T = float(T_fallback)
    symbol = record.get("contractSymbol")
    symbol = None if symbol is None else str(symbol)

    def _drop(reason: str, **detail: float) -> None:
        removals.append(
            RemovalRecord(
                reason=reason,
                option_type=opt_type or "unknown",
                strike=strike if math.isfinite(strike) else float("nan"),
                T=T if math.isfinite(T) else float("nan"),
                contract_symbol=symbol,
                detail={k: float(v) for k, v in detail.items()},
            )
        )

    if opt_type not in (CALL, PUT):
        _drop(REASON_BAD_TYPE)
        return None
    if not math.isfinite(strike) or strike <= 0.0:
        _drop(REASON_BAD_STRIKE, strike=strike)
        return None
    if not math.isfinite(T) or T <= 0.0:
        _drop(REASON_BAD_MATURITY, T=T)
        return None

    bid = _as_float(record.get("bid"))
    ask = _as_float(record.get("ask"))

    # --- quote validity: bid >= 0, ask > 0, ask >= bid, mid > 0 ------------
    if math.isnan(bid) or math.isnan(ask):
        _drop(REASON_MISSING_QUOTE, bid=bid, ask=ask)
        return None
    if bid < 0.0:
        _drop(REASON_NEGATIVE_BID, bid=bid, ask=ask)
        return None
    if ask <= 0.0:
        _drop(REASON_NONPOSITIVE_ASK, bid=bid, ask=ask)
        return None
    if ask < bid:
        _drop(REASON_CROSSED_QUOTE, bid=bid, ask=ask)
        return None

    mid = 0.5 * (bid + ask)
    if not math.isfinite(mid) or mid <= 0.0:
        _drop(REASON_NONPOSITIVE_MID, bid=bid, ask=ask, mid=mid)
        return None

    # --- staleness --------------------------------------------------------
    volume = _as_activity(record.get("volume"))
    open_interest = _as_activity(record.get("openInterest"))
    if config.drop_stale and volume == 0.0 and open_interest == 0.0:
        _drop(REASON_STALE_QUOTE, volume=volume, open_interest=open_interest, mid=mid)
        return None

    spread_abs = ask - bid
    spread_rel = spread_abs / mid

    # --- flags (never removals) -------------------------------------------
    flags: list[str] = []
    one_sided = bid == 0.0
    if one_sided:
        flags.append(FLAG_ONE_SIDED)

    vendor_iv = _as_float(record.get("iv"))
    if math.isnan(vendor_iv):
        flags.append(FLAG_VENDOR_IV_MISSING)
    elif vendor_iv <= float(config.vendor_iv_sentinel):
        flags.append(FLAG_VENDOR_IV_SENTINEL)

    return CleanQuote(
        option_type=opt_type,
        strike=float(strike),
        T=float(T),
        bid=float(bid),
        ask=float(ask),
        mid=float(mid),
        spread_abs=float(spread_abs),
        spread_rel=float(spread_rel),
        volume=float(volume),
        open_interest=float(open_interest),
        vendor_iv=float(vendor_iv),
        one_sided=bool(one_sided),
        contract_symbol=symbol,
        flags=tuple(flags),
    )


def _dedupe_strikes(
    quotes: list[CleanQuote],
    *,
    removals: list[RemovalRecord],
) -> list[CleanQuote]:
    """Keep the best-quality quote per strike; log every discarded duplicate."""
    best: dict[float, CleanQuote] = {}
    order: list[float] = []
    for quote in quotes:
        key = float(quote.strike)
        current = best.get(key)
        if current is None:
            best[key] = quote
            order.append(key)
            continue
        loser = quote if _quality_key(quote) >= _quality_key(current) else current
        winner = current if loser is quote else quote
        best[key] = winner
        removals.append(
            RemovalRecord(
                reason=REASON_DUPLICATE_STRIKE,
                option_type=loser.option_type,
                strike=loser.strike,
                T=loser.T,
                contract_symbol=loser.contract_symbol,
                detail={"spread_rel": loser.spread_rel, "kept_spread_rel": winner.spread_rel},
            )
        )
    return [best[k] for k in order]


def _filter_spread(
    quotes: list[CleanQuote],
    *,
    config: CleaningConfig,
    forward_ref: float,
    removals: list[RemovalRecord],
) -> list[CleanQuote]:
    """
    Drop quotes whose relative spread exceeds the band-dependent ceiling.

    One-sided quotes (``bid == 0``) are exempt unless
    ``CleaningConfig.apply_spread_filter_to_one_sided`` says otherwise: their
    relative spread is identically 2 by construction, so the ratio carries no
    information about quote quality. See :class:`CleaningConfig`.
    """
    kept: list[CleanQuote] = []
    for quote in quotes:
        if quote.one_sided and not config.apply_spread_filter_to_one_sided:
            kept.append(quote)
            continue
        k = _log_moneyness(quote.strike, forward_ref)
        near_atm = math.isfinite(k) and abs(k) <= float(config.atm_band_log)
        s_max = float(config.s_max_atm) if near_atm else float(config.s_max_otm)
        if quote.spread_rel > s_max:
            removals.append(
                RemovalRecord(
                    reason=REASON_WIDE_SPREAD,
                    option_type=quote.option_type,
                    strike=quote.strike,
                    T=quote.T,
                    contract_symbol=quote.contract_symbol,
                    detail={
                        "spread_rel": quote.spread_rel,
                        "s_max": s_max,
                        "log_moneyness": k if math.isfinite(k) else float("nan"),
                    },
                )
            )
            continue
        kept.append(quote)
    return kept


def _truncate_zero_bid_tail(
    quotes: list[CleanQuote],
    *,
    outward_desc: bool,
    config: CleaningConfig,
    forward_ref: float,
    removals: list[RemovalRecord],
) -> list[CleanQuote]:
    """
    Truncate a tail at the CBOE two-consecutive-zero-bid wall.

    Walks *outward from the money* — descending strikes for puts
    (``outward_desc=True``), ascending for calls — and once
    ``config.zero_bid_stop_count`` consecutive zero-bid strikes are seen, drops
    that run and every strike beyond it.

    Why this exists: a zero-bid quote has ``(ask - bid) / mid == 2`` identically,
    so the relative-spread ceiling of :func:`_filter_spread` cannot discriminate
    among one-sided quotes and they are exempt from it. Without a second gate,
    contracts worth ~0 survive at half their tick-wide ask, and their
    ``Q(K) * dK / K**2`` terms inflate the variance-swap rate. The distortion is
    concentrated at short maturities (where far strikes are genuinely worthless),
    so it does not cancel: it tilts the short-maturity slope that spec 4.5 fits
    to extract *H*. The CBOE VIX white paper — the reference named by spec 4.3 —
    resolves exactly this case with the consecutive-zero-bid stop, which is why
    that rule is adopted here rather than an invented price floor.

    Isolated zero-bid quotes *inside* the wall survive, flagged
    :data:`FLAG_ONE_SIDED`, so spec 4.1's "usable for the tails" rule is intact.
    """
    stop = int(config.zero_bid_stop_count)
    if stop <= 0 or not quotes:
        return list(quotes)

    # CBOE walks the OTM side only: puts strictly below the forward, calls
    # strictly above. ITM quotes are never candidates for truncation, so a stray
    # zero-bid pair among them can never amputate the chain.
    def _is_otm(quote: CleanQuote) -> bool:
        return quote.strike < forward_ref if outward_desc else quote.strike > forward_ref

    itm = [q for q in quotes if not _is_otm(q)]
    otm = sorted(
        (q for q in quotes if _is_otm(q)),
        key=lambda q: q.strike,
        reverse=bool(outward_desc),
    )

    cut_index: int | None = None
    run = 0
    for idx, quote in enumerate(otm):
        if quote.one_sided:
            run += 1
            if run >= stop:
                cut_index = idx - run + 1
                break
        else:
            run = 0

    if cut_index is None:
        return list(quotes)

    kept = itm + otm[:cut_index]
    for quote in otm[cut_index:]:
        k = _log_moneyness(quote.strike, forward_ref)
        removals.append(
            RemovalRecord(
                reason=REASON_ZERO_BID_TAIL,
                option_type=quote.option_type,
                strike=quote.strike,
                T=quote.T,
                contract_symbol=quote.contract_symbol,
                detail={
                    "zero_bid_stop_count": float(stop),
                    "mid": quote.mid,
                    "log_moneyness": k if math.isfinite(k) else float("nan"),
                },
            )
        )
    return sorted(kept, key=lambda q: q.strike)


# ---------------------------------------------------------------------------
# Stage 2 — static no-arbitrage repair (per expiry, per type, on mids)
# ---------------------------------------------------------------------------


def _repair_vertical(
    quotes: list[CleanQuote],
    *,
    non_increasing: bool,
    config: CleaningConfig,
    removals: list[RemovalRecord],
) -> list[CleanQuote]:
    """
    Enforce vertical-spread monotonicity on mids.

    Calls must be non-increasing in K, puts non-decreasing. On each violating
    adjacent pair the *worse-quality* quote (widest relative spread, then least
    active, then highest strike) is removed — a deterministic rule, so the
    removal log is reproducible.

    Only the monotonicity leg of the vertical no-arbitrage condition is enforced
    here; the slope bound ``|C(K1) - C(K2)| <= D * (K2 - K1)`` needs the discount
    factor, which belongs to spec 4.2 and is not available at this stage.
    """
    kept = sorted(quotes, key=lambda q: q.strike)
    while True:
        violation = -1
        for i in range(len(kept) - 1):
            left, right = kept[i], kept[i + 1]
            tol = _arb_tol(config, max(left.mid, right.mid))
            gap = (right.mid - left.mid) if non_increasing else (left.mid - right.mid)
            if gap > tol:
                violation = i
                break
        if violation < 0:
            return kept
        left, right = kept[violation], kept[violation + 1]
        loser = right if _quality_key(right) >= _quality_key(left) else left
        other = left if loser is right else right
        removals.append(
            RemovalRecord(
                reason=REASON_VERTICAL_ARBITRAGE,
                option_type=loser.option_type,
                strike=loser.strike,
                T=loser.T,
                contract_symbol=loser.contract_symbol,
                detail={
                    "mid": loser.mid,
                    "neighbour_strike": other.strike,
                    "neighbour_mid": other.mid,
                    "spread_rel": loser.spread_rel,
                },
            )
        )
        kept = [q for q in kept if q is not loser]


def _repair_butterfly(
    quotes: list[CleanQuote],
    *,
    config: CleaningConfig,
    removals: list[RemovalRecord],
) -> list[CleanQuote]:
    """
    Enforce convexity of mids in K (both calls and puts are convex in K).

    For consecutive strikes ``K1 < K2 < K3`` with ``w = (K3 - K2) / (K3 - K1)``
    the butterfly is non-negative iff ``mid(K2) <= w * mid(K1) + (1 - w) * mid(K3)``.
    A violation means the *middle* strike is too expensive relative to its wings,
    so the middle quote is the one removed. Removing an interior point of a
    monotone sequence preserves monotonicity, so no second vertical pass is
    needed.
    """
    kept = sorted(quotes, key=lambda q: q.strike)
    while True:
        violation = -1
        detail: dict[str, float] = {}
        for i in range(len(kept) - 2):
            left, mid_q, right = kept[i], kept[i + 1], kept[i + 2]
            span = right.strike - left.strike
            if span <= 0.0:
                continue
            w = (right.strike - mid_q.strike) / span
            bound = w * left.mid + (1.0 - w) * right.mid
            tol = _arb_tol(config, max(left.mid, mid_q.mid, right.mid))
            if mid_q.mid > bound + tol:
                violation = i + 1
                detail = {
                    "mid": mid_q.mid,
                    "convex_bound": float(bound),
                    "left_strike": left.strike,
                    "right_strike": right.strike,
                    "butterfly": float(bound - mid_q.mid),
                }
                break
        if violation < 0:
            return kept
        loser = kept[violation]
        removals.append(
            RemovalRecord(
                reason=REASON_BUTTERFLY_ARBITRAGE,
                option_type=loser.option_type,
                strike=loser.strike,
                T=loser.T,
                contract_symbol=loser.contract_symbol,
                detail=detail,
            )
        )
        kept = [q for q in kept if q is not loser]


# ---------------------------------------------------------------------------
# Stage 3 — minimum viability
# ---------------------------------------------------------------------------


def evaluate_viability(
    calls: Sequence[CleanQuote],
    puts: Sequence[CleanQuote],
    *,
    forward_ref: float,
    config: CleaningConfig | None = None,
) -> ViabilityReport:
    """
    Spec 4.1, bullet 5.

    ``usable_for_kvar`` requires at least ``n_min_otm_per_side`` OTM quotes on
    each side of ``forward_ref`` (OTM calls: ``K > F``; OTM puts: ``K < F``).
    An expiry that fails that test is still ``usable_for_skew`` when it carries
    at least ``n_min_skew_strikes`` distinct strikes inside
    ``|ln(K / forward_ref)| <= skew_band_log``.

    Exposed separately from :func:`clean_expiry_chain` so spec 4.2 can re-run it
    once the *true* put-call-parity forward is known, instead of the spot proxy.
    """
    cfg = config or CleaningConfig()
    f_ref = float(forward_ref)
    reasons: list[str] = []

    if not math.isfinite(f_ref) or f_ref <= 0.0:
        return ViabilityReport(
            forward_ref=f_ref,
            n_otm_calls=0,
            n_otm_puts=0,
            n_strikes_near_atm=0,
            usable_for_kvar=False,
            usable_for_skew=False,
            reasons=("no_forward_reference",),
        )

    n_otm_calls = sum(1 for q in calls if q.strike > f_ref)
    n_otm_puts = sum(1 for q in puts if q.strike < f_ref)

    near_atm_strikes = {
        float(q.strike)
        for q in list(calls) + list(puts)
        if math.isfinite(_log_moneyness(q.strike, f_ref))
        and abs(_log_moneyness(q.strike, f_ref)) <= float(cfg.skew_band_log)
    }
    n_near = len(near_atm_strikes)

    usable_for_kvar = (
        n_otm_calls >= int(cfg.n_min_otm_per_side) and n_otm_puts >= int(cfg.n_min_otm_per_side)
    )
    if not usable_for_kvar:
        if n_otm_calls < int(cfg.n_min_otm_per_side):
            reasons.append("insufficient_otm_calls")
        if n_otm_puts < int(cfg.n_min_otm_per_side):
            reasons.append("insufficient_otm_puts")

    usable_for_skew = usable_for_kvar or n_near >= int(cfg.n_min_skew_strikes)
    if not usable_for_skew:
        reasons.append("insufficient_atm_strikes")

    return ViabilityReport(
        forward_ref=f_ref,
        n_otm_calls=int(n_otm_calls),
        n_otm_puts=int(n_otm_puts),
        n_strikes_near_atm=int(n_near),
        usable_for_kvar=bool(usable_for_kvar),
        usable_for_skew=bool(usable_for_skew),
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def clean_expiry_chain(
    rows: Any,
    *,
    config: CleaningConfig | None = None,
    forward_ref: float | None = None,
    T: float | None = None,
) -> CleanChain:
    """
    Clean the raw rows of **one** expiry into a :class:`CleanChain`.

    Parameters
    ----------
    rows:
        A ``pandas.DataFrame`` or any iterable of mappings using the
        ``fetch_options_details_yahoo`` row schema (``type``, ``strike``, ``T``,
        ``bid``, ``ask``, ``iv``, ``volume``, ``openInterest``, ``S0``, ...).
        Calls and puts may be mixed.
    config:
        Filter thresholds; defaults to :class:`CleaningConfig`.
    forward_ref:
        Moneyness reference used by the ATM spread band and by the viability
        test. Defaults to the median ``S0`` of the rows — the forward is not
        known until spec 4.2, and the caller is expected to re-run
        :func:`evaluate_viability` with the parity forward afterwards.
    T:
        Year fraction of the expiry; defaults to the median ``T`` of the rows.

    Never raises on malformed input: a NaN-riddled chain yields an empty chain
    flagged :data:`FLAG_EMPTY_CHAIN` with one removal record per rejected row.
    """
    cfg = config or CleaningConfig()
    records = _records(rows)
    removals: list[RemovalRecord] = []

    T_rows = [_as_float(rec.get("T")) for rec in records]
    T_value = float(T) if T is not None and math.isfinite(_as_float(T)) else _median(T_rows)
    if not math.isfinite(T_value):
        T_value = float("nan")

    S0_rows = [_as_float(rec.get("S0")) for rec in records]
    S0_value = _median(S0_rows)

    underlying: str | None = None
    expiry: Any = None
    for rec in records:
        if underlying is None and rec.get("underlying") is not None:
            underlying = str(rec.get("underlying"))
        if expiry is None and rec.get("expiry") is not None:
            expiry = rec.get("expiry")
        if underlying is not None and expiry is not None:
            break

    chain_flags: list[str] = []
    if cfg.assume_european_exercise:
        chain_flags.append(FLAG_AMERICAN_EXERCISE_ASSUMED_EUROPEAN)

    if forward_ref is not None and math.isfinite(_as_float(forward_ref)) and float(forward_ref) > 0:
        f_ref = float(forward_ref)
    else:
        f_ref = S0_value
    if not (math.isfinite(f_ref) and f_ref > 0.0):
        chain_flags.append(FLAG_NO_SPOT)

    parsed: list[CleanQuote] = []
    for rec in records:
        quote = _parse_quote(rec, config=cfg, T_fallback=T_value, removals=removals)
        if quote is not None:
            parsed.append(quote)

    calls = [q for q in parsed if q.option_type == CALL]
    puts = [q for q in parsed if q.option_type == PUT]

    if cfg.drop_duplicate_strikes:
        calls = _dedupe_strikes(calls, removals=removals)
        puts = _dedupe_strikes(puts, removals=removals)

    if math.isfinite(f_ref) and f_ref > 0.0:
        calls = _filter_spread(calls, config=cfg, forward_ref=f_ref, removals=removals)
        puts = _filter_spread(puts, config=cfg, forward_ref=f_ref, removals=removals)
    else:
        # No moneyness reference: apply the (looser) tail ceiling everywhere so
        # the filter still runs rather than being silently skipped.
        tail_only = replace(cfg, s_max_atm=cfg.s_max_otm)
        calls = _filter_spread(calls, config=tail_only, forward_ref=1.0, removals=removals)
        puts = _filter_spread(puts, config=tail_only, forward_ref=1.0, removals=removals)

    # CBOE zero-bid wall — the second gate on one-sided quotes. Needs the
    # moneyness reference to know which way is "outward", so it is skipped (with
    # the chain already flagged FLAG_NO_SPOT) when there is none.
    if math.isfinite(f_ref) and f_ref > 0.0:
        calls = _truncate_zero_bid_tail(
            calls, outward_desc=False, config=cfg, forward_ref=f_ref, removals=removals
        )
        puts = _truncate_zero_bid_tail(
            puts, outward_desc=True, config=cfg, forward_ref=f_ref, removals=removals
        )

    calls = _repair_vertical(calls, non_increasing=True, config=cfg, removals=removals)
    puts = _repair_vertical(puts, non_increasing=False, config=cfg, removals=removals)
    calls = _repair_butterfly(calls, config=cfg, removals=removals)
    puts = _repair_butterfly(puts, config=cfg, removals=removals)

    calls.sort(key=lambda q: q.strike)
    puts.sort(key=lambda q: q.strike)

    viability = evaluate_viability(calls, puts, forward_ref=f_ref, config=cfg)
    if not calls and not puts:
        chain_flags.append(FLAG_EMPTY_CHAIN)

    return CleanChain(
        T=float(T_value),
        underlying=underlying,
        expiry=expiry,
        S0=float(S0_value),
        forward_ref=float(f_ref),
        calls=tuple(calls),
        puts=tuple(puts),
        removals=tuple(removals),
        viability=viability,
        flags=tuple(chain_flags),
    )


def _expiry_key(record: Mapping[str, Any]) -> Any:
    """Deterministic grouping key: expiry timestamp > expiry date > rounded T."""
    ts = _as_float(record.get("expiry_ts"))
    if math.isfinite(ts):
        return ("ts", int(ts))
    expiry = record.get("expiry")
    if expiry is not None:
        return ("expiry", _expiry_to_str(expiry))
    T = _as_float(record.get("T"))
    if math.isfinite(T):
        return ("T", round(T, 12))
    return ("unknown", None)


def clean_option_chains(
    calls: Any = None,
    puts: Any = None,
    *,
    config: CleaningConfig | None = None,
    forward_ref_by_expiry: Mapping[Any, float] | None = None,
    spot: float | None = None,
) -> list[CleanChain]:
    """
    Clean a full multi-expiry chain (the ``(calls_df, puts_df)`` pair returned by
    ``fetch_options_details_yahoo``) into one :class:`CleanChain` per expiry,
    sorted by increasing maturity.

    ``forward_ref_by_expiry`` maps the grouping key (see :func:`_expiry_key`) to
    an already-known forward; ``spot`` overrides the per-row ``S0`` when the rows
    do not carry one. Both are optional: by default each expiry uses its own
    median ``S0`` as the moneyness reference.
    """
    cfg = config or CleaningConfig()
    records = _records(calls) + _records(puts)

    grouped: dict[Any, list[dict[str, Any]]] = {}
    order: list[Any] = []
    for rec in records:
        key = _expiry_key(rec)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(rec)

    chains: list[CleanChain] = []
    for key in order:
        rows = grouped[key]
        ref: float | None = None
        if forward_ref_by_expiry is not None:
            candidate = _as_float(forward_ref_by_expiry.get(key))
            if math.isfinite(candidate) and candidate > 0:
                ref = float(candidate)
        if ref is None and spot is not None:
            candidate = _as_float(spot)
            if math.isfinite(candidate) and candidate > 0:
                ref = float(candidate)
        chains.append(clean_expiry_chain(rows, config=cfg, forward_ref=ref))

    chains.sort(key=lambda c: (not math.isfinite(c.T), c.T))
    return chains


def cleaning_report(chains: Sequence[CleanChain]) -> dict[str, Any]:
    """Aggregate diagnostics across expiries, ready for the Phase-5 report."""
    totals: dict[str, int] = {}
    for chain in chains:
        for reason, count in chain.removals_by_reason().items():
            totals[reason] = totals.get(reason, 0) + count
    return {
        "n_expiries": int(len(chains)),
        "n_quotes_kept": int(sum(c.n_quotes for c in chains)),
        "n_quotes_removed": int(sum(len(c.removals) for c in chains)),
        "removals_by_reason": {str(k): int(v) for k, v in totals.items()},
        "removal_labels_fr": {str(k): REASON_LABELS_FR.get(k, k) for k in totals},
        "n_usable_for_kvar": int(sum(1 for c in chains if c.viability.usable_for_kvar)),
        "n_usable_for_skew": int(sum(1 for c in chains if c.viability.usable_for_skew)),
        "exercise_style_caveat": AMERICAN_EXERCISE_CAVEAT,
        "expiries": [c.diagnostics() for c in chains],
    }


__all__ = [
    "AMERICAN_EXERCISE_CAVEAT",
    "CALL",
    "PUT",
    "CleanChain",
    "CleanQuote",
    "CleaningConfig",
    "FLAG_AMERICAN_EXERCISE_ASSUMED_EUROPEAN",
    "FLAG_EMPTY_CHAIN",
    "FLAG_NO_SPOT",
    "FLAG_ONE_SIDED",
    "FLAG_VENDOR_IV_MISSING",
    "FLAG_VENDOR_IV_SENTINEL",
    "REASON_BAD_MATURITY",
    "REASON_BAD_STRIKE",
    "REASON_BAD_TYPE",
    "REASON_BUTTERFLY_ARBITRAGE",
    "REASON_CROSSED_QUOTE",
    "REASON_DUPLICATE_STRIKE",
    "REASON_LABELS_FR",
    "REASON_MISSING_QUOTE",
    "REASON_NEGATIVE_BID",
    "REASON_NONPOSITIVE_ASK",
    "REASON_NONPOSITIVE_MID",
    "REASON_STALE_QUOTE",
    "REASON_VERTICAL_ARBITRAGE",
    "REASON_WIDE_SPREAD",
    "REASON_ZERO_BID_TAIL",
    "RemovalRecord",
    "ViabilityReport",
    "clean_expiry_chain",
    "clean_option_chains",
    "cleaning_report",
    "evaluate_viability",
]
