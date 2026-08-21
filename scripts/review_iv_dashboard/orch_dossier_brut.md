# DOSSIER BRUT — findings vérifiés (source de vérité pour la rédaction)



## M1 — stale-chain-cache-as-current-iv
- titre: Fallback chain path serves a stale cached chain as today's IV, with expiries shifted by the cache age, and persists it
- ancre: app/model/iv_dashboard/service.py:372
- sévérité finale (vote): M  (votes: M, M, M; finder: M)
- effort: 1 h
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_alpaca_plumbing_snapshot_mock.py

### claim
When the filtered snapshot call fails (network/403/429), fetch_current_atm_iv falls back to download_options_alpaca(), which on its own failure returns the last cached options_alpaca_{SYM}.csv of ANY age; _contracts_from_chain_df (L314) then rebuilds expiry = today + T*365 from a T computed on the cache date, so expiries drift by the cache age, the cached IVs are reported as 'greeks Alpaca' and record_iv_observation writes them as today's observation.

### evidence (finder + orchestrateur)
p1_alpaca_plumbing_snapshot_mock.py scenario D (requests.get raises ConnectionError, 7-day-old cache with iv=0.25, real expiry today+30): log='162 contrats via chaîne Alpaca (cache)' then 'IV ATM 0.2500 (25.00%) — échéance 2026-09-27 (37 j), 126 contrats, méthode : greeks Alpaca'; reported expiry 2026-09-27 vs real 2026-09-20 (shift +7 d); iv_daily_SPY.csv row {'date': '2026-08-21', 'iv': 0.25, 'dte': 37, 'method': 'greeks Alpaca'}.

### impact
Displayed 'IV courante', 'Spread IV − RV', 'Percentile IV' and the accumulated daily IV history can be a days-old number labelled as live Alpaca greeks with a wrong expiry/DTE, exactly on the days the API is unreachable.

### fix_sketch
In the fallback, call download_options_alpaca(..., cache_to_csv=False) (or check the cache file mtime and refuse if older than the session) and carry a real expiry column instead of recomputing it from T; set method='chaîne Alpaca (cache, <age>)' and skip record_iv_observation when the source is the cache.

### panel
- [CODE READING] refuted=false conf=0.92 sev=M
  reason: Reading confirmed end to end. service.py L372 calls download_options_alpaca(sym, feed=..., max_pages=3) with cache_to_csv left at its default True (logic.py L1260). In logic.py, any page failure sets payload=None and breaks (L1405-1413 for 403, L1432-1440 for exceptions) -> df empty -> L1540-1549 returns pd.read_csv(out_path) with NO mtime/TTL check (the 60 s TTL at L1284-1309 only guards the pre-fetch path). The CSV carries T = days_to_expiry/365 computed on the cache date (L1484-1506); _contracts_from_chain_df L314 rebuilds expiry = today + round(T*365), so expiry drifts by the cache age. With iv present, L406-408 count them as n_direct -> method='greeks Alpaca' (L429-430); get_iv_dashboard_data L567-568 then calls record_iv_observation unconditionally. The cache file is also written by calibration_controller.py L257 and options_controller.py L181, so a days-old options_alpaca_SPY.csv is a realistic state, not a contrived one. No guard anywhere in service/controller/view distinguishes the cache source: the log line L374 says '(cache)' but info['method']/['feed'] do not, and the view caption (tab_iv_dashboard.py L243-247) shows method/feed only. Re-ran the mock: scenario D reproduces exactly (iv 0.25, expiry 2026-09-27 vs real 2026-09-20, +7 d, row persisted in iv_daily_SPY.csv with method 'greeks Alpaca').
  corrected_evidence: Re-run 2026-08-21 of p1_alpaca_plumbing_snapshot_mock.py scenario D: log = ['Snapshots filtrés indisponibles (Failed to establish a new connection) ; fallback chaîne complète.', '162 contrats via chaîne Alpaca (cache).', 'IV ATM 0.2500 (25.00%) — échéance 2026-09-27 (37 j), 126 contrats, méthode : greeks Alpaca.']; persisted row {'date': '2026-08-21', 'iv': 0.25, 'dte': 37, 'n_contracts': 126, 'method': 'greeks Alpaca', 'spot': 640.0}. Precondition worth stating: the cached chain must contain IVs (opra feed, or an indicative snapshot that carried greeks); with iv=NaN the chain path yields 'Aucune IV exploitable' instead (mid=None at L322), i.e. the bug is silent-wrong only when the cache is 'rich'.
  fix_ok=true — Proportionate. Simpler variant for the expiry part: the cache CSV already has an `opra` column (logic.py L1497), so _contracts_from_chain_df can decode the real expiry with the existing _decode_opra (L171) instead of recomputing from T — no schema change in logic.py needed. cache_to_csv=False also disables the 60 s TTL pre-read, which is fine here. Tagging method/feed as cache and skipping record_iv_observation for cache-sourced info are both necessary because the view only shows method/feed.
- [REPRODUCTION] refuted=false conf=0.9 sev=M
  reason: Reproduced with my own mock (independent chain, 3 failure modes). service.py L372 calls download_options_alpaca with cache_to_csv=True (default); logic.py L1540-1549 returns the on-disk options_alpaca_SPY.csv of ANY age when the live fetch yields an empty frame (ConnectionError, 403 and 429 all end there). service.py L314 then rebuilds expiry = today + round(T*365) where T was frozen at cache time, so expiry shifts by exactly the cache age; L430 labels it 'greeks Alpaca' and get_iv_dashboard_data L568 persists it as today's row. Measured: 7-day-old cache, real expiry 2026-09-20 -> reported 2026-09-27 (37 j), iv=0.22 from the file, iv_daily_SPY.csv row {'date':'2026-08-21','iv':0.22,'dte':37,'method':'greeks Alpaca'}; 3-day-old + 403 -> +3 d; 10-day-old + 429 -> +10 d. Realistic: fetch_spot_price is Stooq with a permanent cache, so spot is available even with Alpaca unreachable.
  corrected_evidence: Preconditions the finding omits: (1) the cached chain must carry non-NaN IVs — with NaN IVs (indicative feed w/o greeks) the same path ends in 'Aucune IV exploitable...' and info=None (no stale reading); (2) the shifted DTE must stay in [15,60]: cache age 20 d -> dte 50 still accepted, cache age 40 d -> 'Aucun contrat entre 15 et 60 jours' and None. HTTP attempts: 4 for ConnectionError/429 (1 snapshot + 3 retries via download_options_alpaca), 2 for 403. The cached CSV already contains the real expiry in its `opra` column (SPY260920C... -> _decode_opra -> 2026-09-20 vs T-derived 2026-10-30 at 40 d age).
  fix_ok=true — Proportionate. Simplest shift fix: in _contracts_from_chain_df use svc._decode_opra(r['opra'])[1] when available instead of recomputing from T (no schema change: the chain CSV already has `opra`). For staleness: cache_to_csv=False on the fallback call removes the any-age read (it also disables the 60 s TTL reuse, acceptable); then skip record_iv_observation or tag method/feed when contracts came from a cache. Note fetch_spot_price (Stooq, permanent cache) is a second stale source on the same path, out of this finding's scope.
- [IMPACT & SEVERITY] refuted=false conf=0.85 sev=M
  reason: Could not break it. Re-ran the path through the real service/logic code with requests.get mocked (p4_fallback_impact.py S1): ConnectionError on the filtered snapshots + a 10-day-old cache/AlpacaOptionChains/options_alpaca_SPY.csv (iv=0.25) -> fetch_current_atm_iv returns iv=0.25, expiry 2026-09-20 dte=30 whereas the contract's real expiry is 2026-09-10 (true dte 20: shift = cache age), method='greeks Alpaca', and record_iv_observation writes {'date': today, 'iv': 0.25, 'dte': 30, 'method': 'greeks Alpaca'} into iv_daily_SPY.csv. The view caption (tab_iv_dashboard.py L243-247) prints 'méthode : greeks Alpaca · flux : indicative' with no cache/age hint; the only '(cache)' mention is the log line in the collapsed 'Journal' expander (L476-497), so the staleness is silent. Preconditions are realistic: (1) the snapshot endpoint failing was demonstrated live (orch_live_alpaca.out.txt: 401 with revoked keys), and download_options_alpaca shares the same endpoint/credentials so its live leg fails too (logic.py L1405-1437 -> payload None -> L1540-1549 returns the cached CSV of ANY age); (2) the cache is written by the Advanced-calibration 'Load chain' button and the Alpaca-options tab with 'Cache chain to CSV' defaulting to True (tab_alpaca_options.py L388-392) -> a user of the pricers leaves one behind routinely; (3) the calibration tab consumes the chain's iv column, so the cache typically carries IVs. Blast-radius limiter (S3): a cache without iv yields 'Aucune IV exploitable' and no number. Severity: keeps M (wrong displayed IV/DTE + 'Spread IV − RV' + IV signal chip + polluted persisted history, silent, exactly on API-down days) — matches all three M descriptors; not C because it needs a prior failure plus a foreign cache, i.e. it is not the nominal path.
  corrected_evidence: p4_fallback_impact.py S1 (10-day-old cache, iv=0.25, ConnectionError): displayed expiry 2026-09-20 / dte 30 vs real expiry 2026-09-10 / true dte 20 (shift = 10 d = cache age); caption 'méthode : greeks Alpaca · flux : indicative'; iv_daily_SPY.csv row {'date': '2026-08-21', 'iv': 0.25, 'dte': 30, 'n_contracts': 18, 'method': 'greeks Alpaca'}; 4 HTTP attempts / 1.51 s spent before serving the cache. S3 (same cache, iv NaN): current_iv None, 'Aucune IV exploitable…' -> no number displayed. S2: the cache's `opra` column already decodes to the real expiry via svc._decode_opra (e.g. SPY260813C00470000 -> 2026-08-13).
  fix_ok=true — Correct but over-specified. Minimal proportionate fix: pass cache_to_csv=False (kills the any-age cache read at logic.py L1540-1549; also disables the 60 s TTL reuse, negligible) — or, better and consistent with finding fallback-chain-cannot-reach-30dte, delete the fallback block (service.py L369-379) since its live leg never rescues anything. 'Carry a real expiry column' is unnecessary: `opra` already encodes the expiry and svc._decode_opra exists (S2); with cache_to_csv=False the T-recompute is same-day and exact. 'method=chaîne Alpaca (cache, <age>)' / skipping record_iv_observation become moot once the cache path is gone. No regression expected.


## M2 — iv-signal-vrp-bias
- titre: « Signal (IV) » applique la grille mean-reversion à un percentile IV-dans-distribution-RV : biais structurel VRP
- ancre: app/model/iv_dashboard/service.py:571
- sévérité finale (vote): M  (votes: M, M, M; finder: M)
- effort: 1 h
- dimension finder: ? (fusion: iv-signal-chip-structurally-biased)
- repro: scripts/review_iv_dashboard/p1_math_rv_edges_epistemics.py ; scripts/review_iv_dashboard/p1_view_contrast_bias_ranges.py (section B) ; scripts/review_iv_dashboard/p1_view_sinks_and_state.py (clé 'copy')

### claim
iv_regime = classify_regime(percentile_within(RV_tail, IV)) produit le chip « Signal (IV) = MEAN REVERSION ↓ ATTENDUE » dès que l'IV dépasse ~80 % de la distribution RV, ce qui est l'état normal d'un sous-jacent à prime de risque de vol (IV > RV la plupart des jours) — le signal n'est pas un IV rank et ne mesure pas la cherté de l'IV vs son propre historique.

### evidence (finder + orchestrateur)
p1_math_rv_edges_epistemics.py §5 : RV~N(15 %,2 %) sur 252 j, IV=18 % -> percentile 94.0 % -> régime VOL ÉLEVÉE / Signal (IV) = « MEAN REVERSION ↓ ATTENDUE » ; IV=20 % -> 98.8 %. Avec RV lognormale (médiane 13.7 %, q90 21.6 %) : IV=20 % -> 83.3 % -> même signal. Fréquence empirique IV>RV sur SPY : non mesuré (pas de réseau).
--- (doublon ? iv-signal-chip-structurally-biased @ app/vue/tabs/tab_iv_dashboard.py:238) ---
p1_view_contrast_bias_ranges.py §B, GBM sigma=18 % constant, fenêtre 252 : IV = médiane RV +2 pts -> pct 0.643 'AU-DESSUS DE LA MOYENNE/NEUTRE' ; +3 pts -> 0.754 ; +4 pts -> 0.861 'VOL ÉLEVÉE / MEAN REVERSION ↓ ATTENDUE' ; +5 pts -> 0.901. Payload du driver de test (IV = RV+3 pts) rendu via AppTest : métrique 'Percentile IV vs série RV' = '100.0%', chip 'MEAN REVERSION ↓ ATTENDUE' (p1_view_sinks_and_state.py §copy). Niveau réel du VRP SPY : non mesuré (pas de réseau).

### impact
Un trader lit « mean reversion ↓ attendue » sur l'IV quasi tous les jours calmes d'un indice, comme si l'IV était riche alors qu'elle est à son niveau habituel au-dessus de la RV ; le chip pousse à vendre de la vol sans information. Le libellé de la métrique d3 (« Percentile IV vs série RV », help « position de l'IV dans la distribution récente de la RV ») est honnête, mais le chip d4 réutilise mot pour mot le signal de régime sans cette nuance.

### fix_sketch
Option minimale : ne pas dériver de signal mean-reversion de ce percentile — service.py L571 `iv_regime = None` (ou garder seulement le régime sans signal) et tab L237-242 remplacer le chip par un caption « IV au {p:.0%} de la RV récente — prime de risque de vol, pas un IV rank ». Option utile : quand `iv_history` a >= 60 observations, calculer un vrai IV percentile `percentile_within(iv_history['iv'], iv)` et n'afficher le signal que sur celui-ci.

### panel
- [CODE READING] refuted=false conf=0.85 sev=M
  reason: Reading is correct. service.py L569-571: `trailing = series_df['vol'].tail(percentile_window)`; `iv_vs_series_percentile = percentile_within(trailing, current_iv['iv'])`; `iv_regime = classify_regime(iv_vs_series_percentile)` — the same mean-reversion grid (analytics.py L127-132, >0.8 -> 'MEAN REVERSION ↓ ATTENDUE') is applied to an IV-inside-RV-distribution rank. View L237-242 renders `iv_regime['signal_label']` under the caption 'Signal (IV)' with no qualifier; only the d3 metric (L232-235) carries the honest help text. Nothing elsewhere (controller is a pure clamp passthrough, L29-52) re-interprets or suppresses the chip. The legacy script computed its percentile on IV's OWN history (option_trading_dashboard.py L331 `implied_vol.rolling(252).rank(pct=True)`), so the port silently changed the semantics of the signal while keeping the legacy wording — and this departure is not listed in analytics.py's 'Differences vs the legacy script' docstring (L9-13). I could not find a guard, branch or caption that neutralises it. Tried to refute via 'the d3 help makes it clear': the help is on a different widget and the chip is the signal output a trader reads.
  corrected_evidence: p4 probe, independent GBM sigma=18% constant, years=2, pwin=252: RV median 16.17%, q80 18.26%, q90 19.16%. IV = median +2 pts -> pct 0.770 NEUTRE ; +3 pts -> 0.901 'VOL ÉLEVÉE / MEAN REVERSION ↓ ATTENDUE' ; +4 pts -> 0.992 ; +5 pts -> 1.000. Lognormal RV (median 15%, sdlog 0.2): IV 18% -> 0.821 -> signal down ; IV 20% -> 0.929. A 3-pt IV-RV spread (ordinary VRP on an index) already fires the sell-vol signal on a constant-vol process.
  fix_ok=true — Minimal option is proportionate but must touch both sides: setting `iv_regime = None` alone makes view L217 fall back to `{}` and the chip prints 'N/A' (L240) — the chip must be replaced by the caption as the sketch says. tests/integration/_iv_dashboard_render_driver.py L83 builds `iv_regime` itself and would need the same adjustment; tests/test_iv_dashboard_analytics.py only asserts on keys, unaffected. The 'useful' option (percentile within `iv_history` when >= 60 obs) matches the legacy semantics (IV rank on IV history) and is the right long-term direction; note iv_history fills one row per analysis day (service.py L463-481), so the signal would stay hidden for ~3 months of daily use.
- [REPRODUCTION] refuted=false conf=0.8 sev=M
  reason: Reproduced on real data, not GBM: cache/OHLC/stooq_aapl.us_start_end_d.csv (AAPL 2016-01-04 -> 2026-01-08, 2519 RV(20) days, median RV 23.4 %, q10 14.4 %, q90 41.0 %), using the repo's own compute_realized_vol / percentile_within / classify_regime exactly as service.py L569-571 (trailing = last 252 RV rows). A CONSTANT premium carrying zero information about IV richness fires signal_key='down': IV=RV+3 pts -> 33.8 % of days, +5 pts -> 39.7 %, x1.25 -> 43.1 %, x1.4 (SPY-like VIX/RV ratio) -> 54.5 %; the 'up' signal collapses to 9.7 % / 4.8 % / 7.9 % / 3.7 %. The legacy construct (percentile of the same series within ITS OWN trailing 252d, option_trading_dashboard.py L331) gives the symmetric 24.3 % / 24.8 %. So the chip is structurally asymmetric under VRP and does not measure IV richness vs its own history — claim holds. Magnitude correction: the finder's GBM (constant sigma -> RV distribution only ~2-3 pts wide) overstates it; real fat-tailed RV gives 'roughly a third to half of days', not 'quasi tous les jours'. Severity kept at M: it is the headline 'Signal (IV)' chip on the tab, same wording as the RV signal, pushing a sell-vol reading.
  corrected_evidence: Real AAPL RV(20), trailing 252, constant premium: P(down) = 23.9 % (k=0) / 30.3 % (+2 pts) / 33.8 % (+3) / 36.4 % (+4) / 39.7 % (+5) / 50.6 % (+8); multiplicative x1.15 -> 35.4 %, x1.25 -> 43.1 %, x1.4 -> 54.5 %; P(up) falls to 14.0 % / 9.7 % / 7.5 % / 4.8 % / 0.9 %. Legacy IV-own-history construct with the same +3 pts: P(down) 24.3 %, P(up) 24.8 %. GBM sigma=18 %, median over 200 seeds: IV = median RV + 2/3/4/5 pts -> pct 0.766 / 0.849 / 0.913 / 0.956 (finder's single-seed 0.643/0.754/0.861/0.901 are within the q10-q90 spread). SPY not measurable offline (no cached SPY closes).
  fix_ok=true — Both options are proportionate. Minimal option: iv_regime=None renders 'N/A' through `iv_regime or {}` (tab L222, L237-242) with no other consumer of iv_regime in app/ — safe. Useful option (percentile_within(iv_history['iv'], iv) when >= 60 obs) restores the legacy IV-rank semantics; note record_iv_observation upserts one row per analysis day, so the signal stays 'N/A' for the first ~60 trading days of use — acceptable and honest.
- [IMPACT & SEVERITY] refuted=false conf=0.7 sev=M
  reason: Behaviour confirmed (service.py L571 classify_regime(percentile_within(RV tail, IV)) -> tab L238-242 coloured chip 'Signal (IV)'). Legacy option_trading_dashboard.py L331/L395 ranked IV within its OWN 252-day IV history; the port applies the same mean-reversion grid to IV-within-RV-distribution, a quantity that structurally does not revert toward RV (VRP). Impact is real but the phase-1 magnitude is overstated: the evidence used N(15%,2%)/constant-vol GBM with no fat right tail. On real AAPL RV20 (2373 days, p4_impact_g2_smooth_iv.py) MR-down fires 26-45% of days for realistic IV proxies vs 22.8% baseline and only 0-17% on calm days, not 'quasi tous les jours calmes'. What survives: the chip is one-sided (MR-up fires <=4% vs 25.5% baseline) and states an IV dynamic ('mean reversion attendue') that is unfounded, next to an honest d2 spread and d3 percentile label. A coloured directional chip nudging toward short vol without discriminating power = misleading in realistic conditions -> M, not C (no displayed number is wrong, no crash). SPY/QQQ (larger VRP ratio) not measurable offline; likely fires more often there.
  corrected_evidence: Real AAPL RV20 2016-2026, trailing-252 RV distribution, chip signal frequency: IV=RV baseline MR-down 22.8% / MR-up 25.5%; IV=RVx1.2 -> 37.8% / 10.8%; IV=RV+4pts -> 34.7% / 8.4%; IV=max(RV20,mean63)+3pts -> 44.5% / 0.2%; IV=mean126x1.15 -> 28.1% / 0.0%. Conditional on calm days (RV20 <= trailing median, n=1199-1282): MR-down fires 0.3-16.5% depending on proxy. Replace the N(15%,2%) / GBM numbers with these; keep the 'one-sided signal' argument as the core impact.
  fix_ok=true — Option 1 (iv_regime=None / replace chip by a caption 'IV au {p:.0%} de la RV recente - prime de risque de vol, pas un IV rank') is proportionate and regression-free: no test asserts on the chip text or iv_regime signal (tests/test_iv_dashboard_analytics.py:97 checks keys only; tests/integration/_iv_dashboard_render_driver.py:83 builds iv_regime but the view already tolerates None via `or {}`). A cheaper alternative keeping the chip: relabel it as a non-directional IV/RV richness bucket (RICHE / NORMALE / BON MARCHE). Option 2 (true IV percentile on iv_history >= 60 obs) is sound but accumulates one obs per analysis day, i.e. months away for a nomadic user, and mixes ATM IVs at varying DTE; keep it secondary and gated.


## M3 — parity-r-q-zero-bias
- titre: BS inversion with r=0, q=0 (spot, not forward) biases each contract's IV by ~±1 vol pt at 30 DTE; the median only cancels when calls and puts are balanced
- ancre: app/model/iv_dashboard/service.py:416
- sévérité finale (vote): M  (votes: M, M, M; finder: M)
- effort: 1 h
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_alpaca_plumbing_iv_bias.py ; scripts/review_iv_dashboard/p1_alpaca_plumbing_iv_bias_mix.py

### claim
Puts are converted with mid + S − K (q=0, r=0) and every contract is inverted with r=0,q=0 on spot; for SPY (S=640, r=4%, q=1.3%) this under-prices the synthetic call by ≈$1.42 and mis-sets the forward by ≈$1.42, giving −98 bp on ATM puts and +95 bp on ATM calls, up to −361 bp on 5% ITM puts.

### evidence (finder + orchestrateur)
p1_alpaca_plumbing_iv_bias.py [1]: K/S=0.95 call +271 bp / put −48 bp; K/S=1.00 call +95 bp / put −98 bp; K/S=1.05 call +49 bp / put −361 bp (true sigma 16%). p1_alpaca_plumbing_iv_bias_mix.py: median over all calls+puts in ±5% band = +1 bp; calls only +95 bp; puts only −98 bp; calls all + 5 ATM puts +89 bp.
--- (sonde orchestrateur scripts/review_iv_dashboard/orch_probe_iv_bias.py) ---
SPY S=640 r=4% q=1.3% sigma=16% dte=30 : IV recouvrée par le pipeline du code : ATM call +95 bp, ATM put -98 bp ; K/S=0.95 call +271 / put -48 ; K/S=1.05 call +49 / put -361. Médiane calls+puts sur la bande ±5% (14 contrats) : +1 bp (annulation par symétrie) ; |K/S-1|<1.5% : -1 bp. Sensibilité : r=2% -> put -27 bp ; q=0 -> put -142 / call +145. Ask-only : +7 bp (spread $0.10), +20 bp ($0.30), +68 bp ($1.00). T=dte/365 ignore l'intraday : -7 bp (dte 30), -15 bp (dte 15). 252 vs 365 : non-finding (deux conventions 'par an').

### impact
On the inversion path (the one used whenever Alpaca omits greeks — logic.py L1271 states the indicative feed 'typically does NOT include greeks/IV') the headline IV moves by ~±1 vol pt depending on which side happens to have quotes; IV−RV spread and the IV regime signal inherit that error.

### fix_sketch
Use a forward: F = S·exp((r−q)T) with r from a config/secret (e.g. 4%) and q from a per-symbol dividend yield (or 0 for stocks, ~1.3% SPY), convert puts with C = P + S·e^{-qT} − K·e^{-rT}, and call implied_vol_call(C, S, K, T, r, q). Alternatively invert OTM contracts only (put for K<S, call for K>S) with the Black-76 forward, which also removes the ITM drop asymmetry.

### panel
- [CODE READING] refuted=false conf=0.9 sev=M
  reason: Reading is correct. service.py L338 `r_annual: float = 0.0`; the only caller is get_iv_dashboard_data L565 `fetch_current_atm_iv(sym)` (no r passed, grep over app/ and tests/ finds no other caller); L416 put parity `mid + spot - K*exp(-r_annual*T)` omits the S*e^{-qT} leg; L417 `implied_vol_call(call_price, spot, K, T, r_annual, 0.0)` hard-codes q=0 and inverts on spot. No guard elsewhere (controller passes through, view only displays). Driving the REAL fetch_current_atm_iv (not a re-implementation) with a synthetic chain priced at r=4%, q=1.3%, sigma=16%, 30 DTE reproduces the finder's numbers: calls-only +96 bp, puts-only -98 bp, balanced calls+puts ~0 bp; per contract K/S=0.95 call +271/put -48, 1.00 call +95/put -98, 1.05 call +49/put -361. I tried to break the 'imbalance' premise (both types always come back from the same type-agnostic server filter, so why would one side be missing?) and found the imbalance is produced by the code itself: implied_vol_call L29 computes intrinsic with r=q=0 (=0 for K>S) while the true ITM put satisfies P >= K-S-1.42, so the synthetic call P+S-K goes negative for low-time-value ITM puts and is rejected as NaN. At realistic SPY vol levels the puts are dropped and the median no longer cancels: sigma=0.10, 30 DTE -> 52/63 puts survive, median +43 bp; sigma=0.12 -> +44 bp; sigma=0.08 -> +40 bp; 15 DTE -> +23..+29 bp. Only at sigma>=0.16 (30 DTE) do all 126 survive and the median comes back to ~0. The cancellation when balanced is itself governed by the two wing contracts (63rd/64th order statistics = K/S=0.95 put at -48 bp and K/S=1.05 call at +49 bp), not by ATM contracts. Greeks path is unaffected (+0 bp, method 'greeks Alpaca'), so the bug is confined to the inversion path, which is the expected path on the indicative feed (logic.py L1271 comment). Severity M is justified: systematic +40 bp headline bias at SPY-typical IV (today's measured 20d RV is 13.1%), inherited by iv_minus_rv and iv_regime.
  corrected_evidence: Real-function measurements (p4_g3_ivmethod_skeptic.py / _b.py, S=640 r=4% q=1.3%): balanced chain sigma=16% 30 DTE -> +1 bp (126/126 used); calls-only +96 bp; puts-only -98 bp; with r_annual=0.04 but q still 0: calls -50 / puts +44 / mixed -2 bp. Imbalance mechanism measured: sigma=0.08/0.10/0.12 at 30 DTE -> puts surviving 46/52/58 of 63 (ITM puts rejected by the r=q=0 no-arb bound in implied_vol_call L29-L32), median bias +40/+43/+44 bp; 15 DTE -> +23/+25/+26 bp; sigma=0.25 -> 0 bp. Whole-band skewed chain (ATM 16%, -0.6 pt/1%, smile 0.03): +3 bp vs true median. Per-contract extremes confirmed: call +271 bp @0.95, put -361 bp @1.05. DTE scaling of the single-side bias: 15 d +-68, 45 d +116/-121, 60 d +133/-140 bp.
  fix_ok=true — Fix is correct and proportionate. Prefer the second variant (invert OTM contracts only: put for K<F, call for K>F, on the forward) since it also removes the ITM-put rejection that creates the imbalance; with the first variant (full parity with e^{-qT}/e^{-rT}) the dropped-put asymmetry disappears too because the bound becomes exact. r does not need a new secret: the repo already exposes a model-layer `app.model.yieldcurve.service.get_risk_free_rate(T, ensure_cache=True)` (allow_api=False) and options_controller already carries a per-symbol div_yield from load_market_data; reuse those. Also update the docstring at L343-L345 which currently advertises 'parity with q=0'.
- [REPRODUCTION] refuted=false conf=0.9 sev=M
  reason: Reproduced end-to-end through the REAL svc.fetch_current_atm_iv (fetch_spot_price/_fetch_atm_snapshots monkeypatched, synthetic chain priced with r=4%, q=1.3%, flat sigma 16%, 30 DTE). Production always calls fetch_current_atm_iv(sym) with default r_annual=0.0 (service.py L565), so L416 'mid + spot - K*exp(0)' and L417 implied_vol_call(..., 0.0, 0.0) are the live path. Per-contract: K/S=0.95 call +271 / put -48 bp; 1.00 call +95 / put -98; 1.05 call +49 / put -361 — identical to the finder. Headline median: calls+puts +1 bp (n=126), calls-only +95, puts-only -98, calls all + 5 ATM puts +89. The finder's 'cancels only when balanced' is correct but understated: I found a realistic, non-speculative unbalance mechanism — in a low-vol regime the r=q=0 synthetic call of ITM puts goes below intrinsic and implied_vol_call returns nan, so the contracts are silently dropped: sigma=12% drops puts K=667..671 (synthetic call -0.075..-0.517 $) and the headline bias becomes +44 bp; sigma=10% drops 11 puts, +43 bp; sigma=8% +40 bp. With a -0.8 vol-pt/1% skew: +13 bp vs ATM (4 contracts dropped). DTE sensitivity at ATM: 15 d ±69 bp, 30 d ±96 bp, 60 d ±137 bp. Severity M stands: the per-contract inversion is wrong by 0.5-3.6 vol pts, and the headline is only right by accidental symmetry that breaks under low vol (SPY 30D IV of 10-13% is common), skew, or one-sided quoting.
  corrected_evidence: Balanced calls+puts +-5%: +1 bp; calls only +95 bp; puts only -98 bp; calls all + 5 ATM puts +89 bp (all via real fetch_current_atm_iv). NEW: low-vol regime bound-drops unbalance the median without any quoting asymmetry — sigma=12%: 5 ITM puts dropped (synthetic call < 0), headline +44 bp; sigma=10%: 11 dropped, +43 bp; sigma=8%: 17 dropped, +40 bp. Skew -0.8 pt/1%: +13 bp vs ATM. ATM bias vs DTE: 15 d +68/-69, 30 d +95/-98, 60 d +133/-140 bp.
  fix_ok=true — Fix sketch is correct (C = P + S e^{-qT} - K e^{-rT}, invert with r,q; or OTM-only on the forward). A simpler config-free variant that I verified gives 0.0 bp residual at K=608/640/672: imply the forward from ATM parity F = K + e^{rT}(C - P) (any rough r in the discount, residual is negligible), then invert undiscounted prices with S'=F, r=q=0 — no per-symbol dividend yield or rate secret needed. Either way also removes the bound-drop of ITM puts.
- [IMPACT & SEVERITY] refuted=false conf=0.8 sev=M
  reason: Impact survives. service.py L416-417 is called with r_annual=0.0 (get_iv_dashboard_data L565 passes no r) and q=0 on spot; per-contract bias reproduced exactly (ATM call +95 / put -98 bp, K/S=1.05 put -361 bp). The 'median cancels' mitigation is fragile in a way phase 1 did not quantify: the sorted IVs are bimodal and the median sits on a ~100 bp gap (middle values -49/-48/+49/+50 bp), so ONE dropped contract flips the headline by +-49 bp between refreshes, and any call/put imbalance lands the full +-95 bp. A realistic imbalance path exists: the 3000-contract page cap (snapshot-page-cap-silent) truncates the 30-DTE expiry after its calls (OPRA sorts C before P), giving a calls-only ATM set and +95 bp on the displayed IV on ~60% of days under a daily-expiry SPY calendar. Displayed spread IV-RV then reads +3.83 pts instead of +2.88 (RV 13.12% from the live run). Not C: no crash, no data loss, headline error <= 1 vol pt. Caveat lowering confidence: the whole path is dormant if the indicative feed actually returns impliedVolatility (logic.py L1271 says it typically does not; the live probe was 401 so this is unverified).
  corrected_evidence: p4_g3_parity_median.py (real implied_vol_call, service lines replicated): n=130 balanced median +0.6 bp but sorted middle values [-49,-48,+49,+50] bp; drop one put -> +49 bp, drop one call -> -48 bp; calls-only +95 bp, puts-only -98 bp; displayed IV-RV true +2.88 pts vs +3.83 (calls-only) / +1.90 (puts-only). Parity-implied forward fix: F=641.417 vs true 641.422, per-contract bias -31..-2 bp, median -5 bp, calls-only = puts-only = -5 bp. Finding's fix with r=4% but q forgotten (q=0): calls -50 / puts +44 bp; with r left at default 0 the bug is unchanged.
  fix_ok=false — Mathematically correct but disproportionate and fragile: it needs r from a secret (default 0 => no-op) and a per-symbol q (forgetting SPY's 1.3% leaves -+50 bp, measured). A parameter-free fix is smaller and better: take the nearest-ATM call/put pair, F = K0 + C - P, convert puts with C = P + F - K and call implied_vol_call(price, F, K, T, 0, 0) for every contract; measured residual -5 bp median, no config, no regression on the no-greeks path. Alternatively invert OTM contracts only on that same F.


## M4 — local-date-vs-exchange-date
- titre: dt.date.today() (server-local) keys the daily IV cache and drives DTE/T: an Asia user writes the US session under tomorrow's date and overwrites/loses sessions
- ancre: app/model/iv_dashboard/service.py:464
- sévérité finale (vote): M  (votes: M, M, M; finder: M)
- effort: 30 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_alpaca_plumbing_cache.py

### claim
record_iv_observation (L464), _fetch_atm_snapshots (L252) and fetch_current_atm_iv (L359) all use the machine-local date; for a user in UTC+8 the live US session of Aug-20 (15:30 ET = 03:30 SGT Aug-21) is stored as date=2026-08-21, the Aug-21 session analysed at 22:00 SGT overwrites the same key, and DTE is one day short (T undervalued -> IV +26 bp on the inversion path).

### evidence (finder + orchestrateur)
p1_alpaca_plumbing_cache.py [6]: Asia/Singapore local clock 2026-08-21 03:30 while US session date = 2026-08-20 -> key mismatch True (Paris/New York False). p1_alpaca_plumbing_iv_bias.py [3]: true remaining 30 d + 24 h inverted with T=30/365 -> 16.264% (+26 bp).

### impact
The user profile is explicitly nomadic (CLAUDE.md). Two distinct US sessions collapse into one cache row (data loss), the IV history chart is shifted one day, and dte/T are off by one.

### fix_sketch
Define today = datetime.now(ZoneInfo('America/New_York')).date() once in the module (helper _exchange_date()) and use it in L252, L359, L464; optionally key the cache by the exchange session date and keep the UTC timestamp in a column.

### panel
- [CODE READING] refuted=false conf=0.75 sev=M
  reason: Reading confirmed: dt.date.today() (naive, machine-local) at L252 (expiration_date_gte/lte filter), L359 (dte, T and the chain-df expiry reconstruction via _contracts_from_chain_df(chain, today) L367) and L464 (cache key). Nothing upstream normalises the timezone (controller only strips/uppercases the symbol; _fetch_closes_alpaca uses UTC only for bars `end`). For UTC+7/+8 the local date differs from the NY date from 12:00 ET to midnight ET, i.e. the whole local morning (00:00-12:00 local), not just an edge minute — so an Asia-based user doing the analysis in the morning gets every observation labelled +1 day and dte -1 systematically, then the US-open run at 21:30+ local upserts the same key and discards the earlier session (probe [5]: two records -> one row 2026-08-21, iv 0.25, the 0.20 session gone). The IV history is overlaid on the exchange-dated RV series in the same figure (tab_iv_dashboard.py L314-324 vs series.index), so the one-day shift is a visible misalignment. Mitigations I looked for: the DTE error only bites the BS-inversion path (L416-419) when the feed gives no greeks, and T is already truncated to whole days (L400), so the tz error is comparable to the existing discretisation — but it adds in the same direction. Also the view caption (L257-259) frames the cache as 'une par jour d'analyse', so the overwrite per se is by design; what is wrong is the date label and dte. M stands for this user profile (CLAUDE.md: nomad, active Indonesia project).
  corrected_evidence: p4 probe [4]: option expiring 2026-09-19 seen at 15:30 ET 2026-08-20: NY date dte=30 -> inverted IV 16.006% (+0.6 bp vs true 16%); SGT date dte=29 -> 16.279% (+27.9 bp). Local date != session date during 09:30-16:00 ET: UTC+8 True, UTC+7 True, UTC+2 False, UTC-4 False, UTC-7 False. Probe [3]: ZoneInfo('America/New_York') resolves in the venv (tzdata 2026.3 installed), so the fix is feasible on Windows. Probe [5]: two same-local-day upserts collapse to one row.
  fix_ok=true — `_exchange_date()` = datetime.now(ZoneInfo('America/New_York')).date() used at L252, L359 and L464 is correct and proportionate; tzdata is present in the venv (probe [3]) so no new dependency. Keep `today` threaded into _contracts_from_chain_df (L367) so the fallback chain uses the same date. Note the NY calendar date is still not the trading-session date for runs before 09:30 ET (snapshot data is the previous close) — acceptable; keying by true session date would need a market calendar and is out of proportion.
- [REPRODUCTION] refuted=false conf=0.85 sev=M
  reason: Reproduced with a controlled clock (FakeDate whose today() is the local date of a fixed UTC instant in a given IANA zone, injected as service.dt). Two distinct live US sessions (2026-08-20 15:30 ET and 2026-08-21 10:00 ET) recorded under Asia/Singapore or Asia/Makassar (Lombok) both get cache key 2026-08-21 -> 1 row on disk, first session's IV overwritten (ivs kept [0.17]); under Europe/Paris and America/New_York -> 2 rows. The host running this review is itself UTC+8 (time.tzname 'Malaisie', utcoffset +8h), so local date != New-York date 12 h per day, covering 12:00-24:00 ET i.e. the whole second half of the US session. DTE arithmetic of L389-L398 replicated: expiry 2026-09-18 seen from Aug-20 15:30 ET gives dte 29 (exchange) vs 28 (local UTC+8); inverting a 16.00% BS price with the short T yields 16.342% (+34 bp). Bias applies only on the BS-inversion fallback path (direct Alpaca greeks unaffected), cache-key collision applies always. Correctness bug on the primary key of the only persisted dataset of the tab, in the user's actual timezone -> M holds.
  corrected_evidence: Measured (p4 repro): Asia/Singapore & Asia/Makassar -> rows on disk = 1 for 2 distinct US sessions (keys 2026-08-21 / 2026-08-21); Europe/Paris & America/New_York -> rows = 2. Host local tz = UTC+8, host/NY offset difference = +12 h. DTE/T bias: exchange date 2026-08-20 dte=29 vs local 2026-08-21 dte=28 -> true sigma 16.000% inverted with local T = 16.342% (+34.2 bp, ATM, r=4.5%); finder's +26 bp corresponds to 30->31 d, the magnitude scales ~sqrt(dte/(dte-1)). Bias only on the 'inversion Black-Scholes (mid)' / 'mixte' methods.
  fix_ok=true — ZoneInfo('America/New_York') resolves in this venv on Windows (tzdata 2026.3 installed as a hard dependency of pandas 2.1.4, verified via importlib.metadata), so the helper is safe. _contracts_from_chain_df(chain, today) already takes today as a parameter, so one _exchange_date() helper covers L252, L359 and L464. Keying the cache by exchange session date (plus a UTC timestamp column) is the right semantic for 'one observation per trading day'.
- [IMPACT & SEVERITY] refuted=false conf=0.8 sev=M
  reason: Impact is realistic and systematic for this user, not occasional: for Asia/Makassar (Bali/Lombok, UTC+8 — the user's UMKM projects' region) 62% of US RTH minutes (from 12:00 ET onward) have local date != NY date; Asia/Jakarta 46%; Paris/LA/Honolulu 0%. Consequences confirmed by probe and code: (1) cache key collision — two sessions recorded under one local date keep 1 row (second overwrites first, L475), an unrecoverable loss on the free plan; (2) the displayed DTE ('IV ATM ~N j', view L221/L244) is one day short whenever local midnight has passed; (3) the gold IV-history points are dated one day after the RV series they overlay (view L318). None of this changes a trading decision materially (1 day of DTE, one lost point), so not C; but it is wrong-and-silent in the user's normal conditions: M stands. The +26 bp IV bias in the evidence should be de-emphasised: it applies only to the BS-inversion path (with method 'greeks Alpaca' T is unused) and is the same order as the integer-DTE convention's own intraday error.
  corrected_evidence: RTH minutes with local date != NY date on 2026-08-20: Asia/Makassar 241/391 (62%, from 00:00 local = 12:00 ET); Asia/Jakarta 181/391 (46%); Asia/Singapore 241/391 (62%); Europe/Paris, America/Los_Angeles, Pacific/Honolulu 0/391. Two record_iv_observation calls under one local date -> 1 row kept. IV bias 29 vs 30 d = +0.27 pp, BS-inversion path only. ZoneInfo('America/New_York') loads on this Windows venv (tzdata 2026.3 installed as pandas 2.1.4 dependency).
  fix_ok=true — A module-level _exchange_date() = datetime.now(ZoneInfo('America/New_York')).date() used at L252, L359, L464 is correct and proportionate; no regression risk on Windows because tzdata is a hard dependency of pandas>=2 (verified present). Keying the cache by NY date is the right call; keeping a UTC timestamp column is optional. The cutoff at get_iv_dashboard_data (pd.Timestamp.now().normalize()) can stay local — negligible.


## M5 — service-zero-unit-tests
- titre: service.py has no direct tests: 8% statement coverage under the CI gate (303/338 lines never executed)
- ancre: app/model/iv_dashboard/service.py:234
- sévérité finale (vote): M  (votes: M, M, M; finder: M)
- effort: 1 h
- dimension finder: ? (fusion: service-layer-untested-payload-contract-duplicated)
- repro: scripts/review_iv_dashboard/p1_tests_service_sketches.py (run with pytest) and scripts/review_iv_dashboard/p1_tests_cache_dir_patch_target.py ; grep ci-dessus ; scripts/review_iv_dashboard/p1_arch_controller_and_cache.py montre que l'upsert est testable hors-ligne (2 upserts même jour -> 1 ligne, iv=0.25)

### claim
None of the fetchers, parsers, the cache round trip or the orchestrator (_fetch_atm_snapshots pagination, _decode_opra, _snapshot_mid/_snapshot_iv, fetch_current_atm_iv parity branch, record_iv_observation/load_iv_history, fetch_daily_closes fallback chain, get_iv_dashboard_data degradation) is exercised by any test; the only service-adjacent test monkeypatches the whole service away (test_controller_clamps_and_normalizes).

### evidence (finder + orchestrateur)
pytest tests/test_iv_dashboard_analytics.py tests/smoke/test_offline_imports.py --cov=app.model.iv_dashboard -> 'service.py 338 stmts, 303 miss, 8%' (missing 45-590 except module header). tab_iv_dashboard.py 13% under the same CI selection. The 5 proposed tests were written and run: scripts/review_iv_dashboard/p1_tests_service_sketches.py -> '24 passed in 1.25s' (no network, --disable-socket active). Patch-target probe: monkeypatching app.utils.paths.CACHE_IV_HISTORY_DIR does NOT redirect _iv_history_path (False), monkeypatching service.CACHE_IV_HISTORY_DIR does (True) — service.py:33 binds the name at import.
--- (doublon ? service-layer-untested-payload-contract-duplicated @ tests/integration/_iv_dashboard_render_driver.py:33) ---
`grep -rn get_iv_dashboard_data|fetch_daily_closes|fetch_current_atm_iv|record_iv_observation|load_iv_history tests/` -> seules occurrences : test_iv_dashboard_analytics.py:173 (monkeypatch du service, jamais appelé). `_build_payload()` (driver l.33-97) redéfinit les clés symbol/source/.../log/generated_at indépendamment du dict retourné par service.py:590. Tests existants : 23 passed (analytics + controller + render).

### impact
The displayed IV number (median ATM IV, method tag, parity inversion of puts), the daily CSV upsert and the Alpaca->IEX->Stooq fallback are unverified; a regression in pagination (page_token not forwarded), in OPRA parsing (strike /1000) or in the same-day upsert would ship silently and corrupt the IV history the tab accumulates day after day.

### fix_sketch
Create tests/test_iv_dashboard_service.py (pytestmark = pytest.mark.unit). Everything below is proven green in scripts/review_iv_dashboard/p1_tests_service_sketches.py — copy it. Patch `svc.<name>` (module globals), never app.utils.paths.

# 1. _fetch_atm_snapshots (service.py:234-280): pagination, params, page cap, creds
class _Resp:
    def __init__(s, payload, status=200): s._p, s.status_code = payload, status
    def raise_for_status(s):
        if s.status_code >= 400: raise RuntimeError(f'HTTP {s.status_code}')
    def json(s): return s._p

def test_fetch_atm_snapshots_paginates_and_caps(monkeypatch):
    monkeypatch.setattr(svc, '_alpaca_data_headers', lambda: {'k': 'v'})
    calls = []
    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(dict(params)); p = len(calls)
        return _Resp({'snapshots': {f'SPY260918C0045{p}000': {'p': p}}, 'next_page_token': f'tok{p}'})
    monkeypatch.setattr(svc.requests, 'get', fake_get)
    out = svc._fetch_atm_snapshots('spy', feed='indicative', spot=450.0, dte_min=15, dte_max=60)
    assert len(calls) == svc._SNAPSHOT_MAX_PAGES
    assert 'page_token' not in calls[0] and calls[1]['page_token'] == 'tok1'
    assert calls[0]['strike_price_gte'] == pytest.approx(405.0) and calls[0]['strike_price_lte'] == pytest.approx(495.0)
    assert len(out) == 3
# + no-token stops after 1 call; HTTP 403 -> raise_for_status propagates; _alpaca_data_headers()->None -> EnvironmentError and requests.get never called.

# 2. parsers (service.py:171-231) — parametrized tables
@pytest.mark.parametrize('opra, expected', [
    ('SPY260918C00450000', (450.0, dt.date(2026, 9, 18), 'call')),
    ('SPY260918P00450500', (450.5, dt.date(2026, 9, 18), 'put')),
    ('BRKB260918C00450000', (450.0, dt.date(2026, 9, 18), 'call')),
    ('garbage', (None, None, None)), ('', (None, None, None))])
def test_decode_opra(opra, expected): assert svc._decode_opra(opra) == expected
# _snapshot_mid: bid/ask mid; bid=0 -> ask; both 0 -> latestTrade.p; snake_case + string prices; {} -> None
# _snapshot_iv: top-level impliedVolatility, greeks.iv, latestGreeks.impliedVolatility as str, greeks='x' -> nan, {} -> nan

# 3. fetch_current_atm_iv (service.py:330-449) — greeks + parity inversion, median, method tag
def _bs_put(S, K, T, sig, r=0.0):
    d1 = (math.log(S/K) + (r + .5*sig**2)*T)/(sig*math.sqrt(T)); d2 = d1 - sig*math.sqrt(T)
    return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
def test_fetch_current_atm_iv_mixes_greeks_and_parity(monkeypatch):
    today = dt.date.today(); expiry = today + dt.timedelta(days=30); tag = expiry.strftime('%y%m%d')
    put_mid = _bs_put(100.0, 100.0, 30/365, 0.25)
    snaps = {f'SPY{tag}C00100000': {'greeks': {'iv': 0.21}},
             f'SPY{tag}P00100000': {'latestQuote': {'bp': put_mid-.01, 'ap': put_mid+.01}},
             f'SPY{tag}C00130000': {'greeks': {'iv': 0.9}}}  # outside ATM band
    monkeypatch.setattr(svc, 'fetch_spot_price', lambda s: 100.0)
    monkeypatch.setattr(svc, '_fetch_atm_snapshots', lambda *a, **k: snaps)
    monkeypatch.delenv('ALPACA_OPTION_DATA_FEED', raising=False)
    info, log = svc.fetch_current_atm_iv('SPY')
    assert info['method'] == 'mixte (greeks + inversion BS)' and info['n_contracts'] == 2 and info['dte'] == 30
    assert info['iv'] == pytest.approx(np.median([0.21, 0.25]), abs=2e-3)
# + spot None -> (None, 'Spot indisponible…'); snapshots raise + download_options_alpaca -> empty df -> (None, 'Aucun contrat…')

# 4. cache round trip (service.py:455-497)
def test_iv_history_upsert_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, 'CACHE_IV_HISTORY_DIR', tmp_path)   # NOT app.utils.paths
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    pd.DataFrame([{'date': yesterday, 'iv': .19, 'dte': 29, 'n_contracts': 5, 'method': 'x', 'spot': 99.}]).to_csv(tmp_path/'iv_daily_SPY.csv', index=False)
    svc.record_iv_observation(' spy ', {'iv': .21, 'dte': 30, 'n_contracts': 6, 'method': 'g', 'spot': 100.})
    svc.record_iv_observation('SPY',   {'iv': .25, 'dte': 31, 'n_contracts': 4, 'method': 'm', 'spot': 101.})
    hist = svc.load_iv_history('spy')
    assert hist['iv'].tolist() == [0.19, 0.25]          # same-day upsert, older row kept, sorted
    assert list(svc.load_iv_history('ZZZZ').columns) == ['date', 'iv']
    (tmp_path/'iv_daily_BAD.csv').write_text('not,a,csv\n\x00', encoding='utf-8'); assert svc.load_iv_history('BAD').empty
    svc.record_iv_observation('SPY', {'iv': None})      # best-effort, must not raise

# 5. fallback chain + orchestrator degradation (service.py:123-168, 503-590)
def test_fetch_daily_closes_falls_back(monkeypatch):
    feeds = []
    def boom(sym, start, *, feed=None): feeds.append(feed); raise RuntimeError('down')
    monkeypatch.setattr(svc, '_fetch_closes_alpaca', boom)
    monkeypatch.setattr(svc, 'fetch_ohlc_history', lambda sym, period, interval: _closes_df())
    df, source, log = svc.fetch_daily_closes('spy', years=2.0)
    assert feeds == [None, 'iex'] and source == 'fallback (Stooq/Yahoo)' and sum('indisponibles' in m for m in log) == 2
    monkeypatch.setattr(svc, 'fetch_ohlc_history', lambda *a, **k: None)
    assert svc.fetch_daily_closes('spy')[1] == 'none'
def test_get_iv_dashboard_data_degrades(monkeypatch, tmp_path):
    monkeypatch.setattr(svc, 'CACHE_IV_HISTORY_DIR', tmp_path)
    monkeypatch.setattr(svc, 'fetch_daily_closes', lambda sym, **k: (_closes_df(), 'alpaca', ['ok']))
    monkeypatch.setattr(svc, 'fetch_current_atm_iv', lambda sym: (None, ['Spot indisponible (test)']))
    out = svc.get_iv_dashboard_data('spy')
    assert out['current_iv'] is None and out['iv_error'] == 'Spot indisponible (test)' and out['analysis'] is not None
    monkeypatch.setattr(svc, 'fetch_daily_closes', lambda sym, **k: (pd.DataFrame(), 'none', ['nope']))
    with pytest.raises(RuntimeError, match='Aucune donnée de prix'): svc.get_iv_dashboard_data('SPY')

### panel
- [CODE READING] refuted=false conf=0.95 sev=M
  reason: Reading is correct. service.py anchors match the working copy (CACHE_IV_HISTORY_DIR bound at :33; fetch_daily_closes :123-165; _decode_opra/_snapshot_iv/_snapshot_mid :171-231; _fetch_atm_snapshots :234-280 with EnvironmentError at :250, +/-10% band :259-261, 3-page loop :266-279; cache :455-497; orchestrator :503-611). grep over tests/, scripts/ (minus review dir) and conftest.py for any service symbol returns exactly one hit: tests/test_iv_dashboard_analytics.py:173 (monkeypatch of ctrl._svc.get_iv_dashboard_data, never a call). No integration test imports the service either; the render driver builds its payload by hand (_iv_dashboard_render_driver.py:34-95). Re-measured under the CI selection: service.py 338 stmts / 303 miss / 8%, controller 83%, tab 13%. Merged sub-claim (duplicated payload contract): the driver's 20 keys match service.py:590-611 exactly today, so that part is drift risk, not a present mismatch.
  corrected_evidence: pytest -m 'unit or smoke' --cov (analytics + smoke): service.py 338/303/8% missing 45,49-52,63-120,135-165,173-180,184-187,191-205,209-231,248-280,284-286,290-305,309-327,347-449,456-457,462-483,488-497,516-590; tab_iv_dashboard.py 213/180/13%; controller 21/4/83%. Sketch file p1_tests_service_sketches.py: 24 passed in 1.24s (re-run). Driver payload keys == service keys (20) + 'generated_at' added by view:157.
  fix_ok=true — Fix is proportionate and matches the code: patches module globals (svc.requests, svc._alpaca_data_headers, svc.CACHE_IV_HISTORY_DIR) which is the only working patch target given the import-time binding at :33; expected strings 'fallback (Stooq/Yahoo)' / 'none' / feeds [None,'iex'] are the literal values at :143,:161,:165; EnvironmentError at :250; 'Aucune donnée de prix' at :526. pytestmark unit keeps it inside the CI gate.
- [REPRODUCTION] refuted=false conf=0.95 sev=M
  reason: Reproduced independently. Full CI gate run (python -m pytest -m 'unit or smoke' --cov=app.model.iv_dashboard, 676 passed / 2 skipped / 50 deselected in 177 s) measures app/model/iv_dashboard/service.py at 35/338 statements = 7.9% (303 missing, lines 45-590), analytics.py 88.1%, tab_iv_dashboard.py 13.0%. Grep of tests/ for every public/private service symbol returns exactly one line: tests/test_iv_dashboard_analytics.py:173, which monkeypatches get_iv_dashboard_data away. Patch-target claim verified: setting app.utils.paths.CACHE_IV_HISTORY_DIR does not redirect svc._iv_history_path (False) because service.py:33 binds the name at import; setting svc.CACHE_IV_HISTORY_DIR does (True). My own six offline tests written from scratch (pagination x3 pages with page_token forwarding and strike band 405/495, no-token stop, HTTP 403 propagation, EnvironmentError without creds and requests.get never called; _decode_opra/_snapshot_mid/_snapshot_iv tables; same-day upsert round trip [0.19, 0.25] + corrupt CSV -> empty + iv=None best-effort; fallback chain feeds [None,'iex'] -> 'fallback (Stooq/Yahoo)' -> 'none'; fetch_current_atm_iv mixing greeks 0.21 + BS-inverted put 0.25 -> iv 0.2300, method 'mixte (greeks + inversion BS)', n_contracts 2, dte 30; orchestrator degradation current_iv None / iv_error propagated / RuntimeError 'Aucune donnee de prix') all PASS with sockets blocked and no keys. Finder's sketch file also runs: 24 passed in 1.25 s under -m unit. Impact stands: the daily IV CSV that accumulates across sessions and the displayed median IV are unprotected by any test.
  corrected_evidence: Under the real CI selection (whole suite, not just the two files the finder ran): service.py 35/338 stmts = 7.9% (303 miss), tab_iv_dashboard.py 33/213 = 13.0%, analytics.py 89/98 = 88.1%. Patch target: paths-module patch redirects=False, service-module patch redirects=True. Own 6 offline service tests: 6/6 PASS; finder's sketch: 24 passed.
  fix_ok=true — Proportionate: one new unit test file, all sketched tests proven green offline under --disable-socket. Patch svc.<name> module globals (svc.requests.get, svc._alpaca_data_headers, svc.fetch_spot_price, svc._fetch_atm_snapshots, svc._fetch_closes_alpaca, svc.fetch_ohlc_history, svc.CACHE_IV_HISTORY_DIR), never app.utils.paths or app.model.market_data. The parity test depends on dt.date.today() for the expiry tag (dte==30 is computed from today, so it is stable).
- [IMPACT & SEVERITY] refuted=false conf=0.85 sev=M
  reason: Re-measured under the exact CI selection (-m 'unit or smoke' --cov): service.py 338 stmts / 303 miss / 8%, tab 13%. The untested code is not plumbing: service.py:330-449 computes the headline IV number displayed by the tab (median of greeks + parity inversion, service.py:416-437), service.py:460-497 owns the persisted daily IV cache (same-day upsert at :475, silent except at :482), and :123-165 the Alpaca->IEX->Stooq fallback that decides which price series the RV is computed from. The impact is not hypothetical: this same panel found real defects in exactly those regions (iv-cache-corrupt-file-silent-loss, duplicate-date-crash, stale-chain-cache-as-current-iv, snapshot-page-cap-silent, ask-only-and-crossed-mids, yahoo-period-string), all of which a service test file would have made reproducible. M is calibrated: not C (no displayed number is wrong because of the gap itself), but a regression in median/parity/upsert would ship silently and corrupt data the tab accumulates day after day. The sketched tests are offline, fast and stable (24 passed in 1.5-1.6 s on two consecutive runs, socket-blocked).
  corrected_evidence: CI-selection coverage: app/model/iv_dashboard/service.py 338 stmts 303 miss 8% (branches 104/0); tab_iv_dashboard.py 213 stmts 180 miss 13%; analytics.py 88%. Full CI selection = 676 passed, 2 skipped, 50 deselected in 178 s. p1_tests_service_sketches.py: 24 passed in 1.60 s and 1.51 s (two runs).
  fix_ok=true — Proportionate (one unit-marked file, ~1 h, no app change). Three adjustments before copying the sketches verbatim: (1) sketch line 94 pins 'bid=0 -> ask' and line 183 pins 'corrupt CSV -> empty, no raise' — both behaviours are flagged as defects by ask-only-and-crossed-mids and iv-cache-corrupt-file-silent-loss; write those two cases against the agreed target behaviour (or mark xfail) instead of freezing the current one. (2) test_fetch_current_atm_iv asserts dte == 30 from dt.date.today(); if local-date-vs-exchange-date is fixed the service date may differ from the test's local date around midnight for this UTC+8 user — build expiry from the same date helper the service ends up using, or assert dte in {29,30,31}. (3) Keep patching svc.<name> module globals (service.py:32-33 bind fetch_spot_price/CACHE_IV_HISTORY_DIR at import), as the sketch already does.


## m1 — snapshot-page-cap-silent
- titre: _SNAPSHOT_MAX_PAGES×limit = 3000 contracts is below a filtered SPY chain (~4.9k) and the cap hit is never logged
- ancre: app/model/iv_dashboard/service.py:266
- sévérité finale (vote): m  (votes: m, m, M; finder: m)
- effort: 15 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_alpaca_plumbing_snapshot_mock.py

### claim
With honoured filters (15-60 DTE, ±10% strikes, $1 strikes, M/W/F expiries) a SPY-like chain has 4864 contracts; the loop stops after 3 pages without any message; the ~30 DTE expiry still survives only because the response is assumed sorted by symbol ascending (order non mesuré).

### evidence (finder + orchestrateur)
p1_alpaca_plumbing_snapshot_mock.py scenario A: 4864 contracts inside filters, HTTP calls 3 with limits [1000,1000,1000], log='3000 contrats candidats via snapshots filtrés', cap hit True, any log mentioning truncation False; result still 31 DTE / 126 contracts under ascending order.

### impact
Silent truncation; if the server orders differently (or the band is widened) the best_expiry/ATM set can be incomplete with no trace in the journal.

### fix_sketch
Log when page_token is still set after the last page ('pagination tronquée à N contrats'), and tighten the server-side filters to the actual need (expiration window around target_dte, type-agnostic ±5% band) so one page suffices.

### panel
- [CODE READING] refuted=false conf=0.85 sev=m
  reason: Reading is correct. L38 `_SNAPSHOT_MAX_PAGES = 3`, L250 `limit: 1000`, L266-L279: the for-loop exits after 3 iterations with `page_token` still set and nothing is logged or returned about it — `_fetch_atm_snapshots` returns only the dict and has no log channel; the caller at L365 logs only the count ('N contrats candidats'). Nothing in the controller (iv_dashboard_controller.py L48 passes through) or the view inspects this. Verified on the real _fetch_atm_snapshots with requests.get mocked to serve a 4902-contract chain (19 expiries 15..69 DTE x 129 $1 strikes x 2 types, ascending OPRA order): 3 HTTP calls with limits [1000,1000,1000], 3000 fetched, truncated=True, log = '3000 contrats candidats via snapshots filtrés (feed=indicative). | IV ATM ... échéance ... (30 j), 66 contrats' with no truncation mention. Under ascending order the kept expiries span 15..48 DTE and the 30-DTE target survives (the finder's 'survives by luck' claim is reproduced); under descending order only 36..69 DTE survive and the code silently picks 36 DTE on 33 contracts. Alpaca's actual ordering could not be measured offline (the orchestrator's live probe returned 401), and alpaca-py 0.12.0 contains no option-snapshot client (raw REST is used), so there is no library behaviour to check. Severity minor is right: on the default SPY/30 DTE path the result is correct in practice; the defect is the missing trace plus fragility to ordering/band widening.
  corrected_evidence: Mocked chain 4902 contracts (19 expiries x 129 strikes x 2): 3 HTTP calls, limits [1000,1000,1000], fetched 3000, no truncation string in log. Ascending order -> expiries kept 15..48 DTE, chosen 30 DTE, 66 contracts. Descending order -> expiries kept 36..69 DTE, chosen 36 DTE, 33 contracts, still no log. Side note (same constant, fallback branch L372): download_options_alpaca is called with max_pages=3 and no `limit`, so the fallback pulls at most 3 default-size pages of the UNFILTERED chain (nearest expiries first), which for SPY is unlikely to reach 15 DTE — the fallback will most often end in 'Aucun contrat entre 15 et 60 jours'. Not measured live; flagged for whoever owns the fallback finding.
  fix_ok=true — Proportionate. Minimal version: have _fetch_atm_snapshots return (snapshots, truncated: bool) or raise a logging.warning, and append 'pagination tronquée à N contrats' in fetch_current_atm_iv's log. Tightening the server filters is the better root fix: expiration window target_dte +- ~10 days and a type-agnostic +-5% strike band give roughly 7 expiries x 65 strikes x 2 = ~900 contracts, i.e. one page, which also removes the ordering dependency. Keep the ±10% band only as a widening retry if the first query returns nothing.
- [REPRODUCTION] refuted=false conf=0.9 sev=m
  reason: Reproduced with the real fetch_current_atm_iv against a mocked paginated server (svc.requests.get patched, honouring the expiration_date_*/strike_price_* params the code sends). SPY-like listing (M/W/F expiries, $1 strikes): 4902 contracts match the 15-60 DTE / +-10% server filters (finder: 4864; difference is today's weekday); HTTP calls = 3 (limits 1000), fetched = 3000, cap hit = True, and no log line mentions truncation/pagination — only '3000 contrats candidats via snapshots filtrés'. _fetch_atm_snapshots (L266-281) returns the dict and discards page_token with no signal. Result drift by server order: ascending -> expiry 31 DTE, 82 contracts; descending -> expiry 33 DTE, 41 contracts (the 33-DTE expiry itself is half-truncated); shuffled -> 31 DTE, 49 contracts (ATM set thinned by 40%). So the finding is exact: silent cap, correct answer only under ascending symbol order (which is plausible for Alpaca but unmeasured here). Severity m is right — no wrong numbers in the likely ordering, but an untraceable degradation mode.
  corrected_evidence: 4902 contracts match server filters today; 3 calls / 3000 fetched / cap hit, no truncation log. Order sensitivity (real function): asc -> 31 DTE / 82 contracts; desc -> 33 DTE / 41 contracts; shuffled -> 31 DTE / 49 contracts. Filter-tightening arithmetic: +-5% band with 15-60 DTE = 2394 contracts (> 1 page, but fits in the existing 3-page cap); +-5% with 20-40 DTE = 1134 (> 1 page).
  fix_ok=false — Logging the truncation ('pagination tronquée à N contrats' when page_token is still set after the last loop) is correct and proportionate. But the claim that tightening to a type-agnostic +-5% band makes 'one page suffice' is numerically wrong: +-5% over 15-60 DTE is still 2394 contracts and even 20-40 DTE is 1134. Corrected options: (a) narrow the server strike band to the selection band (+-5%) so the chain fits in the existing 3 pages (2394 <= 3000), or (b) raise _SNAPSHOT_MAX_PAGES to 5, or (c) narrow the expiry window to ~target_dte +- 7 days (~7 expiries x 130 = ~900) if one page is really wanted. Note the in-code 0.10 fallback band (L393) would become unreachable with option (a); the nearest-4 fallback still covers it.
- [IMPACT & SEVERITY] refuted=false conf=0.7 sev=M
  reason: Under-rated at m. Measured end-to-end through the real fetch_current_atm_iv with a mocked paginated ascending-OPRA response (ascending order is well-founded: Alpaca's next_page_token is the last symbol of the page): on the default symbol SPY the 3000 cap is hit on 30/30 trading days regardless of calendar assumption (4.9k in filter with M/W/F only, 6.5-6.7k with daily expiries), and the log never mentions it ('3000 contrats candidats'). Whether it reaches the DISPLAYED number depends on how far SPY dailies are listed: with >=5 weeks of daily expiries the truncation point (3000/258 = 11.6 expiries from 15 DTE) falls exactly on the ~30 DTE target, OPRA sorts C before P, so on 18/30 days the ATM set is calls-only (63/0), n_contracts shows 63 instead of 126 and the headline IV carries the +95 bp parity bias; with <=4 weeks of dailies, 0 such days. Silent truncation on every default run plus a concrete, recurring path to a ~1 vol-pt error on the headline meets the M bar (silent failure in realistic conditions); not C because nothing crashes and the error is bounded at ~1 pt.
  corrected_evidence: p4_g3_page_cap_e2e.py <N_DAILY_WEEKS>: N=0 (M/W/F only): in_filter 4902-5160, cap hit 30/30 days, best expiry never cut, IV bias +1 bp. N=3 and N=4: cap hit 30/30, 0 cut days, +1 bp. N=5: in_filter 6450-6708, cap hit 30/30, best expiry (30 DTE) cut on 18/30 days -> ATM C/P = 63/0, n_used 63, displayed IV bias +95 bp; log_mentions_trunc=False on every run. AAPL-like chains ($2.5-5 strikes, weekly Fridays) stay well under 3000.
  fix_ok=false — The log line is right and cheap (check page_token after the loop). The filter-tightening half is partly a regression: narrowing expiration_date_gte/lte 'around target_dte' can miss monthly-only names (3rd Fridays spaced 28-35 days; e.g. 14 and 42 DTE both fall outside a 20-40 window that the current 15-60 catches). Safer and still minimal: raise _SNAPSHOT_MAX_PAGES (each page ~0.5-1 s) and/or tighten only the strike band to the ATM 5% actually used, keep the DTE window. If filters are tightened, also log the fetched/total count.


## m2 — fallback-chain-cannot-reach-30dte-and-masks-root-cause
- titre: Full-chain fallback fetches only 3×100 unfiltered contracts (nearest expiry) and the surfaced iv_error hides the real HTTP failure
- ancre: app/model/iv_dashboard/service.py:385
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 30 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_alpaca_plumbing_snapshot_mock.py

### claim
download_options_alpaca never sends a 'limit' param (server default page 100) so max_pages=3 yields ≤300 symbol-sorted contracts, i.e. the 0-2 DTE expiries for a liquid underlying; all are rejected by the 15-60 DTE filter and the UI shows iv_error = iv_log[-1] = 'Aucun contrat entre 15 et 60 jours d'échéance.' while the 403/429/connection error only sits in the expander log.

### evidence (finder + orchestrateur)
p1_alpaca_plumbing_snapshot_mock.py scenario C (403): UI iv_error = "Aucun contrat entre 15 et 60 jours d'échéance."; scenario E: download_options_alpaca(max_pages=3) -> limit params [None, None, None] -> 0 contracts (300 x 0DTE dropped by min_days_to_expiry=1).

### impact
On an OPRA-agreement 403 or a transient 429 the user is told the chain has no 15-60 DTE contracts (false) instead of the actual cause; the fallback adds latency (retries/backoff) without ever rescuing SPY/QQQ-type names.

### fix_sketch
Surface the first failure message as iv_error (keep exc text), and either drop the fallback or call download_options_alpaca with max_pages=None plus a min_days_to_expiry=dte_min / max_contracts cap; handle 403 explicitly ('feed opra non autorisé -> repasser en indicative').

### panel
- [CODE READING] refuted=false conf=0.88 sev=m
  reason: Both halves hold on reading. (a) logic.py L1394-1398 builds params with only `feed` and `page_token`; there is no `limit` parameter anywhere in download_options_alpaca, so the server default page size applies (Alpaca option snapshots: default 100, max 1000), and service.py L372 caps at _SNAPSHOT_MAX_PAGES=3 -> at most ~300 contracts in OPRA-symbol order, i.e. the nearest expiries; for SPY/QQQ-type chains (hundreds of strikes per daily expiry) these are 0-2 DTE and min_days_to_expiry=1 (L1258, L1484-1486) plus the 15-60 DTE filter at service L382 drop all of them. The qualifier matters: for thin chains (few strikes, weekly/monthly expiries) 300 contracts could reach 15+ DTE, so 'never' is only accurate for liquid names, as the finding itself says. (b) Masking: the only message carrying the HTTP exception text is L368, and it is always followed by L374 ('N contrats via chaîne Alpaca (cache).') or L376; download_options_alpaca swallows 403/429/connection errors internally (logging.warning, returns empty df, never raises), so the path continues to L385 'Aucun contrat entre 15 et 60 jours' which becomes iv_error = iv_log[-1] at L574 and is rendered by st.warning at tab_iv_dashboard.py L249-252. Measured in p4_fallback_probe.py (503 on snapshots): iv_log[-1] = 'Aucun contrat entre 15 et 60 jours d'échéance.', no HTTP text. Latency nuance: on 403 logic breaks immediately (L1405-1413, no retry); retries/backoff (2 x 0.5 s/1 s, L1414-1437) only apply to 429/5xx/connection errors.
  corrected_evidence: p4_fallback_probe.py: download_options_alpaca('SPY', max_pages=3) sends params [{'feed': 'indicative'}, {'feed': 'indicative'}, {'feed': 'indicative'}] — no limit key at all. fetch_current_atm_iv on a 503: log = ['Snapshots filtrés indisponibles (503 Server Error) ; fallback chaîne complète.', '0 contrats via chaîne Alpaca (cache).', 'Aucun contrat entre 15 et 60 jours d'échéance.']; iv_error contains '503'? False. Mock scenario C (403) re-run: identical masking, iv_error = 'Aucun contrat entre 15 et 60 jours d'échéance.'
  fix_ok=true — Surfacing the first failure as iv_error is right. On the fallback itself: max_pages=None without a limit means ~470 requests of 100 for a 47k-contract SPY chain (min_days_to_expiry is a client-side filter, it does not reduce pages) — not acceptable in a Streamlit refresh. Either drop the fallback (simplest, since the filtered call already targets the 15-60 DTE window) or add a `limit` request param (1000) to download_options_alpaca plus max_contracts; note 403 is already detected in logic.py L1405 but only logged, so the explicit 'feed opra non autorisé' message has to come from service.py.
- [REPRODUCTION] refuted=false conf=0.85 sev=m
  reason: Both halves reproduce with my own mock. (a) logic.py L1394-1398 builds params from `feed` and `page_token` only — measured params[0]={'feed':'indicative'}, 'limit' never sent across 3 calls. With a SPY-like chain (daily expiries, 300 strikes x C/P = 600 contracts/expiry, symbol-sorted pages): server default page 100 -> 300 contracts, all 0 DTE, dropped by min_days_to_expiry=1 -> 0 contracts; even with page 1000 (sensitivity) -> 2400 contracts with DTE 3-6 only, 0 in the 15-60 band. (b) 403 on the filtered snapshot: log[0] carries the 403 text, log[-1] = iv_error = "Aucun contrat entre 15 et 60 jours d'échéance." (service.py L385, L574); identical for ConnectionError with no cache. When the fallback is reachable it adds 3 more requests (calls=4) and still ends on the same misleading message.
  corrected_evidence: Alpaca's default page size (100) is documented but not measurable offline; the conclusion is insensitive to it (page 1000 still yields only 3-6 DTE for a liquid name). 'Never rescues' holds for liquid names only: a thin monthly chain (80 contracts/expiry, 3 expiries) is fully fetched in 3 pages and reaches 28/56 DTE (160 usable contracts). 403 path: 2 HTTP calls total (fallback breaks on 403 at logic.py L1404-1412 without retry); 429/ConnectionError: 4 calls.
  fix_ok=false — Surfacing the first failure as iv_error (keep the HTTP status) is right and cheap. But `max_pages=None` is disproportionate: min_days_to_expiry is a client-side filter (logic.py L1489) that does not reduce paging, so a SPY chain (~30k contracts) at 100/page means ~300 sequential requests in the tab. Either drop the fallback (it cannot provide quotes for BS inversion anyway, `mid=None` at L322) or add server-side expiration_date_gte/lte + limit passthrough to download_options_alpaca.
- [IMPACT & SEVERITY] refuted=false conf=0.75 sev=m
  reason: Confirmed and the impact is real but bounded. p4_fallback_impact.py S4 (no cache) for 403, 401-HTML and ConnectionError: the st.warning text is always "IV courante indisponible — Aucun contrat entre 15 et 60 jours d'échéance." (false: nothing was fetched), the real cause only appears in the collapsed expander; the fallback issues 3 extra unfiltered calls with limit=None (server default page 100 -> at most 300 symbol-sorted = nearest-expiry contracts, all < 15 DTE for SPY/QQQ/AAPL) and burns 1.5 s of retry backoff (0 s for 403). Realistic trigger: the tab advertises 'ALPACA_OPTION_DATA_FEED (indicative/opra)' (tab L146) — setting opra on a free plan gives a 403 that is reported as an empty 15-60 DTE window, sending the user to debug the wrong thing; revoked keys (seen live) give the same false message whenever Stooq spot is available. Why m and not M: no displayed number is wrong (IV is correctly shown as unavailable), no trading-decision risk, the truth is one click away in the Journal; it is misleading copy + dead-weight latency, i.e. robustness. Borderline; should be fixed together with stale-chain-cache-as-current-iv since it is the same code block.
  corrected_evidence: S4 measured: [403] 2 HTTP calls (1 unfiltered), 0.00 s; [401html] 4 calls (3 unfiltered, limit params [1000, None, None, None]), 1.50 s; [conn] 4 calls, 1.50 s. iv_error in all three = "Aucun contrat entre 15 et 60 jours d'échéance."; root cause line only in log: 'Snapshots filtrés indisponibles (403 Client Error: forbidden) …' / '(401 Client Error: <html>\n<head><title>401 Authorization Re…'.
  fix_ok=true — 'Surface the first failure message as iv_error' + 'drop the fallback' is the proportionate pair (the live leg can never reach 15-60 DTE and the cache leg is the stale-IV bug). Two caveats: (1) surfacing the raw exception requires the _short_exc helper from html-exception-text-in-log first, otherwise the 8-line nginx 401 page lands in st.warning (S4 shows the root-cause string starts with '401 Client Error: <html>\n<head>…'); (2) the alternative 'max_pages=None' is NOT proportionate — it would page through the entire SPY chain (thousands of contracts) for a 30-DTE ATM median; min_days_to_expiry=dte_min does not help either because server-side paging is symbol-sorted, the filter is client-side. Explicit 403 handling ('feed opra non autorisé -> repasser en indicative') is a nice-to-have, not required.


## m3 — ask-only-and-crossed-mids
- titre: _snapshot_mid accepts ask alone (bid missing/0) and crossed quotes without any sanity check
- ancre: app/model/iv_dashboard/service.py:218
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 15 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_alpaca_plumbing_iv_bias.py

### claim
A one-sided quote is used as the 'mid' (biasing IV up by spread/2 / vega) and bid>ask quotes are averaged silently; a stale latestTrade is used when both sides are 0.

### evidence (finder + orchestrateur)
p1_alpaca_plumbing_iv_bias.py [2]: ATM SPY 30 DTE vega 0.732 $/vol pt (1 $ = 137 bp); ask-only bias +7 bp at $0.10 spread, +34 bp at $0.50, +68 bp at $1.00; crossed bid=5.30/ask=5.10 -> mid 5.20 accepted (no check).

### impact
Pre/after-hours indicative quotes (bid 0) push the inverted IV up by tens of bp; low impact for ATM SPY in session, larger for less liquid names.

### fix_sketch
Require bid>0 and ask>=bid and (ask-bid)/mid below a threshold; otherwise skip the contract (do not fall back to ask or last trade).

### panel
- [CODE READING] refuted=false conf=0.85 sev=m
  reason: Reading is correct. _snapshot_mid L214-L216 returns 0.5*(bid+ask) whenever both are >0 with no ask>=bid check; L217-L218 returns the ask alone when bid is None or <=0; L221-L227 falls back to latestTrade['p'] ignoring its timestamp 't'. Measured on the real function: {bp:0, ap:5.20} -> 5.2; {ap:5.20} (no bp) -> 5.2; crossed {bp:5.30, ap:5.10} -> 5.2; {bp:0, ap:0, latestTrade p=4.00 t=2026-01-02} -> 4.0. Note the asymmetry: {bp:5.10, ap:0} -> None (bid-only is skipped but ask-only is accepted). Downstream guards checked: fetch_current_atm_iv L413 only rejects mid<=0, implied_vol_call only rejects prices outside [intrinsic, S] (L31), L418 only rejects iv outside (0,5) — none of them catches a one-sided or crossed quote at normal price levels. End-to-end on the real function with the whole chain bid=0 (pre-market indicative pattern) the median is pushed up by +12 bp ($0.10 spread), +57 bp ($0.50), +108 bp ($1.00) versus the two-sided chain — larger than the finder's single-ATM-contract numbers because off-ATM contracts have lower vega so spread/2/vega is bigger there and the median mixes them. Impact remains small for in-session SPY ($0.02-0.10 spreads) and the median dampens isolated bad quotes, so minor is the right severity; the fallback to a stale last trade is the more dangerous branch but also the rarest (requires both sides at 0).
  corrected_evidence: Real _snapshot_mid: bid=0/ask=5.20 -> 5.2; bid missing -> 5.2; crossed 5.30/5.10 -> 5.2; bid=ask=0 + stale trade 4.00 -> 4.0; bid=5.10/ask=0 -> None (asymmetric). Whole-chain bid=0 through the real fetch_current_atm_iv (S=640, sigma=16%, 30 DTE, 126 contracts): +12 bp at $0.10 spread, +57 bp at $0.50, +108 bp at $1.00 (two-sided reference ~0 bp). Finder's +7/+34/+68 bp are the single ATM-contract values; the band median is ~1.6x larger.
  fix_ok=true — Require bid>0 and ask>=bid and skip otherwise — correct and cheap. Two caveats: (1) dropping every one-sided quote makes the IV unavailable outside RTH on the indicative feed; that is acceptable but the existing 'Aucune IV exploitable' message (L420-L424) should say why (quotes one-sided / hors séance) so the user does not read it as an outage; (2) the relative-spread threshold is optional — a fixed cut would reject legitimate wide wing quotes on illiquid names; if added, apply it only to contracts whose spread/mid exceeds something like 50%. Also remove or timestamp-gate the latestTrade fallback (snapshot carries 't').
- [REPRODUCTION] refuted=false conf=0.85 sev=m
  reason: Reproduced on the real svc._snapshot_mid (service.py L218-236): bid=0/ask=5.20 -> 5.2; bid missing -> 5.2; bid=None -> 5.2; crossed bid=5.30/ask=5.10 -> 5.2 (no check); locked -> 5.2; bid=0 ask=0 + latestTrade p=4.0 -> 4.0; no quote + trade -> 4.0; bid=0.01/ask=50 -> 25.005; asymmetrically bid=5.10/ask=0 -> None. End-to-end through fetch_current_atm_iv with an r=q=0 market (so only the quote effect is measured), whole +-5% chain bid=0/ask-only: +8 bp at $0.10 spread, +24 at $0.30, +39 at $0.50, +79 at $1.00 (finder's ATM-only numbers +7/+20/+34/+68 are slightly low because OTM contracts in the band have less vega); crossed by $0.20 everywhere: all 126 contracts accepted, none rejected. Whether Alpaca's indicative feed actually returns bid=0 pre-market cannot be checked offline, so the realistic magnitude for in-session SPY ($0.05-0.15 spreads) is <= ~+12 bp; larger for illiquid names. Severity m confirmed.
  corrected_evidence: Ask-only bias measured through the real fetch_current_atm_iv median over the +-5% band: +8 bp ($0.10 spread), +24 bp ($0.30), +39 bp ($0.50), +79 bp ($1.00) — vs finder's ATM-only +7/+20/+34/+68. ATM vega 0.732 $/vol pt (1 $ = 137 bp) confirmed. Crossed $0.20 market: 126/126 accepted. Extra asymmetry: bid>0/ask=0 returns None while bid=0/ask>0 returns ask.
  fix_ok=true — Core of the fix is bid>0 and ask>=bid (rejects one-sided, crossed and zero quotes); dropping the latestTrade fallback is reasonable since a stale print has no place in a live-IV median. The relative-spread threshold is an optional extra knob — if added, keep it loose (e.g. (ask-bid)/mid <= 0.5) so legitimately wide but honest quotes on less liquid names are not discarded.
- [IMPACT & SEVERITY] refuted=false conf=0.75 sev=m
  reason: Kept as m, but the headline claim is the weak part. Ask-only bias at realistic in-session spreads is +3 bp ($0.05), +7 bp ($0.10), +20 bp ($0.30) on ATM SPY/AAPL 30 DTE (vega 0.732 $/pt) -- negligible, and one crossed quote among ~126 contracts cannot move a median. The part with real impact is the third clause, the latestTrade fallback (service.py L229-236): after hours, when bid=ask=0 on the whole book, every 'mid' silently becomes a stale last trade; with a 0.5% spot move since the trade, the median shifts +107 bp and per-contract IVs scatter -977..+527 bp, while the caption still says 'inversion Black-Scholes (mid)'. That is an undisclosed basis change, not a wrong formula, hence m (robustness/transparency), not M.
  corrected_evidence: p4_g3_mid_fallbacks.py (real _snapshot_mid): two-sided 5.10/5.20 -> 5.15; bid=0/ask=5.20 -> 5.2; crossed 5.30/5.10 -> 5.2 (accepted); bid=ask=0 + trade 5.05 -> 5.05; no quote -> None. Ask-only bias: +3/+7/+20/+68 bp at $0.05/$0.10/$0.30/$1.00 spread. After-hours all-zero book with last trades struck at S-3: 126/130 usable, median +107 bp, range -977..+527 bp. Proposed strict fix on that book keeps 0/130 contracts -> fetch_current_atm_iv returns None -> tab shows 'IV courante indisponible'.
  fix_ok=false — The sketch ('skip the contract; do not fall back to ask or last trade') is a regression for the nomadic user: on an after-hours book it keeps 0/130 contracts and the IV block disappears entirely, which also stops the daily iv_daily_*.csv accumulation. Proportionate fix: reject only crossed quotes (ask < bid), keep the ask-only and last-trade fallbacks, count them per contract and surface the count in method/log (e.g. 'inversion BS (mid, 40 ask seul, 86 dernier trade)') so the user can judge the basis. A spread-ratio cap would also drop legitimately wide ITM quotes.


## m4 — iv-cache-corrupt-file-silent-loss
- titre: A 0-byte or header-less iv_daily CSV makes every subsequent observation silently dropped, forever
- ancre: app/model/iv_dashboard/service.py:474
- sévérité finale (vote): m  (votes: m, m, m; finder: M)
- effort: 30 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_alpaca_plumbing_cache.py

### claim
record_iv_observation reads the existing CSV inside the try; pd.read_csv raises EmptyDataError on an empty file (or the df.get default path raises 'Unalignable boolean Series' when 'date' is missing), the except at L482 only logs a warning, the row is not written and the corrupt file is never repaired, so the daily history stops accumulating with no UI signal.

### evidence (finder + orchestrateur)
p1_alpaca_plumbing_cache.py [2]: after writing a 0-byte file and calling record_iv_observation, file size stays 0 bytes, load_iv_history -> 0 rows, only 'LOG WARNING ... No columns to parse from file'. [3]: header 'foo,iv' -> 'Unalignable boolean Series provided as indexer', file unchanged.
--- (sonde orchestrateur orch_probe_plumbing.py §D) ---
Fichier vide -> 'No columns to parse from file' -> warning, observation perdue ; fichier sans colonne date -> le garde df.get('date', Series vide) produit 'Unalignable boolean Series provided as indexer' -> warning, observation perdue : le garde est mort.

### impact
One interrupted to_csv (crash/kill during write, to_csv truncates before writing) loses all future daily IV observations silently; the 'Historique IV local' chart never grows and the user is not told.

### fix_sketch
try: df = pd.read_csv(path) except (pd.errors.EmptyDataError, pd.errors.ParserError): df = pd.DataFrame(columns=row.keys()); if 'date' not in df.columns: df = pd.DataFrame(columns=row.keys()); write via a temp file + os.replace for atomicity; surface the warning in the returned log list.

### panel
- [CODE READING] refuted=false conf=0.85 sev=m
  reason: Reading confirmed. service.py L472-477: `if path.exists(): df = pd.read_csv(path)` sits inside the single try whose except (L482-483) only calls logging.warning. With pandas 2.1.4, a 0-byte or whitespace-only file raises EmptyDataError ('No columns to parse from file') before any write; the guard at L475 `df.get('date', pd.Series(dtype=str))` is dead: when 'date' is missing the empty Series cannot index df and raises 'Unalignable boolean Series provided as indexer'. Nothing repairs the file, so every later call hits the same path. Nothing reaches the UI: record_iv_observation returns None (no log param, probe [6]), the caller at L568 ignores it, load_iv_history (L486-497) swallows the error and returns an empty frame, and the view caption at tab_iv_dashboard.py L255-259 is only rendered when n_iv_obs > 0. Severity: I downgrade to m because the trigger is narrow — to_csv on a ~100-byte file truncates and writes in one buffered call, so the crash/disk-full window is tiny; header-only, truncated-row and Excel-resaved files all still work (probe [1]). The consequence once triggered (permanent, silent) is real, but probability-weighted it is minor; the cache is also documented as best-effort (L482 comment).
  corrected_evidence: p4 probe [1] (pandas 2.1.4, py 3.11.15): EMPTY0 size 0->0 rows=0 written=False; WS (\n\n) 2->2 written=False; NODATECOL ('foo,iv') 22->22 written=False with 'Unalignable boolean Series'; HDRONLY 36->78 rows=1 OK; TRUNC_ROW (cut mid-row) 72->118 rows=2 OK; EXCELDATE (19/08/2026) 75->120 rows=1 OK. Only empty/whitespace or missing-'date'-column files trigger the permanent loss.
  fix_ok=true — Catching EmptyDataError + the `'date' not in df.columns` reset is correct and small; temp-file + os.replace is proportionate. Two cautions: on ParserError, do not silently overwrite — rename the corrupt file to .bak first, since a partially valid file may hold months of history; surfacing the warning needs record_iv_observation to return a message (currently -> None) and the caller at L568 to append it to `log`.
- [REPRODUCTION] refuted=false conf=0.8 sev=m
  reason: Reproduced deterministically with my own probe (CACHE_IV_HISTORY_DIR redirected to a temp dir, 2 calls of record_iv_observation per state): 0-byte file -> size stays 0, load_iv_history 0 rows, only logging.warning 'No columns to parse from file' (service.py L482-483); file 'foo,iv\n1,0.2' -> 'Unalignable boolean Series provided as indexer', file unchanged, 0 rows. The UI is fully silent: tab_iv_dashboard.py L255-256 renders the 'Historique IV local' caption only when n_iv_obs > 0, so a corrupt file shows nothing at all. BUT the trigger is narrower than claimed: an interrupted pandas to_csv leaves a header-only file (measured 9 bytes 'date,iv\r\n' when the body writer is killed), and header-only and partial-body files both self-heal (header-only 'date,iv' -> 85 bytes / 1 row; partial last row -> parsed, 3 rows). A 0-byte file needs the kill between open('w') and the header write or a power loss before page-cache flush; a header-less file needs a manual edit. Real defect (catch-all except, never repaired, silent), rare trigger, best-effort cache -> minor rather than major.
  corrected_evidence: Measured (p4 repro): baseline absent file -> 81 bytes, 1 row (2 same-day calls upsert correctly). 0-byte -> size 0->0, rows 0, 2 warnings 'No columns to parse from file'. 'foo,iv\n1,0.2\n' -> size 13->13, rows 0, 'Unalignable boolean Series provided as indexer'. 'date,iv\n' header-only -> 8->85 bytes, rows 1 (self-heals). Partial last row '...\n2026-08-19,0.1' -> 91->151 bytes, rows 3 (parses; the truncated value 0.1 is silently kept as a data point). Binary garbage -> UnicodeDecodeError, rows 0. Interrupted to_csv (body writer raises) leaves 'date,iv\r\n' (9 bytes), i.e. a header-only file, not a 0-byte file.
  fix_ok=true — Catching pd.errors.EmptyDataError and resetting df when 'date' not in df.columns is sufficient and proportionate. temp-file + os.replace is optional (header-only and partial states already self-heal). 'Surface the warning in the returned log list' needs a signature change: record_iv_observation returns None and is called at L568 without a log sink — either keep logging or adapt the caller. UnicodeDecodeError is not covered by (EmptyDataError, ParserError) but that state is unrealistic.
- [IMPACT & SEVERITY] refuted=false conf=0.6 sev=m
  reason: Behaviour confirmed: a 0-byte iv_daily CSV makes record_iv_observation fail on every later run (service.py L474 read_csv -> EmptyDataError -> L482 logging.warning only), the file is never repaired, and the UI gives no signal (tab_iv_dashboard.py L254-260 only renders the 'Historique IV local' caption when n_iv_obs > 0; the warning is not appended to result['log']). Impact when triggered is permanent and, on the free Alpaca plan, unrecoverable since this cache is the only IV-history source. But the trigger is rare: measured to_csv truncation window for a 500-row (2-year) history is 3.06 ms median / 32 ms max per analysis, so 'one interrupted to_csv' needs a kill/power loss inside a ~3 ms window; concurrent Streamlit reruns only produce a transient failure (the other writer completes). Header-only files work (79 bytes, 1 row after record), and the dead df.get('date') guard only bites when a date-less header ALSO has data rows (manual edit). This is a robustness defect with a rare trigger, not a realistic-conditions failure: m, not M.
  corrected_evidence: to_csv window for 500-row history: median 3.06 ms, max 32.45 ms (20 runs). 0-byte file -> size stays 0, load rows 0, only logging.warning. Header-only file -> record OK (79 bytes, 1 row). Header-less garbage with no data rows -> record OK (guard only fails when data rows exist). UI: caption conditional on n_iv_obs>0, warning never reaches result['log'].
  fix_ok=true — Catching EmptyDataError/ParserError and resetting to an empty frame is correct and cheap; temp-file + os.replace is the part that actually prevents the 0-byte state and works on Windows when the target is not locked (an Excel lock would then raise at replace instead of at open — same outcome as today). Surfacing the warning requires changing record_iv_observation's return to a log list and the caller at L569; acceptable but optional. Also drop the dead df.get('date') guard in favour of the explicit 'date' in df.columns check.


## m5 — iv-history-filename-unsanitized-symbol
- titre: Nom de fichier du cache IV dérivé du symbole brut (pas de sanitisation, contrairement au reste du modèle)
- ancre: app/model/iv_dashboard/service.py:457
- sévérité finale (vote): m  (votes: m, m, m/réfuté; finder: m)
- effort: 10 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_arch_controller_and_cache.py

### claim
`_iv_history_path` fait `CACHE_IV_HISTORY_DIR / f"iv_daily_{sym}.csv"` avec `sym` seulement strip/upper ; un séparateur dans le symbole crée un sous-répertoire au lieu d'un fichier, alors que market_data.py (l.361, l.483) et options/logic.py (l.1107) passent le ticker dans `re.sub(r"[^A-Za-z0-9._-]", "_", ...)`.

### evidence (finder + orchestrateur)
Sonde p1_arch_controller_and_cache.py : `'A/B'` -> `iv_daily_A/B.csv` (parent='iv_daily_A'), `'../evil'` -> `iv_daily_../EVIL.csv` ; après `record_iv_observation('A/B', ...)` les fichiers écrits sous le répertoire temp sont `['iv_daily_A\\B.csv', 'iv_daily_SPY.csv']` (sous-dossier créé par `path.parent.mkdir(parents=True)` l.480). Tous les chemins testés restent sous IVHistory (resolve().relative_to OK) — pas de traversal mesuré.

### impact
Faible en pratique (l'écriture n'a lieu qu'après un fetch Alpaca réussi pour ce symbole, donc un symbole valide) ; incohérence de convention avec les autres caches et pollution possible de `cache/IVHistory/` par des sous-dossiers.

### fix_sketch
Dans `_iv_history_path` : `safe = re.sub(r"[^A-Za-z0-9._-]", "_", sym) or "SYMBOL"` puis `CACHE_IV_HISTORY_DIR / f"iv_daily_{safe}.csv"` (même regex que options/logic.py:1107).

### panel
- [CODE READING] refuted=false conf=0.9 sev=m
  reason: Reading confirmed: _iv_history_path L455-457 only strips/uppercases and interpolates into a Path; sibling caches sanitise (market_data.py L361 replace('/', '_'), L483 `_safe_key_fragment`, options/logic.py L1107 re.sub). The same module escapes the symbol for URLs (sym_url L283-286, quote_plus) but not for the filesystem. Measured: 'A/B' and 'A\\B' -> IVHistory/iv_daily_A/B.csv, the parent dir is created by mkdir(parents=True) at L480; no traversal ('..' and '../evil' become literal components 'iv_daily_..', resolve().relative_to(tmp) OK). Extra Windows colour: 'A:B' writes an NTFS alternate data stream on the 'iv_daily_A' entry (record and load both succeed with no visible file); 'A?B' raises OSError, caught by the best-effort except. load_iv_history never crashes on odd names (Path.exists() returns False for invalid Windows names). Reachability confirms minor: record_iv_observation is only called at L568 when fetch_current_atm_iv returned a result, which needs fetch_spot_price(sym) and Alpaca snapshots/chain to succeed for that literal symbol, so a separator-bearing symbol essentially never reaches the write.
  corrected_evidence: p4 probe [2]: 'A/B' -> 'iv_daily_A\\B.csv' parent='iv_daily_A'; '../evil' -> 'iv_daily_..\\EVIL.csv' inside tmp (no traversal); after record('A/B'|'A:B'|'A?B'): files = ['iv_daily_A', 'iv_daily_A\\B.csv', ...] — 'A:B' stored as ADS (load_iv_history('A:B') rows=1, no file listed), 'A?B' -> '[Errno 22] Invalid argument' swallowed by the except. 'BRK.B', 'BRK-B', '^VIX', 'ES=F' stay flat files.
  fix_ok=true — `re.sub(r'[^A-Za-z0-9._-]', '_', sym) or 'SYMBOL'` matches options/logic.py L1107 and is proportionate (one line, needs `import re`). Apply it in _iv_history_path so both record and load share the key; existing files for plain tickers are unaffected.
- [REPRODUCTION] refuted=false conf=0.8 sev=m
  reason: Reproduced: _iv_history_path('A/B') and ('BRK/B') -> <IVHistory>/iv_daily_A/B.csv; record_iv_observation creates the sub-directory via path.parent.mkdir(parents=True) (L480) — measured files 'IVHistory\\iv_daily_A\\B.csv', 'IVHistory\\iv_daily_BRK\\B.csv'. '../evil' -> iv_daily_../EVIL.csv fails with ENOENT (warning only, nothing written). The finder's 'no traversal measured' is incomplete on Windows: '/../../ESCAPE' -> 'IVHistory/iv_daily_/../../ESCAPE.csv' is normalised lexically by Win32 and ESCAPE.csv is written OUTSIDE IVHistory (measured at the sandbox root, resolve().relative_to fails). However both the write (L568, only after fetch_current_atm_iv returned info, which needs fetch_spot_price + Alpaca snapshots for that exact string) and the read (L576, only after fetch_daily_closes succeeded) are gated by a successful market fetch for the symbol, so such strings are unreachable in practice. Convention mismatch with the other caches is real; impact stays low -> m confirmed.
  corrected_evidence: Measured (p4 repro, sandbox IVHistory): 'A/B' -> iv_daily_A\\B.csv (subdir created); 'BRK/B' -> iv_daily_BRK\\B.csv; '../evil' -> ENOENT warning, nothing written; '/../../ESCAPE' -> ESCAPE.csv written one level ABOVE IVHistory (Windows lexical '..' normalisation), load_iv_history reads it back (1 row); '..\\..\\ESCAPE2' -> IVHistory\\ESCAPE2.csv (stays inside); 'A B' -> 'iv_daily_A B.csv'; '' -> 'iv_daily_.csv'. Evidence nit: market_data.py:361 uses .replace('/', '_').replace(' ', '_'), not re.sub; only market_data.py:483 (regex [^A-Za-z0-9_-]+) and options/logic.py:1107 ([^A-Za-z0-9._-]) use re.sub.
  fix_ok=true — re.sub(r'[^A-Za-z0-9._-]', '_', sym) or 'SYMBOL' is correct and proportionate: it removes both separators (so '/../../X' becomes '_.._.._X', no traversal on either OS), keeps BRK.B intact, and matches options/logic.py:1107. Apply it once in _iv_history_path so both record_iv_observation and load_iv_history agree.
- [IMPACT & SEVERITY] refuted=true conf=0.75 sev=m
  reason: Impact is negligible in any reachable path. The write path (record_iv_observation) only runs after fetch_current_atm_iv returned a non-None info (service.py L567-569), i.e. after Alpaca served option snapshots or a chain for that exact symbol — a symbol with a path separator never gets there. The read path (load_iv_history, called unconditionally at L573) was probed on Windows with 13 hostile symbols ('A/B', 'A\\B', 'A:B', 'A?B', 'A*B', 'A|B', 'A<B>', '"A"', '../X', 'SPY,QQQ', 'NUL', 'CON', 'BRK.B'): every call returns 0 rows and nothing raises (Path.exists() swallows WinError 123). Nothing displayed is wrong, nothing crashes, no data is lost; the only observable effect would be a stray subfolder/NTFS alternate stream in cache/IVHistory/ via an unreachable path. The finding itself rates its impact 'faible en pratique'; it is a style-consistency nit to fold into any future touch of _iv_history_path, not a review finding.
  corrected_evidence: load_iv_history on 13 hostile symbols (Windows): rows=0 each, no exception. Unreachable write path curiosity: 'A:B' writes into an NTFS alternate data stream of file 'iv_daily_A' (0-byte main stream), after which 'A/B' fails with WinError 183 — both only reachable if Alpaca returned snapshots for such a symbol, which it cannot.
  fix_ok=true — The one-line re.sub to the same regex as options/logic.py:1107 is correct and keeps existing filenames for all legitimate symbols (dot in BRK.B is allowed, so no cache migration). Harmless to apply opportunistically; does not justify its own finding.


## m6 — html-exception-text-in-log
- titre: Texte brut d'exception (page HTML 401 nginx, 8 lignes) injecté tel quel dans le journal et dans iv_error
- ancre: app/model/iv_dashboard/service.py:147
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 15 min
- dimension finder: orchestrator
- repro: scripts/review_iv_dashboard/orch_live_alpaca.out.txt

### claim
Les messages de log concaténent str(exc) sans troncature ni nettoyage ; une réponse HTML (401/403 edge) s'étale sur 8 lignes par feed dans l'expander « Journal » (st.code), et iv_error = iv_log[-1] peut être une phrase longue/technique affichée en warning.

### evidence (finder + orchestrateur)
Run live orch_live_alpaca.py (clés révoquées -> 401 HTML) : log contient 2 blocs '<html>...<center>nginx</center></body></html>' (feed default + iex), 534 barres via fallback Stooq/Yahoo ; voir scripts/review_iv_dashboard/orch_live_alpaca.out.txt.

### impact
Journal illisible quand Alpaca refuse la clé (cas fréquent : clé révoquée/expirée) ; le message utile ('401 Unauthorized') est noyé.

### fix_sketch
Helper `_short_exc(exc, n=160)` : `' '.join(str(exc).split())[:n]` appliqué dans fetch_daily_closes L147, fetch_current_atm_iv L368/375 ; pour requests.HTTPError utiliser `f'HTTP {resp.status_code}'`.

### panel
- [CODE READING] refuted=false conf=0.85 sev=m
  reason: Log half confirmed, iv_error half mostly unreachable. alpaca-py 0.12.0 common/rest.py L203-205 raises APIError(response.text, http_error) and APIError.__init__ passes that text to Exception (exceptions.py L10-11), so str(exc) is the raw nginx HTML body; service.py L149 (not L147 as anchored — L147 is the success return) appends f'Barres {tag} indisponibles : {exc}' for both feeds, get_iv_dashboard_data forwards it (L522-523) and the view dumps it through st.code (tab_iv_dashboard.py L495-497). The live run output (orch_live_alpaca.out.txt L63-78) shows the two 8-line HTML blocks. For iv_error: it is iv_log[-1] (L574); the message holding str(exc) at L368 is never last (always followed by L374 or L376), and L376's exc2 comes from download_options_alpaca which swallows HTTP/network errors internally, so the long-HTML-in-st.warning case is practically unreachable — measured: on a 503 iv_error = 'Aucun contrat entre 15 et 60 jours d'échéance.'. Severity m stands on the journal-readability impact alone.
  corrected_evidence: Anchor should be service.py L149 (concat site), fix targets L149 / L368 / L376 (finding says L147 / L368 / L375). p4_fallback_probe.py: on a 503, iv_log[-1] carries no exception text ('503' in log[-1] -> False). Live run: 2 x 8-line '<html>...<center>nginx</center>...' blocks in result['log'] (orch_live_alpaca.out.txt).
  fix_ok=true — Proportionate. For the alpaca-py leg use exc.status_code (APIError property, exceptions.py L21-24) rather than parsing text; for requests.HTTPError use exc.response.status_code. Whitespace-collapse + 160-char cap handles the HTML case well enough.
- [REPRODUCTION] refuted=false conf=0.85 sev=m
  reason: Half reproduces, half does not. Daily-bars path: alpaca-py 0.12.0 RESTClient._one_request raises APIError(response.text) on any non-retry HTTP error, so str(exc) is the raw nginx page; service.py L149 appends it verbatim. Measured with a mocked 401 HTML response: log[0] 202 chars / 8 lines, log[1] 208 chars / 8 lines, 2 messages containing '<html>', dumped as-is by the view's st.code (tab_iv_dashboard.py L495-497). That part stands. The claim that iv_error can be this long HTML does NOT reproduce: on the same 401 the snapshot path (L368) logs a single 217-char line ('401 Client Error: Unauthorized for url: ...') and iv_error = iv_log[-1] = "Aucun contrat entre 15 et 60 jours d'échéance."; L376 (exc2) is practically unreachable because download_options_alpaca swallows its own errors.
  corrected_evidence: HTML reaches only the expander log via fetch_daily_closes L149 (2 x 8 lines), never iv_error. L368's message is one line but carries the full URL with query string (217 chars measured). iv_error misdirection is the separate 'fallback-chain...masks-root-cause' finding, not an HTML-length problem.
  fix_ok=true — Truncation alone (' '.join(str(exc).split())[:160]) still prints '<html> <head><title>401 Authorization Required...' tags. Better: alpaca APIError exposes .status_code (alpaca/common/exceptions.py) -> log f'HTTP {exc.status_code}' when available, and for requests.HTTPError use exc.response.status_code; fall back to the whitespace-collapsed truncated text otherwise. Apply at L149 and L368; L376 is low value.
- [IMPACT & SEVERITY] refuted=false conf=0.85 sev=m
  reason: Behaviour confirmed, impact cosmetic: p4_fallback_impact.py S5 reproduces the live 401 case (alpaca-py raises with the nginx HTML body as message): the two bars entries are 8 lines each, the Journal st.code block renders 17 lines for 3 log entries. This lives in the collapsed 'Journal & détails des régressions' expander (tab L476-497), does not touch any displayed number, and st.code renders the tags literally (no injection). The 'iv_error may be a long technical sentence' half is currently unreachable on the snapshot path: S4 shows iv_error is always the short (but false) 'Aucun contrat entre 15 et 60 jours…' because of the fallback; it only becomes reachable once fallback-chain-cannot-reach-30dte's fix surfaces the first exception — which makes this helper a prerequisite of that fix rather than a standalone nuisance. m is calibrated; not a duplicate of anything surfaced to the user.
  corrected_evidence: S5: log entries = 3, rendered journal lines = 17 (8 + 8 + 1); with ' '.join(str(exc).split())[:160] each entry collapses to one line starting 'Barres alpaca indisponibles : <html> <head><title>401 Authorization Required</title></head>…'. S4: the 401 root-cause string reaching the log starts '401 Client Error: <html>\n<head>…' (would reach st.warning once the first exception is surfaced as iv_error).
  fix_ok=true — Proportionate and regression-free. Apply _short_exc at service.py L147 (bars), L368 (snapshots) and L375 (chain). For requests.HTTPError prefer exc.response.status_code ('HTTP 401') over the body; alpaca-py APIError exposes status_code too when available. Land it before (or with) the fix that surfaces the first exception as iv_error.


## m7 — linreg-scipy-valueerror-coincidence
- titre: Série RV constante : ValueError scipy attrapée par coïncidence, message anglais brut affiché
- ancre: app/model/iv_dashboard/analytics.py:139
- sévérité finale (vote): m  (votes: m, m, m/réfuté; finder: m)
- effort: 30 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_math_service_constant_series.py

### claim
_linreg appelle stats.linregress sans garde ; une série RV à valeur unique (clôtures constantes ou alternance à 2 prix) lève ValueError('Cannot calculate a linear regression if all x values are identical') APRÈS le contrôle MIN_ANALYSIS_POINTS, et service.py L586 l'attrape comme si c'était le signal « série insuffisante » (seul cas documenté par la docstring L166).

### evidence (finder + orchestrateur)
p1_math_service_constant_series.py (service stubbé, clôtures 10.00/10.01 alternées, 262 lignes, vol nunique=1) : analysis=None, analysis_error='Cannot calculate a linear regression if all x values are identical' -> affiché tel quel dans st.info (tab L542) et dans le journal. Clôtures constantes : idem (current_vol=0.0, percentile 0.502 -> NORMALE/NEUTRE). Chemin atteint uniquement si TOUTE la série n'a qu'une valeur distincte : un halt de 300 j suivi de 100 j bruités passe (n=350, OK), 200 séries OU mean-reverting passent, 200 graines avec halt de 60 j ne déclenchent pas le cas « sous-ensemble de régime à x identiques » (p1_math_degenerate_regressions.py, p1_math_regime_subset_degenerate.py).
--- (sonde orchestrateur scripts/review_iv_dashboard/orch_probe_math.py §B) ---
Clôtures alternant 1.00/1.01 tous les 7 jours sur 300 j -> RV à 2 valeurs distinctes (n=280) : la régression GLOBALE passe (2 niveaux) mais la régression PAR RÉGIME (reg_high/reg_low, >10 pts, x identiques) lève ValueError('Cannot calculate a linear regression if all x values are identical') -> attrapée L586 -> analysis=None : reg_forward/reg_diff pourtant calculables sont perdus. Le cas n'exige donc PAS une série entièrement constante, seulement <=2 niveaux distincts de RV.

### impact
Cas rare (symbole sans mouvement sur toute la durée) : l'analyse est effectivement impossible, donc aucune perte d'information, mais l'utilisateur voit un message scipy en anglais et un régime « NORMALE » (percentile 50.2 % = moyenne des ex æquo) sur une vol strictement nulle.

### fix_sketch
Dans analyze_forward_vol, après le dropna L182 : `if df['current_vol'].nunique() < 2: raise ValueError('Série de vol constante : régression impossible.')` ; et dans _linreg, retourner un dict de NaN (slope/intercept/r2/p_value NaN, n) quand `x.nunique() < 2` pour que les régressions par régime ne puissent jamais faire tomber reg_forward/reg_diff déjà valides.

### panel
- [CODE READING] refuted=false conf=0.9 sev=m
  reason: Reading is correct. analytics.py L139 calls stats.linregress with no guard; scipy 1.11.4 (venv: scipy/stats/_stats_mstats_common.py L156-158) raises ValueError('Cannot calculate a linear regression if all x values are identical') when amax(x)==amin(x). The MIN_ANALYSIS_POINTS check (L183) runs before, so the constant-series case passes it. service.py L586 catches bare ValueError and stores str(exc) as analysis_error; tab_iv_dashboard.py L540-543 prints it raw in st.info, L590+ in the log. Measured: constant closes -> RV nunique=1, last pct 0.502 -> classify_regime 'NORMALE', ValueError caught with the English scipy text. The orchestrator's 2-level case also reproduces: closes alternating 1.00/1.01 every 7d -> global regressions pass, regime regression (>10 identical x) raises -> whole analysis lost. Realistic 60d halt inside a noisy series: 0/50 seeds trigger, consistent with finder -> rare case, severity m is right.
  corrected_evidence: p4 probe: constant closes -> rv nunique=1, last rv 0.0, pct 0.502, regime NORMALE, ValueError 'Cannot calculate a linear regression if all x values are identical' (caught by service L586). 2-level closes -> n=280, ValueError from regime regression. 60d halt, 50 seeds -> 0 hits.
  fix_ok=true — First half (nunique<2 guard after L182 raising a French ValueError) is fine. Second half should make _linreg return None rather than a NaN dict for the regime regressions: the view already handles None (_format_reg L465 'données insuffisantes', _render_diff_chart L433 'if not reg'), whereas a NaN dict would print 'pente nan' in the log and add an empty plotly trace. Alternatively guard at the call sites L203-212 with `and df.loc[mask,'current_vol'].nunique() > 1`.
- [REPRODUCTION] refuted=false conf=0.9 sev=m
  reason: Reproduced independently (p4 script, scipy 1.11.4). analytics.py L139 calls stats.linregress unguarded; L182-187 only checks row count. (a) constant closes x400 -> RV == 0.0 on 380 rows -> ValueError('Cannot calculate a linear regression if all x values are identical') caught by service.py L586 `except ValueError` and stored verbatim in analysis_error -> tab L542 st.info; meanwhile current_vol=0.0, percentile 0.50198 -> regime 'NORMALE'. (b) closes 1.00/1.01 alternating every 7 d -> RV nunique=3 (2 rounded levels): global reg OK, per-regime reg (L203-212) raises the same ValueError -> whole analysis lost although reg_forward/reg_diff were computable. (c) 200 simulated illiquid paths (flat >=20 d stretches mixed with noise): 1/200 hits. Mechanism is real; trigger realism is low (near-dead tickers), consistent with severity m.
  corrected_evidence: Constant closes: ValueError msg exactly 'Cannot calculate a linear regression if all x values are identical', current_vol=0.0, current_pct=0.501984, regime=NORMALE. 1.00/1.01 every 7 d: rv nunique=3 (levels 0.051248 / 0.06224), still raises in per-regime reg. 10.00/10.01 daily: rv nunique=1 -> raises. Illiquid sims: 1/200 runs raise (seed 2: 277/380 zero-RV rows).
  fix_ok=true — nunique<2 guard in analyze_forward_vol with a French message + NaN-dict return in _linreg is proportionate; view _format_reg formats NaN fine. Ensure slope1 NaN path (L193-198) falls back to median — it already does.
- [IMPACT & SEVERITY] refuted=true conf=0.75 sev=m
  reason: Behaviour confirmed (scipy 1.11.4 raises ValueError, not a warning, on identical x) but the impact is negligible in realistic use. Trigger requires the ENTIRE vol series (or an entire >10-point regime) to have <=2 distinct RV values: on real AAPL data every one of 471 sliding 2y windows has nunique(vol) >= 498/504 and the per-regime subsets have 186/288 distinct values — a halted/dead ticker for the whole duration is needed, and such a ticker has no options market, so no trading decision is at stake. When it does trigger nothing is silent: the user sees a vol of 0.0% and an explicit st.info 'Analyse forward vol indisponible : Cannot calculate a linear regression if all x values are identical' plus the journal line. The only real defects are copy (English scipy text) and the 'NORMALE' label on a zero vol — polish. Severity m is calibrated; under the impact lens it does not survive as a finding worth blocking on.
  corrected_evidence: p4_impact_math.py S3: stats.linregress(np.ones(12), arange) -> ValueError 'Cannot calculate a linear regression if all x values are identical' (confirmed). Real-data distance from the trigger: min nunique(vol) over 471 real 2y windows = 498 of 504; latest 2y window nunique=474/474, high regime 186 distinct, low regime 288 distinct. Default symbols SPY/QQQ/AAPL cannot reach nunique<2.
  fix_ok=true — Both halves are cheap and safe: the nunique<2 guard in analyze_forward_vol gives a French message through the existing ValueError path; returning a NaN dict from _linreg for degenerate x makes slope1 NaN -> median split (already handled L195-198) and the view prints 'nan' fields without crashing (plotly drops NaN lines). Minor: ensure _format_reg tolerates NaN (it uses :.4f, which prints 'nan' — fine). Optional: classify_regime could return 'unknown' when current_vol == 0 to avoid the 'NORMALE' label, but that is polish.


## m8 — regime-split-out-of-range
- titre: Intersection y=x hors de la plage des données (pente > 1) : split de régime négatif affiché, un régime vide, reg_high = reg_diff
- ancre: app/model/iv_dashboard/analytics.py:194
- sévérité finale (vote): m  (votes: m, m, M; finder: m)
- effort: 30 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_math_degenerate_regressions.py

### claim
intersection = intercept/(1-slope) n'est pas contrainte à la plage de current_vol ; avec slope > 1 et intercept > 0 (série à rupture de régime : halt puis activité) elle devient négative, tout l'échantillon tombe dans le régime haut, reg_low=None, n_low=0 et reg_high duplique exactement reg_diff. La garde `abs(1-slope) > 1e-12` est équivalente au `!= 1` legacy (l'explosion pour slope≈1 n'est pas évitée).

### evidence (finder + orchestrateur)
p1_math_degenerate_regressions.py : halt 300 j + 100 j bruités -> slope 1.0436, intercept 0.0131, intersection -0.3008, n_high=350, n_low=0, reg_low=None, reg_high == reg_diff (mêmes slope/intercept/r2). Le journal affiche « Régime VOL BASSE (vol ≤ -0.301, n=0) » et le graphe diff une vline « Split régimes (-0.301) » hors axe. Sur 200 séries RV mean-reverting synthétiques (OU log-vol) : pente dans [0.262, 0.870], 0 intersection hors plage, 0 régime vide — le cas est confiné aux ruptures structurelles.

### impact
Lecture trompeuse du split (un seuil de vol négatif) et d'une « régression régime haute » qui n'est que la régression globale ; pas de crash (vue guardée par `not int(mask.sum())`). Comportement identique au legacy.

### fix_sketch
Après L198 : `lo, hi = df['current_vol'].min(), df['current_vol'].max(); if not (lo < intersection < hi): intersection = float(df['current_vol'].median()); insights.append('Pente ≥ 1 : pas d'intersection avec y=x dans la plage, split à la médiane.')`.

### panel
- [CODE READING] refuted=false conf=0.9 sev=m
  reason: Reading is correct. analytics.py L193-198: intersection = intercept/(1-slope) is only protected against non-finite values and |1-slope|<=1e-12; no check against the current_vol range. With slope>1 and intercept>0 it goes negative, high_mask (L200) is all True, reg_low=None (L208-212, sum 0 <= 10), reg_high is computed on df.loc[all-True] == df so it equals reg_diff exactly. View: _render_diff_chart L433 guards empty masks (no crash) but L447-451 still draws the vline at the negative x and _render_log L484-491 prints 'Régime VOL BASSE (vol <= -0.301, n=0)'. The 1e-12 guard is indeed equivalent to the legacy `slope1 != 1` (option_trading_dashboard.py L477-480): slope=1-1e-9 passes and gives intersection 1e7. Legacy had no range check either, so behaviour is identical to legacy.
  corrected_evidence: p4 probe (halt 300d + 100d noise, seed 0): slope1 1.0436, intercept 0.0131, intersection -0.3008, vol range [0.0000, 0.3388], n_high 350, n_low 0, reg_low None, reg_high == reg_diff True. slope=1-1e-9 -> abs(1-slope)>1e-12 True -> intersection 1.0e7.
  fix_ok=true — Clamping to the median when intersection is outside [min,max] is proportionate and mirrors the existing slope==1 fallback (L196); it also neutralises the slope~1 blow-up since a 1e7 intersection is out of range. Keep the insight message so the user knows the split is not a y=x intersection.
- [REPRODUCTION] refuted=false conf=0.9 sev=m
  reason: Reproduced. Finder's probe p1_math_degenerate_regressions.py re-run gives exactly slope 1.04356, intercept 0.01310, intersection -0.30076, n_high=350, n_low=0, reg_low=None, reg_high identical to reg_diff (same slope/intercept/r2/p). My own halt300+noisy100 draw (seed 1) did NOT reproduce (slope 0.943, intersection 0.126 in range) so the halt case is seed-dependent; but the mechanism is broader than the title claims: smooth trending log-vol with slope 0.9839 < 1 gives intersection 0.3229 above max 0.2367 -> n_high=0, reg_low == reg_diff; drift-down case gives intersection 0.0243 below min 0.0339 -> n_low=0, reg_high == reg_diff; 9/200 near-random-walk log-vol paths (theta=0.002) yield an out-of-range intersection and an empty regime. Guard |1-slope|>1e-12 (L193) with slope=1-1e-6 passes and gives intersection 1e4, so it is functionally the legacy `!= 1`. View L447-452 draws the vline at that value and L486/490 print 'n=0' regimes. No crash; identical to legacy L477-480.
  corrected_evidence: Not confined to slope > 1 nor to structural breaks: slope 0.984/intercept 0.0052 -> intersection 0.323 vs data range [0.136, 0.237] (n_high=0, reg_low duplicates reg_diff); slope 0.927/intercept 0.0018 -> intersection 0.024 vs [0.034, 0.150] (n_low=0). OU log-vol theta=0.02: 0/200 out of range (matches finder); theta=0.002: 9/200 out of range with an empty regime. Halt300+noisy100 is seed-dependent (seed 1: slope 0.943, in range).
  fix_ok=true — Range clamp to median is proportionate and also neutralises the slope~1 explosion. But the sketched insight text 'Pente >= 1 : pas d'intersection...' is wrong in the slope<1 cases measured above; word it as 'intersection y=x hors de la plage des données, split à la médiane'.
- [IMPACT & SEVERITY] refuted=false conf=0.7 sev=M
  reason: At DEFAULT parameters the impact is nil: on real AAPL history (Stooq 1984-2026, incl. 1987/2008/COVID) the pipeline never degenerates — 0/471 sliding 2y windows, 0/241 1y windows, 0/217 5y windows have slope>=1, an out-of-range intersection or an empty regime (slope always in [0.04, 0.93], min(n_high,n_low)>=18). BUT the form offers rv_window up to 120 and forward up to 90, and at those reachable settings the degeneracy is frequent on real data: rv=120/fwd=30/1 an -> 67/238 windows (28%) out of range + empty regime, 36 with reg_high==reg_diff, displayed split up to 273.910 (=27391% vol) and 'Régime VOL BASSE (vol <= 273.910, n=238)'; rv=120/fwd=90/1 an -> 62/238 (26%); rv=60/fwd=90/1 an -> 14/240 (6%); rv=120/fwd=90/2 ans -> 24/232 (10%, intersection up to 88.98). That is 'misleading in realistic conditions' (a displayed threshold and a 'regime' regression that is the global one, vline off-axis) => M, not m. Mitigating: the primary signals (current percentile, regime label, mean-reversion insights) are unaffected, the absurd threshold is visibly garbage, and behaviour is legacy-identical. Also confirmed the 1e-12 guard is useless: windows with slope exactly 1.000/0.998 produce intersections of 273.9/13.2.
  corrected_evidence: Real-data sweep (p4_regime_split_param_sweep.py, AAPL 1984-2026, step 42 td): defaults rv=20/fwd=30: 0 out-of-range over 235 (2y), 241 (1y, one 0.928-slope window out of range with an empty regime), 217 (5y) windows. rv=120/fwd=30/1y: slope>=1 in 14/238, intersection out of [min,max] in 67/238 (28%), empty regime 67, reg_high==reg_diff 36, intersection range [-3.418, 273.910]. rv=120/fwd=90/1y: 10 slope>=1, 62/238 out (26%), 35 duplicated regressions. rv=60/fwd=90/1y: 14/240 out (6%). rv=120/fwd=90/2y: 24/232 out (10%), intersection up to 88.98. IMPORTANT: most out-of-range cases have slope < 1 (e.g. 0.599, 0.283, 0.509) — the cause is a narrow RV range with a smooth long-window series, not only 'pente > 1'. The finding's restriction to structural breaks / slope>1 is too narrow.
  fix_ok=true — Range check + median fallback is proportionate and cannot regress (intersection stays finite, test_iv_dashboard_analytics L123 still passes; masks computed identically). Two corrections: (1) the insight text must NOT say 'Pente >= 1' — in the sweep the majority of out-of-range cases have slope < 1; use a generic message such as 'Intersection y=x hors de la plage des vols observées : split à la médiane.' (2) Prefer a clamp to the inner percentiles or median rather than strict 'lo < x < hi' alone — an intersection at 0.1% from the edge still yields n_high of a handful of points; consider also falling back when min(n_high,n_low) <= MIN_REGIME_POINTS so the per-regime regressions never silently duplicate reg_diff.


## m9 — percentile-label-vs-effective-window
- titre: Libellé « Percentile (N j) » = fenêtre demandée, pas la fenêtre effective (min_periods=60 + lookback = durée seulement)
- ancre: app/model/iv_dashboard/service.py:521
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 30 min
- dimension finder: ? (fusion: percentile-window-not-in-lookback)
- repro: scripts/review_iv_dashboard/p1_math_effective_window.py ; scripts/review_iv_dashboard/p1_alpaca_plumbing_warmup.py

### claim
extra_days ne couvre que le warm-up RV (≈47 j calendaires), pas la fenêtre percentile ; quand percentile_window > nombre de lignes RV disponibles (1 an + 504/756, 2 ans + 756, combinaisons autorisées par l'UI et le controller), le percentile affiché est calculé sur toutes les lignes disponibles, mais la métrique reste étiquetée « Percentile (756 j) ».

### evidence (finder + orchestrateur)
p1_math_effective_window.py (service stubbé, série couvrant exactement le lookback demandé) : 1 an/756 -> 275 points effectifs (label 756) ; 1 an/504 -> 275 ; 2 ans/756 -> 536. Combinaisons par défaut 2 ans/252 -> 252 points (exact). Jour 60 d'historique : percentile = rang parmi 60 points (p1_math_percentile_forward.py §C, première valeur non-NaN à la position 59).
--- (doublon ? percentile-window-not-in-lookback @ app/model/iv_dashboard/service.py:521) ---
p1_alpaca_plumbing_warmup.py (business-day approximation, holidays ignored): years=1/pct=504 -> 275/504 obs (55%); years=1/pct=756 -> 275/756 (36%); years=2/pct=756 -> 536/756 (71%); defaults years=2/pct=252 -> 252/252 OK; years=1/pct=252 -> 252 (barely, before holidays).

### impact
Pour les combinaisons non-défaut, le rang affiché est un percentile sur ~1 an étiqueté 3 ans — lecture faussée du régime ; l'écart legacy (strict 252 -> NaN) est déclaré mais la relaxation n'est pas rendue visible.

### fix_sketch
Soit étendre le lookback : `extra_days = int(rv_window*1.6) + int(percentile_window*1.5) + 15` (fetch_daily_closes L140) ; soit exposer le nombre effectif : service retourne `percentile_effective_n = int(series_df['vol'].tail(pwin).shape[0])` et la vue (L190) affiche « Percentile (252 j, 275 pts) » quand n < fenêtre.

### panel
- [CODE READING] refuted=false conf=0.9 sev=m
  reason: Reading is correct. service.py L521 `extra_days = int(rv_window*1.6)+15` only budgets the RV warm-up; L140 `lookback_days = years*365.25 + extra_days` never looks at `percentile_window`. L531 `compute_percentile_series(rv, percentile_window)` uses `rolling(window, min_periods=60).rank(pct=True)` (analytics.py L75-77), so when fewer rows than `window` exist the rank is taken over whatever is available, silently. View L190 labels the metric `Percentile ({percentile_window} j)` from the requested parameter (L596) and the help text (L192) does not mention an effective count. UI offers Durée 1 an with percentile up to 756 (L137-139) and controller clamps keep those combinations (years 0.5..10, pwin 60..756, L49-51). Second sink: the IV-vs-RV percentile (L569) uses `series_df['vol'].tail(pwin)` AFTER the cutoff filter (L536), so it has even fewer points than the rolling percentile.
  corrected_evidence: p4 probe, pandas `rv.rolling(pwin, min_periods=60).count().iloc[-1]` on a business-day stub covering exactly the requested lookback (rv_window=20): 1y/504 -> 275 pts (label 504) ; 1y/756 -> 275 (label 756) ; 2y/756 -> 536 (label 756). IV tail post-cutoff for the same combos: 262 / 262 / 523 pts. All other UI combinations (1y/60, 1y/252, 2y/60-504, 3y+ any) have count == label. Finder's numbers reproduced exactly.
  fix_ok=true — Both sketches are proportionate. Caveat on option 1 (extend lookback): it fixes the rolling percentile but NOT the IV-vs-RV tail, because series_df is cut at `cutoff` (L536) before `.tail(pwin)` (L569); for 1y/504 the IV percentile would still use ~262 points. Option 2 (expose `percentile_effective_n` and show it in the label when n < window) covers both sinks and is the cleaner fix; combine with the same count for the IV tail if the IV chip survives the iv-signal-vrp-bias fix.
- [REPRODUCTION] refuted=false conf=0.9 sev=m
  reason: Reproduced end-to-end through S.get_iv_dashboard_data with fetch_daily_closes stubbed on the REAL AAPL trading calendar (Stooq file shifted to end today, holidays included) and load_iv_history stubbed. Hand oracle (rank of last RV among the last min(pwin, n_rv) rows) equals the service's current_percentile to 1e-12 in all 16 (years, pwin) combos, so the effective counts are exact, not estimates. extra_days = int(20*1.6)+15 = 47 calendar days (service L521). Mismatching combos reachable from the UI (Durée 1/2/3/5 ans x Fenêtre percentile 60..756): 1 an/504 -> 261 pts labeled 'Percentile (504 j)', 1 an/756 -> 261 pts labeled 756 j, 2 ans/756 -> 512 pts labeled 756 j; the other 13 combos (incl. defaults 2 ans/252) are exact. pandas oracle: rolling(756, min_periods=60).rank(pct=True) on 265 rows == plain rank among 265 (0.516981 both). Finder's 275/536 were bdate_range (no holidays) overestimates; real calendar gives 261/512. Severity m is right: only non-default advanced settings, and the value is the best available percentile — only the label overstates the window.
  corrected_evidence: Real calendar (not bdate_range): years=1 -> 281 closes / 261 RV rows: pwin 504 -> 261 eff (52 %), pwin 756 -> 261 eff (35 %); years=2 -> 532 closes / 512 RV rows: pwin 756 -> 512 eff (68 %); years=3 -> 763 RV rows: all windows exact; years=5 -> 1267 RV rows: all exact; 1 an/252 -> 252/252 exact (261 rows available). 3 of 16 UI-reachable combos mismatch, none of the defaults. Also: 1 an / 756 leaves only 251 series rows after the cutoff for the IV-vs-RV .tail(756) at L569.
  fix_ok=false — Direction right, formula wrong. Option 2's `percentile_effective_n = int(series_df['vol'].tail(pwin).shape[0])` is evaluated AFTER the L536 cutoff, but the rolling percentile at L531 is computed on the pre-cutoff rv; for 1 an/252 it would display '252 j, 251 pts' although the window really saw 252 rows. Use `min(int(percentile_window), int(rv.notna().sum()))` computed before the cutoff (or `int(pct.notna().sum()) + min_periods - 1`-style accounting on rv). Option 1 (extend extra_days by ~1.5*pwin) must also widen the Stooq/Yahoo fallback `period` at L152, which is derived from `years` only; otherwise the fallback path keeps the mismatch.
- [IMPACT & SEVERITY] refuted=false conf=0.65 sev=m
  reason: Behaviour confirmed through the real service path with fetch stubbed on AAPL closes (p4_impact_g2_epistemics.py section 2): extra_days = rv_window*1.6+15 (service.py L521) never covers the percentile window, compute_percentile_series min_periods=60 (analytics.py L61-73) silently relaxes, and tab L190 labels the metric with the requested window. Defaults 2 ans/252 are exact (252/252) and 1 an/252 gives 251 (negligible), so the default user is unaffected. The issue only arises for a self-inconsistent combination (window longer than the fetched duration) chosen inside the collapsed 'Paramettres avances' expander (tab L131-139, no help text warning). When it does arise the displayed rank is materially different from what the label promises (on AAPL, 252-pt vs 756-pt rank: median gap 0.093, p90 0.238, regime bucket differs on 46.7% of days) and the legacy would have shown NaN. Silent + advanced-only + still a valid rank on available data -> m is calibrated; not M because realistic conditions (defaults) are exact.
  corrected_evidence: Service path on real AAPL calendar: 2 ans/252 -> 252 pts (exact); 1 an/252 -> 251; 1 an/504 -> 251 (label 504); 1 an/756 -> 251 (label 756); 2 ans/504 -> 502; 2 ans/756 -> 502 (label 756); 3 ans/756 -> 753; 5 ans/756 -> 756. Shown vs true pct at data end: 1 an/756 0.027 vs 0.016; 2 ans/756 0.023 vs 0.016. Over 1996 days, |pct252 - pct756|: median 0.093, p90 0.238, max 0.378; regime bucket differs 46.7% of days, mean-reversion signal 25.5%.
  fix_ok=true — Both sketches are proportionate. The label fix (service returns percentile_effective_n, view shows 'Percentile (756 j, 251 pts)' when n < window) is the surgical one and cannot regress. The lookback-extension fix also repairs the expanding-window (60..252 pts) percentile series in the first year of a default run, at the cost of fetching ~1.5x percentile_window extra calendar days (fine for daily bars); caveat: the Stooq/Yahoo fallback period at service.py L153 is ceil(years)+1 y and ignores extra_days, so the extension would not apply on the fallback path for 1 an/756. Simplest honest alternative: clamp percentile_window to the available rows and log it.


## m10 — duplicate-date-crash
- titre: Index de dates dupliqué -> ValueError « cannot reindex on an axis with duplicate labels » au build de series_df
- ancre: app/model/iv_dashboard/service.py:534
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 15 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_math_rv_edges_epistemics.py

### claim
pd.DataFrame({'close': closes, 'vol': rv, 'vol_percentile': pct}) aligne trois Series sur l'index dates ; une date dupliquée dans les clôtures (aucune déduplication dans fetch_daily_closes ni compute_*) fait lever pandas, l'onglet affiche « Analyse impossible : cannot reindex on an axis with duplicate labels ». En amont, compute_log_returns avec doublon injecte un rendement exactement 0 (shift positionnel).

### evidence (finder + orchestrateur)
p1_math_rv_edges_epistemics.py §3 : 301 lignes / 300 dates uniques -> compute_realized_vol OK (rendement 0.0 injecté au doublon), puis construction DataFrame -> ValueError 'cannot reindex on an axis with duplicate labels'. Qu'une source réelle (barres Alpaca normalisées, fallback Stooq/Yahoo) produise un doublon : non mesuré.

### impact
Crash de l'onglet avec message opaque sur une anomalie de données bénigne ; la ligne dupliquée biaise aussi la RV vers le bas (rendement nul artificiel).

### fix_sketch
Dans fetch_daily_closes (ou juste avant L533) : `closes = closes[~closes.index.duplicated(keep='last')]` ; idem `out.drop_duplicates(subset='Date', keep='last')` dans _fetch_closes_alpaca L116 et le fallback L156.

### panel
- [CODE READING] refuted=false conf=0.8 sev=m
  reason: Reading is correct. service.py L534 builds the DataFrame from three Series; closes keeps the duplicate label while rv (first row dropped by L37) has a different index, so pandas 2.1.4 takes the union and reindexes -> ValueError 'cannot reindex on an axis with duplicate labels' (measured; with identical duplicated indexes pandas does not raise, so the crash is exactly the union path). L534 is outside any try, the controller adds nothing, and tab L167-168 shows 'Analyse impossible : cannot reindex on an axis with duplicate labels'. No dedup anywhere: grep 'duplicated'/'drop_duplicates' is False in service.py and market_data.py (_fetch_closes_alpaca L116, fallback L152, _standardize_ohlc L273 only dropna+sort). compute_log_returns yields 0.0 at the duplicate only when the duplicate carries the same close (measured [0.00833, 0.0]); a revised-bar duplicate would give a non-zero return instead. Whether Alpaca/Stooq/Yahoo ever emit a duplicate day is unmeasured (finder already says so), which keeps this at minor.
  corrected_evidence: p4 probe: 301 rows / 300 unique dates -> compute_realized_vol OK, returns at dup date [0.00833343, 0.0]; pd.DataFrame({'close','vol','vol_percentile'}) -> ValueError 'cannot reindex on an axis with duplicate labels'; pd.DataFrame({'a': c_dup, 'b': c_dup}) does not raise. No 'duplicated'/'drop_duplicates' in service.py or market_data.py.
  fix_ok=true — A single `closes = closes[~closes.index.duplicated(keep='last')]` before L533 is sufficient and covers all three sources; deduping inside the two fetchers is optional belt-and-braces.
- [REPRODUCTION] refuted=false conf=0.85 sev=m
  reason: Reproduced offline, including the full service path with fetch_daily_closes monkeypatched (no network). 301 rows / 300 unique dates (dup mid, dup last): compute_realized_vol OK (len 300), compute_percentile_series OK, then service.py L534 pd.DataFrame({'close','vol','vol_percentile'}) raises ValueError 'cannot reindex on an axis with duplicate labels' (pandas 2.1.4) because rv/pct indexes differ from closes (first row dropped) and need a reindex on a duplicated axis. get_iv_dashboard_data does not catch it (only analyze_forward_vol is try/except'ed, L583-588) -> tab L167-168 st.error('Analyse impossible : cannot reindex ...'). No dedup anywhere in _fetch_closes_alpaca L116, fallback L156, or market_data._standardize_ohlc. Realism: Alpaca daily bars are one-per-day; the Yahoo chart v8 fallback (used when both Alpaca feeds fail, e.g. no keys and Stooq down) is known to occasionally emit today's bar twice -> low but non-zero. Crash-with-opaque-message on benign data anomaly -> m.
  corrected_evidence: Return injected at the duplicate is exactly 0.0 only when the duplicated row carries the same close (measured [-0.002888, 0.0]); with a different close it injects log(c2/c1) (measured 0.009083 for +1.0). In both cases the DataFrame build crashes before anything is displayed, so the 'RV biased down' secondary impact is moot until the crash is fixed. Full service path: ValueError propagates uncaught to the controller.
  fix_ok=true — One-line keep='last' dedup before L533 (or in fetch_daily_closes) is proportionate and removes both the crash and the spurious return.
- [IMPACT & SEVERITY] refuted=false conf=0.6 sev=m
  reason: Behaviour confirmed (ValueError 'cannot reindex on an axis with duplicate labels' at the series_df build; an artificial 0.0 return is injected at the duplicate). When triggered the whole analysis for that symbol is replaced by st.error with an opaque pandas message (tab_iv_dashboard.py L167-168) — loud, not silent, and the tab itself keeps working, so not C. Likelihood is low but not provably zero: Alpaca daily bars are normalised in UTC (04:00/05:00Z -> one date, no DST collision), 0 duplicate dates in 10 415 real Stooq rows (p4_real_cache_scan.py), but the Yahoo chart fallback is known to occasionally return today's live bar alongside the daily bar during US session hours (yfinance dedups for this reason) and the live probe shows the app currently runs on the fallback chain (Alpaca 401). A 1-line defensive dedup at a single chokepoint removes a feature-killing failure mode for the price of nothing => keep as m (robustness). Would become M if the Yahoo live-session duplicate is confirmed with real traffic.
  corrected_evidence: p4_impact_math.py S4 on a real 2y AAPL window + one duplicated date: DataFrame build -> ValueError 'cannot reindex on an axis with duplicate labels'; injected log-return at the duplicate = 0.000000; after closes[~closes.index.duplicated(keep='last')] the build succeeds and the RV series is numerically identical to the clean one (np.allclose True). Real-data likelihood: 0 duplicate dates across 4 cached price files / 10 415 Stooq rows; Alpaca timestamps normalise to distinct dates; Yahoo live-session duplicate unmeasured (no network).
  fix_ok=true — Correct and proportionate, but apply it ONCE at the chokepoint (in fetch_daily_closes just before returning, or in get_iv_dashboard_data before L533: closes = closes[~closes.index.duplicated(keep='last')]) rather than in three places; keep='last' is the right choice for a live-bar duplicate (most recent wins). Also append a log line when duplicates were removed so the journal shows it. No regression risk: a deduplicated monotonic index is exactly what analytics already assumes.


## m11 — rv-bad-close-silent-drop
- titre: Clôture ≤ 0 : deux rendements supprimés silencieusement, la fenêtre RV enjambe le trou (pas de NaN)
- ancre: app/model/iv_dashboard/analytics.py:37
- sévérité finale (vote): m  (votes: m, m, m/réfuté; finder: m)
- effort: 30 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_math_rv_edges_epistemics.py

### claim
px.where(px>0) puis .dropna() retire les rendements t et t+1 au lieu de les laisser NaN : la RV n'est pas NaN sur 20 jours (hypothèse de l'attaque réfutée) mais est calculée sur 22 lignes calendaires avec 20 rendements, et le saut réel t-1 -> t+1 est perdu ; 2 dates disparaissent de la série.

### evidence (finder + orchestrateur)
p1_math_rv_edges_epistemics.py §1-2 : une clôture 0.0 (ou -5.0, même résultat) au milieu de 300 jours -> len(rv) 299 -> 297, RV absente le jour J et J+1, 19 valeurs RV modifiées, écart max 1.72 pts de vol (synthétique σ=1 %/j).

### impact
Une ligne corrompue dans les barres décale jusqu'à `window-1` valeurs de RV sans trace dans le journal ; effet borné (≈1.7 pt ici) et source peu probable côté Alpaca.

### fix_sketch
Garder l'alignement : `rets = np.log(px / px.shift(1))` sans dropna, `rets.iloc[1:]`, puis rolling(min_periods=window) -> NaN explicite autour du trou ; ou au minimum logger `n_bad = int((px <= 0).sum())` dans le service.

### panel
- [CODE READING] refuted=false conf=0.85 sev=m
  reason: Reading is correct. analytics.py L36 `px.where(px > 0)` turns the bad close into NaN, L37 `np.log(px/px.shift(1)).dropna()` then removes the two NaN returns (t: px NaN, t+1: shift NaN), so the return index loses 2 dates and the rolling std at L55 (positional, min_periods=window) spans the hole with no NaN. Nothing upstream filters close<=0: _fetch_closes_alpaca L116 and the fallback L152 only dropna. Measured: 299->297 returns, dates J and J+1 missing, RV has only the 19 warm-up NaN, 19 RV rows altered, max 1.08 vol pts (finder 1.72 with another seed); the two dropped returns sum exactly to the true J-1->J+1 jump, so it is that jump that is silently lost. Negative close behaves the same. Trigger is improbable with Alpaca daily bars, hence minor.
  corrected_evidence: p4 probe (sigma 1%/d, seed 1): len rets 299 -> 297; missing 2024-07-29 and 2024-07-30; rv_bad NaN count 19 (warm-up only); 19 RV rows changed; max abs diff 1.083 vol pts; sum of dropped rets 0.00243 == log(c[151]/c[149]).
  fix_ok=true — Both variants work. The minimal one (log `n_bad` in the service) is proportionate given the improbable trigger. The 'keep alignment' variant changes the compute_log_returns contract (docstring 'first row dropped') and produces ~window NaN RV rows that service L535 then drops, so the forward window in analyze_forward_vol (positional shift L171) would span the gap anyway; no test pins the bad-close behaviour (tests/test_iv_dashboard_analytics.py L28-56).
- [REPRODUCTION] refuted=false conf=0.8 sev=m
  reason: Reproduced. analytics.py L36-37: px.where(px>0) then .dropna() removes returns t and t+1 instead of leaving NaN. One close 0.0 (or -5.0) at row 150 of 300 (sigma 1%/d, seed 42): len(rv) 299 -> 297, dates 2024-07-30 and 2024-07-31 vanish, 0 NaN beyond warm-up, 19 RV values change (2024-08-01..2024-08-27), max |diff| 4.92 vol pts on a ~14.5% RV level — larger than the finder's 1.72 (seed-specific). The 20-return window at the first changed date spans 22 calendar rows (2024-07-03..2024-08-01). Service leaves no trace (log at service.py L543-545 only reports current RV). Realism is low: service.py L116/L156 already drop NaN closes and Alpaca daily bars do not carry close<=0, so only a corrupted fallback row can trigger it; bounded effect -> m.
  corrected_evidence: Max RV distortion 4.92 vol pts (not 1.72) on a 14.5% level, 19 values over 2024-08-01..2024-08-27, 2 dates dropped, 0 NaN. NaN/non-numeric closes behave identically in analytics but are already filtered upstream by service.py dropna(subset=['Close']).
  fix_ok=true — Prefer the 'log n_bad' option. The NaN-propagating variant blanks 21 RV rows (measured) instead of biasing 19, and analyze_forward_vol L169 v.dropna() still skips the gap positionally for the forward window, so it does not fully remove the 'enjambement'.
- [IMPACT & SEVERITY] refuted=true conf=0.7 sev=m
  reason: Behaviour confirmed (one close <= 0 removes returns t and t+1, the t-1->t+1 jump is lost, 19 RV values shift by up to 2.12 vol pts on real AAPL data, 2 dates vanish, no journal trace) but the condition cannot realistically reach analytics: both the Alpaca path (service.py L116) and the fallback (L156) dropna(subset=['Close']) before analytics, so NaN/missing closes never arrive, and a strictly non-positive close is not something Alpaca, Stooq or Yahoo emit for listed equities — 0 occurrences in 10 415 real Stooq rows (p4_real_cache_scan.py). The finding itself concedes 'source peu probable côté Alpaca'. With defaults SPY/QQQ/AAPL the probability is effectively zero and the effect, if it ever happened, is bounded (~2 vol pts on 19 of 504 points) and does not flip a regime classification. Robustness polish at most.
  corrected_evidence: p4_impact_math.py S2 on the latest real 2y AAPL window with one 0.0 close injected: current code rv len 519->517 (2 dates lost), 19 values changed, max |dRV| = 2.12 vol pts, lost real 2-day return -0.1055%. Finding's NaN-hole fix: 21 RV days become NaN for ONE bad row (then dropped by series_df.dropna). Filter-first alternative (px = px[px > 0]): 1 date lost, 20 values changed, max |dRV| = 1.89 vol pts, no hole, 2-day return kept. Upstream: Alpaca L116 and fallback L156 already dropna on Close, so only a strictly <= 0 close can reach analytics; 0 such rows in 10 415 cached real rows.
  fix_ok=false — The primary fix_sketch (keep NaN alignment + rolling(min_periods=window)) is disproportionate: one bad row blanks 21 RV days (measured), which series_df.dropna then removes, costing the forward regression 21 points and punching a hole in the percentile window — worse than today's bounded 2-pt drift, and it changes compute_log_returns' documented 'first row dropped' length contract that tests rely on. If anything is done, prefer the minimal and consistent approach: drop non-positive rows FIRST (px = px[px > 0]) so the t-1->t+1 return survives as a 2-day return (same treatment upstream already gives missing rows), and log n_bad = int((px <= 0).sum()) in the service when > 0. The 'au minimum logger' half of the sketch is fine on its own.


## m12 — iv-disabled-says-alpaca-inaccessible
- titre: IV désactivée par l'utilisateur -> warning « chaîne d'options Alpaca inaccessible »
- ancre: app/vue/tabs/tab_iv_dashboard.py:249
- sévérité finale (vote): m  (votes: m, m, m; finder: M)
- effort: 15 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_view_sinks_and_state.py (clé 'iv_disabled_warning')

### claim
Quand la case « IV courante via options Alpaca » est décochée, le service laisse `current_iv=None` et `iv_error=None` (service.py L564-574) ; la vue (L249-252) tombe alors sur le texte par défaut « IV courante indisponible — chaîne d'options Alpaca inaccessible. », un diagnostic de panne faux puisque rien n'a été interrogé.

### evidence (finder + orchestrateur)
p1_view_sinks_and_state.py §iv_disabled_warning : payload with_iv=False, iv_error=None rendu via AppTest -> warnings = ["IV courante indisponible — chaîne d'options Alpaca inaccessible."], 0 exception.

### impact
L'utilisateur qui a volontairement désactivé l'appel options croit à une panne Alpaca / clés invalides et part déboguer une connexion qui n'a pas été testée.

### fix_sketch
Dans `_render_metrics` : `if current_iv: ... elif result.get('iv_error'): st.warning('IV courante indisponible — ' + iv_error) else: st.caption('IV courante désactivée (case « IV courante via options Alpaca »).')`. Option : le service renvoie `iv_error='désactivée'` explicitement.

### panel
- [CODE READING] refuted=false conf=0.95 sev=m
  reason: Reading confirmed. service.py L560 `iv_error = None`, L564 `if include_current_iv:` has no else branch, so with the box unchecked the payload carries current_iv=None AND iv_error=None (and no `include_current_iv` flag at all, L591-609). View L249-252 then renders `"IV courante indisponible — " + str(iv_error or "chaîne d'options Alpaca inaccessible.")`. Nothing in the controller (iv_dashboard_controller.py L38-54, pure pass-through) or elsewhere in the view distinguishes 'disabled' from 'failed'. Probe A: service patched (fetch_current_atm_iv asserts it is never called) with include_current_iv=False -> current_iv=None, iv_error=None, payload_has_include_flag=False; _render_metrics on that payload emits exactly one st.warning: 'IV courante indisponible — chaîne d'options Alpaca inaccessible.' Not in the legacy Tkinter script (no IV toggle there), so a port-introduced wording defect. I propose downgrading M -> m: it is a misleading label only, no wrong number; the user unchecked the box themselves in the same form and the checkbox keeps its unchecked state across the rerun, so the signal to the user is mostly self-evident.
  corrected_evidence: p4 probe A: service(include_current_iv=False) -> current_iv=None, iv_error=None, payload has no include flag, series index tz-naive; _render_metrics -> warnings=['IV courante indisponible — chaîne d'options Alpaca inaccessible.'] (1 warning, 0 exceptions).
  fix_ok=true — The `elif result.get('iv_error')` / `else: caption('désactivée')` split is sound: every None-return path of fetch_current_atm_iv appends a non-empty log line first (service.py L356, L376, L385, L423-427) and L574 falls back to 'IV indisponible.', so iv_error is None only when the fetch was skipped. Slightly more robust (1 line): also emit `"include_current_iv": bool(include_current_iv)` in the payload and branch on it in the view.
- [IMPACT & SEVERITY (skeptic)] refuted=false conf=0.85 sev=m
  reason: Behaviour confirmed by code and probe: service.py L564-574 leaves iv_error=None only when include_current_iv=False; whenever IV was actually attempted iv_error is always non-empty (L574 falls back to 'IV indisponible.'), so the view's default string at tab_iv_dashboard.py L251 ('chaîne d'options Alpaca inaccessible.') is reached exclusively in the user-disabled case — it is a pure copy mismatch. Impact is bounded: no displayed number is affected, no trading decision depends on it, and the user triggered it by unchecking a box in the advanced expander seconds earlier (default is checked). A false 'inaccessible' diagnosis is misleading copy, not a silent failure -> severity M is over-calibrated; m (copy) is right.
  corrected_evidence: p4_g6_viewA_impact.py §A: include_iv=False -> {current_iv: None, iv_error: None}; include_iv=True with empty log -> iv_error='IV indisponible.'; with log -> iv_error=last log line. The L251 fallback text is therefore dead code except for the disabled case. Real cache/IVHistory is empty (no live run yet).
  fix_ok=true — Fix is proportionate: a 3-branch if/elif/else in _render_metrics (st.caption 'IV courante désactivée' when iv_error is None) relies on the service invariant proven in §A. Slightly more robust: have the service set iv_error='désactivée par l'utilisateur' or add an 'iv_requested' flag to the payload so the view does not infer intent from a None. No regression either way.
- [REPRODUCTION] refuted=false conf=0.92 sev=m
  reason: Reproduced end-to-end through the REAL service (not a hand-built payload): get_iv_dashboard_data('SPY', include_current_iv=False) with fetch_daily_closes patched offline -> fetch_current_atm_iv called 0 times (spy raises if called), payload current_iv=None, iv_error=None, and no IV line at all in payload['log'] (service.py L560-574 only set iv_error inside the `if include_current_iv` branch). Rendering that payload through the real view with AppTest gives at.warning == ['IV courante indisponible — chaîne d\'options Alpaca inaccessible.'] (tab_iv_dashboard.py L249-252 default string). Control: with include_current_iv=True and the chain returning (None, ['... 401 Unauthorized.']) the warning carries the real error. So the copy asserts a connectivity failure for a call that was never made. Severity argument for m rather than M: only reachable via an explicit opt-out in the 'Paramètres avancés' expander, it is a st.warning not an error, nothing else in the payload/charts is affected, and the journal does not contradict the user; it is a wrong message, not wrong data.
  corrected_evidence: service path: fetch_current_atm_iv_calls=0, current_iv=None, iv_error=None, log_mentions_iv=[]; view warnings=['IV courante indisponible — chaîne d\'options Alpaca inaccessible.'], exceptions=[]; control with real chain failure: warnings=['IV courante indisponible — Chaîne d\'options Alpaca indisponible : 401 Unauthorized.']
  fix_ok=true — Three-way branch in _render_metrics (current_iv / iv_error / disabled caption) is the minimal fix; needs a way to tell 'disabled' from 'no error recorded' — either the service sets iv_error to an explicit 'désactivée' string (single-line change at L560-574) or the view reads result['include_current_iv'] which the payload does not carry today (would need adding to the return dict at L589-609). Either is proportionate (~15 min).


## m13 — stale-result-under-error-and-params-drift
- titre: Résultat périmé rendu sous l'erreur ; paramètres du formulaire ≠ paramètres du résultat affiché
- ancre: app/vue/tabs/tab_iv_dashboard.py:513
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 15 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_view_sinks_and_state.py (clé 'stale_after_error')

### claim
Après un échec de re-soumission, `st.error` s'affiche (L168) mais `session_state['iv_dashboard_result']` n'est pas invalidé : les 3 graphes et métriques du symbole précédent restent rendus sous l'erreur (L513-545). La « Durée » (years) du résultat n'apparaît nulle part dans la vue (caption L523-526 : symbole, source, nb points, heure) alors que le selectbox peut afficher une autre valeur ; `generated_at` n'a pas de date (L165).

### evidence (finder + orchestrateur)
p1_view_sinks_and_state.py §stale_after_error : état seedé SPY/2 ans, contrôleur patché pour lever RuntimeError, text_input -> '<img…>', selectbox -> '5 ans', clic Analyser : errors=['Analyse impossible : Aucune donnee de prix disponible pour <IMG SRC=X ONERROR=ALERT(1)> …'], n_charts_after_error=3, caption='**SPY** · source série : alpaca · 540 points · généré à 12:00:00', selectbox_now='5 ans', result_years_in_state=2.0.

### impact
Écran mixte : erreur rouge pour XYZ + dashboard complet de SPY en dessous ; un utilisateur qui change « Durée » sans recliquer lit des régimes calculés sur 2 ans en croyant voir 5 ans (seul indice : le nombre de points).

### fix_sketch
L168 : `st.session_state.pop(_STATE_KEY, None)` avant `st.error` (ou afficher le résultat précédent grisé avec un `st.warning('Résultat précédent …')`). Caption L523 : ajouter `f"· {result.get('years'):g} an(s) · RV {rv_window} j"` et horodater avec la date (`'%Y-%m-%d %H:%M:%S'`).

### panel
- [CODE READING] refuted=false conf=0.9 sev=m
  reason: Reading confirmed. tab_iv_dashboard.py L166-168: the except branch only calls st.error; `st.session_state[_STATE_KEY]` is assigned at L166 only on success and never popped. render_tab L513 reads the key unconditionally and renders metrics + charts (L523-545). Caption L523-526 shows symbol/source/len(series)/generated_at; `years` is echoed nowhere in the view (rv_window is shown in the chip label L183, percentile_window at L189, forward_window in the chart title L387 — only Durée has no echo). L165 `strftime('%H:%M:%S')` has no date. Probe B (AppTest, controller patched to raise, state seeded SPY/2 ans, text_input->XYZ, selectbox->'5 ans', click): errors=['Analyse impossible : boom'], caption still '**SPY** · source série : alpaca · 252 points · généré à 12:00:00', selectbox_now='5 ans', state_years=2.0, result_still_rendered=True, years_shown_anywhere=False. The 'form shows different params than the result' part is inherent to any st.form-based tab (form widgets update locally without rerun); the genuine defect is the missing Durée echo plus the stale render under the error. Severity m is right.
  corrected_evidence: p4 probe B: after failed resubmit -> errors=['Analyse impossible : boom'], caption='**SPY** · source série : alpaca · 252 points · généré à 12:00:00', selectbox_now='5 ans', state_years=2.0, result_still_rendered=True, years_shown_anywhere=False, 0 exceptions.
  fix_ok=true — `st.session_state.pop(_STATE_KEY, None)` before st.error is correct and proportionate (or keep the result with an explicit st.warning 'résultat précédent'). In the caption fix, only `{years:g} an(s)` is actually missing — `RV {rv_window} j` is already shown in the chip label at L183, adding it is redundant but harmless. Adding the date to generated_at ('%Y-%m-%d %H:%M:%S') is fine.
- [IMPACT & SEVERITY (skeptic)] refuted=false conf=0.6 sev=m
  reason: Three sub-claims, uneven impact. (a) Stale dashboard under st.error: real (L166-168 never invalidates _STATE_KEY, probe B state_pop_on_error=False), but the stale block is headed by a bold caption with its own symbol (L523-526), so the screen is labelled, and keeping the last good result is the common Streamlit pattern — low impact. (b) Params drift: rv_window, percentile_window and forward_window ARE displayed (L184, L190, L386); only 'years' is missing, and the series chart x-axis is a dated DatetimeIndex (L277), so the actual window is visible on the first chart. Widget-vs-result drift is inherent to st.form (no rerun on widget change) and applies to every form tab. (c) generated_at without date (L165 '%H:%M:%S'): genuinely ambiguous for a nomadic user who reopens a day-old session — minor. Nothing here corrupts a number; m is correctly calibrated and the finding is not a duplicate.
  corrected_evidence: p4_g6_viewA_impact.py §B: rv_window=True, percentile_window=True, forward_window=True, years=False, symbol_in_caption=True, series_x_axis_is_dates=True, generated_at_fmt='%H:%M:%S', state_pop_on_error=False.
  fix_ok=true — Caption additions (years, RV window, full '%Y-%m-%d %H:%M:%S') are correct and harmless. Prefer the second variant (keep the previous result and show st.warning('Résultat précédent …') or a 'stale' caption) over st.session_state.pop(): popping on any exception discards a valid result on a transient Alpaca/network failure — a small regression for a nomadic user on flaky connectivity.
- [REPRODUCTION] refuted=false conf=0.9 sev=m
  reason: Reproduced with an independent harness driving the real controller+view through the form (model fetchers patched offline): (A) SPY/'2 ans' ok -> 3 plotly charts, state years=2.0. (B) submit XYZ/'5 ans' with the model raising RuntimeError -> at.error=['Analyse impossible : Aucune donnée de prix disponible pour XYZ ...'], n_plotly_charts=3, header caption still '**SPY** · source série : synthetic · 523 points · généré à HH:MM:SS', form shows XYZ / '5 ans', state still SPY / years=2.0. Line 167-168 catches and shows the error without touching session_state[_STATE_KEY] (L166 only written on success). (C) NEW, worse than the finder stated: one more plain rerun after the failed submit -> at.error=[] and the stale SPY dashboard is shown with NO hint while the form still reads XYZ / '5 ans' (st.error only lives in the submit run). Across all snapshots no rendered text (markdown/caption/metric/warning/error/info/code) contains ' an', ' ans', '1 an'..'5 ans' or 'Durée' -> the duration of the displayed result is never shown (view only uses `years` at L159). generated_at is '%H:%M:%S' (L165) — note this matches the legacy Tk script (option_trading_dashboard.py L187), so it is port parity, not a regression. Severity stays m: symbol IS shown in bold in the caption so the symbol mismatch is visible; the years mismatch is the only silent part and the point count (523 vs 1287) is a weak hint.
  corrected_evidence: A: errors=[], charts=3, caption '**SPY** · ... 523 points', state years=2.0. B (XYZ/5 ans, failure): errors=1, charts=3, caption unchanged (SPY, 523 points), form XYZ/'5 ans', state SPY/2.0, rendered_text_mentions_years=False. C (plain rerun): errors=0, charts=3, caption SPY, form XYZ/'5 ans'. D control (QQQ/5 ans ok): caption '**QQQ** · ... 1287 points', state years=5.0. generated_at='19:02:36'.
  fix_ok=true — st.session_state.pop(_STATE_KEY, None) before st.error at L168 is the minimal, correct fix and also removes variant C (nothing stale left to render). Adding `· {years:g} an(s) · RV {rv_window} j` to the caption at L523-526 is proportionate. Adding the date to generated_at is harmless but optional (legacy parity).


## m14 — iv-history-overlay-unbounded-x-range
- titre: Overlay « IV ATM (historique local) » non borné à la fenêtre affichée : écrase l'axe x et relie des observations éparses
- ancre: app/vue/tabs/tab_iv_dashboard.py:314
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 15 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_view_contrast_bias_ranges.py (section C)

### claim
`_render_series_chart` trace `iv_history` entier (L314-326) en `lines+markers` ; `load_iv_history` (service.py L486-497) ne filtre pas par date. Une observation IV ancienne (analyse faite il y a des mois/années) étend l'autoscale x bien avant le début de la série RV demandée et est reliée par un segment droit aux observations récentes.

### evidence (finder + orchestrateur)
p1_view_contrast_bias_ranges.py §C : série « 1 an » 2025-09-04 -> 2026-08-21 (351 j) + une observation iv_history du 2023-03-01 -> axe x de 1269 j, la série RV n'occupe plus que 28 % de l'axe. Rendu réel après usage longue durée : non mesuré (cache vide en revue).

### impact
Après quelques mois d'utilisation intermittente, le choix « 1 an » ne donne plus un graphe sur 1 an, et les segments dorés entre observations distantes suggèrent une trajectoire d'IV qui n'a jamais été observée.

### fix_sketch
Avant le tracé : `iv_history = iv_history[iv_history['date'] >= series.index[0]]` et `mode='markers'` (ou `connectgaps=False` après reindex sur les jours ouvrés) ; afficher le nombre d'observations hors fenêtre dans la caption L257.

### panel
- [CODE READING] refuted=false conf=0.85 sev=m
  reason: Reading confirmed. load_iv_history (service.py L486-497) returns the whole CSV with no date filter; get_iv_dashboard_data L576 passes it through unfiltered even though series_df was cut at `cutoff` (L533-535). View L313-326 plots iv_history['date']/['iv'] with mode='lines+markers'; neither _base_layout (L95-103) nor the chart sets xaxis.range, so Plotly autoranges over all traces. Probe C: 1-year synthetic series (2025-09-04 -> 2026-08-21, 351 d) + one IV observation 2023-03-01 -> xaxis_range_set=False, combined x-extent 1269 d, RV occupies 27.7 % of the axis; iv_mode='lines+markers'. Also note record_iv_observation writes one row per analysis day (L460-483), so the history is sparse by construction and the 'lines' segment joins non-adjacent days from the very first two uses, not only after months. Legacy script plotted a dense IB IV series (option_trading_dashboard.py L566), where lines were appropriate; the port changed the data density but kept the line mode. tz check for the proposed filter: series index is tz-naive (service.py L113 tz_convert(None).normalize(), L156 normalize()) and iv_history dates are naive (L492), so the comparison `iv_history['date'] >= series.index[0]` works (probe: fix_filter_comparison_ok=True). Severity m agreed: visual artifact growing with cache age, no wrong metric.
  corrected_evidence: p4 probe C: xaxis_range_set=False, xaxis_autorange=None, iv_mode='lines+markers', rv_start=2025-09-04, iv_min=2023-03-01, span_rv_days=351, span_all_days=1269, rv_share_of_axis_pct=27.7, fix filter comparison tz-safe=True.
  fix_ok=true — Filtering `iv_history[iv_history['date'] >= series.index[0]]` in the view and switching to mode='markers' is minimal and tz-safe (both sides naive). Skip the reindex/connectgaps variant — needless. If filtering in the view, keep the caption at L255-258 consistent (count in-window vs total).
- [IMPACT & SEVERITY (skeptic)] refuted=false conf=0.6 sev=m
  reason: Code confirmed: load_iv_history (service.py L486-497) returns the whole CSV and _render_series_chart L314-326 plots it unfiltered in lines+markers. But the impact timeline is much later than the finding implies: the view's minimum duration is '1 an' (L29) and the default is '2 ans', observations are keyed to analysis days <= today, so the x-axis can only be extended once the oldest observation is older than the chosen window — i.e. after >= 1 year (1 an) or >= 2 years (default) of accumulated use. The real cache is empty today. The sparse-segment half (straight gold line across a 30-45 d gap between intermittent analyses) is immediate, but the caption at L257-260 already says 'une par jour d'analyse', so the sparsity is surfaced. Latent, certain to appear with long-term use, 15-min fix -> keep as m, not higher.
  corrected_evidence: p4_g6_viewA_impact.py §C (observations every 30 d): years=1 -> x-axis unaffected up to 365 d of use, RV share 0.529 after 730 d, 0.348 after 1095 d; years=2 -> unaffected up to 730 d, RV share 0.695 after 1095 d. §D: cache/IVHistory contains 0 CSV files. The phase-1 '28 %' figure assumes a 2023 observation with a 1-year window, i.e. ~3.5 years of use.
  fix_ok=true — Filter `iv_history[iv_history['date'] >= series.index[0]]` is type-safe (date is datetime64 after load_iv_history L493; the `.empty` guard at L315 precedes it) and mode='markers' removes the fabricated trajectory; no regression. Counting out-of-window observations in the L257 caption is optional polish.
- [REPRODUCTION] refuted=false conf=0.7 sev=m
  reason: Mechanism confirmed with the REAL cache writer/reader/renderer: history files produced by record_iv_observation (dt.date.today patched to walk past days), read by load_iv_history (signature takes only `symbol`, no date filter, service.py L486-497), rendered by _render_series_chart with st.plotly_chart intercepted: overlay trace mode='lines+markers', connectgaps unset, fig.layout.xaxis.range=None, so plotly autoscale spans min(all traces) and the RV series no longer fills the axis. The finder's headline number reproduces exactly (1269-day axis, RV share 28.8%) BUT only with an observation dated 2023-03-01 — before this tab existed — which is contrived. Realistic magnitudes are much milder: '1 an' selected after 14 months of intermittent use -> 429-day axis, RV covers 85.1%, one golden segment spanning 380 days from an out-of-window point to today; default '2 ans' after 26 months -> 800-day axis, 91.2%, segment 751 days. The spanning straight segment is the genuinely misleading element (a trajectory never observed); the axis squeeze is cosmetic and slow-developing (needs usage older than the selected window; the tab is brand new so real impact today is nil and grows with time). In-window sparse use (monthly) gives 100% axis but still connects points 33 days apart — that part is a design choice of lines+markers, not a defect. Keep as m with corrected, realistic numbers.
  corrected_evidence: S0 control (5 recent obs): axis 365 d, RV share 100%. S1 finder-style (obs 2023-03-01 + 5 recent, '1 an'): axis 1269 d, RV share 28.8%, largest connected gap 1265 d — unrealistic input. S2 realistic ('1 an', weekly use 14..12.5 months ago then today): 8 of 9 obs before window, axis 429 d, RV share 85.1%, connected gap 380 d. S3 realistic (default '2 ans', use 26..24.5 months ago then today): axis 800 d, RV share 91.2%, gap 751 d. S4 in-window monthly use: axis 365 d, 100%, gap 33 d. overlay_mode='lines+markers', xaxis.range=None in all cases.
  fix_ok=true — Filtering `iv_history[iv_history['date'] >= series.index[0]]` before the trace at L314-326 is a correct one-liner and removes both the axis stretch and the spanning segment. Switching to mode='markers' is optional/taste (sparse daily observations are arguably clearer as markers); `connectgaps=False` only helps after a reindex on business days, which is more work than the issue warrants. Showing the out-of-window count in the caption at L257 is fine but optional.


## m15 — chart-title-overprinted-by-legend
- titre: Titre Plotly recouvert par la légende horizontale (3 graphiques)
- ancre: app/vue/tabs/tab_iv_dashboard.py:100
- sévérité finale (vote): m  (votes: M, m, m; finder: m)
- effort: 30 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_view_render_charts.py <out_dir> puis ouvrir les .html (plotly.js embarqué)

### claim
`_base_layout` place la légende en `orientation='h', y=1.02, yanchor='bottom'` avec `margin.t=48` et un titre en position auto (L97-104) ; les libellés longs (7 entrées sur le graphe série, 4 entrées « Régime vol haute (n=…) » sur les demi-colonnes) passent sur 2 lignes et recouvrent le titre, qui devient illisible.

### evidence (finder + orchestrateur)
Figures capturées depuis le code de la vue (p1_view_render_charts.py) et rendues localement (plotly.js embarqué, loopback) : série @900 px -> bbox titre y[23,43], légende y[18,66] (2 lignes), recouvrement vertical 20 px ; forward @560 px -> 21 px (recouvrement horizontal 402 px) ; diff @560 px -> 21 px. Captures : scratchpad/ivcharts/playwright-mcp-out/iv_series.png, iv_diff.png — le texte du titre est visiblement superposé aux entrées de légende. Rendu via le template Streamlit : non mesuré (géométrie titre/légende identique, template n'y touche pas).

### impact
Titres « Série de volatilité et bandes de régime », « Vol forward 30 j vs vol courante — y = …x + … » (qui porte l'équation de régression) et « Diff de vol … » illisibles aux largeurs de colonne réelles.

### fix_sketch
Dans `_base_layout` : `margin=dict(l=10, r=10, t=90, b=10)`, `title=dict(y=0.98, yanchor='top')`, `legend=dict(orientation='h', yanchor='top', y=-0.18, ...)` (légende sous l'axe x) ou réduire à 4 entrées en supprimant les traces fantômes et en annotant les hlines.

### panel
- [IMPACT & SEVERITY] refuted=false conf=0.65 sev=M
  reason: Survives and is stronger than claimed. (1) It is STRUCTURAL, not width-specific: my probe reproduces `_base_layout` (tab_iv_dashboard.py L97-104) with the geometry Streamlit's frontend forces (applyStreamlitTheme in static/js/index.CxIUUfab.js sets title xanchor='left', x=0, font 16px and wraps the title in <b>) and measures, at 1400 px with the legend on ONE row (7 or 5 entries), title bbox y[16,35] vs legend y[20,49] -> 15 px vertical / 283 px horizontal overlap. Cause: plotly centres the auto-positioned title in the user margin t=48 while the legend at y=1.02/yanchor=bottom auto-expands the top margin into the same band (plot top moves from 56 to 72 px). So every run, every width, all 3 charts. (2) The Streamlit template does not rescue it: streamlit_plotly_theme.py only sets colorway/colorscales; the JS theme moves the title to x=0 (same left edge as the legend), making the overlap with the first legend entries complete rather than partial. (3) Impact on displayed numbers: the forward-chart title carries the regression equation (L385-388) and the overlapped legend entries carry R^2 and n= (L366, L413/L422/L441) -> the tab's headline quantitative outputs are illegible on the default path (SPY/2 ans). They remain recoverable in the collapsed Journal expander, which is why this is M and not C. (4) Not a duplicate of anything surfaced to the user.
  corrected_evidence: Streamlit-like geometry (title left/bold/16px, legend h y=1.02): series@1400px 7 entries, legend 1 row: title y[16,35], legend y[20,49], overlap 15 px vertical / 283 px horizontal; series@1400px 5 entries (no IV layer): identical 15 px; series@900px 2 rows: 19 px; forward@560px 2 rows: 19 px / 402 px horizontal; forward@700px 1 row: 14 px / 425 px. Plot top measured at 72 px vs 56 px expected from t=48 -> legend automargin expansion, title not repositioned.
  fix_ok=true — Primary branch (legend below the x-axis, repo convention: calibration_diagnostics.py L131 y=-0.15, model_comparison.py L158 y=-0.2) is correct; but then margin t=90 + title y=0.98 is unnecessary (t=48 suffices once the legend leaves the top band) — keep the change to a single line in _base_layout. The alternative branch 'réduire à 4 entrées / supprimer les traces fantômes' does NOT fix it: measured 15 px overlap with 5 entries on one row. Regression risk to check: forward/diff charts have an x-axis title; at y=-0.18 the legend can collide with 'Vol courante' under Streamlit's xaxis automargin/standoff — use y~-0.25 or verify visually.
- [REPRODUCTION] refuted=false conf=0.95 sev=m
  reason: Reproduced independently: figures built through the view's own _render_*_chart functions, the Streamlit 1.51 'streamlit' plotly template re-applied as the frontend does (TF()/yF() in static/js/index.CxIUUfab.js: title 16px bold, x=0 xanchor=left, legend font 12px, axes automargin), rendered in headless Chrome from a local file (plotly.js embedded, no network) and SVG bboxes of .gtitle / g.legend measured. With margin.t=48 (L99), title.y auto (baseline at margin.t/2 = 24px) and legend y=1.02/yanchor=bottom (L100), the legend box [~21,50] always straddles the title box [18,35]. The overlap is NOT caused by 2-row wrapping as the finder claims: a single-row legend already covers 79-84% of the title height (forward chart at 560/586/671/863 px; series chart at 1189/1359/1743 px), 2 rows cover 100% (series @900, diff @560-671). Horizontal overlap 218-326 px because both title and legend are left-anchored under the Streamlit template. Screenshots confirm the title text struck through by legend entries on all three charts. Finder's plain-template numbers (20/21 px @900/560) match mine (21 px).
  corrected_evidence: Streamlit-template render, title bbox vs legend bbox (y px from top): series w=1189/1359/1743 -> rows=1, title y[18,35], legend y[21,50], 14px = 84% of title height covered, x-overlap 218px; series w=900 -> rows=2, legend y[18,66], 100% covered. forward w=560..863 -> rows=1, title y[18,35], legend y[22,51], 13px = 79% covered, x-overlap 326px (all widths). diff w=560/586/671 -> rows=2, legend y[18,66], 100% covered; diff w=863 -> rows=1, 79% covered. margin.t stays 48 in every case (legend does not push automargin). Title is drawn after the legend in the SVG (overprint, not hidden).
  fix_ok=true — margin.t~90 + title=dict(y=0.98, yanchor='top') (or legend below the x-axis with a larger margin.b) is correct and proportionate. The alternative 'réduire à 4 entrées' is NOT sufficient on its own: a one-row legend already covers ~80% of the title, so the geometry of _base_layout must change regardless of the number of entries.
- [CODE READING] refuted=false conf=0.9 sev=m
  reason: Reading is correct and the finder UNDER-stated it. (1) app/vue/tabs/tab_iv_dashboard.py L97-104: margin.t=48, legend orientation='h', yanchor='bottom', y=1.02, x=0; titles set as plain strings at L340, L384-389, L454 (title.y left to default 'auto'). (2) Installed plotly 5.22.0 bundle (package_data/plotly.min.js): main-title y is `"auto"===r.y ? n.t/2 : ...` i.e. the title is drawn at the vertical middle of the *computed* top margin — exactly the band the legend occupies (legend pushes margin.t via autoMargin and sits in it). (3) Streamlit 1.51.0 frontend (static/js/index.CxIUUfab.js, applyStreamlitTheme -> layout.template.layout) forces `title.xanchor='left', x=0`, wraps the title in <b>..</b>, and does NOT touch title.y / margin.t / legend position — so in the real app title and legend share the same left edge; overlap is worse than in the finder's plain-plotly render (centered title). A parallel-probe screenshot at 1359 px with a ONE-row legend (scratchpad/p4_overlap/series_1359_st.png) shows title bbox y 18-35 under legend bbox y 20.6-70, x-overlap 85-315 px: the overlap is not a 2-row-only phenomenon, it happens at every width. Severity m is right: the regression equation lost in the forward-chart title is also in the log expander (L469-471).
  corrected_evidence: Streamlit-rendered geometry (bold left-aligned title, 1-row legend, 1359 px): title [x12,y18,w303,h17], legend [x85,y20.6,w980,h49.6] -> overlap even with one legend row; at 671 px (diff chart) title [12,25.5,335,42.5] vs legend [95,17.9,670,65.9]. Streamlit template: title xanchor=left,x=0; margin override only {pad,r:0,l:0}.
  fix_ok=false — Direction OK but the sketch needs tuning: (a) legend below the x-axis at y=-0.18 with yanchor='top' collides with the x-axis title on the forward/diff charts (tick labels + Streamlit standoff xl + title ≈ 55 px vs 0.18*~220 px plot ≈ 40 px); use y≈-0.3 with an explicit margin.b≈80, or keep the legend on top and pin the title with title=dict(y=1, yanchor='top', pad_t=6) plus margin.t sized for 2 legend rows (≈100). (b) The alternative 'réduire à 4 entrées' does NOT fix anything: the overlap exists with a single legend row (measured above) because plotly centres the title in the same margin band the legend lives in.


## m16 — split-annotation-units-mismatch
- titre: Annotation « Split régimes (0.165) » en décimal sur un axe formaté en %
- ancre: app/vue/tabs/tab_iv_dashboard.py:451
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 10 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_view_render_charts.py

### claim
La vline du graphe diff est annotée `f"Split régimes ({intersection:.3f})"` (décimal) alors que `fig.update_xaxes(tickformat='.0%')` (L456) affiche l'axe en pourcentage ; même mélange dans le journal (L482-490, `.4f`/`.3f`) et le titre du graphe forward (L385-388, intercept en décimal).

### evidence (finder + orchestrateur)
Capture scratchpad/ivcharts/playwright-mcp-out/iv_diff.png : annotation « Split régimes (0.165) » posée sur une vline située entre les ticks « 15% » et « 20% ».

### impact
Lecture à double conversion pour le trader ; cohérence d'unités entre métriques (%) et annotations (décimal).

### fix_sketch
`annotation_text=f"Split régimes ({intersection:.1%})"` ; dans le titre forward afficher l'intercept en % (`{reg['intercept']:.1%}`) ou préciser « (vol en décimal) ».

### panel
- [IMPACT & SEVERITY] refuted=false conf=0.7 sev=m
  reason: Real but copy-level. Within the same diff chart the x-axis shows '10% 15% 20%' (L456 tickformat='.0%') while the vline annotation reads 'Split régimes (0.165)' (L451); the forward title shows 'y = 0.602x + 0.066' (L385-388) over %-formatted axes; the metrics panel is 100% in percent via _fmt_pct (L183-238). The mismatch is a port artefact: the legacy Tkinter script (option_trading_dashboard.py L523, L557, L597-617) used decimals everywhere, axes included; the port converted axes and metrics to % but not annotations/title/journal. Impact is limited: 0.165 vs 16.5% is a trivial conversion that vol traders read natively, no number is wrong, no trading decision is at material risk -> m is correctly calibrated, not M. Not a duplicate.
  corrected_evidence: Units audit of the tab: percent = chips/st.metric/range (L183-238 via _fmt_pct), all axis ticks (L342, L391-392, L456-457), hovertemplates (.2%); decimal = vline annotation (L451, .3f), forward title intercept (L388, .3f), Journal (L468-490, .4f/.3f). Legacy script: decimal everywhere (L523, L557, L599, L607, L615), so the inconsistency was introduced by the port.
  fix_ok=true — `{intersection:.1%}` for the annotation is correct and proportionate. For the title, only the intercept may be converted (slope is dimensionless and must stay '0.602'); 'y = 0.602x + 6.6%' is dimensionally right but slightly unusual — the alternative 'vol en décimal' caption is equally acceptable. Leave the Journal in decimal (it mirrors the legacy log and is a diagnostic block, not a headline number).
- [REPRODUCTION] refuted=false conf=0.95 sev=m
  reason: Reproduced from the figure objects built by the view on a synthetic 3-year series: diff chart layout.xaxis.tickformat='.0%' (L456), layout.yaxis.tickformat='.1%', hovertemplate 'courante %{x:.2%}', yet the vline annotation reads 'Split régimes (0.162)' (L451, decimal); screenshot shows it between the 16% and 18% ticks. Forward chart title 'Vol forward 30 j vs vol courante — y = 0.233x + 0.124' (L385-388) with both axes in '.0%'. Journal (L482-490): 'intersection avec y=x à vol = 0.1621', 'VOL HAUTE (vol > 0.162, ...)'. The legacy Tkinter script (option_trading_dashboard.py L556-557) used the same '.3f' label but its axes were also decimal (no PercentFormatter): the port changed only the axes, creating the mismatch. Impact is cosmetic/readability, severity m is right.
  corrected_evidence: intersection=0.162115; xaxis.tickformat='.0%'; annotations=['Split régimes (0.162)']; forward title 'y = 0.233x + 0.124' with xaxis.tickformat='.0%'; journal lines decimal (.4f/.3f). Legacy: percent formatter on axes = False, split label '.3f' = True (internally consistent in decimal).
  fix_ok=true — f"Split régimes ({intersection:.1%})" is the right one-liner. For the forward title prefer 'y = {slope:.3f}·x + {intercept:.1%}' (slope is unitless, intercept is a vol) rather than leaving a decimal; the journal can stay decimal if it is meant as raw regression output, but say so or convert it too for consistency.
- [CODE READING] refuted=false conf=0.95 sev=m
  reason: Reading confirmed line by line: L451 `annotation_text=f"Split régimes ({intersection:.3f})"` (decimal) vs L456 `fig.update_xaxes(tickformat=".0%")`; forward title L386-388 prints the intercept (a vol level) as `.3f` while both axes are `.0%` (L391-392); hovertemplates use `.2%` (L360, L412, L422) and metrics use `_fmt_pct` — the annotation/title are the only decimal-unit texts on %-formatted charts. Legacy reference (Downloads/option_trading_dashboard.py L557, L523, L599-615) used decimal on BOTH axes and labels (no PercentFormatter/tickformat anywhere), so the mismatch was introduced by the port when the axes were switched to %. The log lines L482-490 are decimal like the legacy log — parity, lower priority. Parallel-probe screenshot diff_671_st.png shows 'Split régimes (0.162)' sitting between the 16% and 18% ticks.
  corrected_evidence: Legacy script: no PercentFormatter/tickformat; annotation 'Regime Split (Vol={intersection_x:.3f})', title 'y={slope1:.3f}x+{intercept1:.3f}' — consistent decimal. Port: axes .0%/.1% (L391-392, L456-457), annotation .3f (L451), intercept .3f (L387).
  fix_ok=true — `{intersection:.1%}` for the annotation and `{reg['intercept']:.1%}` in the forward title are correct (slope must stay dimensionless `.3f`). Log lines may stay decimal for legacy parity or follow the same convention — 10 min either way.


## m17 — copy-en-dessous-hyphen
- titre: Libellé de régime « EN-DESSOUS DE LA MOYENNE » : trait d'union fautif
- ancre: app/model/iv_dashboard/analytics.py:124
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 5 min
- dimension finder: ?
- repro: grep -n 'EN-DESSOUS' app/model/iv_dashboard/analytics.py

### claim
La locution française s'écrit « en dessous » (sans trait d'union) ; seul « au-dessous » en prend un. Le libellé est affiché tel quel par le chip « Régime courant » (tab_iv_dashboard.py L195-199).

### evidence (finder + orchestrateur)
analytics.py L124 : `key, label = "below", "EN-DESSOUS DE LA MOYENNE"` ; L120 « AU-DESSUS DE LA MOYENNE » (correct). Rendu confirmé dans p1_view_sinks_and_state.py (chips extraits du markdown).

### impact
Coquille visible sur le chip principal dans ~20 % des états (0.2 < percentile ≤ 0.4).

### fix_sketch
`"EN DESSOUS DE LA MOYENNE"` (et adapter tout test qui compare la chaîne).

### panel
- [CODE READING] refuted=false conf=0.9 sev=m
  reason: analytics.py L124 reads exactly `key, label = "below", "EN-DESSOUS DE LA MOYENNE"` (verified by grep and by calling classify_regime(0.3) -> 'EN-DESSOUS DE LA MOYENNE'). Standard French orthography is 'en dessous' (no hyphen); the hyphen belongs to 'au-dessous' / 'ci-dessous' / 'là-dessous', which is why L120 'AU-DESSUS DE LA MOYENNE' is correct and L124 is not. The string is rendered verbatim by the 'Régime courant' chip (tab_iv_dashboard.py L195-199) for 0.2 < p <= 0.4; no other code path rewrites the label. Only the RV-side chip shows it (the IV side only uses signal_label, L240), which the finder's impact already implies.
  corrected_evidence: grep across app/ and tests/ finds the hyphenated string only at analytics.py L124; tests/test_iv_dashboard_analytics.py L82-97 asserts on `key`/`signal_key` and `label` truthiness only, never on the label text, and the integration driver does not compare labels either.
  fix_ok=true — One-token change is exactly proportionate. The clause 'adapter tout test qui compare la chaîne' is moot: no test compares the label string, so nothing else needs to move.
- [REPRODUCTION] refuted=false conf=0.95 sev=m
  reason: Confirmed: analytics.py L124 returns label 'EN-DESSOUS DE LA MOYENNE' for 0.2 < p <= 0.4 (measured p=0.21, 0.3, 0.4 -> that label; 0.2 -> VOL FAIBLE; 0.41 -> NORMALE); share of p in [0,1] at step 1e-3 hitting it = 20.0 %, matching the impact claim. tab_iv_dashboard.py L197 renders `str(regime.get('label', 'N/A'))` verbatim in the 'Régime courant' chip. French orthography: 'en dessous' is the only standard spelling (BDL/Larousse/Robert: no hyphen, unlike au-dessous / ci-dessous / là-dessous / par-dessous); the sibling 'AU-DESSUS DE LA MOYENNE' (L120) is correctly hyphenated, so the inconsistency is real. git grep -i 'EN-DESSOUS' over tracked files finds exactly this one occurrence; tests/test_iv_dashboard_analytics.py L88-94 asserts on keys ('below') not labels, so the fix breaks no test.
  corrected_evidence: Single occurrence (analytics.py L124); 20.0 % of percentile range (0.201..0.400); zero tests compare the label string (test_iv_dashboard_analytics.py checks key='below'), so 'adapter tout test' in the fix is a no-op.
  fix_ok=true — One-token change to 'EN DESSOUS DE LA MOYENNE'; no test update needed.
- [IMPACT & SEVERITY] refuted=false conf=0.95 sev=m
  reason: Confirmed at analytics.py L124 ('EN-DESSOUS DE LA MOYENNE'), rendered verbatim by the 'Regime courant' chip (tab L195-199). Standard French is 'en dessous' (no hyphen; only au-dessous / ci-dessous / la-dessous / par-dessous take one). Visible on the main chip for 0.2 < p <= 0.4, i.e. ~20% of states by construction of a percentile rank. Pure copy issue, no numeric or decision impact -> m per the rules (polish/copy); not negligible enough to refute since it sits on the headline regime chip.
  corrected_evidence: 
  fix_ok=true — One-string change; no test compares the label (tests/test_iv_dashboard_analytics.py:85-97 asserts keys and truthy label only; grep for 'DESSOUS' in tests/ and docs/ returns nothing). No regression risk.


## m18 — forbid-streamlit-script-trips-on-docstring
- titre: scripts/precommit_forbid_streamlit.py passe au rouge à cause du docstring de service.py
- ancre: app/model/iv_dashboard/service.py:15
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 10 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_arch_import_cycles.py + `python scripts/precommit_forbid_streamlit.py`

### claim
Le docstring « must not import Streamlit. » contient la sous-chaîne `import streamlit` (scan textuel lowercase) ; le script de gate `scripts/precommit_forbid_streamlit.py` s'arrête en erreur sur ce fichier alors qu'aucun import Streamlit réel n'existe.

### evidence (finder + orchestrateur)
`python scripts/precommit_forbid_streamlit.py` -> `[ERROR] Streamlit found in model: app\model\iv_dashboard\service.py` (exit 1). `git grep -il 'import streamlit' HEAD~1 -- app/model` -> aucun hit (rc=1) ; sur HEAD -> uniquement service.py. Sonde p1_arch_import_cycles.py : import de service/analytics/controller ne charge PAS streamlit dans sys.modules (False). Le script n'est ni dans .github/workflows/tests.yml ni dans les hooks git (post-checkout/post-commit = graphify), donc CI verte ; check_mvc_integrity.py (AST) est OK.

### impact
Faux positif sur un gate local du repo : quiconque lance le script pré-commit historique voit le modèle « violer » MVC ; bruit et perte de confiance dans le gate, pas d'impact sur les chiffres affichés.

### fix_sketch
Reformuler le docstring (ex. « ne doit pas dépendre de Streamlit ») OU durcir le script pour matcher `^\s*(import|from)\s+streamlit` par ligne de code (regex multiline) au lieu d'un `in text.lower()`.

### panel
- [CODE READING] refuted=false conf=0.95 sev=m
  reason: Reading is correct. app/model/iv_dashboard/service.py:15 contains the docstring text 'must not import Streamlit.'; scripts/precommit_forbid_streamlit.py (lines 6-10) does `text.lower()` then `'import streamlit' in text` — a substring test over the whole file including docstrings, so the docstring alone trips it. Measured: script rc=1 with '[ERROR] Streamlit found in model: app\model\iv_dashboard\service.py'; AST walk of service.py finds 0 real streamlit Import/ImportFrom nodes; `git grep -il 'import streamlit' HEAD~1 -- app/model` -> rc=1 (no hit before this commit), so the regression is introduced by this PR. Other model files phrase the same rule without the trigger substring (calibration/diagnostics.py:3 'No Streamlit imports allowed', volatility_models/__init__.py:4 'no Streamlit'). Scope check also holds: .github/workflows/tests.yml runs scripts/check_mvc_integrity.py (rc=0 here) and scan_secrets.py only, not the forbid script, so CI stays green — severity m is right.
  corrected_evidence: precommit_forbid_streamlit.py rc=1 on HEAD; textual hit lines in service.py = [15] only; AST streamlit imports = 0; check_mvc_integrity.py rc=0; forbid script not referenced in .github/workflows/tests.yml (only check_mvc_integrity.py and scan_secrets.py).
  fix_ok=true — Surgical fix for this PR is the docstring reword (e.g. 'no Streamlit dependency', matching calibration/diagnostics.py:3 wording). Hardening the script to a per-line `^\s*(import|from)\s+streamlit` regex is correct but is a pre-existing weakness of a script outside this PR's scope; fine as an optional follow-up.
- [IMPACT & SEVERITY] refuted=false conf=0.8 sev=m
  reason: Behaviour confirmed: `python scripts/precommit_forbid_streamlit.py` exits 1 on app/model/iv_dashboard/service.py:15 while the file has 0 real `import/from streamlit` statements (regex count) and the canonical AST gate `scripts/check_mvc_integrity.py` (the one wired in .github/workflows/tests.yml:26 and scripts/test_mvc_rules.py) exits 0. Impact is tightly bounded: the legacy script is referenced by NOTHING (not tests.yml, run_me.ps1, run_tests.ps1, pyproject, README; git hooks are only graphify post-commit/post-checkout), so CI stays green and no displayed number, data or tab behaviour is touched. It survives only as polish: a script whose sole job is to flag model->Streamlit imports now lies on this file, which is exactly the kind of noise that erodes trust in a gate. Calibrated at m; not C/M under any reading.
  corrected_evidence: legacy gate rc=1 ([ERROR] ... service.py); check_mvc_integrity rc=0; references to precommit_forbid_streamlit across CI/runners/README = none; real streamlit import statements in service.py = 0; pre-existing model docstrings on HEAD~1 (calibration/diagnostics.py:3 'No Streamlit imports allowed here') use a wording the substring scan does not trip on, so this PR is the first to break it.
  fix_ok=true — Take the first option only: reword the docstring (e.g. 'must not depend on Streamlit') — one line, zero regression risk, matches existing model docstrings. Do NOT harden the legacy script in this PR (surgical-change rule; the AST gate is already the single source of truth per docs/mvc_import_matrix.md).
- [REPRODUCTION] refuted=false conf=0.95 sev=m
  reason: Reproduced independently. Running `scripts/precommit_forbid_streamlit.py` from the repo root: rc=1, stdout `[ERROR] Streamlit found in model: app\model\iv_dashboard\service.py`. The script (lines 7-8) does `text.lower()` then `'import streamlit' in text` — a substring match over the whole file. My AST oracle over every app/model/**/*.py finds ZERO real `import streamlit`/`from streamlit` nodes; the only textual hit is service.py line 15 `'must not import Streamlit.'`, which ast.get_docstring confirms is inside the module docstring. Importing app.model.iv_dashboard.service leaves `'streamlit' in sys.modules` == False. Mitigating fact (keeps it minor): the script is wired nowhere — not in .github/workflows/tests.yml (which runs check_mvc_integrity.py, AST-based for app/model/iv_dashboard, and it passes), not in run_tests.ps1, no git hooks, no .pre-commit-config.yaml; `scripts/test_mvc_rules.py` delegates to check_mvc_integrity. So CI stays green; only a developer who runs the legacy script by hand sees the false red.
  corrected_evidence: gate rc=1 on service.py; AST scan of app/model: 0 real streamlit imports; text hit = docstring line 15 only; streamlit not in sys.modules after importing service; script referenced by 0 CI/hook/ps1 files (orphan gate).
  fix_ok=true — Either branch of the sketch works; rewording the docstring (e.g. 'must not depend on Streamlit') is the 1-line proportionate fix. Hardening the script with a per-line `^\s*(import|from)\s+streamlit` regex is also fine but touches a gate nobody runs; if kept, consider wiring it or deleting it rather than leaving an orphan.


## m19 — controller-bounds-duplicated-silent-clamp
- titre: Bornes de validation dupliquées vue/contrôleur, clamp silencieux
- ancre: app/controller/iv_dashboard_controller.py:50
- sévérité finale (vote): m  (votes: m, m/réfuté, m; finder: m)
- effort: 20 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_arch_controller_and_cache.py

### claim
Les bornes (rv 5..120, forward 5..90, percentile 60..756, years 0.5..10) sont codées en dur deux fois — widgets de la vue (tab_iv_dashboard.py:118-139) et `_clamp_*` du contrôleur — sans constante partagée ; le contrôleur tronque silencieusement sans log ni retour, donc un élargissement futur des widgets serait neutralisé sans signal.

### evidence (finder + orchestrateur)
Sonde p1_arch_controller_and_cache.py : `ctrl.get_iv_analysis('spy', years=0.1, rv_window=2, forward_window=500, percentile_window=30)` transmet au modèle `{'years': 0.5, 'rv_window': 5, 'forward_window': 90, 'percentile_window': 60}` sans aucun avertissement. Extraction statique : view bounds = [(RV,5,120),(forward,5,90),(percentile,60,756)], controller bounds = [(years,0.5,10.0),(rv_window,5,120),(forward_window,5,90),(percentile_window,60,756)] ; `_DURATION_CHOICES` vue = {1,2,3,5} ans vs clamp 0.5..10 côté contrôleur.

### impact
Aujourd'hui cohérent (aucun chiffre affiché faux : les labels « Percentile (X j) » lisent la valeur clampée renvoyée par le payload). Risque de dérive : si la vue autorise demain RV=150 j, l'analyse tournera à 120 j et seul le caption le trahira.

### fix_sketch
Exposer les bornes depuis le contrôleur (ex. `PARAM_BOUNDS = {'rv_window': (5, 120), ...}`) et faire consommer `min_value/max_value` des `st.number_input` depuis `ctrl.PARAM_BOUNDS` ; optionnellement logger quand un clamp modifie la valeur.

### panel
- [CODE READING] refuted=false conf=0.7 sev=m
  reason: Reading is correct but the impact is purely prospective. Controller app/controller/iv_dashboard_controller.py:50-53 hard-codes (0.5,10.0),(5,120),(5,90),(60,756); view app/vue/tabs/tab_iv_dashboard.py:120-121, 134, 138 hard-codes min/max (5,120),(5,90),(60,756) and _DURATION_CHOICES line 29 = {1,2,3,5} years. No shared constant exists (grep PARAM_BOUNDS/_BOUNDS in app/controller, app/vue -> only Heston DEFAULT_BOUNDS in calibration_controller, unrelated). _clamp_int/_clamp_float (lines 15-28) return max(lo,min(v,hi)) with no logging: measured with the service mocked, get_iv_analysis('spy', years=0.1, rv_window=2, forward_window=500, percentile_window=30) forwards {'years': 0.5, 'rv_window': 5, 'forward_window': 90, 'percentile_window': 60} and logger.warning is called 0 times. Mitigation the finder already acknowledged is real: the view labels read result.get('rv_window') / result.get('percentile_window') (tab_iv_dashboard.py:184, 190, 280) and analysis['forward_window'] (line 350), so any clamp is visible in the captions. Note this duplication is the repo-wide convention (30 `min_value=` widgets across app/vue/tabs, none sourced from a controller), so this is a maintainability nit, not a defect of this PR specifically; today the two bound sets are identical and every UI-reachable value passes the clamp unchanged.
  corrected_evidence: Clamp probe: kwargs to model = {'years': 0.5, 'rv_window': 5, 'forward_window': 90, 'percentile_window': 60}; logger.warning calls = 0. View bounds [(5,120),(5,90),(60,756)] vs controller [(0.5,10.0),(5,120),(5,90),(60,756)] — identical on the three shared params. Payload keys echoed by view labels: rv_window, percentile_window (+ analysis['forward_window'] at line 350).
  fix_ok=true — Exposing PARAM_BOUNDS from the controller and consuming it in the view respects the MVC import direction (view -> controller) and is proportionate. The 'log when clamp changes the value' part is optional; a logged warning is invisible in Streamlit anyway, the payload echo already surfaces it.
- [IMPACT & SEVERITY] refuted=true conf=0.75 sev=m
  reason: Impact is nil in realistic and even adversarial UI use. (1) The view widgets enforce the identical bounds (tab_iv_dashboard.py:118-139 -> (5,120),(5,90),(60,756); controller:50-53 -> same plus years 0.5..10 covering _DURATION_CHOICES {1,2,3,5}), so the controller clamp is unreachable from the UI — it is defense-in-depth for programmatic callers. (2) The clamp is not silent where it matters: service.py:593-596 echoes the clamped values into the payload and the view labels read them (tab:184 'RV courante (X j)', :190 'Percentile (X j)', :280 'Vol réalisée (X j)', :350 forward_window), so any future widget/controller drift would be visible on screen, not hidden. (3) Silent clamping in the controller is the established house convention (hedger_v2_controller.py:206, 224, 230 use the same max(lo, min(v, hi)) pattern without logging), so this PR introduces no new risk. The finding itself concedes 'aujourd'hui cohérent, aucun chiffre affiché faux'; what remains is a hypothetical future-drift comment, not a defect of the change under review.
  corrected_evidence: Probe: get_iv_analysis('spy', years=0.1, rv_window=2, forward_window=500, percentile_window=30) -> model receives {years:0.5, rv_window:5, forward_window:90, percentile_window:60} and the payload echoes exactly those clamped values (True); view widget bounds extracted = [(5,120),(5,90),(60,756)] == controller bounds; 3 label sites read result.get('rv_window'/'percentile_window'); hedger_v2_controller has 3 identical silent clamps.
  fix_ok=true — A shared PARAM_BOUNDS on the controller consumed by the view is MVC-legal (view->controller) and regression-free, but it is unrequested abstraction for 4 parameters with zero impact today; skip the logging part entirely. Not worth doing in this PR.
- [REPRODUCTION] refuted=false conf=0.85 sev=m
  reason: Finder's numbers confirmed exactly. With `_svc.get_iv_dashboard_data` monkeypatched to capture kwargs, `ctrl.get_iv_analysis('spy', years=0.1, rv_window=2, forward_window=500, percentile_window=30)` forwards `{'years': 0.5, 'rv_window': 5, 'forward_window': 90, 'percentile_window': 60}`; junk inputs ('abc', None, 'x') silently fall back to defaults (2.0, 20, 30). Zero warnings, zero log lines, no 'clamped'/'warnings' key in the payload, and the controller imports neither logging nor warnings. Static extraction: view bounds {rv 5..120, fwd 5..90, pct 60..756, years 1..5 via _DURATION_CHOICES} vs controller bounds {years 0.5..10, rv 5..120, fwd 5..90, pct 60..756}; the view reads no UPPER_CASE constant from ctrl (no shared source of truth). Skeptic caveat that bounds the severity: the clamp is NOT reachable through the real UI today — every view widget range is a subset of the controller range, and the labels 'RV courante (X j)' / 'Percentile (X j)' read result.get('rv_window'/'percentile_window') from the payload, so no displayed number is wrong. This is latent maintenance drift, not a defect; minor is correct and should not be promoted. Also note hedger_v2_controller already clamps the same way and no view in the repo consumes controller constants, so the duplication matches existing repo practice rather than being a new deviation.
  corrected_evidence: Forwarded kwargs for (0.1, 2, 500, 30) = (0.5, 5, 90, 60); for (99, 1e4, 0, -5) = (10.0, 120, 5, 60); junk -> defaults (2.0, 20, 30, 252). Warnings=0, log lines='' . UI reachability of clamp: False for all 4 params. Precedent: hedger_v2_controller also clamps; 0 views read ctrl.UPPER constants.
  fix_ok=true — Exposing `PARAM_BOUNDS` on the controller and having the view's number_input min/max read from it is MVC-legal (view -> controller) and small. It introduces a pattern with no precedent in the repo (no other view consumes controller constants), so it is optional hygiene rather than a required change; keep the defensive clamp either way. Logging on clamp is optional and should not raise.


## m20 — smoke-tab-count-and-inventory
- titre: Smoke edit: stale test name (ten vs 11), CONTROLLERS inventory not extended, and the hard count pin already collides with the sibling branch
- ancre: tests/smoke/test_offline_imports.py:101
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 15 min
- dimension finder: ? (fusion: smoke-controllers-list-not-updated)
- repro: ls 'C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/app/vue/tabs' | grep -c '^tab_'  (12) vs worktree (11) ; grep CONTROLLERS tests/smoke/test_offline_imports.py ; ls app/controller

### claim
`test_all_ten_tabs_present` now asserts 11; the explicit CONTROLLERS list (lines 40-48) was not extended with `iv_dashboard_controller` (it is only imported transitively through the tab); and pinning the count means every concurrently developed tab breaks the smoke gate on merge, which test_app_boot.py's docstring explicitly argues against ('The tab count is deliberately not pinned').

### evidence (finder + orchestrateur)
Worktree: 11 tab_*.py modules -> assertion holds (47 passed). Main checkout of the same repo (branch feat/rbergomi-hurst-pipeline) today lists 12 tab_ modules (tab_kalman_filters.py, tab_rough_vol.py, no tab_iv_dashboard.py) -> 13 after both branches merge, `== 11` goes red. grep 'iv_dashboard' tests/smoke/test_offline_imports.py -> only lines 99 and 102 (membership), not in CONTROLLERS. The assertion message `mods` does print the module list (helpful).
--- (doublon ? smoke-controllers-list-not-updated @ tests/smoke/test_offline_imports.py:38) ---
CONTROLLERS = [calibration, dashboard_v2, hedger_v2, options, portfolio_and_risk, trading, yieldcurve] (l.38-46) ; `ls app/controller/` = ces 7 + iv_dashboard_controller.py. Couverture transitive seulement : `test_tab_imports_offline[tab_iv_dashboard]` importe le contrôleur via la vue (23 passed). Sonde p1_arch_import_cycles.py : `controller_alone` importe OK sans clés ni réseau.

### impact
Predictable red smoke gate on the next merge for a reason unrelated to the change being merged; the controller inventory drifts from reality.

### fix_sketch
-def test_all_ten_tabs_present():
+def test_tab_inventory():
     mods = _tab_modules()
-    assert len(mods) == 11, mods
     assert "app.vue.tabs.tab_iv_dashboard" in mods, mods
     assert "app.vue.tabs.tab_bots" not in mods, mods
     assert "app.vue.tabs.tab_exercices" not in mods, mods
(if the count must stay pinned, keep it but update the name and the comment); and
 CONTROLLERS = [
     "calibration_controller",
     "dashboard_v2_controller",
     "hedger_v2_controller",
+    "iv_dashboard_controller",
     "options_controller", ...

### panel
- [CODE READING] refuted=false conf=0.95 sev=m
  reason: Reading is correct and the situation is worse than described. tests/smoke/test_offline_imports.py:96 is still named test_all_ten_tabs_present while line 101 asserts len(mods) == 11 (git show HEAD confirms only the count/comment changed, not the name). CONTROLLERS (lines 40-48) lists 7 controllers; ls app/controller has 8 — iv_dashboard_controller.py is absent from the list and only covered transitively via test_tab_imports_offline[tab_iv_dashboard]. On the merge-collision part the finder measured the wrong target: origin/main is not 'the sibling branch' but already contains PR #15 merged (4e1ff90, 9 commits ahead of merge-base ed657c6), its smoke test is test_all_twelve_tabs_present with `== 12`, and its CONTROLLERS was extended with kalman_controller. `git merge-tree --write-tree origin/main HEAD` reports CONFLICT (content) in tests/smoke/test_offline_imports.py (same hunk edited on both sides) and in app/vue/main_app.py. So the outcome is not a silently red gate but a guaranteed textual merge conflict that must be hand-resolved to 13 tabs and 9 controllers. main's own resolution pattern (rename test + extend CONTROLLERS) confirms the finder's reading of the file's convention. Test-only impact, severity m stands.
  corrected_evidence: origin/main = 4e1ff90 (PR #15 merged), 9 commits ahead of merge-base ed657c6; origin/main smoke test: def test_all_twelve_tabs_present, len(mods) == 12, CONTROLLERS includes kalman_controller. `git merge-tree --write-tree --name-only origin/main HEAD` -> CONFLICT (content) in tests/smoke/test_offline_imports.py and app/vue/main_app.py. Worktree: 11 tab_ modules, controllers on disk not in CONTROLLERS = ['iv_dashboard_controller'].
  fix_ok=true — Renaming the test and adding iv_dashboard_controller to CONTROLLERS is proportionate and matches what main did for kalman (test_all_twelve_tabs_present + kalman_controller). Dropping the `== N` pin is a judgement call: it contradicts this file's own comment 'Locks the tab-count invariant' while agreeing with tests/integration/test_app_boot.py:27-28; whichever is chosen, the merge conflict forces a manual resolution anyway (13 tabs, 9 controllers), so the rebase onto origin/main is the real fix.
- [IMPACT & SEVERITY] refuted=false conf=0.8 sev=m
  reason: Two of the three sub-claims are real but cosmetic; the third is overstated. (a) Stale name: worktree keeps `test_all_ten_tabs_present` while asserting 11 (tests/smoke/test_offline_imports.py:96,101); the sibling branch in the main checkout renamed its copy to `test_all_twelve_tabs_present` and asserts 12, so the repo convention is to rename on each bump — this PR did not. (b) CONTROLLERS inventory: worktree list (lines 40-48) lacks iv_dashboard_controller while the sibling branch did add kalman_controller to its list; coverage impact is nil because tab_iv_dashboard.py:22 imports the controller at module level, so test_tab_imports_offline[tab_iv_dashboard] already performs the offline import (20 passed), but the explicit parametrised id is missing. (c) Merge collision: real (union of tab modules = 13) but both branches edit the same overlapping hunk (lines 96-103), so the merge produces a textual conflict that a human resolves to 13 — not a silently-green merge that turns the gate red. Pinning the count is pre-existing design (commits 1de0298 -> 12, a23eed9 -> 10) that the sibling branch also kept; arguing against it via test_app_boot.py:28 is a repo-level design question, not a defect introduced here. Nothing touches displayed numbers, data or tab runtime: m.
  corrected_evidence: worktree tabs=11, main checkout tabs=12, union=13; worktree: 'def test_all_ten_tabs_present' + 'assert len(mods) == 11'; sibling: 'def test_all_twelve_tabs_present' + 'assert len(mods) == 12' and CONTROLLERS contains kalman_controller; worktree CONTROLLERS lacks iv_dashboard_controller; tab_iv_dashboard.py:22 `from app.controller import iv_dashboard_controller as ctrl` (module-level) -> transitive offline coverage exists; pytest tests/smoke/test_offline_imports.py -> 20 passed. The overlapping hunks mean the merge yields a conflict, not a clean merge followed by a red gate.
  fix_ok=true — Apply the two cheap parts: rename the test (e.g. test_tab_inventory or follow the sibling's numeric convention) and add 'iv_dashboard_controller' to CONTROLLERS — no regression possible. Dropping `len(mods) == N` is defensible (membership asserts keep the intent and it matches test_app_boot's philosophy) but it is a design change that the sibling branch did not make; decide it once at merge time rather than unilaterally in this PR, since the hunk will conflict and be hand-resolved anyway.
- [REPRODUCTION] refuted=false conf=0.9 sev=m
  reason: All three sub-claims reproduce from filesystem reads alone. (1) Stale name: worktree test is `test_all_ten_tabs_present` asserting `len(mods) == 11`. (2) Inventory drift: app/controller has 8 *_controller.py files, CONTROLLERS lists 7, missing = ['iv_dashboard_controller']; coverage is only transitive (importing tab_iv_dashboard puts app.controller.iv_dashboard_controller in sys.modules -> True, under the same offline fixture, so functionally covered but the explicit inventory is wrong). (3) Collision: the main checkout (feat/rbergomi-hurst-pipeline) has 12 tab modules (+tab_kalman_filters, +tab_rough_vol, no tab_iv_dashboard) and its own copy of the test is already renamed `test_all_twelve_tabs_present` with `== 12` AND its CONTROLLERS already includes `kalman_controller`; union after merge = 13, so neither branch's pin holds. Nuance the finder missed: because both branches edit the same assertion line, git will raise a textual conflict rather than silently auto-merging to a red gate — the resolver is forced to look, but must still hand-compute 13. The sibling branch followed the rename+extend-CONTROLLERS convention, which makes this branch's omissions a real inconsistency. test_app_boot.py line 27-28 does say the count is 'deliberately not pinned'. The pin itself pre-exists this diff (was `== 10`), so only the name/inventory drift is attributable to this change.
  corrected_evidence: Worktree: 11 tabs, pin 11, name 'ten'. Main checkout: 12 tabs, pin 12, name 'twelve', CONTROLLERS=8 incl. kalman_controller. Union = 13 (both pins fail). Worktree CONTROLLERS missing iv_dashboard_controller (8 on disk vs 7 listed). Finder's '47 passed' matches neither the file (20 passed) nor tests/smoke (132 passed, 1 skipped).
  fix_ok=true — Adding `iv_dashboard_controller` to CONTROLLERS is unambiguous and should be done. For the count: either drop the pin (as the sketch does, consistent with test_app_boot's stance) or keep it and rename to `test_all_eleven_tabs_present` to match the sibling branch's convention — note the sibling kept its pin at 12, so whichever choice is made, the merge will still conflict on these lines and needs a manual 13 or a pin-free inventory test on both sides.


## m21 — render-guard-happy-path-only
- titre: Render guard never renders the degraded states the tab shows without Alpaca keys (current_iv None, analysis None, empty iv_history, form submit)
- ancre: tests/integration/_iv_dashboard_render_driver.py:104
- sévérité finale (vote): m  (votes: M, m, m; finder: M)
- effort: 1 h
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_tests_view_branch_coverage.py and scripts/review_iv_dashboard/p1_tests_form_submit_feasibility.py

### claim
The seeded payload always has current_iv, a 5-row iv_history, an analysis with both regime regressions, and the second run has no payload at all; the warning branch (view:249), the analysis-missing info (view:540), the empty-history skips (view:256->exit, 315->327), the reg=None fallbacks (view:432, 466) and the whole form-submit path (view:150-168, incl. the st.error on controller failure) are not executed by any test.

### evidence (finder + orchestrateur)
coverage of app/vue/tabs/tab_iv_dashboard.py under the driver's two runs (p1_tests_view_branch_coverage.py): 85%, missing lines [78,80,82,89,90,92,151-157,165-168,249,432,466,540,550], missing arcs (213->249),(256->exit),(315->327),(328->340),(431->432),(465->466),(480->495),(533->540). Seeded run: charts=3 metrics=3 warnings=0 infos=0 errors=0; payload reg_high/reg_low both present (n_high=224, n_low=286), len(iv_history)=5. Feasibility measured (p1_tests_form_submit_feasibility.py): degraded payload renders charts=1 metrics=1 warnings=['IV courante indisponible — …'] infos=1 exc=[]; at.button[0].click().run() with controller raising -> errors=['Analyse impossible : Alpaca HS (test)']; with controller returning a payload -> charts=3, session_state symbol 'QQQ'; blank symbol -> warning 'Entre un symbole…'.

### impact
A user without APCA keys (the documented offline mode) hits exactly the untested branches; a crash there (e.g. float(None) on iv_minus_rv, formatting of a None reg) would reach the screen while the guard stays green. The controller-error path that the view relies on to show 'Analyse impossible' is also unverified.

### fix_sketch
In _iv_dashboard_render_driver.py add a third and fourth run and report them:

# Run 3: degraded payload (no IV, no analysis, empty history)
p = _build_payload()
p.update(current_iv=None, iv_error='chaîne Alpaca inaccessible (test)', iv_regime=None, iv_minus_rv=None,
         iv_vs_series_percentile=float('nan'), iv_history=pd.DataFrame(columns=['date','iv']),
         analysis=None, analysis_error='Série insuffisante (test)')
at3 = AppTest.from_function(_tab_script, default_timeout=120); at3.session_state['iv_dashboard_result'] = p; at3.run()
degraded = {'exceptions': [...], 'n_charts': len(at3.get('plotly_chart')), 'n_warnings': len(at3.warning), 'n_infos': len(at3.info)}

# Run 4: form submit with the controller failing (module object is shared with the AppTest script)
from app.controller import iv_dashboard_controller as ctrl
ctrl.get_iv_analysis = lambda symbol, **kw: (_ for _ in ()).throw(RuntimeError('Alpaca HS (test)'))
at4 = AppTest.from_function(_tab_script, default_timeout=120); at4.run(); at4.button[0].click().run()
submit_error = {'exceptions': [...], 'errors': [e.value for e in at4.error]}

print(RESULT_MARKER + json.dumps({'seeded': seeded, 'empty': empty, 'degraded': degraded, 'submit_error': submit_error}))

In test_iv_dashboard_render.py:
def test_tab_degraded_payload_renders_series_only(render_result):
    d = render_result['degraded']; assert not d['exceptions'] and d['n_charts'] == 1 and d['n_warnings'] == 1 and d['n_infos'] == 1
def test_form_submit_surfaces_controller_error(render_result):
    s = render_result['submit_error']; assert not s['exceptions'] and any('Analyse impossible' in e for e in s['errors'])
Also tighten the seeded test: assert render_result['seeded']['n_warnings'] == 0 (currently unobserved). Side note: the fixture strips only APCA_API_KEY_ID/APCA_API_SECRET_KEY/OPENAI_API_KEY (line 39) while test_app_boot.py and the smoke test strip six names incl. ALPACA_API_KEY/ALPACA_SECRET_KEY — harmless today because the driver never calls the service, but align the list if run 4 ever exercises it.

### panel
- [CODE READING] refuted=false conf=0.85 sev=M
  reason: Reading is correct. Driver main() (:104-124) renders only a full payload (current_iv set, 5-row iv_history, analysis with reg_high/reg_low) and an empty session; it never clicks the form. Re-ran the driver under coverage in a pristine subprocess: 85%, missing lines [78,80,82,89,90,92,151-157,165-168,249,432,466,540,550] and arcs (213->249),(256->exit),(315->327),(328->340),(431->432),(465->466),(480->495),(533->540) -- every line the finder cites is in the missing set, none is covered. One correction to the impact text: its concrete crash examples are already guarded (view:228 'if spread is not None', view:465 'if not reg'), and the degraded payload renders today with exc=[] (charts=1, metrics=1, warnings=1, infos=1). So the finding is a test-gap on the default offline path and on the form-submit path (view:150-168, the only way a payload ever enters the tab, 0% covered anywhere), not a latent crash.
  corrected_evidence: Driver under coverage: 213 stmts, 22 miss, 40 branches, 13 partial, 85%. Degraded payload (pristine process, keys stripped): exc=[] charts=1 metrics=1 warnings=['IV courante indisponible - ...'] infos=1. at.button has 1 item labelled '🔍 Analyser' (form_submit_button is exposed via at.button in streamlit 1.51.0); click with controller raising -> errors=['Analyse impossible : Alpaca HS (test)'], exc=[]; click with controller OK -> charts=3, session symbol 'QQQ'; blank symbol -> warning. Impact examples float(None)/None-reg are guarded at view:228 and view:465.
  fix_ok=true — Runs 3 and 4 are feasible exactly as sketched (verified in a pristine process). Patching ctrl.get_iv_analysis on the module object works because view:22 imports the module ('from app.controller import iv_dashboard_controller as ctrl') and calls ctrl.get_iv_analysis at :157. Minor: the reported exceptions list must be collected with str(e.value) as the driver already does; tighten seeded n_warnings==0 is cheap. Note the guard stays outside CI unless render-guard-not-in-ci-gate is also acted on.
- [REPRODUCTION] refuted=false conf=0.85 sev=m
  reason: Reproduced with my own coverage harness wrapping the real driver's _build_payload/_tab_script: the two existing runs leave exactly the finder's missing lines [78,80,82,89,90,92,151-157,165-168,249,432,466,540,550] and missing arcs (213->249),(256->exit),(315->327),(328->340),(431->432),(465->466),(480->495),(533->540); seeded payload has reg_high and reg_low present, len(iv_history)=5, warnings=0, infos=0. So the degraded-IV warning, analysis-missing info, empty-history skips, reg=None fallbacks and the entire form-submit path are indeed unexecuted by the guard. However, I also rendered every one of those paths: degraded payload (current_iv None, analysis None, empty history) -> exceptions=[], charts=1, metrics=1, warnings=['IV courante indisponible - ...'], infos=1; reg_high/reg_low None + iv_history None -> exceptions=[], charts=3; submit with controller raising -> errors=['Analyse impossible : Alpaca HS (test)'], exceptions=[]; submit with controller returning -> charts=3, session_state symbol 'QQQ', kwargs {forward_window, include_current_iv, percentile_window, rv_window, years}; blank symbol -> warning 'Entre un symbole...'. No latent crash exists in the untested branches today, so the finding is a regression-protection gap rather than a defect, and the guard it would strengthen is not even in the CI gate (see render-guard-not-in-ci-gate). Downgrading to minor on impact.
  corrected_evidence: Line-based statement coverage of tab_iv_dashboard.py under the two driver runs is 191/213 = 89.7% (the finder's 85% is the branch-weighted figure; the missing-line and missing-arc sets are identical). All four extra scenarios (degraded, reg=None, submit-error, submit-ok, blank) render with zero exceptions; after them only lines 78,80,82,89,90,92,550 remain unexecuted.
  fix_ok=true — Feasible as sketched: AppTest.from_function shares the process, so rebinding app.controller.iv_dashboard_controller.get_iv_analysis is seen by the view (verified); the tab has a single button ('Analyser') so at.button[0] is the submit. Suggest also asserting the successful-submit path (controller called with symbol 'QQQ' and the five kwargs, result lands in session_state) since that is the only way a real user gets a payload; it costs one more run. Keep the env-strip list aligned with test_app_boot.py if run 4 ever reaches the service.
- [IMPACT & SEVERITY] refuted=false conf=0.7 sev=m
  reason: The coverage gap is real, but the impact is over-stated: I rendered every degraded state the finding names in a pristine socket-blocked interpreter and none crashes today — (A) current_iv=None + analysis=None + empty iv_history: exc=[], charts=1, metrics=1, 1 warning, 1 info; (B) IV checkbox off: exc=[], charts=3; (C) reg_high=None: exc=[], charts=3; (D) iv_history=None + sparse current_iv: exc=[]; (E) form submit with controller raising: errors=['Analyse impossible : Alpaca HS (test)']; (F) submit OK -> charts=3, state symbol 'QQQ'; (G) blank symbol -> warning. So nothing displayed is wrong, nothing fails silently, no trading decision is affected: this is future-proofing of a view, i.e. robustness -> m by the panel's own rules. Two further reasons it is not M: the only defect that actually lives on the untested branch (view:249 copy when IV is unticked, finding iv-disabled-says-alpaca-inaccessible) would NOT be caught by the proposed count-based assertions (warnings==1 passes with the wrong text); and the render guard itself is outside the CI gate (render-guard-not-in-ci-gate), so until that is fixed the extra runs only protect local full runs.
  corrected_evidence: p4_g9_tests_render_degraded_probe.py (APCA keys stripped, sockets blocked): A.degraded {exc:[], charts:1, metrics:1, warnings:['IV courante indisponible — Snapshots filtrés indisponibles (Clés Alpaca absentes) ; fallback chaîne complète.'], infos:1}; B.iv_disabled {exc:[], charts:3, metrics:1, warnings:['IV courante indisponible — chaîne d\'options Alpaca inaccessible.']}; C.reg_high_none {exc:[], charts:3}; D.hist_none {exc:[], charts:3}; E.submit_controller_raises {exc:[], errors:['Analyse impossible : Alpaca HS (test)']}; F.submit_ok {charts:3, state_symbol:'QQQ'}; G.submit_blank {warnings:['Entre un symbole avant de lancer l\'analyse.']}. Each extra AppTest run costs 0.1-0.6 s.
  fix_ok=true — Correct and cheap (each added run is 0.1-0.6 s; at.button[0] is the form submit button and the controller module object is shared with the AppTest script — verified). Keep run 4 last or restore ctrl.get_iv_analysis afterwards, since the monkeypatch is process-global in the driver. If the degraded test is meant to protect the no-keys path, assert on the warning text prefix ('IV courante indisponible') rather than only n_warnings==1, otherwise it stays blind to the copy defect already found on that branch.


## m22 — render-guard-not-in-ci-gate
- titre: The render guard is `integration`-only, so the CI gate (-m 'unit or smoke') never runs it although it is offline and takes 1.4 s
- ancre: tests/integration/test_iv_dashboard_render.py:26
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 15 min
- dimension finder: ?
- repro: "C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.venv/Scripts/python.exe" -m pytest -m 'unit or smoke' --co -q | grep iv_dashboard_render

### claim
pytestmark = pytest.mark.integration excludes the three render tests from the only automated gate (.github/workflows/tests.yml:35 runs `-m "unit or smoke"`), so the view is protected only when a developer runs bare pytest locally; the test is socket-blocked and fast enough to qualify as smoke.

### evidence (finder + orchestrateur)
pytest -m 'unit or smoke' --co -q | grep iv_dashboard_render -> no match (exit 1); pytest -m integration --co -q -> 3 items. pytest tests/integration/test_iv_dashboard_render.py --durations=5 -> '3 passed in 1.44s', slowest = 1.42s setup (module-scoped subprocess). Direct driver run: exit=0, stdout = single result line, stderr = 2 'missing ScriptRunContext' warnings. View coverage under the CI selection: 13% (tab_iv_dashboard.py 213 stmts, 180 miss).

### impact
A rendering regression in the tab is not caught by CI — only by a local full run; the docstring's 'render guard' wording overstates the protection actually in place.

### fix_sketch
Either `pytestmark = [pytest.mark.integration, pytest.mark.smoke]` in tests/integration/test_iv_dashboard_render.py (the repo's marker policy allows several markers; the test is offline by construction) or widen the CI step to `-m "unit or smoke or integration"` once test_app_boot.py's runtime is acceptable. Note that requirements/test.txt already installs streamlit and plotly (via runtime.txt), so CI can run it.

### panel
- [CODE READING] refuted=false conf=0.8 sev=m
  reason: Reading is correct. tests/integration/test_iv_dashboard_render.py:26 sets pytestmark = pytest.mark.integration only; .github/workflows/tests.yml:35 runs -m 'unit or smoke'; PLAN.md:27,32 name that selection as the gate; run_tests.ps1 only iterates scripts/test_*.py (:64-66), so there is no second automated gate. Measured: -m 'unit or smoke' --co collects no iv_dashboard_render item, -m integration collects 3, -m smoke collects none from tests/integration. Context that bounds the severity: test_app_boot.py:92 is also integration-only, so the whole AppTest-subprocess family is outside CI by existing repo convention; this PR follows that convention rather than introducing the gap. Still factually a gap for the view.
  corrected_evidence: Render test on this machine: 3 passed in 2.60s (setup 2.56s), wall 3.4s -- not 1.44s. test_app_boot.py: 3 passed in 5.51s, wall 6.05s. requirements/test.txt -> runtime.txt pins streamlit==1.51.0 and plotly==5.22.0, so CI can run it. test_app_boot.py (analogous boot guard) is equally excluded from CI.
  fix_ok=true — Both options are correct and cheap; pytest accepts a list for pytestmark under --strict-markers and the 'smoke: offline import/boot guards' marker text fits. Recommend deciding it as a repo-level policy (apply the same treatment to test_app_boot.py, ~6s, or widen CI to integration) rather than marking only this file, otherwise the two offline guards diverge.
- [REPRODUCTION] refuted=false conf=0.9 sev=m
  reason: Reproduced: python -m pytest -m 'unit or smoke' --co -q collects 678/728 items (50 deselected), 20 of them iv_dashboard-related (analytics + controller + the offline import smoke) and none from tests/integration/test_iv_dashboard_render.py; -m integration --co lists its 3 items. .github/workflows/tests.yml runs exactly -m 'unit or smoke', so the render guard never runs in CI. The guard alone passes in 1.91 s (3 passed, 1.89 s module-scoped setup), offline by construction (driver blocks socket.connect). Mitigating context: this follows the repo convention - all five tests/integration/*.py files including test_app_boot.py are integration-only and equally outside the gate - so it is a policy gap, not an oversight specific to this tab; impact is limited to local-only protection of the view. Minor stands.
  corrected_evidence: CI selection: 678 collected, 0 from test_iv_dashboard_render.py (20 iv_dashboard items all from analytics/smoke). Guard runtime measured 1.91 s (finder: 1.44 s). 37 integration items across 5 files, all excluded from CI; requirements/test.txt -> runtime.txt pins streamlit==1.51.0 and plotly==5.22.0, so CI has the deps.
  fix_ok=true — pytestmark = [pytest.mark.integration, pytest.mark.smoke] is valid under --strict-markers (both registered in pyproject) and is the surgical option; widening the gate to integration pulls in test_app_boot.py (default_timeout=300 full-app boot) and should be a separate decision. Note the subprocess child is not coverage-tracked, so the fail_under floor is unaffected either way. Linux behaviour of AppTest+plotly in CI not verified here (Windows only).
- [IMPACT & SEVERITY] refuted=false conf=0.8 sev=m
  reason: Confirmed: `pytest -m 'unit or smoke' --co` selects 20 iv_dashboard items, none from test_iv_dashboard_render.py; `-m integration` selects the 3. Wall time of the guard is 2.0-3.6 s (module-scoped subprocess), versus 178 s for the whole CI selection, so cost is negligible. The proposed marker change works under --strict-markers: a copy with pytestmark=[integration, smoke] is selected by the CI expression and passes (3 passed in 2.3 s). Impact is real but small: CI is the merge gate (PLAN.md:27-32 defines it as `pytest -m 'unit or smoke'`), and the only test of the 550-line view is outside it — but the tab's import is covered by the smoke matrix (test_offline_imports), the repo's other integration tests (test_app_boot, 4.3 s) follow the same integration-only convention, so this is a consistent pre-existing policy rather than a regression introduced by the PR. m is calibrated.
  corrected_evidence: -m 'unit or smoke' --co: 20 iv_dashboard items, render guard absent; render guard alone: 3 passed in 2.02 s (wall 3.6 s); test_app_boot.py: 3 passed in 4.30 s; copy with pytestmark=[pytest.mark.integration, pytest.mark.smoke] run with -m 'unit or smoke': 3 passed in 2.30 s. Whole CI selection: 676 passed in 178 s.
  fix_ok=true — Prefer the dual marker over widening the CI expression: it matches the registered 'smoke: offline import/boot guards' definition (the guard is offline by construction and is a boot guard of one tab), costs ~3 s, and does not drag the other integration tests in. Widening to 'unit or smoke or integration' is also viable (test_app_boot is only 4-5 s) but is a repo-policy change out of scope for this PR. Note the child process is not under coverage, so the fail_under=12 floor is unaffected either way.


## m23 — weak-realized-vol-oracle
- titre: test_realized_vol_known_magnitude's expected value is the ddof=0 number and rel=0.05 accepts ddof=0/1, sqrt(256), sqrt(260) and simple returns alike; warm-up test asserts one NaN fewer than exist
- ancre: tests/test_iv_dashboard_analytics.py:45
- sévérité finale (vote): m  (votes: m, m, m; finder: m)
- effort: 15 min
- dimension finder: ?
- repro: scripts/review_iv_dashboard/p1_tests_analytics_assertion_strength.py

### claim
expected = 0.01*sqrt(252) is the population-std value while the implementation (analytics.py:49-55) documents ddof=1; the 5% tolerance hides that 2.6% discrepancy and would also let annualization or return-definition regressions through. test_realized_vol_warmup_is_nan (line 55) checks rv.iloc[:18] although 19 leading NaNs exist and its own comment says 'first (window - 1) return rows'.

### evidence (finder + orchestrateur)
p1_tests_analytics_assertion_strength.py: actual=0.162869 expected=0.158745 rel_err=2.5978%; variants passing rel=0.05: ddof=1*sqrt(252) (impl, 2.60%), ddof=0*sqrt(252) (0.00%), ddof=1*sqrt(256) (3.41%), ddof=1*sqrt(260) (4.21%), simple returns (2.60%); only sqrt(365) fails (23.5%). Warm-up: len(rv)=29, leading NaNs=19, test asserts the first 18 rows only.

### impact
The RV number shown as 'RV courante' could silently change by up to 5% (ddof or annualization convention drift) without any test going red; the warm-up test would not notice a window shortened by one day.

### fix_sketch
-    expected = 0.01 * np.sqrt(252)
-    assert rv.iloc[-1] == pytest.approx(expected, rel=0.05)
+    # alternating ±1% log returns: sample std over 20 = 0.01*sqrt(20/19) (ddof=1), annualized with sqrt(252)
+    expected = 0.01 * np.sqrt(20 / 19) * np.sqrt(252)
+    assert rv.iloc[-1] == pytest.approx(expected, rel=1e-9)

-    # first (window - 1) return rows have no full window yet
-    assert rv.iloc[: 20 - 2].isna().all()
+    # 29 return rows; the first (window - 1) = 19 have no full window, the 20th is the first value
+    assert rv.iloc[: 20 - 1].isna().all()
+    assert rv.iloc[20 - 1:].notna().all()

### panel
- [CODE READING] refuted=false conf=0.95 sev=m
  reason: Reading is correct. analytics.py:49-55 documents and uses pandas default ddof=1 (rolling().std()), while tests/test_iv_dashboard_analytics.py:45 sets expected = 0.01*sqrt(252), the ddof=0 value, and :46 accepts rel=0.05. Measured actual=0.162869 vs expected 0.158745 (rel 2.598%); ddof=1 closed form 0.01*sqrt(20/19)*sqrt(252) matches actual to 1.4e-14. Warm-up test :55 slices iloc[:18] while the series has 19 leading NaNs (len 29, first value at iloc[19]); the comment at :54 says '(window - 1)' = 19, contradicting the code's 20-2.
  corrected_evidence: Variants passing rel=0.05 against the test's expected: ddof=1*sqrt(252) 2.598%, ddof=0 0.000%, sqrt(256) 3.409%, sqrt(260) 4.214%, simple returns 2.600%, and ALSO window=15 3.28% / 19 2.60% / 21 2.35% / 25 1.98% (the test pins neither ddof, annualisation nor window); only sqrt(365) fails (23.5%). Warm-up: window=19 -> 18 leading NaNs and the current assert iloc[:18].isna().all() is True (regression undetected); window=18 -> False. Proposed oracle |actual/exp-1| = 1.38e-14 (passes rel=1e-9); proposed iloc[:19].isna().all() and iloc[19:].notna().all() both True.
  fix_ok=true — Fix is exact and proportionate: any 20-window of alternating +/-0.01 has mean 0, so sample var = 20e-4/19 and the closed form is exact; rel=1e-9 is safe (observed 1e-14, deterministic sum of 20 terms). The warm-up correction to window-1 = 19 plus the notna tail assert pins the window length to the day.
- [REPRODUCTION] refuted=false conf=0.95 sev=m
  reason: Reproduced exactly. Implementation (analytics.py:55, pandas rolling std ddof=1 * sqrt(252)) returns 0.162869 on the test's alternating +/-1% series while the test's expected value 0.01*sqrt(252) = 0.158745 is the ddof=0 number (rel err 2.598%). With rel=0.05 the assertion also accepts ddof=0 (0.00%), sqrt(256) (3.41%), sqrt(260) (4.21%), simple returns (2.60%), window 19/21/30/100 and ddof=0*sqrt(270) (3.51%); only sqrt(365) (23.5%) and window=10 (5.41%) fail. Warm-up: len(rv)=29, 19 leading NaNs, test asserts only iloc[:18]; a window=19 regression (18 leading NaNs) still passes the current test while window=18 and a min_periods=2 regression fail it. Finder's numbers are correct to the digit.
  corrected_evidence: actual=0.162869, expected_in_test=0.158745, rel_err=2.5978%; variants passing rel=0.05: ddof=0 sqrt252 (0.000%), ddof=1 sqrt240 (0.125%), window=100 (0.504%), ddof=0 sqrt260 (1.575%), window=30 (1.710%), window=21 (2.353%), impl (2.598%), window=19 (2.598%), simple returns (2.600%), ddof=1 sqrt256 (3.409%), ddof=0 sqrt270 (3.510%), ddof=1 sqrt260 (4.214%); failing: window=10 (5.409%), sqrt365 (23.477%). Warm-up: 19 leading NaNs of 29 rows, test checks 18; window=19 passes current test.
  fix_ok=true — Proposed oracle 0.01*sqrt(20/19)*sqrt(252) matches the implementation at rel 1.4e-14 (rel=1e-9 is safe) and rejects every variant above except window=19, whose sample variance coincides with window=20 on an alternating series (both 1e-4*20/19) - the proposed warm-up assertions (iloc[:19] NaN, iloc[19:] not NaN) catch that window=19 case, so the two fixes together are complete. Both edits are two-line test changes, proportionate.
- [IMPACT & SEVERITY] refuted=false conf=0.9 sev=m
  reason: All numbers confirmed independently: the test's expected 0.158745 is the ddof=0 value, the implementation returns 0.162869 (rel 2.598%), and rel=0.05 also accepts sqrt(256) (3.41%), sqrt(260) (4.21%), simple returns (2.60%) and a window off by one in either direction (19: 2.60%, 21: 2.35%); only sqrt(365) fails. Warm-up: 29 return rows, 19 leading NaNs, the test checks 18 — and a window shortened to 19 still passes the current assertion (18 leading NaNs). Impact on displayed numbers if a ddof/annualization drift slipped through: on a realistic series RV 15.17% -> 14.79%, IV-RV spread +4.83 -> +5.21 pts, RV percentile unchanged (rank is scale-invariant) but the IV-vs-RV percentile moves 0.794 -> 0.825, i.e. it can cross the 0.8 'high' threshold and flip the displayed Signal (IV). That is a plausible but hypothetical regression behind a test, so m is calibrated — not M, since nothing is wrong today.
  corrected_evidence: actual=0.162869, expected(test)=0.158745, rel_err=2.5978%; passes rel=0.05: ddof=0 (0.000%), sqrt(256) (3.409%), sqrt(260) (4.214%), simple returns (2.600%), window=19 (2.598%), window=21 (2.353%); fails: sqrt(365) (23.477%). Proposed exact oracle 0.01*sqrt(20/19)*sqrt(252)=0.162869014, rel_err 1.4e-14 (passes rel=1e-9). Warm-up: len=29, leading NaNs=19, test checks first 18; window=19 -> 18 NaNs and the current assertion still passes. ddof drift impact: IV-vs-RV percentile 0.794 -> 0.825 (crosses the 0.8 regime threshold).
  fix_ok=true — Both hunks are correct: the exact oracle is numerically achievable (rel 1.4e-14, so rel=1e-9 is safe) and the warm-up fix (first 19 NaN, 20th onward not NaN) is what catches the window off-by-one. Caveat worth a comment: with this alternating ±1% fixture, window=19 yields exactly the same sample std as window=20 (10 and 9 alternating terms give 0.01*sqrt(20/19) too), so the magnitude test alone still cannot see a 19-day window — the tightened warm-up assertion is what provides that protection, keep both.


# TUÉS PAR LE PANEL / L'ORCHESTRATEUR

## hardcoded-dark-palette-light-theme — Palette sombre codée en dur : illisible si l'utilisateur bascule le thème en clair
- ancre app/vue/tabs/tab_iv_dashboard.py:31
- claim: Les couleurs de chips, de traces et de lignes de régime sont des constantes pensées pour `#0E1117` (L31-53) ; `.streamlit/config.toml` force `base='dark'` mais le menu Réglages de Streamlit permet toujours de choisir « Light », où `_COL_NEUTRAL` (chip « NORMALE », ligne « Moyenne ») et `_COL_GOLD` (IV) deviennent invisibles.
- [IMPACT & SEVERITY] refuted=true: Negligible in realistic use and not tab-specific. The app is dark-only by design at the repo level: .streamlit/config.toml pins base='dark'/backgroundColor='#0E1117'; main_app.py L17 `alt.themes.enable('dark')`; theme_animated.css force-darkens body (L1-7), page header (L24-75), inputs/textarea/select (`background:#1c1c20 !important; color:#dedede !important`, L109-115), tabs (L121-140) and makes plotly SVG transparent (L153-156). A user who toggles Light in the Settings menu gets a broken hybrid on EVERY tab (white page with dark widgets and dark cards), so the IV tab's chip/trace palette is not the failure a user would notice or attribute to this tab. The same hex values are already used e
- [REPRODUCTION] refuted=true: The contrast arithmetic reproduces exactly (on #FFFFFF: #e5e7eb 1.24, #fbbf24 1.67, #34d399 1.92, #fb923c 2.26, #9ca3af 2.54, #60a5fa 2.54, #f87171 2.77; on #0E1117 all >= 6.83), but the premise 'le menu Réglages permet toujours de choisir Light' is false for the shipped Streamlit 1.51.0. In static/js/index.CAj-7vWz.js, processThemeInput() calls createCustomThemes(themeInput) then addThemes(N,{keepPresetThemes:!1}) — presets (Use system setting/Light/Dark) are dropped whenever config.toml has a [theme] section; with no [theme.light]/[theme.dark] sections (this repo's config) createCustomThemes returns a single 'Custom Theme', and the Settings dialog selectbox is rendered with disabled: activ
- [CODE READING] refuted=true: Premise false for the pinned Streamlit 1.51.0. .streamlit/config.toml defines a `[theme]` section (base='dark', backgroundColor='#0E1117') with no `[theme.light]`/`[theme.dark]` sub-sections. In the frontend bundle (static/js/index.CAj-7vWz.js): `createCustomThemes` then yields ONE theme named 'Custom Theme'; `processThemeInput` calls `addThemes(N,{keepPresetThemes:!1})` (System/Light/Dark presets are removed from availableThemes) and `setAndSendTheme(N[0])` when no cached preference; the Settings dialog selectbox is rendered with `disabled: activeTheme.name===CUSTOM_THEME_NAME`. So a user cannot 'choisir Light' from the Réglages menu; the only way to get a light background is a developer ed

## yahoo-period-string — Fallback period f'{years+1}y' produces '3y'/'4y'/'6y', which are not valid Yahoo chart 'range' values
- ancre app/model/iv_dashboard/service.py:152
- claim: fetch_ohlc_history passes the period verbatim as Yahoo's range (market_data.py L302); Yahoo accepts 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max, so for the default years=2 the Yahoo leg of the last-resort fallback is likely rejected (Stooq is tried first and is unaffected).
- tué par orchestrateur: Mesure live Yahoo v8 chart SPY : range=2y/3y/4y/5y/6y -> HTTP 200, error=null, 501/753/1003/1254/1506 barres. Prémisse fausse.
- [CODE READING] refuted=false: Code path verified exactly as claimed: service.py L152 period = f'{ceil(years)+1}y' -> fetch_ohlc_history (market_data.py L336) -> Stooq leg first (L346-352, served from an infinite-TTL disk cache in market_data/service.py L74-89 when present) -> _fetch_yahoo_ohlc L291-302 passes rng verbatim as params={'range': rng}. Nothing normalises the string; _period_to_days (L192-205) parses '3y' fine but is only used for the local date filter. Measured with requests.get patched: years 1/2/3/5 (the four UI choices, tab_iv_dashboard.py L29, default '2 ans') send range '2y'/'3y'/'4y'/'6y' — three of the four UI choices send a value outside Yahoo's documented set {1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max}.
- [REPRODUCTION] refuted=false: Code path reproduces deterministically offline: with both Alpaca feeds and Stooq failing, fetch_daily_closes sends Yahoo params {'range': '3y', 'interval': '1d'} for the tab default years=2.0 (service.py L152 -> market_data.py L302), and get_iv_dashboard_data raises 'Aucune donnée de prix disponible pour SPY' when Yahoo answers its result=null error envelope. Whether Yahoo's v8 chart endpoint actually rejects '3y' is NOT measurable here (no network); it is outside the yfinance-documented set {1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max}, so I cannot refute, but I also cannot confirm the external behaviour. Realistic trigger requires a triple failure (Alpaca default + IEX + Stooq), i.e. rare.
- [IMPACT & SEVERITY] refuted=true: Refuted empirically from the existing live-run artefacts (p4_yahoo_range_evidence.py). orch_live_alpaca.out.txt (written 2026-08-21 16:04:36 local) shows fetch_daily_closes(years=2) served '534 barres daily via fallback Stooq/Yahoo' with period='3y' (service.py L152: ceil(2)+1). In that same run Stooq was down/rate-limited: fetch_spot_price fell through to _fetch_stooq_spot and returned None ('Spot indisponible'), and no stooq_spy.us_* cache file was created in cache/OHLC (the directory mtime is 15:48:55, before the run, and it only contains the two AAPL files), whereas market_data/service.py L97-102 writes that file on every Stooq success. So the 534 bars (= the expected ~536 trading days o


# CLEAN CHECKS / NON COUVERT par dimension (Phase 1)

## §4.3 — Architecture / MVC
### clean_checks
- Gates repo : `scripts/check_mvc_integrity.py` -> `[MVC] OK — no integrity violations detected.` ; `scripts/scan_secrets.py` -> `[secret-scan] OK — no secrets found in tracked files.` ; `scripts/test_mvc_rules.py` passe.
- Vue : n'importe que `app.controller.iv_dashboard_controller` + `app.vue.components.page_utils.render_page_header` — motif identique aux 10 autres onglets (tab_dashboard_v2, tab_yieldcurve, tab_hedging_systems… importent tous page_utils). Pas d'import app.model / app.utils dans la vue.
- Contrôleur : 58 lignes, pas de logique métier (strip/upper + clamps + délégation) ; n'importe ni streamlit ni app.vue. Pas de `streamlit` dans app/model/iv_dashboard ni app/controller (grep + probe sys.modules = False).
- Cycles d'import : sonde p1_arch_import_cycles.py — service→logic, logic→service, market_data→service, analytics seul, controller seul : tous OK en sous-processus frais, sans clés, sockets bloqués ; l'import lazy de `app.model.options.logic` dans le fallback ne crée pas de cycle (logic n'importe pas iv_dashboard).
- Signatures inter-modules vérifiées : `implied_vol_call(price, S0, K, t, r, q)`, `fetch_ohlc_history(symbol, *, period, interval)`, `fetch_spot_price(symbol)`, `download_options_alpaca(symbol, *, feed, max_pages, ...)` correspondent aux appels de service.py.
- paths.py : `CACHE_IV_HISTORY_DIR` suit exactement le motif des 4 autres CACHE_* (déclaration sous CACHE_CSV_DIR + mkdir(parents=True, exist_ok=True) à l'import) ; `git check-ignore -v cache/IVHistory` -> `.gitignore:158:cache/` ; README liste le nouveau dossier dans la section « Cache layout » — exact.
- Câblage main_app : sonde p1_arch_tab_wiring.py — `autodiscover_tabs()` découvre `tab_iv_dashboard` par nom de fichier (pkgutil), `DEFAULT_LABEL_OVERRIDES['tab_iv_dashboard'] == TAB_LABEL == '🌡️ Vol Implicite'`, le label figure dans TAB_GROUPS['🧩 Modèles'], `ordered_tab_labels` le place en position 8/8, render_fn = tab_iv_dashboard.render_tab ; 11 modules découverts. `tests/integration/test_app_boot.py` + `tests/smoke/test_offline_imports.py` : 23 passed.
- Clés d'état : `iv_dashboard_result`, form `iv_dashboard_form`, plotly keys `iv_dash_series_chart|forward_chart|diff_chart` — aucune collision (grep 'iv_dash' hors change set = uniquement main_app override).
- Écriture disque : le modèle n'écrit que sous CACHE_IV_HISTORY_DIR (sonde : redirection vers tmp, fichiers écrits uniquement là ; upsert même jour -> 1 ligne). Aucun autre write dans service.py.
- Logging : `logging.warning(f"[iv-dashboard] ...")` sur le root logger = convention dominante de app/model (market_data.py:306/619/666, options/logic.py:74/120/925…, yieldcurve/builder.py:72) ; un seul module utilise getLogger(__name__). Cohérent.
- Env var `ALPACA_OPTION_DATA_FEED` lue via os.getenv : même motif que `ALPACA_STOCK_DATA_FEED` (market_data.py:147) et `ALPACA_OPTION_CHAIN_CACHE_TTL_SEC` ; non documentée dans .env.example, comme les deux autres.
- Dead code / lint : pyflakes 3.4.0 sur les 5 modules + 3 fichiers de tests -> 0 warning. L'alias `render()` (vue l.548) est une convention présente dans 9/11 onglets (utilisé par tab_trading pour ses sous-onglets). `sym_url` (service.py:283) est public sans underscore et absent de `__all__` — nommage seulement, utilisé en interne.
- Marqueurs pytest : analytics = unit, rendu = integration (même choix que test_app_boot) ; `--strict-markers` respecté ; aucun test du change set ne requiert le réseau (--disable-socket actif).
### out_of_scope_notes
- docs/architecture_mvc.md:66-72 énumère des contrôleurs disparus (portfolio_controller, buy_sell_controller, backtest_controller, hedger_controller, dashboard_controller) — doc déjà obsolète avant ce commit.
- scripts/precommit_forbid_streamlit.py n'est branché ni en CI ni en hook git (seul check_mvc_integrity.py l'est) — gate orphelin préexistant.
- app/model/options/logic.py:1280 construit aussi un nom de cache depuis le symbole non sanitisé (`options_alpaca_{sym}.csv`) — même faiblesse, préexistante.
- Les env vars non-secrètes de feed (ALPACA_STOCK_DATA_FEED, ALPACA_OPTION_CHAIN_CACHE_TTL_SEC) ne sont pas documentées dans .env.example — lacune préexistante que la nouvelle ALPACA_OPTION_DATA_FEED prolonge.
- Le sous-processus Python du repo sort en cp1252 sur Windows : les labels emoji font planter tout `print` sans PYTHONIOENCODING=utf-8 (affecte scripts/ et tests qui impriment des labels), hors change set.
### not_covered
- Exécution réelle de `streamlit run` avec la nouvelle tab (seulement AppTest via le test de rendu existant et le probe d'autodiscovery).
- Comportement de `get_iv_dashboard_data` bout-en-bout avec fetchers stubbés (non mesuré ici — c'est précisément la lacune de test signalée ; relève des dimensions data/correctness).
- Conformité du payload Alpaca v1beta1 (greeks/latestQuote) et de la pagination : hors périmètre architecture, aucun appel réseau effectué.
- Impact perf de l'import de `alpaca.data.*` (lazy dans `_fetch_closes_alpaca`) et de `app.model.options.logic` (lazy, ~1500 lignes) au premier clic — non chronométré.
### probe_scripts
- p1_arch_import_cycles.py
- p1_arch_tab_wiring.py
- p1_arch_controller_and_cache.py

## §4.1 Mathematics — app/model/iv_dashboard/analytics.py + call sites service.py::get_iv_dashboard_data
### clean_checks
- Forward construction (analytics.py L171) : rolling(fw, min_periods=1).mean().shift(-fw) — mesuré : aucune moyenne partielle n'entre dans l'échantillon de régression (la moyenne partielle n'existe qu'aux 29 premières positions AVANT le shift, qui tombent hors indice) ; ligne i = mean(v[i+1..i+fw]) exactement ; les fw dernières lignes sont NaN et supprimées (p1_math_percentile_forward.py §E).
- Look-ahead confiné : current_vol (service L541), current_percentile (L542-544), regime (L546), iv_* (L569-573) sont tous lus dans series_df avant la construction de `analysis` (L583) ; `analysis['df']` n'alimente que les deux graphes de régression et le journal (vue L346-456). Vérifié par lecture.
- Comparabilité des deux percentiles : rank(pct=True, method='average') incluant la valeur courante vs percentile_within mi-rang — écart exact = 1/(2n) = 0.198 pt pour n=252 (mesuré 49.21 % vs 49.01 %) ; plancher 1/n=0.4 % pour rank vs 0 % pour percentile_within ; fenêtre trailing tail(252) de series_df = mêmes 252 valeurs RV (courante incluse) que la fenêtre roulante dès que len(series_df) ≥ 252 (cas par défaut).
- Ex æquo : fenêtre entièrement constante -> rank(pct) 0.505 (n=100) / 0.502 (n=252), percentile_within 0.5 -> régime NORMALE/NEUTRE ; cohérent avec method='average'.
- Annualisation RV sqrt(252)/ddof=1 vs T=dte/365 côté IV : non-finding. Les deux sont des taux « par an » ; BS en années calendaires est la convention de place. Mesuré : même prix inversé avec T=30/365 vs T=21/252 -> 20.000 % vs 19.863 %, soit 0.137 pt (0.69 % relatif) ; sans effet quand les greeks Alpaca sont utilisés directement. Horizon RV 20 j de bourse ≈ 28 j calendaires vs IV ~30 j : cohérent.
- Seuils de régime et MIN_REGIME_POINTS : `> 10` strict (L205/L210) = legacy L487/L495 ; buckets >0.8/>0.6/>0.4/>0.2 et signaux >0.8/<0.2 identiques au legacy L398-422 ; low_mask = ~high_mask ≡ `<= intersection` legacy (pas de NaN après dropna).
- Stabilité de l'intersection sur séries réalistes : 200 séries RV OU-log-vol -> pente ∈ [0.262, 0.870], 0 intersection hors plage, 0 régime vide.
- linregress : 2 points -> slope 1 / p 0 (pas d'exception) ; 1 point, NaN, inf -> NaN + RuntimeWarning mais inatteignables (MIN_ANALYSIS_POINTS=30, >10 par régime, dropna L182, RV finie) ; y constant -> slope 0 / p 1 sans exception. Aucune fuite de NaN/RuntimeWarning observée dans le journal des régressions sur les cas testés.
- Comparaison formule à formule avec le legacy (option_trading_dashboard.py L326/331/446-500/622-631 ; dashboard.py identique) : déviations = (1) suppression de `close*sqrt(252)` [déclarée], (2) min_periods=60 [déclarée], (3) dropna(subset=...) au lieu de dropna() [déclarée], (4) `len<30 -> return` remplacé par raise ValueError [même seuil], (5) gardes isfinite sur slope/intersection [ajout bénin], (6) insight else-branch « pente ≥ 1 » au lieu de « slope > 1 » [plus exact]. Aucune déviation non déclarée sur les formules elles-mêmes.
- Série gappy (60 NaN au milieu) : analyze_forward_vol reconstruit sur offsets de lignes après dropna (n=310) — conforme au legacy, pas de crash.
- tests/test_iv_dashboard_analytics.py : 19 passed (venv du repo).
### out_of_scope_notes
- app/model/market_data/market_data.py::fetch_ohlc_history (fallback Stooq/Yahoo) : déduplication / tri des dates non vérifiés — pré-existant, hors change set.
- app/model/calibration/implied_vol.py::implied_vol_call : précision/bornes de l'inversion non auditées ici (utilisée par service L408) — pré-existant.
- Convention de parité put-call avec q=0 (service L407) relève de la dimension data/IV, pas de analytics.py ; non évaluée ici.
### not_covered
- Fréquence empirique IV > RV (VRP) sur SPY/QQQ réels — non mesurée (pas d'appel réseau) ; l'impact du finding iv-signal-vrp-bias repose sur un exemple synthétique et la littérature.
- Existence réelle de dates dupliquées ou de clôtures ≤ 0 dans les barres Alpaca ou le fallback Stooq/Yahoo — non mesurée.
- Rendu Streamlit effectif des cas dégénérés (n_low=0, analysis_error scipy) — vérifié par lecture de tab_iv_dashboard.py, pas via le driver d'intégration.
- Précision de l'inversion BS et choix r_annual côté service (dimension IV/data, pas analytics).
### probe_scripts
- p1_math_degenerate_regressions.py
- p1_math_regime_subset_degenerate.py
- p1_math_service_constant_series.py
- p1_math_percentile_forward.py
- p1_math_effective_window.py
- p1_math_rv_edges_epistemics.py

## §4.4 View — app/vue/tabs/tab_iv_dashboard.py
### clean_checks
- width="stretch" sur st.form_submit_button : signature installée streamlit 1.51.0 `.venv/.../streamlit/elements/form.py` L239-252 (`width: Width = "content"`, docstring accepte "content" | "stretch" | int) ; cohérent avec 20+ usages `width="stretch"` dans app/vue (router.py, surface_ui.py, tab_alpaca_*.py, tab_advanced_calibration.py) ; `use_container_width` n'est pas utilisé dans la vue sous revue.
- XSS / sinks HTML : symbole hostile '<img src=x onerror=alert(1)>' injecté dans le résultat et soumis via le formulaire -> n'atteint que st.caption (proto.allow_html=False), st.code (journal) et st.error (allow_html=False) ; aucun des 5 st.markdown(unsafe_allow_html=True) (`_chip`) ne contient de chaîne utilisateur — leurs entrées sont des libellés constants d'analytics, des ints (rv_window, percentile_window, dte) et des sorties de `_fmt_pct`. `render_page_header` : chaînes statiques via st.html.
- Traces fantômes à un point (L291-301) : mode='lines' + hoverinfo='skip' -> aucun segment dessiné, aucun hover ; valeurs q25/q75/mean toujours dans [min,max] de la série -> autoscale y inchangé (mesuré : range y [0.0865,0.2316] englobe données + étoile IV).
- Clés Plotly : 3 clés statiques distinctes ; render_tab est la seule entrée utilisée par main_app.autodiscover_tabs (render() n'est pris que si render_tab est absent) ; st.tabs rend chaque onglet une fois par run ; double `at.run()` -> 0 exception, ids `$$ID-…-iv_dash_*` uniques.
- Perf : aucun iterrows / boucle par point dans les 3 fonctions de graphe ; construction des 3 figures pour 1 260 points (5 ans) en ~125 ms (série 94.8 ms, forward+diff 29.4 ms).
- Contraste thème sombre par défaut (config.toml base='dark', bg #0E1117) : tous les tons ≥ 6.83:1 (#9ca3af 7.44, #60a5fa 7.43, #e5e7eb 15.26) — caption de chip 0.78rem en #9ca3af passe AA.
- Formulaire : bornes number_input (RV 5..120, forward 5..90, percentile 60..756) identiques aux clamps du contrôleur ; choix Durée {1,2,3,5} dans le clamp 0.5..10 ; aucun label de widget identique dans un autre onglet (pas de DuplicateWidgetID) ; st.columns + st.expander dans st.form rendus sans exception sous 1.51.
- États vides/partiels : analysis None -> st.info avec analysis_error ; percentile NaN -> métrique 'N/A' et chip 'N/A' gris ; iv_history vide -> overlay et caption sautés (L315, L256) ; current_iv absent avec iv_error renseigné -> warning avec le dernier message du log.
- Formatage des nombres : 0.18 -> '18.00%' ; percentile -> '95.6%' ; spread -> '+3.00 pts' (signe explicite) ; caption échéance 'Échéance 2026-09-20 (30 j) · 6 contrats · méthode : greeks Alpaca · flux : indicative'. Libellé 'Percentile IV vs série RV' + aide explicite : honnête.
- Intersection/vline hors plage : sur 5 séries GBM lissées (rv_window=120, forward_window=5, pente 0.88-0.97) l'intersection reste dans la plage des données (inside=True) — pas de cas dégénéré observé pour la vline côté vue.
### out_of_scope_notes
- La palette sombre codée en dur est partagée par les autres onglets (commentaire L31 « aligned with the other tabs ») — le défaut thème clair est un pattern pré-existant de l'app, pas introduit ici.
- analytics.classify_regime applique les seuils legacy à n'importe quel percentile : le biais IV-vs-RV (finding M) a sa racine dans service.py L569-571 (dimension modèle).
- service.load_iv_history ne filtre pas par date et record_iv_observation ne déduplique que sur la date du jour : la croissance du CSV est la cause côté modèle du finding overlay.
- tests/integration/_iv_dashboard_render_driver.py : l'assertion `n_metrics >= 3` est lâche (3 st.metric attendus exactement) et ne teste ni l'état « IV désactivée » ni la re-soumission en erreur.
- Les hovertemplates `%{x|%d %b %Y}` affichent les mois en anglais (locale Plotly par défaut) dans une UI française — cosmétique, non retenu.
- Outillage : un dossier `.playwright-mcp/` a été créé dans la worktree par le MCP Playwright pendant la capture ; déplacé vers le scratchpad (`scratchpad/ivcharts/playwright-mcp-out/`), `git status` ne montre aucun fichier suivi modifié.
### not_covered
- Rendu à travers le template Plotly « streamlit » (st.plotly_chart theme='streamlit') : les géométries titre/légende ont été mesurées sur un rendu plotly.js standalone (template plotly_dark), pas dans le frontend Streamlit.
- Bascule effective vers le thème clair via le menu Réglages de Streamlit 1.51 avec un thème custom en config : non vérifiée dans le frontend.
- Niveau réel de la prime de risque de vol (IV ATM − RV20) sur SPY : non mesuré (aucun appel réseau autorisé) ; le biais du chip est démontré sur séries synthétiques et sur le payload du driver de test.
- Comportement sous charge réelle (latence de l'appel Alpaca bloquant le rendu de tous les onglets pendant le spinner) : non mesuré.
### probe_scripts
- p1_view_sinks_and_state.py
- p1_view_contrast_bias_ranges.py
- p1_view_render_charts.py

## §4.5 Tests — new tests reviewed as code (tests/test_iv_dashboard_analytics.py, tests/integration/test_iv_dashboard_render.py + _iv_dashboard_render_driver.py, tests/smoke/test_offline_imports.py edit)
### clean_checks
- Marker policy: both new test files carry pytestmark (unit / integration), conftest's every-test-needs-a-known-marker hook and --strict-markers are satisfied; the driver file is not collected (python_files = test_*.py); 47 tests (bridge helpers + analytics + smoke) green in 2.1 s, render test 3 passed in 1.44 s.
- Subprocess rationale is accurate, not overstated: measured in one process — AppTest on a trivial script before importing controller_bridge renders 1 metric/1 info/1 markdown; after `import app.vue.components.options.controller_bridge` the same AppTest raises "'types.SimpleNamespace' object does not support the context manager protocol" on `with c1:` and renders 0 elements (st.columns replaced by a lambda, st.session_state by FakeState, st._codex_fake_streamlit=True, guard at controller_bridge.py:118 only skips when a ScriptRunContext exists).
- Driver vs test_app_boot.py: identical fixture pattern (env strip, PYTHONIOENCODING=utf-8, subprocess, marker parse, timeout=600) — ~35 duplicated lines; test_app_boot inlines the child as a string, the new test uses a file. Acceptable duplication, noted not reported.
- Stdout-marker protocol (parser copied verbatim, p1_tests_driver_protocol.py): warnings before the marker, marker printed twice (first wins), marker string inside a JSON value, newline inside an exception text (json-escaped), CRLF — all parsed correctly; a marker not at column 0 yields the 'no result line' assertion whose message embeds full stdout+stderr (diagnosable). Driver stdout is in fact a single line.
- Encoding on Windows: PYTHONIOENCODING=utf-8 is both set by the fixture and necessary — a child printing the tab emoji without it dies with UnicodeEncodeError (rc=1, stdout=''); with it rc=0. json.dumps' ensure_ascii keeps the result line pure ASCII anyway; parent decodes utf-8 with errors='replace'.
- timeout=600 is coherent with two AppTest runs at default_timeout=120 plus imports; measured wall time 1.4 s. Return code not asserted: measured that a child printing the marker then exiting 3 would pass — harmless because the print is the last statement of main(); asserting rc==0 could add false failures from interpreter-shutdown noise.
- Socket block is installed before any numpy/pandas/streamlit/app import (driver lines 22-27 vs 29-31/35/99); class-level socket.socket.connect patch also covers ssl.SSLSocket, urllib3 and requests; getaddrinfo is not blocked but that is DNS only.
- Seeded payload does exercise analysis-present, current_iv-present, the iv_history overlay (5 rows) and both regime regressions (n_high=224, n_low=286); seeded run shows 0 exceptions / 0 warnings / 0 errors. n_charts == 3 and n_metrics >= 3 are discriminating: a degraded payload renders 1 chart and 1 metric.
- Analytics tests: test_analyze_forward_vol_mean_reverting_series is deterministic (1 distinct (slope1, slope2) pair over 20 runs: 0.2016 / -0.7984, margin to 1.0 = 0.80; no seed in 0..49 flips slope1 >= 1); the 'mean' insight assertion only matches the mean-reversion branches (analytics.py:216-223); test_percentile_within mid value is exactly 0.5; insufficient-data test raises on the length check before linregress.
- Controller test patches ctrl._svc.get_iv_dashboard_data correctly (call-time lookup through the module); controller propagates service exceptions unchanged (measured RuntimeError re-raised) and the view catches them at tab_iv_dashboard.py:167 -> st.error (measured via AppTest click).
- Cache redirection for future tests: monkeypatch service.CACHE_IV_HISTORY_DIR (works), not app.utils.paths.CACHE_IV_HISTORY_DIR (does not, import-time binding at service.py:33); record/load round trip, same-day upsert, older-row preservation, corrupt-file and missing-file paths all behave as documented (real cache/IVHistory untouched by the probes).
- Smoke count 11 matches the 11 tab_*.py files of the worktree and the assertion message prints the module list.
### out_of_scope_notes
- tests/integration/test_app_boot.py already embodies the same subprocess/marker fixture; a shared helper (tests/integration/_subprocess_guard.py) would remove ~35 duplicated lines — pre-existing pattern.
- CI gate excluding every `integration` test is a pre-existing policy (.github/workflows/tests.yml:35); test_app_boot.py is in the same situation as the new render guard.
- analytics.analyze_forward_vol on a constant vol series with >= 30 points raises scipy's ValueError ('all x values are identical'); service.py:586 catches ValueError so the tab degrades — analytics/service dimension, no test either way.
- record_iv_observation with iv=None logs a warning whose accented text renders as mojibake on a cp1252 console ('�criture du cache') — logging encoding, cosmetic.
- fetch_current_atm_iv calls fetch_spot_price outside any try (service.py:352); market_data.fetch_spot_price is itself defensive (returns None on failure), so no crash path was found — service dimension.
### not_covered
- Full test-suite run and cross-test ordering effects (only the new files + bridge helpers + smoke were run; no pytest-randomly/xdist available).
- Encoding behaviour measured on Windows only; Linux CI (utf-8 locale) not exercised.
- test_app_boot.py runtime not measured (relevant only to the 'widen CI to integration' option).
- _fetch_closes_alpaca's alpaca-py client path (service.py:58-120: default feed -> IEX retry, DataFrame shaping) was only bypassed by monkeypatching the wrapper, not unit-tested with a fake StockHistoricalDataClient.
- No live Alpaca/Stooq/Yahoo call was made (hard rule); the fixture shapes used in the sketches follow the code's own field names, not a recorded cassette.
### probe_scripts
- p1_tests_bridge_stub_apptest.py
- p1_tests_view_branch_coverage.py
- p1_tests_analytics_assertion_strength.py
- p1_tests_driver_protocol.py
- p1_tests_cache_dir_patch_target.py
- p1_tests_form_submit_feasibility.py
- p1_tests_service_sketches.py

## §4.2 Alpaca plumbing (app/model/iv_dashboard/service.py)
### clean_checks
- _decode_opra right-anchored slicing: 19-case attack set (SPXW, XSP, BRKB, AAPL1 adjusted root, 5-digit strike 12345.5, mini root, lowercase, no root, polygon prefix, garbage/None/int/trailing space/month 13/7-digit strike) — all valid decode correctly, all garbage returns None; only a non-C/P letter defaults to 'put' (Alpaca only emits C/P).
- alpaca-py 0.12.0 StockBarsRequest accepts feed (DataFeed) and adjustment (Adjustment) fields; DataFeed('iex') and Adjustment.SPLIT construct; tz-aware end is converted to naive UTC and serialized RFC3339; feed fallback to a raw string only triggers for non-enum names (never with 'iex'); MultiIndex bars.df path parsed correctly (mocked BarSet).
- Split handling on the primary path: adjustment=split is actually sent (measured in the request object); a fake 4:1 unadjusted split would inflate RV20 from 0.115 to 4.963 (x43) — only relevant if the Stooq/Yahoo fallback were unadjusted (non mesuré).
- Secrets: keys only via get_secret (L44-45); headers dict never logged; requests.HTTPError text contains the URL with query params (feed/limit/dates, no key); alpaca APIError str is the response body only; no print/logging of headers anywhere in the module.
- download_options_alpaca(sym, feed=..., max_pages=...) signature exists (logic.py L1252-1256) — no TypeError swallowed; fetch_ohlc_history(sym, period=, interval=) signature matches (market_data.py L336).
- fetch_spot_price None/invalid path handled (L355-357) — returns None with an explicit log line.
- Pagination is bounded by _SNAPSHOT_MAX_PAGES (no infinite loop on a repeated next_page_token); filtered-snapshot call has no retry/backoff/403 special-casing (accepted given the fallback), and a filter ignored by the server fails LOUDLY (3000 nearest-expiry contracts -> 'Aucun contrat entre 15 et 60 jours'), not with a wrong number (mock scenario B).
- Alpaca bar timestamps (04:00Z/05:00Z) -> tz_convert(None).normalize() give the correct ET session date; cutoff on local date only affects the lower bound by ≤1 day.
- IV validity filter 0<iv<5 consistent with implied_vol_call bounds (vol_max=5, NaN outside arbitrage bounds); best_expiry tie (28 vs 32 DTE) deterministically picks the earlier expiry.
- T=max(dte,1)/365 intraday error: +7 bp at 6.5 h remaining (below the 20 bp threshold); band-median vs true ATM with a linear −0.5 pt/1% skew on symmetric $1 strikes: +0 bp direct, −1 bp inverted (±5%), +11 bp with smile convexity.
- record_iv_observation: DataFrame.get('date', default) returns the column Series as intended; reload dtype datetime64[ns]; same-day upsert keeps exactly one row; 8 concurrent same-date writers produced 1 row (race not reproduced — non mesuré under real multi-process Streamlit).
### out_of_scope_notes
- app/model/options/logic.py::download_options_alpaca never sends a 'limit' query param (server default 100/page) — pre-existing, it is what makes the new fallback's max_pages=3 cover only ~300 contracts.
- app/model/options/logic.py::download_options_alpaca returns a cached options_alpaca_{SYM}.csv of any age when the download fails — pre-existing behaviour that the new fallback inherits (see finding stale-chain-cache-as-current-iv).
- market_data.fetch_spot_price defaults to the IEX latest trade (ALPACA_STOCK_DATA_FEED) — IEX prints can lag/deviate from SIP for thin names; pre-existing.
- market_data._fetch_yahoo_ohlc passes 'period' straight as Yahoo 'range' (no validation) — pre-existing; new code only exposes it via the '3y' string.
- alpaca-py is pinned at 0.12.0, which has no alpaca.data.historical.option / OptionChainRequest at all — the raw requests approach is forced and there is no in-venv model to validate the v1beta1 option query params.
### not_covered
- Live acceptance of expiration_date_gte/lte, strike_price_gte/lte, limit=1000 and feed by GET /v1beta1/options/snapshots/{underlying} — not measurable offline (alpaca-py 0.12.0 has no option request model to proxy); names match the documented OptionChainRequest fields of later alpaca-py releases from memory only.
- Whether the 'indicative' feed returns greeks/impliedVolatility (decides how often the BS-inversion path and its ±1 vol pt bias are actually used).
- Whether Alpaca returns today's in-progress daily bar with end=now−16 min on the free plan (the code keeps it if returned — consistent with the legacy IB endDateTime='' behaviour).
- Ordering of the snapshot response keys (the page-cap truncation is only harmless under ascending symbol order).
- Split/dividend adjustment of the Stooq and Yahoo fallback histories.
- Real multi-process Streamlit race on the IV CSV (only threads tested).
- 429 rate-limit behaviour of the filtered call (no retry; falls to the fallback chain).
### probe_scripts
- p1_alpaca_plumbing_opra_decode.py
- p1_alpaca_plumbing_iv_bias.py
- p1_alpaca_plumbing_iv_bias_mix.py
- p1_alpaca_plumbing_snapshot_mock.py
- p1_alpaca_plumbing_cache.py
- p1_alpaca_plumbing_bars_request.py
- p1_alpaca_plumbing_warmup.py


# SCRIPTS p4 (panel)
- p1_alpaca_plumbing_snapshot_mock.py (re-run, no network: requests.get patched module-wide)
- p4_ask-only-and-crossed-mids_repro.py
- p4_cache_group_code_reading.py
- p4_chart-title-overprinted-by-legend_repro.py
- p4_controller-bounds-duplicated-silent-clamp_repro.py
- p4_copy-en-dessous-hyphen_repro.py
- p4_duplicate-date-crash_repro.py
- p4_fallback-chain-cannot-reach-30dte-and-masks-root-cause_repro.py
- p4_fallback_impact.py
- p4_fallback_probe.py
- p4_forbid-streamlit-script-trips-on-docstring_repro.py
- p4_g1_math_code_reading.py
- p4_g2_epistemics_probe.py
- p4_g3_ivmethod_skeptic.py
- p4_g3_ivmethod_skeptic_b.py
- p4_g3_mid_fallbacks.py
- p4_g3_page_cap_e2e.py
- p4_g3_parity_median.py
- p4_g5_cache_impact.py
- p4_g6_viewA_impact.py
- p4_g6_viewA_probe.py
- p4_g7_viewB_code_reading.py
- p4_g7_viewB_title_legend_overlap.py
- p4_g8_arch_impact_probe.py
- p4_g8_arch_probe.py
- p4_g9_tests_ci_gate_probe.py
- p4_g9_tests_oracle_probe.py
- p4_g9_tests_probe.out.txt
- p4_g9_tests_probe.py
- p4_g9_tests_render_degraded_probe.py
- p4_hardcoded-dark-palette-light-theme_repro.py
- p4_html-exception-text-in-log_repro.py
- p4_impact_g2_epistemics.py
- p4_impact_g2_smooth_iv.py
- p4_impact_math.py
- p4_iv-cache-corrupt-file-silent-loss_repro.py
- p4_iv-disabled-says-alpaca-inaccessible_repro.py
- p4_iv-history-filename-unsanitized-symbol_repro.py
- p4_iv-history-overlay-unbounded-x-range_repro.py
- p4_iv-signal-vrp-bias_repro.py
- p4_linreg-scipy-valueerror-coincidence_repro.py
- p4_local-date-vs-exchange-date_repro.py
- p4_parity-r-q-zero-bias_repro.py
- p4_parity-r-q-zero-bias_repro2.py
- p4_percentile-label-vs-effective-window_repro.py
- p4_real_cache_scan.py
- p4_regime-split-out-of-range_repro.py
- p4_regime_split_param_sweep.py
- p4_render-guard-happy-path-only_repro.py
- p4_render-guard-not-in-ci-gate_repro.py
- p4_rv-bad-close-silent-drop_repro.py
- p4_service-zero-unit-tests_repro.py
- p4_smoke-tab-count-and-inventory_repro.py
- p4_snapshot-page-cap-silent_repro.py
- p4_split-annotation-units-mismatch_repro.py
- p4_stale-chain-cache-as-current-iv_repro.py
- p4_stale-result-under-error-and-params-drift_repro.py
- p4_weak-realized-vol-oracle_repro.py
- p4_yahoo-period-string_repro.py
- p4_yahoo_range_evidence.py
