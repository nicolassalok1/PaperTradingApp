# Review adversariale — onglet 🌡️ Vol Implicite (dashboard IV sur Alpaca)

- **Date** : 2026-08-21
- **Cible** : commit `cd278ec` — « Add 🌡️ Vol Implicite tab: IV regime dashboard ported from IB/Tkinter to Alpaca » (12 fichiers, +1887/−2). Le commit n'existait ni en local ni sur `origin` (session cloud dont le push a été refusé, 403) ; le patch livré (`iv-dashboard-alpaca.patch`, 73 170 octets) a été rejoué sur `origin/main` (`ed657c6`) dans la worktree `.claude/worktrees/feature+iv-dashboard-alpaca` → commit local **`f2d0812`**, empreinte strictement identique (12 fichiers, 1887 insertions, 2 suppressions). Tous les ancrages de lignes correspondent à la ligne près.
- **Méthode** : ultracode review-only — Phase 1 : 5 finders (un par dimension §4.1–4.5) ; Phase 2 : sondes de l'orchestrateur écrites indépendamment des agents ; Phase 3 (live Alpaca) : sautée, clés révoquées — Annexe B ; Phase 4 : 27 sceptiques (9 groupes × 3 angles : relecture du code / reproduction indépendante / impact-sévérité). Un finding survit si ≥ 2 sceptiques sur 3 ne le réfutent pas ; sévérité = vote majoritaire. Aucun fichier sous `app/` ni `tests/` n'a été modifié ; toutes les sondes vivent sous `scripts/review_iv_dashboard/`.
- **Interpréteur / versions** : `.venv/Scripts/python.exe` — Python 3.11.15, numpy 1.26.4, scipy 1.11.4, pandas 2.1.4, streamlit 1.51.0, plotly 5.22.0, alpaca-py 0.12.0 (versions épinglées confirmées). Mesures effectuées sous Windows 11 uniquement.

---

## 1. Résumé exécutif

**Verdict global : corrections requises.** Aucun bloquant : l'onglet ne plante pas dans le chemin nominal, aucune donnée n'est perdue quand Alpaca répond, aucun chiffre affiché n'est faux dans les conditions par défaut. Mais quatre défauts majeurs rendent l'IV affichée ou persistée fausse ou trompeuse dans des conditions réalistes, et le cinquième laisse tout cela sans filet.

- **M1** — Quand les snapshots filtrés échouent, le fallback « chaîne complète » sert un `options_alpaca_{SYM}.csv` de n'importe quel âge comme IV du jour, décale les échéances de l'âge du cache (+7 j mesuré), l'étiquette « greeks Alpaca » et l'écrit dans l'historique IV.
- **M2** — Le chip « Signal (IV) » applique la grille mean-reversion au percentile de l'IV dans la distribution de la **RV** : la prime de risque de vol (IV > RV structurellement) déclenche « MEAN REVERSION ↓ ATTENDUE » sur un tiers à la moitié des jours sur données réelles, sans information sur la cherté de l'IV.
- **M3** — L'inversion BS se fait avec r = 0, q = 0 sur le spot : ±1 pt de vol par contrat à 30 DTE (ATM call +95 bp / put −98 bp) ; la médiane ne s'annule que par symétrie calls/puts, symétrie qui casse en régime de basse vol (+40 à +44 bp mesurés à σ = 8–12 %).
- **M4** — `dt.date.today()` (heure machine) sert de clé au cache IV quotidien et pilote DTE/T : pour un utilisateur en UTC+8 (profil nomade), 62 % des minutes de séance US tombent sur la date locale du lendemain → deux sessions écrasées en une ligne, DTE −1.
- **M5** — `service.py` n'a aucun test direct : 8 % de couverture sous le gate CI (303/338 instructions jamais exécutées), alors que c'est là que vivent le chiffre d'IV affiché, l'upsert CSV et la chaîne de fallback. Les 5 tests esquissés sont prouvés verts (24 passed, hors ligne).

**Phase 3 live sautée** : les seules clés Alpaca disponibles sont révoquées (HTTP 401 sur `/v2/clock` et `data.alpaca.markets`, en-têtes edge Alpaca présents). Conséquence : le **contrat API des filtres serveur des snapshots reste *unverified-live*** (m1 et l'hypothèse « filtres honorés »), tout comme la présence des greeks sur le flux `indicative`, le comportement `end − 16 min` et l'IV headline vs référence externe. Le script `orch_live_alpaca.py <.env>` est prêt à rejouer ces quatre mesures avec des clés valides.

Au total : **0 C, 5 M, 23 m** (28 findings) ; 2 candidats tués (palette thème clair : prémisse fausse ; `range=3y` Yahoo : accepté en live).

---

## 2. Verdict par dimension

| Dimension | Verdict | Justification courte | Findings |
|---|---|---|---|
| §4.1 Maths (`analytics.py`) | acceptable avec réserves | aucune erreur de formule ; déviations legacy déclarées tiennent ; `shift(-fw)` et 252/365 = non-findings ; M2 (épistémique VRP) naît ici mais s'affiche dans la vue | M2, m7–m11 |
| §4.2 Plomberie Alpaca (`service.py`) | **corrections requises** | M1, M3, M4 touchent le chiffre affiché ou persisté ; contrat API **unverified-live** | M1, M3, M4, m1–m6 |
| §4.3 Architecture / MVC | acceptable en l'état | gates verts, vue → contrôleur seul, pas de Streamlit dans modèle/contrôleur, pas de cycle d'import | m18–m20 |
| §4.4 Vue (`tab_iv_dashboard.py`) | acceptable avec réserves | pas de sink HTML ; défauts de lisibilité et de copy | m12–m17 (+ chip M2) |
| §4.5 Tests | **corrections requises** | M5 (service sans test direct, 8 % de couverture sous le gate CI) | M5, m21–m23 |
| §4.6 Live | non vérifié | clés révoquées ; script prêt | — (Phase 3 sautée — Annexe B) |

---

## 3. Findings

### Bloquants (C) : aucun

Critères C retenus : crash de l'onglet dans le chemin nominal, perte de données sans trace, ou chiffre affiché faux dans les conditions par défaut avec Alpaca joignable. Aucun finding ne les remplit : M1–M4 exigent une condition réaliste mais non nominale (API indisponible + cache étranger, flux sans greeks, utilisateur hors fuseau US, ou une lecture épistémique contestable), et le seul crash trouvé (m10) exige une anomalie de données non observée sur 10 415 lignes réelles.

---

### M1 — Fallback chaîne : un cache périmé servi comme IV du jour, échéances décalées, et persisté

- **Ancre** : `app/model/iv_dashboard/service.py:372` (secondaires : L314 `_contracts_from_chain_df`, L429-430 étiquette `method`, L567-568 `record_iv_observation` ; `app/model/options/logic.py:1540-1549` lecture du cache sans TTL).
- **Constat** : quand l'appel snapshots filtrés échoue (réseau, 403, 429), `fetch_current_atm_iv` retombe sur `download_options_alpaca(sym, feed=..., max_pages=3)` avec `cache_to_csv=True` par défaut ; sur son propre échec, `logic.py` renvoie le dernier `options_alpaca_{SYM}.csv` de **n'importe quel âge** (le TTL de 60 s ne protège que le chemin pré-fetch). `_contracts_from_chain_df` reconstruit `expiry = today + round(T*365)` à partir d'un T figé à la date du cache, donc les échéances dérivent de l'âge du cache ; les IV du fichier sont comptées comme `n_direct` → `method='greeks Alpaca'` ; `get_iv_dashboard_data` appelle `record_iv_observation` sans condition. Ce cache est écrit couramment par l'onglet calibration (`calibration_controller.py:257`) et l'onglet options Alpaca (`options_controller.py:181`, « Cache chain to CSV » coché par défaut).
- **Évidence mesurée** : `p1_alpaca_plumbing_snapshot_mock.py` scénario D (ConnectionError, cache de 7 jours, iv = 0,25, vraie échéance today+30) : log `'162 contrats via chaîne Alpaca (cache).'` puis `'IV ATM 0.2500 (25.00%) — échéance 2026-09-27 (37 j), 126 contrats, méthode : greeks Alpaca.'` ; échéance rapportée 2026-09-27 vs réelle 2026-09-20 (+7 j) ; ligne persistée `{'date': '2026-08-21', 'iv': 0.25, 'dte': 37, 'n_contracts': 126, 'method': 'greeks Alpaca', 'spot': 640.0}`. Panel : cache 3 j + 403 → +3 j ; 10 j + 429 → +10 j ; `p4_fallback_impact.py` S1 : 4 tentatives HTTP / 1,51 s avant de servir le cache ; caption vue `'méthode : greeks Alpaca · flux : indicative'` sans mention de cache. Préconditions relevées par le panel : le cache doit contenir des IV non-NaN (sinon `'Aucune IV exploitable'`, pas de chiffre) et le DTE décalé doit rester dans [15, 60] (âge 20 j → dte 50 accepté ; âge 40 j → `'Aucun contrat entre 15 et 60 jours'`). La colonne `opra` du CSV contient déjà la vraie échéance (`SPY260920C…` → `_decode_opra` → 2026-09-20).
- **Impact utilisateur / trading** : « IV courante », « Spread IV − RV », « Percentile IV » et l'historique IV accumulé peuvent être un chiffre vieux de plusieurs jours étiqueté greeks Alpaca live, avec échéance/DTE faux — précisément les jours où l'API est injoignable. La seule trace (« (cache) ») est dans l'expander « Journal » replié.
- **Correctif proposé** (intègre les notes du panel : `opra` suffit, pas de changement de schéma) :

```diff
--- a/app/model/iv_dashboard/service.py
@@ def fetch_current_atm_iv(...):
-            chain = download_options_alpaca(sym, feed=feed_val, max_pages=_SNAPSHOT_MAX_PAGES)
+            # Option A (minimale) : jamais de cache d'âge arbitraire sur ce chemin
+            chain = download_options_alpaca(sym, feed=feed_val, max_pages=_SNAPSHOT_MAX_PAGES,
+                                            cache_to_csv=False)
+            # Option B (cohérente avec m2) : supprimer le bloc fallback L369-379,
+            # sa branche live n'atteint jamais 15-60 DTE (voir m2).
@@ def _contracts_from_chain_df(chain, today):
-            expiry = today + dt.timedelta(days=int(round(float(r["T"]) * 365.0)))
+            _, expiry, _ = _decode_opra(str(r.get("opra") or ""))
+            if expiry is None:
+                expiry = today + dt.timedelta(days=int(round(float(r["T"]) * 365.0)))
```

  Si le chemin cache est conservé : taguer `info['method'] = 'chaîne Alpaca (cache, <âge>)'` et ne pas appeler `record_iv_observation` quand la source est le cache.
- **Effort estimé** : 1 h.
- **Panel** : confirmé 3/3 (M, M, M). Le sceptique impact préfère la suppression pure du bloc fallback (même bloc que m2) à la variante « colonne d'échéance réelle », jugée sur-spécifiée.

---

### M2 — « Signal (IV) » : grille mean-reversion appliquée à un percentile IV-dans-RV — biais structurel VRP

- **Ancre** : `app/model/iv_dashboard/service.py:571` (secondaires : L569-570 ; `analytics.py:127-132` ; `tab_iv_dashboard.py:237-242` chip).
- **Constat** : `iv_regime = classify_regime(percentile_within(series_df['vol'].tail(pwin), current_iv['iv']))` — la grille de régime/signal (> 0,8 → « MEAN REVERSION ↓ ATTENDUE ») est appliquée au rang de l'IV **dans la distribution de la RV**. Le legacy (`option_trading_dashboard.py:331`) rangeait l'IV dans son **propre** historique 252 j. Le port a changé la sémantique en gardant le libellé, et cette déviation n'est pas listée dans le docstring « Differences vs the legacy script » (`analytics.py:9-13`). Seule la métrique d3 (« Percentile IV vs série RV ») porte un texte d'aide honnête ; le chip d4 reprend mot pour mot le signal de régime.
- **Évidence mesurée** : finder (`p1_math_rv_edges_epistemics.py` §5, `p1_view_contrast_bias_ranges.py` §B) : GBM σ = 18 % constant, IV = médiane RV +2/+3/+4/+5 pts → percentile 0,643 / 0,754 / 0,861 / 0,901 ; payload du driver de test (IV = RV+3 pts) rendu via AppTest : métrique « Percentile IV vs série RV » = `'100.0%'`, chip `'MEAN REVERSION ↓ ATTENDUE'`. Sonde orchestrateur G : IV 17 % dans une RV uniforme 12–18 % → 0,857 → « VOL ÉLEVÉE / MEAN REVERSION ↓ ATTENDUE ». **Correction de magnitude par le panel (données réelles AAPL 2016–2026, RV20, trailing 252)** : prime constante IV = RV+3 pts → signal « down » 33,8 % des jours ; +5 pts → 39,7 % ; ×1,25 → 43,1 % ; ×1,4 (ratio VIX/RV type SPY) → 54,5 % ; le signal « up » s'effondre à 9,7 / 4,8 / 7,9 / 3,7 %. Construction legacy (IV dans son propre historique) avec le même +3 pts : 24,3 % / 24,8 % (symétrique). Conditionnel aux jours calmes (RV20 ≤ médiane glissante) : 0,3–16,5 % selon le proxy. GBM médiane sur 200 graines : +2/+3/+4/+5 pts → 0,766 / 0,849 / 0,913 / 0,956. Fréquence réelle sur SPY : non mesurée (pas de données IV).
- **Impact utilisateur / trading** : un chip directionnel coloré pousse à vendre de la vol « un tiers à la moitié des jours » sur une prime qui ne porte aucune information de cherté — pas « quasi tous les jours » comme le disait le finder, mais structurellement asymétrique (« up » ≤ 4 % vs 25,5 % de base). Aucun chiffre n'est faux (d2 spread et d3 percentile sont honnêtes) ; c'est le libellé dynamique (« mean reversion attendue ») qui est infondé.
- **Correctif proposé** (les deux côtés doivent bouger, sinon le chip affiche `'N/A'`) :

```diff
--- a/app/model/iv_dashboard/service.py
@@
-            iv_regime = classify_regime(iv_vs_series_percentile)
+            iv_regime = None  # pas de signal mean-reversion sur un rang IV-dans-RV (VRP)
--- a/app/vue/tabs/tab_iv_dashboard.py
@@ def _render_metrics(...):
-        _chip("Signal (IV)", iv_regime.get("signal_label", "N/A"), ...)
+        st.caption(f"IV au {p:.0%} de la RV récente — prime de risque de vol, pas un IV rank.")
```

  Option utile (sémantique legacy restaurée) : quand `iv_history` compte ≥ 60 observations, calculer `percentile_within(iv_history['iv'], iv)` et n'afficher un signal que sur ce percentile (restera `N/A` ~60 jours d'analyse ; mélange des DTE à documenter). Alternative moins chère : relabeller le chip en bucket non directionnel (RICHE / NORMALE / BON MARCHÉ). Mettre à jour `_iv_dashboard_render_driver.py:83` qui construit `iv_regime` lui-même.
- **Effort estimé** : 1 h.
- **Panel** : confirmé 3/3 (M, M, M) ; les trois corrigent la magnitude du finder (GBM à vol constante surestime) mais gardent l'argument central « signal à sens unique ».

---

### M3 — Inversion BS avec r = 0, q = 0 sur le spot : ±1 pt de vol par contrat, médiane juste par accident

- **Ancre** : `app/model/iv_dashboard/service.py:416` (secondaires : L338 `r_annual: float = 0.0`, L417 `implied_vol_call(..., r_annual, 0.0)`, L565 seul appelant sans `r` ; `app/model/calibration/implied_vol.py:29-32` borne d'arbitrage).
- **Constat** : les puts sont convertis par `mid + spot − K·exp(−r_annual·T)` sans la jambe `S·e^{−qT}`, et chaque contrat est inversé avec r = 0, q = 0 sur le spot (pas sur le forward). Pour SPY (S = 640, r = 4 %, q = 1,3 %, 30 DTE) le call synthétique est sous-évalué d'≈1,42 $ et le forward mal posé d'autant. Le chemin est actif dès qu'Alpaca omet les greeks — ce que `logic.py:1271` annonce comme typique du flux `indicative` (non vérifié en live).
- **Évidence mesurée** (sonde orchestrateur `orch_probe_iv_bias.py`, finder `p1_alpaca_plumbing_iv_bias*.py`, panel `p4_g3_*.py` via la vraie `fetch_current_atm_iv`) : σ vraie 16 %, dte 30, T = 0,0822. Biais par contrat : K/S = 0,95 call +271 bp / put −48 ; 0,97 : +168 / −62 ; 0,99 : +112 / −84 ; **1,00 : +95 / −98** ; 1,01 : +82 / −118 ; 1,03 : +62 / −187 ; 1,05 : +49 / −361. Médiane calls+puts sur ±5 % (14 contrats) : +1 bp ; |K/S−1| < 1,5 % : −1 bp ; calls seuls +95 ; puts seuls −98 ; calls + 5 puts ATM +89. Sensibilité ATM put : r = 0 % → +46 bp ; 2 % → −27 ; 4 % → −98 ; 5 % → −133 ; q = 0 → put −142 / call +145 ; q = 3 % → −40 / +31. DTE : 15 j ±68, 45 j +116/−121, 60 j +133/−140. **Mécanisme de déséquilibre découvert par le panel** : en basse vol, le call synthétique r = q = 0 des puts ITM passe sous l'intrinsèque et `implied_vol_call` renvoie NaN → contrats rejetés silencieusement : σ = 12 % → 5 puts écartés, médiane **+44 bp** ; σ = 10 % → 11 écartés, +43 ; σ = 8 % → 17 écartés, +40 ; 15 DTE → +23…+29. Skew −0,8 pt/1 % : +13 bp. Les IV triées sont bimodales autour de la médiane (valeurs centrales −49/−48/+49/+50 bp) : un seul contrat perdu déplace le headline de ±49 bp. Fix « r = 4 % mais q oublié » : calls −50 / puts +44. Forward impliqué par parité : F = 641,417 vs vrai 641,422 → résidu −5 bp (calls seuls = puts seuls = −5 bp). Greeks path : +0 bp.
- **Impact utilisateur / trading** : sur le chemin inversion, l'IV headline bouge de ~±1 pt selon le côté coté, et de +40 bp systématiquement à des niveaux d'IV SPY courants (RV20 mesurée en live : 13,12 %) ; spread IV−RV lu +3,83 pts au lieu de +2,88 (calls seuls) ou +1,90 (puts seuls). `iv_minus_rv` et `iv_regime` héritent de l'erreur.
- **Correctif proposé** — variante sans paramètre préférée par le panel (aucun secret r, aucun q par symbole, résidu −5 bp mesuré) :

```python
# service.py, fetch_current_atm_iv — avant la boucle d'inversion
k0 = min(strikes, key=lambda k: abs(k - spot))
c0, p0 = mids[(k0, "call")], mids[(k0, "put")]       # paire ATM la plus proche
F = k0 + (c0 - p0)                                    # forward par parité (r≈0 dans l'actualisation)
for K, typ, mid in contracts:
    price = mid if typ == "call" else mid + F - K     # parité sur le forward
    iv = implied_vol_call(price, F, K, T, 0.0, 0.0)   # inversion sur F, non sur S
```

  Variante équivalente : n'inverser que les OTM (put si K < F, call si K > F) sur ce même F — supprime aussi le rejet des puts ITM. Variante « complète » (r via `app.model.yieldcurve.service.get_risk_free_rate(T, ensure_cache=True)`, q via le `div_yield` déjà porté par `options_controller`) : correcte mais fragile (q oublié = ±50 bp). Mettre à jour le docstring L343-345 (« parity with q=0 »).
- **Effort estimé** : 1 h.
- **Panel** : confirmé 3/3 (M, M, M). Désaccord sur l'esquisse : le sceptique impact juge le fix du finder (secret r + q par symbole) disproportionné et propose le forward par parité, que le sceptique reproduction a vérifié à 0,0 bp de résidu sur K = 608/640/672.

---

### M4 — `dt.date.today()` (heure machine) comme clé du cache IV et pilote de DTE/T

- **Ancre** : `app/model/iv_dashboard/service.py:464` (secondaires : L252 filtre `expiration_date_gte/lte`, L359 dte/T, L367 `_contracts_from_chain_df(chain, today)`, L475 upsert).
- **Constat** : trois sites utilisent la date locale naïve de la machine. Pour un utilisateur en UTC+7/+8 la date locale diffère de la date New-York de 12 h00 ET à minuit ET, soit toute la seconde moitié de la séance US : la session du 20 août analysée à 15 h30 ET (03 h30 SGT le 21) est stockée sous `2026-08-21`, la session du 21 août analysée à 22 h SGT écrase la même clé, et le DTE est un jour trop court. Rien en amont ne normalise le fuseau (le contrôleur ne fait que strip/upper ; `_fetch_closes_alpaca` n'utilise UTC que pour `end`). L'historique IV est superposé à la série RV datée séance (vue L314-324) : le décalage d'un jour y est visible.
- **Évidence mesurée** : `p1_alpaca_plumbing_cache.py` [6] : horloge Asia/Singapore 2026-08-21 03:30 vs date de séance US 2026-08-20 → `key mismatch True` (Paris / New York : False). Panel : deux sessions US distinctes (20/08 15 h30 ET et 21/08 10 h00 ET) sous Asia/Singapore ou Asia/Makassar → **1 ligne sur disque** (clés 2026-08-21 / 2026-08-21, IV de la première session perdue) ; Europe/Paris et America/New_York → 2 lignes. Minutes de séance (RTH) avec date locale ≠ date NY le 2026-08-20 : Asia/Makassar 241/391 (**62 %**), Asia/Jakarta 181/391 (46 %), Asia/Singapore 241/391 ; Paris / Los Angeles / Honolulu 0/391. La machine de revue est elle-même en UTC+8. Biais DTE sur le chemin inversion : 30 → 29 j, σ vraie 16 % inversée avec le T court → 16,279 % (+27,9 bp) ; 29 → 28 j : 16,342 % (+34 bp) ; finder : +26 bp. `ZoneInfo('America/New_York')` se résout dans le venv (tzdata 2026.3, dépendance dure de pandas 2.1.4).
- **Impact utilisateur / trading** : pour ce profil (nomade, projets en Indonésie), systématique et silencieux : collision de clé = perte irrécupérable d'une session par jour sur le plan gratuit (seule source d'historique IV), DTE affiché « IV ATM ~N j » un jour trop court, points dorés décalés d'un jour sur le graphe. Ne change pas une décision matériellement (1 j de DTE) → M, pas C.
- **Correctif proposé** :

```diff
--- a/app/model/iv_dashboard/service.py
+from zoneinfo import ZoneInfo
+
+def _exchange_date() -> dt.date:
+    """Date calendaire New-York (séance US), indépendante du fuseau de la machine."""
+    return dt.datetime.now(ZoneInfo("America/New_York")).date()
@@ _fetch_atm_snapshots
-    today = dt.date.today()
+    today = _exchange_date()
@@ fetch_current_atm_iv
-    today = dt.date.today()
+    today = _exchange_date()
@@ record_iv_observation
-    row = {"date": dt.date.today().isoformat(), ...}
+    row = {"date": _exchange_date().isoformat(), ...}
```

  Garder `today` passé à `_contracts_from_chain_df`. Optionnel : colonne horodatage UTC dans le CSV. Le cutoff de `get_iv_dashboard_data` (`pd.Timestamp.now().normalize()`) peut rester local. Avant 09 h30 ET la date NY n'est pas encore la date de séance (données = clôture veille) — acceptable, un calendrier de marché serait disproportionné.
- **Effort estimé** : 30 min.
- **Panel** : confirmé 3/3 (M, M, M) ; le sceptique impact demande de dé-emphasiser le biais +26 bp (chemin inversion seulement, du même ordre que l'erreur intraday de la convention DTE entier).

---

### M5 — `service.py` sans test direct : 8 % de couverture sous le gate CI

- **Ancre** : `app/model/iv_dashboard/service.py:234` (module entier : L45-590 non couverts ; seul test adjacent `tests/test_iv_dashboard_analytics.py:173`, qui monkeypatche le service sans l'appeler).
- **Constat** : aucun fetcher, parseur, aller-retour cache ni orchestrateur n'est exercé (`_fetch_atm_snapshots` et sa pagination, `_decode_opra`, `_snapshot_mid/_snapshot_iv`, branche parité de `fetch_current_atm_iv`, `record_iv_observation/load_iv_history`, chaîne de fallback de `fetch_daily_closes`, dégradation de `get_iv_dashboard_data`). Le driver de rendu redéfinit le payload à la main (`_iv_dashboard_render_driver.py:34-95`, 20 clés identiques aujourd'hui → risque de dérive, pas de désaccord actuel). Piège de patch : `CACHE_IV_HISTORY_DIR` est lié à l'import (`service.py:33`) — patcher `app.utils.paths` ne redirige pas.
- **Évidence mesurée** : `pytest -m "unit or smoke" --cov=app.model.iv_dashboard` (sélection CI complète, 676 passed / 2 skipped / 50 deselected, 177–178 s) : `service.py 338 stmts, 303 miss, 8%` (missing 45,49-52,63-120,135-165,173-180,184-187,191-205,209-231,248-280,284-286,290-305,309-327,347-449,456-457,462-483,488-497,516-590) ; `tab_iv_dashboard.py 213/180/13%` ; `analytics.py 88%` ; contrôleur 83 %. `grep -rn get_iv_dashboard_data|fetch_daily_closes|fetch_current_atm_iv|record_iv_observation|load_iv_history tests/` → une seule occurrence (L173, monkeypatch). Patch-target : `app.utils.paths.CACHE_IV_HISTORY_DIR` redirige = False ; `svc.CACHE_IV_HISTORY_DIR` = True. **`scripts/review_iv_dashboard/p1_tests_service_sketches.py` → `24 passed in 1.25s`** (sockets bloqués, sans clés ; re-runs panel 1,24 / 1,51 / 1,60 s). Le sceptique reproduction a écrit 6 tests indépendants : 6/6 PASS.
- **Impact utilisateur / trading** : le chiffre d'IV affiché (médiane ATM, parité, étiquette méthode), l'upsert CSV quotidien et la chaîne Alpaca → IEX → Stooq sont sans filet : une régression de pagination (`page_token` non transmis), du parsing OPRA (strike /1000) ou de l'upsert corromprait silencieusement l'historique accumulé jour après jour. Ce même panel a trouvé M1, m1, m3, m4, m10 exactement dans ces régions.
- **Correctif proposé** : créer `tests/test_iv_dashboard_service.py` (`pytestmark = pytest.mark.unit`) en copiant les sketches, patcher `svc.<nom>` (globaux du module), jamais `app.utils.paths`. Les 5 sketches (prouvés verts) :

```python
# 1. _fetch_atm_snapshots (service.py:234-280) — pagination, params, page cap, creds
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
# + sans token : 1 seul appel ; HTTP 403 -> raise_for_status propage ;
#   _alpaca_data_headers() -> None -> EnvironmentError et requests.get jamais appelé.

# 2. parseurs (service.py:171-231) — tables paramétrées
@pytest.mark.parametrize('opra, expected', [
    ('SPY260918C00450000', (450.0, dt.date(2026, 9, 18), 'call')),
    ('SPY260918P00450500', (450.5, dt.date(2026, 9, 18), 'put')),
    ('BRKB260918C00450000', (450.0, dt.date(2026, 9, 18), 'call')),
    ('garbage', (None, None, None)), ('', (None, None, None))])
def test_decode_opra(opra, expected): assert svc._decode_opra(opra) == expected
# _snapshot_mid : bid/ask -> mid ; bid=0 -> ask ; les deux à 0 -> latestTrade.p ; snake_case + prix str ; {} -> None
# _snapshot_iv : impliedVolatility top-level, greeks.iv, latestGreeks.impliedVolatility en str, greeks='x' -> nan, {} -> nan

# 3. fetch_current_atm_iv (service.py:330-449) — greeks + inversion parité, médiane, étiquette
def _bs_put(S, K, T, sig, r=0.0):
    d1 = (math.log(S/K) + (r + .5*sig**2)*T)/(sig*math.sqrt(T)); d2 = d1 - sig*math.sqrt(T)
    return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
def test_fetch_current_atm_iv_mixes_greeks_and_parity(monkeypatch):
    today = dt.date.today(); expiry = today + dt.timedelta(days=30); tag = expiry.strftime('%y%m%d')
    put_mid = _bs_put(100.0, 100.0, 30/365, 0.25)
    snaps = {f'SPY{tag}C00100000': {'greeks': {'iv': 0.21}},
             f'SPY{tag}P00100000': {'latestQuote': {'bp': put_mid-.01, 'ap': put_mid+.01}},
             f'SPY{tag}C00130000': {'greeks': {'iv': 0.9}}}  # hors bande ATM
    monkeypatch.setattr(svc, 'fetch_spot_price', lambda s: 100.0)
    monkeypatch.setattr(svc, '_fetch_atm_snapshots', lambda *a, **k: snaps)
    monkeypatch.delenv('ALPACA_OPTION_DATA_FEED', raising=False)
    info, log = svc.fetch_current_atm_iv('SPY')
    assert info['method'] == 'mixte (greeks + inversion BS)' and info['n_contracts'] == 2 and info['dte'] == 30
    assert info['iv'] == pytest.approx(np.median([0.21, 0.25]), abs=2e-3)
# + spot None -> (None, 'Spot indisponible…') ; snapshots lèvent + download_options_alpaca -> df vide -> (None, 'Aucun contrat…')

# 4. aller-retour cache (service.py:455-497)
def test_iv_history_upsert_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, 'CACHE_IV_HISTORY_DIR', tmp_path)   # PAS app.utils.paths
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    pd.DataFrame([{'date': yesterday, 'iv': .19, 'dte': 29, 'n_contracts': 5, 'method': 'x', 'spot': 99.}]).to_csv(tmp_path/'iv_daily_SPY.csv', index=False)
    svc.record_iv_observation(' spy ', {'iv': .21, 'dte': 30, 'n_contracts': 6, 'method': 'g', 'spot': 100.})
    svc.record_iv_observation('SPY',   {'iv': .25, 'dte': 31, 'n_contracts': 4, 'method': 'm', 'spot': 101.})
    hist = svc.load_iv_history('spy')
    assert hist['iv'].tolist() == [0.19, 0.25]          # upsert même jour, ancienne ligne conservée, trié
    assert list(svc.load_iv_history('ZZZZ').columns) == ['date', 'iv']
    (tmp_path/'iv_daily_BAD.csv').write_text('not,a,csv\n\x00', encoding='utf-8'); assert svc.load_iv_history('BAD').empty
    svc.record_iv_observation('SPY', {'iv': None})      # best-effort, ne doit pas lever

# 5. chaîne de fallback + dégradation de l'orchestrateur (service.py:123-168, 503-590)
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
```

  Trois ajustements demandés par le panel avant copie : (1) les cas « bid = 0 → ask » (sketch 2) et « CSV corrompu → vide sans lever » (sketch 4) figent des comportements signalés comme défauts par m3 et m4 — les écrire contre le comportement cible ou les marquer `xfail` ; (2) `dte == 30` dépend de `dt.date.today()` : si M4 est corrigé, construire l'échéance avec le même helper que le service ou accepter `dte in {29, 30, 31}` ; (3) continuer à patcher `svc.<nom>`.
- **Effort estimé** : 1 h.
- **Panel** : confirmé 3/3 (M, M, M).

---

### m1 — Plafond de pagination 3 × 1000 sous une chaîne SPY filtrée (~4,9 k), jamais journalisé

- **Ancre** : `app/model/iv_dashboard/service.py:266` (L38 `_SNAPSHOT_MAX_PAGES = 3`, L250 `limit: 1000`, L266-279 boucle ; L365 log du seul compte).
- **Constat** : avec les filtres serveur tels qu'envoyés (15–60 DTE, ±10 % de strikes), une chaîne type SPY (échéances L/M/V, strikes à 1 $) compte 4 864–4 902 contrats ; la boucle s'arrête après 3 pages avec `page_token` encore renseigné, sans message ni retour. L'échéance ~30 DTE survit seulement si la réponse est triée par symbole croissant (plausible — `next_page_token` est le dernier symbole de la page — mais **non mesuré en live**).
- **Évidence mesurée** : `p1_alpaca_plumbing_snapshot_mock.py` A : 4864 contrats dans les filtres, 3 appels, limits [1000, 1000, 1000], log `'3000 contrats candidats via snapshots filtrés'`, cap hit True, aucune mention de troncature ; résultat 31 DTE / 126 contrats en ordre croissant. Panel (vraie `fetch_current_atm_iv`, serveur mocké honorant les params) : 4902 contrats ; croissant → 31 DTE / 82 contrats ; décroissant → 33 DTE / 41 (échéance elle-même tronquée) ou 36 DTE / 33 ; mélangé → 31 DTE / 49 (set ATM aminci de 40 %). `p4_g3_page_cap_e2e.py` : cap atteint **30/30 jours** sur SPY quel que soit le calendrier ; avec ≥ 5 semaines d'échéances quotidiennes (6 450–6 708 contrats) le point de troncature tombe sur l'échéance cible **18/30 jours** → set ATM calls seuls (63/0), n_contracts 63, biais headline +95 bp (cf. M3) ; ≤ 4 semaines : 0 jour. Arithmétique des filtres : ±5 % sur 15–60 DTE = 2 394 contrats (> 1 page, ≤ 3 pages) ; ±5 % sur 20–40 DTE = 1 134. Chaînes type AAPL (strikes 2,5–5 $, hebdo) restent < 3000.
- **Impact utilisateur / trading** : troncature silencieuse à chaque run par défaut ; dégradation introuvable si le serveur trie autrement ou si la bande est élargie ; chemin concret vers un biais d'~1 pt combiné à M3 si les échéances quotidiennes listées dépassent 5 semaines.
- **Correctif proposé** (corrigé par le panel : « une page suffit » est faux numériquement ; resserrer la fenêtre DTE peut manquer les noms mensuels) :

```diff
@@ _fetch_atm_snapshots — après la boucle
+    truncated = bool(page_token)
+    return snapshots, truncated
@@ fetch_current_atm_iv
-    snapshots = _fetch_atm_snapshots(...)
+    snapshots, truncated = _fetch_atm_snapshots(...)
+    if truncated:
+        log.append(f"Pagination tronquée à {len(snapshots)} contrats (_SNAPSHOT_MAX_PAGES).")
```

  Plus, au choix : (a) bande de strikes serveur = bande de sélection ±5 % (la chaîne tient dans les 3 pages existantes : 2 394 ≤ 3000 ; la bande de repli 0,10 de L393 devient inatteignable, le repli « 4 plus proches » couvre) ; (b) `_SNAPSHOT_MAX_PAGES = 5` (~0,5–1 s par page). Garder la fenêtre 15–60 DTE.
- **Effort estimé** : 15 min.
- **Panel** : confirmé 3/3 (m, m, **M**) — le sceptique impact proposait M (troncature sur 30/30 jours + chemin récurrent vers +95 bp) ; vote majoritaire m. Deux sceptiques jugent la moitié « resserrer les filtres » de l'esquisse fausse ou régressive.

---

### m2 — Le fallback chaîne complète ne peut pas atteindre 30 DTE et masque la cause racine

- **Ancre** : `app/model/iv_dashboard/service.py:385` (L368-376 fallback ; L574 `iv_error = iv_log[-1]` ; `logic.py:1394-1398` params sans `limit`, L1405-1437 erreurs avalées).
- **Constat** : `download_options_alpaca` n'envoie jamais de paramètre `limit` (page serveur par défaut 100), donc `max_pages=3` rapporte ≤ 300 contrats triés par symbole = les 0–2 DTE d'un sous-jacent liquide, tous rejetés par `min_days_to_expiry=1` et le filtre 15–60 DTE. L'UI affiche alors `iv_error = "Aucun contrat entre 15 et 60 jours d'échéance."` (faux : rien n'a été récupéré) tandis que le 403/429/ConnectionError réel ne vit que dans l'expander. `download_options_alpaca` avale ses erreurs HTTP (logging.warning, df vide) ; donc L376 (`exc2`) est pratiquement inatteignable.
- **Évidence mesurée** : scénario C (403) : `iv_error = "Aucun contrat entre 15 et 60 jours d'échéance."` ; scénario E : params `[None, None, None]` pour `limit`, 0 contrat (300 × 0 DTE). `p4_fallback_probe.py` : params envoyés `[{'feed': 'indicative'}] × 3` ; sur 503 : log `['Snapshots filtrés indisponibles (503 Server Error) ; fallback chaîne complète.', '0 contrats via chaîne Alpaca (cache).', "Aucun contrat entre 15 et 60 jours d'échéance."]`, `'503' in iv_error` → False. `p4_fallback_impact.py` S4 : [403] 2 appels HTTP, 0,00 s ; [401 html] 4 appels (limits `[1000, None, None, None]`), 1,50 s ; [conn] 4 appels, 1,50 s. Sensibilité : même avec page 1000, 2 400 contrats à 3–6 DTE, 0 dans 15–60. Une chaîne mensuelle mince (80 contrats × 3 échéances) serait, elle, entièrement récupérée (28/56 DTE).
- **Impact utilisateur / trading** : sur un 403 OPRA (l'onglet documente `ALPACA_OPTION_DATA_FEED (indicative/opra)`, L146) ou un 429 transitoire, l'utilisateur est envoyé déboguer une fenêtre d'échéances vide au lieu de ses droits/clés ; latence morte jusqu'à 1,5 s. Aucun chiffre faux (l'IV est correctement absente) → m.
- **Correctif proposé** (« `max_pages=None` » rejeté par le panel : `min_days_to_expiry` est un filtre client, ~300–470 requêtes séquentielles pour SPY) :

```diff
@@ fetch_current_atm_iv
     except Exception as exc:
-        log.append(f"Snapshots filtrés indisponibles ({exc}) ; fallback chaîne complète.")
-        chain = download_options_alpaca(sym, feed=feed_val, max_pages=_SNAPSHOT_MAX_PAGES)
-        ...
+        log.append(f"Chaîne d'options Alpaca indisponible : {_short_exc(exc)}")  # cf. m6
+        return None, log          # iv_error = cause racine, plus de fallback
```

  Option « nice-to-have » : détecter `403` et suggérer « feed opra non autorisé → repasser en indicative ». Landing conjoint avec M1 (même bloc) et m6 (prérequis pour ne pas pousser 8 lignes HTML dans `st.warning`).
- **Effort estimé** : 30 min.
- **Panel** : confirmé 3/3 (m, m, m). Deux sceptiques notent que la variante `max_pages=None` de l'esquisse est disproportionnée.

---

### m3 — `_snapshot_mid` accepte l'ask seul, les cotations croisées et un dernier trade périmé

- **Ancre** : `app/model/iv_dashboard/service.py:218` (L214-216 mid sans test `ask ≥ bid`, L217-218 ask seul, L221-227/229-236 `latestTrade['p']` sans regarder `'t'`).
- **Constat** : une cotation unilatérale (bid absent ou 0) devient le « mid » (biais +spread/2/vega), bid > ask est moyenné sans contrôle, et quand les deux côtés sont à 0 le dernier trade est pris quel que soit son âge. Asymétrie : bid > 0 / ask = 0 → None, mais bid = 0 / ask > 0 → ask. Aucune garde aval ne rattrape (L413 `mid ≤ 0`, borne d'arbitrage, `0 < iv < 5`).
- **Évidence mesurée** : vraie `_snapshot_mid` : bid 0 / ask 5,20 → 5,2 ; bid absent → 5,2 ; croisé 5,30/5,10 → 5,2 ; bid = ask = 0 + trade 4,00 (t = 2026-01-02) → 4,0 ; bid 0,01 / ask 50 → 25,005 ; bid 5,10 / ask 0 → None. Vega ATM SPY 30 DTE 0,732 $/pt (1 $ = 137 bp). Biais ask-seul, contrat ATM (sonde orchestrateur) : spread 0,02 $ → +1 bp ; 0,10 $ → +7 ; 0,30 $ → +20 ; 1,00 $ → +68. Médiane sur toute la bande ±5 % avec bid = 0 partout (vraie `fetch_current_atm_iv`) : +8/+12 bp (0,10 $), +24 (0,30 $), +39/+57 (0,50 $), +79/+108 (1,00 $) — ~1,6× les valeurs ATM (les ailes ont moins de vega). Croisé de 0,20 $ partout : 126/126 acceptés. **Carnet après séance tout à 0 avec derniers trades à S−3** (`p4_g3_mid_fallbacks.py`) : 126/130 utilisables, médiane **+107 bp**, dispersion −977…+527 bp par contrat, caption toujours « inversion Black-Scholes (mid) ». Le fix strict du finder sur ce carnet : 0/130 contrats → IV indisponible.
- **Impact utilisateur / trading** : en séance sur SPY (spreads 0,02–0,15 $) ≤ ~+12 bp, négligeable ; hors séance le « mid » devient silencieusement un dernier trade, soit un changement de base non annoncé de ~1 pt — robustesse/transparence, pas formule fausse.
- **Correctif proposé** (version proportionnée retenue par le panel : rejeter les croisés, garder les replis mais les compter) :

```diff
@@ _snapshot_mid
-    if bid is not None and bid > 0 and ask is not None and ask > 0:
+    if bid is not None and bid > 0 and ask is not None and ask >= bid:
         return 0.5 * (bid + ask)
+    if bid is not None and ask is not None and 0 < ask < bid:
+        return None                                   # cotation croisée : on écarte
@@ fetch_current_atm_iv — compteur de base par contrat
+    n_ask_only, n_last_trade = ..., ...
+    info["method"] = f"inversion BS (mid, {n_ask_only} ask seul, {n_last_trade} dernier trade)"
```

  Variante stricte (`bid > 0 and ask >= bid`, pas de repli trade) : correcte mais rend l'IV indisponible hors RTH sur `indicative` et stoppe l'accumulation d'`iv_daily_*.csv` — dans ce cas expliciter « cotations unilatérales / hors séance » dans `'Aucune IV exploitable'`. Si un seuil de spread relatif est ajouté, le garder lâche (≤ 50 %). Gater le repli `latestTrade` sur son horodatage `'t'`.
- **Effort estimé** : 15 min.
- **Panel** : confirmé 3/3 (m, m, m) ; le sceptique impact juge l'esquisse du finder (« ne jamais retomber sur ask ou dernier trade ») régressive pour l'utilisateur nomade.

---

### m4 — CSV `iv_daily` vide ou sans en-tête : toute observation ultérieure perdue, pour toujours

- **Ancre** : `app/model/iv_dashboard/service.py:474` (L472-477 lecture dans le `try`, L475 garde `df.get('date', pd.Series(dtype=str))` morte, L482-483 `except` → `logging.warning` seulement ; vue L255-259 caption conditionnelle à `n_iv_obs > 0`).
- **Constat** : `pd.read_csv` lève `EmptyDataError` sur un fichier vide, et l'absence de colonne `date` fait lever « Unalignable boolean Series provided as indexer » (la Series vide par défaut ne peut pas indexer `df`). L'`except` journalise et abandonne ; le fichier n'est jamais réparé, le retour est `None` (pas de canal de log), l'appelant L568 ignore, `load_iv_history` avale et renvoie vide, la vue n'affiche rien. Le panel restreint le déclencheur : un `to_csv` interrompu laisse un fichier **en-tête seul** (9 octets `'date,iv\r\n'`), qui s'auto-répare ; le 0 octet exige un kill entre `open('w')` et l'écriture de l'en-tête (fenêtre ~3 ms) ou une coupure avant flush ; l'absence de colonne `date` avec des lignes de données exige une édition manuelle.
- **Évidence mesurée** : `p1_alpaca_plumbing_cache.py` [2]-[3] et sonde orchestrateur §D : fichier 0 octet → taille 0 → 0, 0 ligne, `'No columns to parse from file'` ; `'foo,iv'` → `'Unalignable boolean Series provided as indexer'`, fichier inchangé. Panel : EMPTY0 0→0 written=False ; WS (`\n\n`) 2→2 False ; NODATECOL 22→22 False ; HDRONLY 36→78 1 ligne OK ; TRUNC_ROW 72→118 2 lignes OK ; EXCELDATE 75→120 OK ; garbage binaire → `UnicodeDecodeError`, 0 ligne. Fenêtre `to_csv` pour 500 lignes : médiane 3,06 ms, max 32,45 ms (20 runs). Ligne finale tronquée `'2026-08-19,0.1'` : parsée, la valeur tronquée 0,1 est gardée comme point.
- **Impact utilisateur / trading** : une fois déclenché, permanent, silencieux et irrécupérable sur le plan gratuit (seule source d'historique IV) ; le graphe « Historique IV local » ne grandit plus. Déclencheur rare → m (le finder disait M).
- **Correctif proposé** :

```diff
@@ record_iv_observation
-        if path.exists():
-            df = pd.read_csv(path)
-        ...
-        df = df[df.get("date", pd.Series(dtype=str)).astype(str) != today]
+        df = pd.DataFrame(columns=list(row.keys()))
+        if path.exists():
+            try:
+                df = pd.read_csv(path)
+            except pd.errors.EmptyDataError:
+                pass                                   # fichier vide : on repart de zéro
+            except pd.errors.ParserError:
+                path.rename(path.with_suffix(".csv.bak"))  # ne jamais écraser des mois d'historique
+        if "date" not in df.columns:
+            df = pd.DataFrame(columns=list(row.keys()))
+        df = df[df["date"] != row["date"]]
@@ écriture atomique
-        df.to_csv(path, index=False)
+        tmp = path.with_suffix(".csv.tmp"); df.to_csv(tmp, index=False); os.replace(tmp, path)
```

  Remonter l'avertissement dans `result['log']` demande que `record_iv_observation` renvoie un message et que L568 l'ajoute — optionnel.
- **Effort estimé** : 30 min.
- **Panel** : confirmé 3/3 (m, m, m) ; finder M, rétrogradé à m par les trois sceptiques (déclencheur étroit).

---

### m5 — Nom de fichier du cache IV dérivé du symbole brut (pas de sanitisation)

- **Ancre** : `app/model/iv_dashboard/service.py:457` (L455-457 `_iv_history_path`, L480 `mkdir(parents=True)`).
- **Constat** : `CACHE_IV_HISTORY_DIR / f"iv_daily_{sym}.csv"` avec `sym` seulement strip/upper, alors que `market_data.py:483` (`[^A-Za-z0-9_-]+`) et `options/logic.py:1107` (`[^A-Za-z0-9._-]`) passent le ticker par `re.sub` (`market_data.py:361` utilise `.replace('/', '_')`). Le même module échappe le symbole pour l'URL (`quote_plus`, L283-286) mais pas pour le système de fichiers. Écriture et lecture sont gatées par un fetch marché réussi pour la chaîne exacte : un symbole avec séparateur n'y arrive pas en pratique.
- **Évidence mesurée** : `p1_arch_controller_and_cache.py` : `'A/B'` → `iv_daily_A/B.csv` (sous-dossier créé) ; `'../evil'` → `iv_daily_../EVIL.csv` (ENOENT, rien écrit). Panel Windows : `'A:B'` → flux NTFS alternatif sur l'entrée `iv_daily_A` (record et load réussissent sans fichier visible) ; `'A?B'` → `[Errno 22]` avalé ; **`'/../../ESCAPE'` → `ESCAPE.csv` écrit un niveau au-dessus d'`IVHistory`** (normalisation lexicale Win32, `resolve().relative_to` échoue) ; `'BRK.B'`, `'BRK-B'`, `'^VIX'`, `'ES=F'` restent des fichiers plats ; `load_iv_history` sur 13 symboles hostiles → 0 ligne, aucune exception.
- **Impact utilisateur / trading** : négligeable sur tout chemin atteignable ; incohérence de convention et pollution possible de `cache/IVHistory/`.
- **Correctif proposé** :

```diff
+import re
@@ def _iv_history_path(symbol):
-    sym = symbol.strip().upper()
-    return CACHE_IV_HISTORY_DIR / f"iv_daily_{sym}.csv"
+    safe = re.sub(r"[^A-Za-z0-9._-]", "_", symbol.strip().upper()) or "SYMBOL"
+    return CACHE_IV_HISTORY_DIR / f"iv_daily_{safe}.csv"
```

  Même regex qu'`options/logic.py:1107` ; `BRK.B` conservé, donc aucune migration de cache.
- **Effort estimé** : 10 min.
- **Panel** : confirmé 2/3 — réfutation (angle impact) : impact nul sur tout chemin atteignable, à replier dans une future retouche de `_iv_history_path` plutôt que finding autonome.

---

### m6 — Texte brut d'exception (page HTML 401 nginx, 8 lignes) dans le journal

- **Ancre** : `app/model/iv_dashboard/service.py:147` (le panel précise que le site de concaténation est **L149** ; L147 est le `return` de succès) ; secondaires L368, L376 ; vue L495-497 `st.code`.
- **Constat** : alpaca-py 0.12.0 (`common/rest.py:203-205`) lève `APIError(response.text, http_error)` dont `str()` est le corps HTML brut ; `fetch_daily_closes` concatène `f'Barres {tag} indisponibles : {exc}'` pour chaque feed sans nettoyage ni troncature. La moitié « `iv_error` peut être cette phrase longue » ne se reproduit pas : `iv_error = iv_log[-1]` n'est jamais L368 (toujours suivi de L374/L376) et L376 est inatteignable (`download_options_alpaca` avale) — elle ne le deviendra qu'une fois m2 corrigé, ce qui fait de ce helper un prérequis de m2.
- **Évidence mesurée** : run live `orch_live_alpaca.out.txt` : deux blocs de 8 lignes `<html>…<center>nginx</center>…</html>` (feed défaut + iex). Panel (401 mocké) : log[0] 202 car./8 lignes, log[1] 208 car./8 lignes ; journal rendu 17 lignes pour 3 entrées ; L368 produit une ligne de 217 caractères avec l'URL et sa query string ; sur 503, `'503' in iv_log[-1]` → False.
- **Impact utilisateur / trading** : journal illisible quand Alpaca refuse la clé (clé révoquée/expirée, cas fréquent) ; `st.code` rend les balises littéralement (pas d'injection) ; cosmétique.
- **Correctif proposé** :

```python
def _short_exc(exc: BaseException, n: int = 160) -> str:
    status = getattr(exc, "status_code", None) \
        or getattr(getattr(exc, "response", None), "status_code", None)
    if status:
        return f"HTTP {status}"                      # alpaca APIError / requests.HTTPError
    return " ".join(str(exc).split())[:n]
```

  Appliquer à L149 (barres), L368 (snapshots), L376 (chaîne, faible valeur). La troncature seule laisserait `'<html> <head><title>401 Authorization Required…'` : préférer le code HTTP.
- **Effort estimé** : 15 min.
- **Panel** : confirmé 3/3 (m, m, m) ; correction d'ancre (L149) et de portée (journal seulement, pas `iv_error`).

---

### m7 — Série RV constante : `ValueError` scipy attrapée par coïncidence, message anglais brut

- **Ancre** : `app/model/iv_dashboard/analytics.py:139` (`_linreg` sans garde ; L182-187 seul contrôle d'effectif ; `service.py:586` `except ValueError` ; vue L540-543 `st.info`).
- **Constat** : scipy 1.11.4 lève `ValueError('Cannot calculate a linear regression if all x values are identical')` après le contrôle `MIN_ANALYSIS_POINTS` ; `service.py:586` l'attrape comme si c'était « série insuffisante » (seul cas documenté, L166) et stocke le texte anglais dans `analysis_error`. La sonde orchestrateur montre que le cas n'exige pas une série entièrement constante : ≤ 2 niveaux distincts de RV suffisent pour que la régression **par régime** (> 10 points à x identiques) lève et fasse perdre `reg_forward`/`reg_diff` pourtant calculables.
- **Évidence mesurée** : `p1_math_service_constant_series.py` : clôtures 10,00/10,01 alternées (262 lignes, `nunique=1`) → `analysis=None`, `analysis_error='Cannot calculate a linear regression if all x values are identical'` ; clôtures constantes → `current_vol=0.0`, percentile 0,502 → NORMALE/NEUTRE. `orch_probe_math.py` B : 1,00/1,01 tous les 7 j sur 300 j → n = 280, régression globale OK, régression par régime lève. Réalisme : halt 60 j dans une série bruitée → 0/50 et 0/200 graines ; chemins illiquides simulés → 1/200 ; sur AAPL réel, min `nunique(vol)` sur 471 fenêtres 2 ans = 498/504, régimes 186/288 valeurs distinctes.
- **Impact utilisateur / trading** : cas rare (ticker mort sur toute la durée, sans marché d'options) ; rien de silencieux ; défauts de copy (texte scipy) et étiquette « NORMALE » sur une vol nulle.
- **Correctif proposé** (intègre la note du panel : `None` plutôt qu'un dict NaN — la vue gère déjà `None` via `_format_reg` L465 « données insuffisantes » et `_render_diff_chart` L433 `if not reg`, alors qu'un dict NaN imprimerait « pente nan » et ajouterait une trace vide) :

```diff
@@ analyze_forward_vol — après le dropna L182
+    if df["current_vol"].nunique() < 2:
+        raise ValueError("Série de vol constante : régression impossible.")
@@ _linreg
-    res = stats.linregress(x, y)
+    if pd.Series(x).nunique() < 2:
+        return None                      # régimes dégénérés : la vue affiche « données insuffisantes »
+    res = stats.linregress(x, y)
```

  Alternative : garder le garde aux sites d'appel L203-212 (`and df.loc[mask, 'current_vol'].nunique() > 1`). Polish optionnel : `classify_regime` → `'unknown'` quand `current_vol == 0`.
- **Effort estimé** : 30 min.
- **Panel** : confirmé 2/3 — réfutation (angle impact) : inatteignable sur SPY/QQQ/AAPL, rien de silencieux, polish seulement.

---

### m8 — Intersection y = x hors plage des données : split négatif affiché, régime vide, `reg_high == reg_diff`

- **Ancre** : `app/model/iv_dashboard/analytics.py:194` (L193-198 intersection ; L200-212 masques ; vue L447-451 vline, L484-491 journal).
- **Constat** : `intersection = intercept/(1−slope)` n'est contrainte ni à la plage de `current_vol` ni utilement pour slope ≈ 1 (`abs(1−slope) > 1e-12` ≡ `!= 1` legacy : slope = 1−1e-9 passe et donne 1e7). Hors plage, tout l'échantillon tombe dans un régime, l'autre est `None` (n = 0) et la régression « régime » duplique exactement `reg_diff`. Le panel élargit la cause : **la majorité des cas hors plage ont slope < 1** (série longue et lisse à plage de RV étroite), pas seulement « pente > 1 / rupture structurelle ».
- **Évidence mesurée** : `p1_math_degenerate_regressions.py` (halt 300 j + 100 j bruités, seed 0) : slope 1,0436, intercept 0,0131, intersection −0,3008 pour une plage [0,0000 ; 0,3388], n_high 350, n_low 0, `reg_low=None`, `reg_high == reg_diff` ; journal « Régime VOL BASSE (vol ≤ −0.301, n=0) », vline « Split régimes (−0.301) » hors axe. Sonde orchestrateur I : intersection 3,2427 pour [0,0999 ; 0,2007] → n_high 0. 200 séries OU log-vol (θ = 0,02) : pente ∈ [0,262 ; 0,870], 0 hors plage ; θ = 0,002 : 9/200. Panel : slope 0,984 / intercept 0,0052 → 0,323 vs [0,136 ; 0,237] ; slope 0,927 → 0,024 vs [0,034 ; 0,150]. **Balayage réel AAPL 1984–2026** (`p4_regime_split_param_sweep.py`) : défauts rv 20 / fwd 30 → 0/235 (2 ans), 0/217 (5 ans), 1/241 (1 an) ; **rv 120 / fwd 30 / 1 an → 67/238 (28 %)** hors plage + régime vide, 36 régressions dupliquées, split affiché jusqu'à 273,910 (= 27 391 % de vol), « Régime VOL BASSE (vol ≤ 273.910, n=238) » ; rv 120 / fwd 90 / 1 an → 62/238 (26 %) ; rv 60 / fwd 90 / 1 an → 14/240 (6 %) ; rv 120 / fwd 90 / 2 ans → 24/232 (10 %, jusqu'à 88,98). Fenêtres à slope 1,000/0,998 → intersections 273,9/13,2 (garde 1e-12 inutile).
- **Impact utilisateur / trading** : nul aux paramètres par défaut ; fréquent (6–28 %) aux réglages atteignables du formulaire (RV jusqu'à 120, forward jusqu'à 90) : seuil absurde, « régression régime haute » qui n'est que la globale, vline hors axe. Signaux primaires (percentile, régime, insights) non affectés ; pas de crash (vue gardée par `not int(mask.sum())`) ; identique au legacy.
- **Correctif proposé** (clamp à la médiane — neutralise aussi l'explosion slope ≈ 1 puisque 1e7 est hors plage ; message corrigé par le panel, pas de « Pente ≥ 1 ») :

```diff
@@ analyze_forward_vol — après L198
+    lo, hi = float(df["current_vol"].min()), float(df["current_vol"].max())
+    if not (lo < intersection < hi):
+        intersection = float(df["current_vol"].median())
+        insights.append("Intersection y=x hors de la plage des vols observées : split à la médiane.")
```

  Durcissement suggéré : retomber aussi sur la médiane quand `min(n_high, n_low) <= MIN_REGIME_POINTS`, pour que les régressions par régime ne dupliquent jamais silencieusement `reg_diff`. `test_iv_dashboard_analytics.py:123` reste vert.
- **Effort estimé** : 30 min.
- **Panel** : confirmé 3/3 (m, m, **M**) — le sceptique impact proposait M (28 % des fenêtres réelles à rv = 120) ; vote majoritaire m.

---

### m9 — Libellé « Percentile (N j) » = fenêtre demandée, pas fenêtre effective

- **Ancre** : `app/model/iv_dashboard/service.py:521` (`extra_days = int(rv_window*1.6)+15` ; L140 lookback ; L531 `compute_percentile_series` ; L569 `.tail(pwin)` post-cutoff L536 ; vue L190 libellé, L192 aide).
- **Constat** : le lookback ne budgète que le warm-up RV (47 j calendaires), jamais `percentile_window` ; `rolling(window, min_periods=60)` se relâche silencieusement quand il y a moins de lignes que la fenêtre ; le libellé lit le paramètre demandé. Second puits : le percentile IV-vs-RV (L569) utilise `series_df['vol'].tail(pwin)` **après** le cutoff, donc encore moins de points. Seules 3 des 16 combinaisons atteignables dans l'UI (Durée 1/2/3/5 ans × fenêtre 60..756) sont touchées, aucune par défaut.
- **Évidence mesurée** : finder (bdate_range, sans jours fériés) : 1 an/504 → 275 pts (label 504) ; 1 an/756 → 275 (label 756) ; 2 ans/756 → 536. Panel, **vrai calendrier AAPL** (oracle manuel = service à 1e-12 sur 16 combos) : 1 an → 281 clôtures / 261 lignes RV : pwin 504 → 261 (52 %), 756 → 261 (35 %) ; 2 ans → 532 / 512 : 756 → 512 (68 %) ; 3 ans (763 RV) et 5 ans (1 267) : exacts ; 1 an/252 → 252/252 (autre sceptique : 251). Tail IV post-cutoff : 262/262/523 (1 an/756 : 251). Écart de lecture sur AAPL, |pct252 − pct756| sur 1 996 jours : médiane 0,093, p90 0,238, max 0,378 ; bucket de régime différent 46,7 % des jours, signal MR 25,5 %. `extra_days` = 47 j.
- **Impact utilisateur / trading** : pour les combinaisons non-défaut (expander « Paramètres avancés », sans texte d'aide), un rang sur ~1 an étiqueté 3 ans ; la relaxation legacy (strict 252 → NaN) est déclarée mais invisible.
- **Correctif proposé** (formule corrigée par le panel : compter **avant** le cutoff) :

```diff
@@ get_iv_dashboard_data — avant le cutoff L536
+    percentile_effective_n = int(min(int(percentile_window), int(rv.notna().sum())))
@@ payload
+    "percentile_effective_n": percentile_effective_n,
--- a/app/vue/tabs/tab_iv_dashboard.py
@@ L190
-    label = f"Percentile ({percentile_window} j)"
+    n_eff = result.get("percentile_effective_n", percentile_window)
+    label = f"Percentile ({percentile_window} j" + (f", {n_eff} pts)" if n_eff < percentile_window else ")")
```

  Option « étendre le lookback » (`extra_days += int(percentile_window*1.5)`) : doit aussi élargir `period` du fallback Stooq/Yahoo (L152-153, dérivé de `years` seul) et ne répare pas le tail IV. Alternative honnête minimale : clamper `percentile_window` aux lignes disponibles et le journaliser.
- **Effort estimé** : 30 min.
- **Panel** : confirmé 3/3 (m, m, m) ; un sceptique juge la formule `tail(pwin).shape[0]` de l'esquisse fausse (post-cutoff).

---

### m10 — Index de dates dupliqué → `ValueError « cannot reindex on an axis with duplicate labels »`

- **Ancre** : `app/model/iv_dashboard/service.py:534` (construction de `series_df` hors `try` ; aucune dédup dans `_fetch_closes_alpaca` L116, fallback L152-156, `market_data._standardize_ohlc` L273 ; vue L167-168 `st.error`).
- **Constat** : `pd.DataFrame({'close': closes, 'vol': rv, 'vol_percentile': pct})` aligne trois Series ; `rv` a perdu sa première ligne, donc pandas prend l'union et réindexe sur un axe dupliqué → lève (avec des index dupliqués identiques, pandas ne lève pas : c'est exactement le chemin union). Non attrapé (seul `analyze_forward_vol` est dans un `try`) → l'onglet affiche « Analyse impossible : cannot reindex on an axis with duplicate labels ». En amont `compute_log_returns` injecte un rendement 0,0 au doublon (si même clôture ; `log(c2/c1)` sinon).
- **Évidence mesurée** : `p1_math_rv_edges_epistemics.py` §3 : 301 lignes / 300 dates uniques → `compute_realized_vol` OK (rendements `[0.00833343, 0.0]` au doublon), construction → `ValueError`. Panel : chemin service complet avec `fetch_daily_closes` patché → exception propagée au contrôleur ; sur fenêtre AAPL 2 ans + 1 doublon, après `closes[~closes.index.duplicated(keep='last')]` la RV est numériquement identique à la série propre (`np.allclose True`). Réalisme : 0 date dupliquée sur 10 415 lignes Stooq réelles (`p4_real_cache_scan.py`) ; barres Alpaca normalisées UTC → dates distinctes ; le chart v8 Yahoo est connu pour parfois renvoyer la barre live du jour en double en séance (yfinance dédoublonne pour cela) — non mesuré, et le run live montre que l'app tourne aujourd'hui sur ce fallback.
- **Impact utilisateur / trading** : toute l'analyse du symbole remplacée par un message pandas opaque — bruyant, pas silencieux, onglet fonctionnel → m (deviendrait M si le doublon Yahoo était confirmé en trafic réel).
- **Correctif proposé** (un seul point d'étranglement, `keep='last'` = barre la plus récente gagne) :

```diff
@@ get_iv_dashboard_data — avant L533
+    n_dup = int(closes.index.duplicated().sum())
+    if n_dup:
+        closes = closes[~closes.index.duplicated(keep="last")]
+        log.append(f"{n_dup} date(s) dupliquée(s) supprimée(s) (dernière barre conservée).")
```

- **Effort estimé** : 15 min.
- **Panel** : confirmé 3/3 (m, m, m).

---

### m11 — Clôture ≤ 0 : deux rendements supprimés silencieusement, la fenêtre RV enjambe le trou

- **Ancre** : `app/model/iv_dashboard/analytics.py:37` (L36 `px.where(px > 0)`, L37 `.dropna()`, L55 rolling positionnel).
- **Constat** : la clôture fautive devient NaN, puis les rendements t et t+1 sont retirés (pas laissés NaN) : l'index perd 2 dates, la fenêtre de 20 rendements couvre 22 lignes calendaires sans NaN, et le saut réel t−1 → t+1 est perdu (la somme des deux rendements supprimés est exactement `log(c[t+1]/c[t−1])`). En amont, `service.py` L116/L156 ne font que `dropna(subset=['Close'])` : seule une clôture strictement ≤ 0 peut atteindre analytics.
- **Évidence mesurée** : finder : 1 clôture 0,0 (ou −5,0) au milieu de 300 j → `len(rv)` 299 → 297, RV absente J et J+1, 19 valeurs modifiées, écart max 1,72 pt (σ = 1 %/j). Panel : 1,083 pt (seed 1), 4,92 pt sur un niveau 14,5 % (seed 42), 2,12 pt sur AAPL réel (perte du rendement 2 j −0,1055 %) ; la variante NaN du finder blanchit 21 lignes RV pour une seule mauvaise ligne ; filtre-d'abord (`px = px[px > 0]`) : 1 date perdue, 20 valeurs, max 1,89 pt, pas de trou. **0 clôture ≤ 0 sur 10 415 lignes réelles.**
- **Impact utilisateur / trading** : borné (~2 pts sur 19 points sur 504), sans trace ; probabilité effectivement nulle avec SPY/QQQ/AAPL → robustesse.
- **Correctif proposé** (la variante « garder l'alignement NaN » est jugée disproportionnée par le panel : 21 lignes blanchies puis supprimées, contrat `compute_log_returns` changé) :

```diff
@@ compute_log_returns
-    px = px.where(px > 0)
+    px = px[px > 0]                     # le rendement t-1 -> t+1 survit comme rendement 2 jours
@@ service.get_iv_dashboard_data
+    n_bad = int((closes <= 0).sum())
+    if n_bad:
+        log.append(f"{n_bad} clôture(s) ≤ 0 ignorée(s).")
```

- **Effort estimé** : 30 min (ou 5 min pour la seule ligne de log).
- **Panel** : confirmé 2/3 — réfutation (angle impact) : condition inatteignable avec les sources réelles (0/10 415), effet borné, polish.

---

### m12 — IV désactivée par l'utilisateur → warning « chaîne d'options Alpaca inaccessible »

- **Ancre** : `app/vue/tabs/tab_iv_dashboard.py:249` (L249-252 ; `service.py:560-574`).
- **Constat** : case « IV courante via options Alpaca » décochée → le service laisse `current_iv=None` **et** `iv_error=None` (pas de branche `else`, aucun drapeau `include_current_iv` dans le payload) ; la vue retombe sur le texte par défaut `"IV courante indisponible — chaîne d'options Alpaca inaccessible."`. Quand l'IV a réellement été tentée, `iv_error` n'est jamais vide (L574 retombe sur `'IV indisponible.'`) : le texte par défaut L251 est donc du code mort sauf dans le cas « désactivée ».
- **Évidence mesurée** : `p1_view_sinks_and_state.py` § `iv_disabled_warning` : `warnings = ["IV courante indisponible — chaîne d'options Alpaca inaccessible."]`, 0 exception. Panel (vrai service, `fetch_current_atm_iv` instrumenté) : 0 appel, `current_iv=None`, `iv_error=None`, aucune ligne IV dans le log ; contrôle avec échec réel : `"IV courante indisponible — Chaîne d'options Alpaca indisponible : 401 Unauthorized."`.
- **Impact utilisateur / trading** : l'utilisateur croit à une panne / clés invalides et débogue une connexion jamais testée ; aucun chiffre touché ; opt-out explicite quelques secondes plus tôt → m (finder M).
- **Correctif proposé** :

```diff
@@ _render_metrics
-    else:
-        st.warning("IV courante indisponible — " + str(iv_error or "chaîne d'options Alpaca inaccessible."))
+    elif result.get("iv_error"):
+        st.warning("IV courante indisponible — " + str(result["iv_error"]))
+    else:
+        st.caption("IV courante désactivée (case « IV courante via options Alpaca »).")
```

  Plus robuste d'une ligne : `"include_current_iv": bool(include_current_iv)` dans le payload (L589-609) et brancher dessus.
- **Effort estimé** : 15 min.
- **Panel** : confirmé 3/3 (m, m, m) ; finder M, rétrogradé à m par les trois.

---

### m13 — Résultat périmé rendu sous l'erreur ; paramètres du formulaire ≠ paramètres du résultat

- **Ancre** : `app/vue/tabs/tab_iv_dashboard.py:513` (L166-168 `except` sans invalidation de `_STATE_KEY` ; L523-526 caption ; L165 `strftime('%H:%M:%S')`).
- **Constat** : après un échec de re-soumission, `st.error` s'affiche mais `session_state['iv_dashboard_result']` reste : métriques et 3 graphes du symbole précédent sous l'erreur. `rv_window`, `percentile_window`, `forward_window` sont bien échos (L184, L190, L386) ; seule **Durée (`years`)** n'apparaît nulle part ; `generated_at` n'a pas de date (parité legacy `option_trading_dashboard.py:187`). La dérive widget/résultat est inhérente à `st.form`. Variante découverte par le panel : un rerun simple après l'échec → `st.error` disparaît, le dashboard périmé reste, le formulaire affiche toujours XYZ / 5 ans.
- **Évidence mesurée** : `p1_view_sinks_and_state.py` § `stale_after_error` : contrôleur levant, state SPY/2 ans, saisie `<img…>` / 5 ans → `errors=['Analyse impossible : Aucune donnee de prix disponible pour <IMG SRC=X ONERROR=ALERT(1)> …']`, `n_charts_after_error=3`, caption `'**SPY** · source série : alpaca · 540 points · généré à 12:00:00'`, selectbox `'5 ans'`, `result_years_in_state=2.0`. Panel : `years_shown_anywhere=False` sur tout le texte rendu ; C (rerun) : errors 0, charts 3, caption SPY, formulaire XYZ/5 ans ; D contrôle QQQ/5 ans : 1 287 points.
- **Impact utilisateur / trading** : écran mixte ; un utilisateur qui change « Durée » sans recliquer lit des régimes sur 2 ans en croyant 5 ans (seul indice : le nombre de points). Le symbole est en gras dans la caption → visible.
- **Correctif proposé** :

```diff
@@ L166-168
     except Exception as exc:
+        st.session_state.pop(_STATE_KEY, None)        # ou : garder + st.warning("Résultat précédent …")
         st.error(f"Analyse impossible : {exc}")
@@ L165
-    result["generated_at"] = pd.Timestamp.now().strftime("%H:%M:%S")
+    result["generated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
@@ caption L523
-    f"**{symbol}** · source série : {source} · {n} points · généré à {generated_at}"
+    f"**{symbol}** · {result.get('years'):g} an(s) · source série : {source} · {n} points · généré à {generated_at}"
```

  Le sceptique impact préfère garder le résultat précédent avec un `st.warning` explicite (un `pop` jette un résultat valide sur un échec réseau transitoire) ; le sceptique reproduction tranche pour le `pop` (supprime aussi la variante C) ; le sceptique relecture accepte les deux options.
- **Effort estimé** : 15 min.
- **Panel** : confirmé 3/3 (m, m, m).

---

### m14 — Overlay « IV ATM (historique local) » non borné à la fenêtre affichée

- **Ancre** : `app/vue/tabs/tab_iv_dashboard.py:314` (L313-326 `lines+markers` ; `service.py:486-497` `load_iv_history` sans filtre de date ; L576 passage brut).
- **Constat** : tout le CSV est tracé, avec segments ; aucun `xaxis.range` n'est posé, Plotly autoscale sur l'union des traces. Une observation antérieure à la fenêtre choisie étire l'axe ; les segments entre observations espacées (une par jour d'analyse, par construction) dessinent une trajectoire jamais observée. Le legacy traçait une série IB dense, où les lignes avaient un sens. Comparaison tz-safe (les deux côtés naïfs).
- **Évidence mesurée** : finder : série 1 an (351 j) + 1 observation du 2023-03-01 → axe 1 269 j, RV = 28 % de l'axe (contrefait : antérieur à l'onglet). Panel, magnitudes réalistes : « 1 an » après 14 mois d'usage intermittent → axe 429 j, RV 85,1 %, un segment doré de 380 j ; « 2 ans » (défaut) après 26 mois → 800 j, 91,2 %, segment 751 j ; usage mensuel en fenêtre → 100 %, segments de 33 j. Observations tous les 30 j : years = 1 inchangé jusqu'à 365 j d'usage (RV 0,529 à 730 j, 0,348 à 1 095 j) ; years = 2 inchangé jusqu'à 730 j. `cache/IVHistory` : 0 fichier aujourd'hui.
- **Impact utilisateur / trading** : latent (n'apparaît qu'après un usage plus ancien que la fenêtre), certain à long terme ; le segment traversant est l'élément trompeur, la compression de l'axe est cosmétique.
- **Correctif proposé** :

```diff
@@ _render_series_chart
     if iv_history is not None and not iv_history.empty:
+        iv_history = iv_history[iv_history["date"] >= series.index[0]]
         fig.add_trace(go.Scatter(x=iv_history["date"], y=iv_history["iv"],
-                                 mode="lines+markers", ...))
+                                 mode="markers", ...))
```

  Optionnel : compter les observations hors fenêtre dans la caption L257. `connectgaps=False` nécessiterait un reindex jours ouvrés — inutile.
- **Effort estimé** : 15 min.
- **Panel** : confirmé 3/3 (m, m, m), chiffres du finder remplacés par des magnitudes réalistes.

---

### m15 — Titre Plotly recouvert par la légende horizontale (3 graphiques)

- **Ancre** : `app/vue/tabs/tab_iv_dashboard.py:100` (`_base_layout` L97-104 : `margin.t=48`, `legend(orientation='h', y=1.02, yanchor='bottom', x=0)`, titres en `title.y` auto L340/L384-389/L454).
- **Constat** : plotly 5.22.0 dessine le titre auto au milieu du marge haute calculée (`"auto"===r.y ? n.t/2`), exactement la bande qu'occupe la légende. Le panel corrige le mécanisme : **ce n'est pas un phénomène de repli sur 2 lignes** — une légende sur une seule ligne couvre déjà 79–84 % de la hauteur du titre, à toutes les largeurs ; et le template Streamlit 1.51 (`applyStreamlitTheme`, `static/js/index.CxIUUfab.js`) force `title.xanchor='left', x=0`, 16 px gras : titre et légende partagent le même bord gauche, le recouvrement est pire qu'en plotly nu.
- **Évidence mesurée** : finder (plotly.js standalone, `p1_view_render_charts.py`) : série @900 px titre y[23,43] vs légende y[18,66] (2 lignes), recouvrement 20 px ; forward @560 : 21 px (402 px horizontal) ; diff @560 : 21 px. Panel (template Streamlit ré-appliqué, Chrome headless, bboxes SVG) : série w = 1 189/1 359/1 743 → 1 ligne, titre y[18,35], légende y[21,50], 14 px = 84 % couverts, x-overlap 218 px ; série @900 → 2 lignes, 100 % ; forward 560…863 → 1 ligne, 13 px = 79 %, x-overlap 326 px ; diff 560/586/671 → 100 % ; série @1 400, 5 entrées (sans couche IV) : 15 px — **réduire le nombre d'entrées ne change rien**. Le titre est dessiné après la légende (surimpression).
- **Impact utilisateur / trading** : les titres « Série de volatilité et bandes de régime », « Vol forward 30 j vs vol courante — y = …x + … » (qui porte l'équation de régression) et « Diff de vol … » sont illisibles sur le chemin par défaut ; R² et n= sont dans les entrées de légende recouvertes. Récupérables dans le Journal replié.
- **Correctif proposé** (convention du repo : légende sous l'axe, `calibration_diagnostics.py:131` y=−0,15, `model_comparison.py:158` y=−0,2 ; valeurs ajustées par le panel pour ne pas heurter le titre d'axe x) :

```diff
@@ _base_layout
-    margin=dict(l=10, r=10, t=48, b=10),
-    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, ...),
+    margin=dict(l=10, r=10, t=48, b=80),
+    legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, ...),
```

  Alternative (légende en haut) : `title=dict(y=1, yanchor='top', pad_t=6)` + `margin.t≈100` (2 lignes de légende). « Réduire à 4 entrées » **ne corrige pas** (mesuré). Vérifier visuellement forward/diff (titre d'axe x « Vol courante »).
- **Effort estimé** : 30 min.
- **Panel** : confirmé 3/3 (**M**, m, m) — le sceptique impact proposait M (équation et R² illisibles sur le chemin par défaut) ; vote majoritaire m. Un sceptique juge y = −0,18 de l'esquisse insuffisant.

---

### m16 — Annotation « Split régimes (0.165) » en décimal sur un axe en %

- **Ancre** : `app/vue/tabs/tab_iv_dashboard.py:451` (`.3f`) vs L456 `tickformat=".0%"` ; titre forward L385-388 (intercept `.3f`, axes `.0%` L391-392) ; journal L482-490 (`.4f`/`.3f`).
- **Constat** : le port a converti axes, hovertemplates (`.2%`) et métriques (`_fmt_pct`) en pourcentage mais pas l'annotation de la vline ni l'intercept du titre ; le legacy était en décimal partout (axes compris, pas de `PercentFormatter`), donc cohérent — l'incohérence est introduite par le port. Le journal décimal = parité legacy (bloc diagnostique).
- **Évidence mesurée** : capture `iv_diff.png` : « Split régimes (0.165) » entre les ticks « 15% » et « 20% » ; panel : `intersection=0.162115`, `annotations=['Split régimes (0.162)']`, titre `'y = 0.233x + 0.124'` avec `xaxis.tickformat='.0%'` ; capture `diff_671_st.png` entre 16 % et 18 %.
- **Impact utilisateur / trading** : double conversion mentale ; aucun chiffre faux.
- **Correctif proposé** :

```diff
-    annotation_text=f"Split régimes ({intersection:.3f})",
+    annotation_text=f"Split régimes ({intersection:.1%})",
@@ titre forward
-    f"… — y = {reg['slope']:.3f}x + {reg['intercept']:.3f}"
+    f"… — y = {reg['slope']:.3f}·x + {reg['intercept']:.1%}"     # pente sans unité, intercept = vol
```

  Journal : laisser en décimal (parité legacy) ou convertir pour cohérence — 10 min dans les deux cas.
- **Effort estimé** : 10 min.
- **Panel** : confirmé 3/3 (m, m, m).

---

### m17 — Libellé « EN-DESSOUS DE LA MOYENNE » : trait d'union fautif

- **Ancre** : `app/model/iv_dashboard/analytics.py:124` (rendu verbatim par le chip « Régime courant », `tab_iv_dashboard.py:195-199`).
- **Constat** : « en dessous » s'écrit sans trait d'union (seuls au-dessous / ci-dessous / là-dessous / par-dessous en prennent) ; L120 « AU-DESSUS DE LA MOYENNE » est correct. Occurrence unique dans le repo ; aucun test ne compare la chaîne (`test_iv_dashboard_analytics.py:82-97` n'asserte que `key`/`signal_key`).
- **Évidence mesurée** : `classify_regime(0.3) → 'EN-DESSOUS DE LA MOYENNE'` ; plage déclenchante 0,2 < p ≤ 0,4 = 20,0 % du rang percentile (pas 1e-3).
- **Impact utilisateur / trading** : coquille sur le chip principal dans ~20 % des états.
- **Correctif proposé** :

```diff
-    key, label = "below", "EN-DESSOUS DE LA MOYENNE"
+    key, label = "below", "EN DESSOUS DE LA MOYENNE"
```

- **Effort estimé** : 5 min.
- **Panel** : confirmé 3/3 (m, m, m).

---

### m18 — `scripts/precommit_forbid_streamlit.py` passe au rouge à cause du docstring de `service.py`

- **Ancre** : `app/model/iv_dashboard/service.py:15` (docstring « must not import Streamlit. ») ; `scripts/precommit_forbid_streamlit.py:6-10` (`'import streamlit' in text.lower()`).
- **Constat** : scan textuel par sous-chaîne sur tout le fichier, docstrings compris. Le gate canonique (`scripts/check_mvc_integrity.py`, AST, câblé dans `.github/workflows/tests.yml:26` et `scripts/test_mvc_rules.py`) est vert ; le script legacy n'est référencé nulle part (ni CI, ni `run_tests.ps1`, ni hooks — gate orphelin). Les autres docstrings de modèle évitent la sous-chaîne (`calibration/diagnostics.py:3` « No Streamlit imports allowed »). `git grep -il 'import streamlit' HEAD~1 -- app/model` → rc=1 : régression introduite par ce commit.
- **Évidence mesurée** : `python scripts/precommit_forbid_streamlit.py` → `[ERROR] Streamlit found in model: app\model\iv_dashboard\service.py` (exit 1) ; AST : 0 nœud `Import/ImportFrom` streamlit dans `app/model` ; `'streamlit' in sys.modules` après import du service → False ; `check_mvc_integrity.py` rc=0.
- **Impact utilisateur / trading** : faux positif sur un gate local ; bruit, érosion de confiance ; aucun impact sur les chiffres.
- **Correctif proposé** (un seul côté, règle du changement chirurgical) :

```diff
--- a/app/model/iv_dashboard/service.py
-... must not import Streamlit.
+... must not depend on Streamlit.
```

  Durcir le script (`^\s*(import|from)\s+streamlit` par ligne) est une faiblesse préexistante hors périmètre ; le câbler ou le supprimer plutôt que le laisser orphelin.
- **Effort estimé** : 10 min.
- **Panel** : confirmé 3/3 (m, m, m).

---

### m19 — Bornes de validation dupliquées vue/contrôleur, clamp silencieux

- **Ancre** : `app/controller/iv_dashboard_controller.py:50` (L50-53 bornes, L15-28 `_clamp_*` sans log) ; vue L118-139 et `_DURATION_CHOICES` L29.
- **Constat** : (rv 5..120, forward 5..90, percentile 60..756, years 0,5..10) codés deux fois sans constante partagée ; le contrôleur tronque sans log ni retour. Aujourd'hui identiques ; le clamp est inatteignable depuis l'UI (les bornes des widgets sont un sous-ensemble) et le payload écho les valeurs clampées que les libellés lisent (L184, L190, L280, L350) : toute dérive future serait visible à l'écran. Convention maison (30 `min_value=` dans `app/vue/tabs`, aucun sourcé d'un contrôleur ; `hedger_v2_controller.py:206/224/230` clampe pareil).
- **Évidence mesurée** : `p1_arch_controller_and_cache.py` : `get_iv_analysis('spy', years=0.1, rv_window=2, forward_window=500, percentile_window=30)` → modèle reçoit `{'years': 0.5, 'rv_window': 5, 'forward_window': 90, 'percentile_window': 60}`, 0 warning ; panel : `(99, 1e4, 0, -5)` → `(10.0, 120, 5, 60)` ; entrées junk → défauts `(2.0, 20, 30, 252)` ; écho payload = valeurs clampées (True).
- **Impact utilisateur / trading** : nul aujourd'hui ; risque de dérive de maintenance.
- **Correctif proposé** (MVC-légal, vue → contrôleur ; sans précédent dans le repo, donc optionnel) :

```python
# iv_dashboard_controller.py
PARAM_BOUNDS = {"years": (0.5, 10.0), "rv_window": (5, 120), "forward_window": (5, 90), "percentile_window": (60, 756)}
# tab_iv_dashboard.py
lo, hi = ctrl.PARAM_BOUNDS["rv_window"]
st.number_input("Fenêtre RV (j)", min_value=lo, max_value=hi, ...)
```

  Ne pas ajouter de logging (invisible dans Streamlit, l'écho payload suffit).
- **Effort estimé** : 20 min.
- **Panel** : confirmé 2/3 — réfutation (angle impact) : clamp inatteignable depuis l'UI, écho visible, convention maison ; « pas la peine dans cette PR ».

---

### m20 — Smoke : nom de test périmé (ten vs 11), `CONTROLLERS` non étendu, pin de compte en collision

- **Ancre** : `tests/smoke/test_offline_imports.py:101` (L96 `test_all_ten_tabs_present` ; L40-48 `CONTROLLERS`).
- **Constat** : le test s'appelle encore `test_all_ten_tabs_present` et asserte `== 11` ; `CONTROLLERS` liste 7 contrôleurs sur 8 (`iv_dashboard_controller` absent, couvert seulement transitivement via `test_tab_imports_offline[tab_iv_dashboard]`). Le panel corrige la cible de collision : `origin/main` est déjà à `4e1ff90` (PR #15 mergée, 9 commits après la base `ed657c6`) avec `test_all_twelve_tabs_present`, `== 12` et `kalman_controller` ajouté ; `git merge-tree --write-tree origin/main HEAD` → **CONFLICT (content)** dans `tests/smoke/test_offline_imports.py` et `app/vue/main_app.py`. Donc pas un gate silencieusement rouge mais un conflit textuel à résoudre à la main (13 onglets, 9 contrôleurs). Le pin lui-même préexiste (`== 10`), seuls nom et inventaire sont attribuables à ce commit.
- **Évidence mesurée** : worktree 11 `tab_*.py` ; checkout principal 12 (`tab_kalman_filters.py`, `tab_rough_vol.py`, pas de `tab_iv_dashboard.py`) ; union 13 ; `grep 'iv_dashboard' tests/smoke/test_offline_imports.py` → L99 et L102 seulement ; `pytest tests/smoke/test_offline_imports.py` → 20 passed (le « 47 passed » du finder ne correspond pas à ce fichier).
- **Impact utilisateur / trading** : test-only ; conflit de merge garanti et inventaire des contrôleurs dérivant de la réalité.
- **Correctif proposé** (la convention de la branche sœur est rename + extend ; le pin est une décision à prendre une fois, au merge) :

```diff
-def test_all_ten_tabs_present():
+def test_tab_inventory():        # ou test_all_eleven_tabs_present si le pin est conservé
     mods = _tab_modules()
-    assert len(mods) == 11, mods
     assert "app.vue.tabs.tab_iv_dashboard" in mods, mods
     assert "app.vue.tabs.tab_bots" not in mods, mods
     assert "app.vue.tabs.tab_exercices" not in mods, mods
@@ CONTROLLERS
     "hedger_v2_controller",
+    "iv_dashboard_controller",
     "options_controller",
```

  Le vrai fix est le rebase sur `origin/main`.
- **Effort estimé** : 15 min.
- **Panel** : confirmé 3/3 (m, m, m).

---

### m21 — Le render guard ne rend jamais les états dégradés

- **Ancre** : `tests/integration/_iv_dashboard_render_driver.py:104` (main L104-124 : payload complet + session vide, jamais de clic).
- **Constat** : le payload seedé a toujours `current_iv`, 5 lignes d'`iv_history`, une analyse avec les deux régressions de régime ; les branches warning (vue L249), info analyse absente (L540), historique vide (L256→exit, L315→327), `reg=None` (L432, L466) et tout le chemin de soumission du formulaire (L150-168, seule façon réelle d'obtenir un payload) ne sont exécutés par aucun test. Correction du panel sur l'impact : les exemples de crash du finder (`float(None)`, reg `None`) sont déjà gardés (L228, L465) — **aucun crash latent aujourd'hui** dans ces branches (toutes rendues sans exception) ; c'est un trou de protection contre les régressions, pas un défaut.
- **Évidence mesurée** : couverture de `tab_iv_dashboard.py` sous les deux runs du driver (`p1_tests_view_branch_coverage.py`) : 85 % (pondéré branches ; 191/213 = 89,7 % en lignes), lignes manquantes `[78,80,82,89,90,92,151-157,165-168,249,432,466,540,550]`, arcs `(213->249),(256->exit),(315->327),(328->340),(431->432),(465->466),(480->495),(533->540)`. Faisabilité (`p1_tests_form_submit_feasibility.py`, `p4_g9_tests_render_degraded_probe.py`) : payload dégradé → charts 1, metrics 1, warnings 1, infos 1, exc [] ; `at.button[0].click().run()` avec contrôleur levant → `errors=['Analyse impossible : Alpaca HS (test)']` ; contrôleur OK → charts 3, state symbol `'QQQ'` ; symbole vide → warning `'Entre un symbole avant de lancer l'analyse.'` ; chaque run supplémentaire 0,1–0,6 s.
- **Impact utilisateur / trading** : un utilisateur sans clés (mode hors ligne documenté) touche exactement les branches non testées ; une régression future y arriverait à l'écran avec un guard vert — et le guard n'est de toute façon pas dans le gate CI (m22).
- **Correctif proposé** (runs 3 et 4 vérifiés faisables ; patch du module contrôleur partagé avec le script AppTest) :

```python
# _iv_dashboard_render_driver.py — Run 3 : payload dégradé (sans IV, sans analyse, historique vide)
p = _build_payload()
p.update(current_iv=None, iv_error='chaîne Alpaca inaccessible (test)', iv_regime=None, iv_minus_rv=None,
         iv_vs_series_percentile=float('nan'), iv_history=pd.DataFrame(columns=['date', 'iv']),
         analysis=None, analysis_error='Série insuffisante (test)')
at3 = AppTest.from_function(_tab_script, default_timeout=120); at3.session_state['iv_dashboard_result'] = p; at3.run()
degraded = {'exceptions': [str(e.value) for e in at3.exception], 'n_charts': len(at3.get('plotly_chart')),
            'n_warnings': len(at3.warning), 'warning_text': [w.value for w in at3.warning], 'n_infos': len(at3.info)}

# Run 4 (en dernier : le patch est global au process) : soumission avec contrôleur en échec
from app.controller import iv_dashboard_controller as ctrl
ctrl.get_iv_analysis = lambda symbol, **kw: (_ for _ in ()).throw(RuntimeError('Alpaca HS (test)'))
at4 = AppTest.from_function(_tab_script, default_timeout=120); at4.run(); at4.button[0].click().run()
submit_error = {'exceptions': [str(e.value) for e in at4.exception], 'errors': [e.value for e in at4.error]}

# test_iv_dashboard_render.py
def test_tab_degraded_payload_renders_series_only(render_result):
    d = render_result['degraded']
    assert not d['exceptions'] and d['n_charts'] == 1 and d['n_warnings'] == 1 and d['n_infos'] == 1
    assert d['warning_text'][0].startswith('IV courante indisponible')   # sinon aveugle au copy de m12
def test_form_submit_surfaces_controller_error(render_result):
    s = render_result['submit_error']
    assert not s['exceptions'] and any('Analyse impossible' in e for e in s['errors'])
```

  Aussi : `assert render_result['seeded']['n_warnings'] == 0` ; un run « soumission OK » (contrôleur appelé avec `'QQQ'` + 5 kwargs, résultat en session) ; aligner la liste d'env à stripper sur `test_app_boot.py` (6 noms) si le run 4 atteint un jour le service.
- **Effort estimé** : 1 h.
- **Panel** : confirmé 3/3 (M, m, m) — finder M et sceptique relecture M ; reproduction et impact rétrogradent à m (aucun crash latent, guard hors CI) ; vote m.

---

### m22 — Le render guard est `integration`-only : jamais exécuté par le gate CI

- **Ancre** : `tests/integration/test_iv_dashboard_render.py:26` (`pytestmark = pytest.mark.integration`) ; `.github/workflows/tests.yml:35` (`-m "unit or smoke"`).
- **Constat** : le seul test de la vue (550 lignes) est hors du seul gate automatisé ; il est hors ligne par construction (socket bloqué) et rapide. Convention préexistante : les 5 fichiers `tests/integration/*.py` (37 items, dont `test_app_boot.py`) sont tous exclus — trou de politique, pas oubli spécifique. `requirements/test.txt → runtime.txt` épingle streamlit 1.51.0 et plotly 5.22.0 : la CI a les dépendances.
- **Évidence mesurée** : `pytest -m 'unit or smoke' --co -q | grep iv_dashboard_render` → aucun (exit 1) ; `-m integration --co` → 3 items ; sélection CI : 678 collectés, 20 items iv_dashboard (analytics + contrôleur + smoke), 0 du render. Durée du guard seul : 1,44 s (finder), 1,91 / 2,02 / 2,60 s (panel, wall 3,4–3,6 s) vs 178 s pour la sélection CI ; `test_app_boot.py` : 4,30–6,05 s. Copie avec `pytestmark=[integration, smoke]` lancée en `-m 'unit or smoke'` : 3 passed in 2.30 s. Couverture vue sous la sélection CI : 13 %. L'enfant subprocess n'est pas sous coverage → `fail_under=12` inchangé dans les deux options.
- **Impact utilisateur / trading** : une régression de rendu n'est attrapée que par un run local complet ; le mot « render guard » surestime la protection.
- **Correctif proposé** :

```diff
--- a/tests/integration/test_iv_dashboard_render.py
-pytestmark = pytest.mark.integration
+pytestmark = [pytest.mark.integration, pytest.mark.smoke]   # hors ligne, boot guard d'un onglet (~3 s)
```

  Élargir la CI à `-m "unit or smoke or integration"` est viable (`test_app_boot` 4–6 s) mais c'est une décision de politique repo, à prendre pour les deux guards ensemble. Comportement AppTest + plotly sous Linux non vérifié.
- **Effort estimé** : 15 min.
- **Panel** : confirmé 3/3 (m, m, m).

---

### m23 — Oracle faible de `test_realized_vol_known_magnitude` ; test de warm-up un NaN trop court

- **Ancre** : `tests/test_iv_dashboard_analytics.py:45` (expected `0.01*sqrt(252)`, `rel=0.05`) et L55 (`rv.iloc[:18]`, commentaire « (window − 1) »).
- **Constat** : l'implémentation (`analytics.py:49-55`) documente et utilise `ddof=1` ; l'oracle est la valeur `ddof=0` ; la tolérance 5 % masque l'écart de 2,6 % et laisserait passer toute dérive d'annualisation, de définition du rendement ou de fenêtre. Le test de warm-up vérifie 18 lignes alors que 19 NaN existent : une fenêtre raccourcie à 19 passe.
- **Évidence mesurée** : `p1_tests_analytics_assertion_strength.py` : actual 0,162869, expected 0,158745, rel 2,5978 %. Passent à `rel=0.05` : `ddof=0` (0,000 %), `ddof=1·sqrt(240)` (0,125 %), window 100 (0,504 %), `ddof=0·sqrt(260)` (1,575 %), window 30 (1,710 %), window 21 (2,353 %), impl (2,598 %), window 19 (2,598 %), rendements simples (2,600 %), `sqrt(256)` (3,409 %), `ddof=0·sqrt(270)` (3,510 %), `sqrt(260)` (4,214 %) ; échouent : window 10 (5,409 %), `sqrt(365)` (23,477 %). Oracle proposé `0.01·sqrt(20/19)·sqrt(252) = 0.162869014`, écart relatif 1,4e-14. Warm-up : 29 lignes, 19 NaN, test sur 18 ; window 19 → 18 NaN → test actuel passe ; window 18 → échoue. Impact d'une dérive ddof non détectée : RV 15,17 % → 14,79 %, spread IV−RV +4,83 → +5,21 pts, percentile IV-vs-RV 0,794 → 0,825 (franchit le seuil 0,8).
- **Impact utilisateur / trading** : « RV courante » pourrait changer de 5 % sans test rouge ; le signal IV peut basculer.
- **Correctif proposé** (les deux ensemble : sur une série alternée ±1 %, window 19 donne exactement le même écart-type que 20 — seul le warm-up resserré l'attrape) :

```diff
-    expected = 0.01 * np.sqrt(252)
-    assert rv.iloc[-1] == pytest.approx(expected, rel=0.05)
+    # rendements log alternés ±1 % : écart-type échantillon sur 20 = 0.01*sqrt(20/19) (ddof=1), annualisé sqrt(252)
+    expected = 0.01 * np.sqrt(20 / 19) * np.sqrt(252)
+    assert rv.iloc[-1] == pytest.approx(expected, rel=1e-9)

-    # first (window - 1) return rows have no full window yet
-    assert rv.iloc[: 20 - 2].isna().all()
+    # 29 lignes de rendements ; les (window - 1) = 19 premières n'ont pas de fenêtre pleine, la 20e est la première valeur
+    assert rv.iloc[: 20 - 1].isna().all()
+    assert rv.iloc[20 - 1:].notna().all()
```

- **Effort estimé** : 15 min.
- **Panel** : confirmé 3/3 (m, m, m).

---

## 4. Non-findings notables (attaqué, et ça tient)

- **Forward `rolling(fw, min_periods=1).mean().shift(-fw)`** : 0 ligne à moyenne partielle parmi les lignes conservées (les moyennes partielles n'existent qu'aux 29 premières positions avant le shift et sortent de l'index) ; `forward[t] = mean(v[t+1..t+fw])` exactement ; les `fw` dernières lignes sont NaN et supprimées (`orch_probe_math.py` E, `p1_math_percentile_forward.py` §E). La crainte du prompt est réfutée.
- **252 vs 365** : RV annualisée « par an » (`sqrt(252)`, `ddof=1`) et `T = dte/365` sont la même unité ; BS en années calendaires est la convention de place. Même prix inversé avec T = 30/365 vs 21/252 : 20,000 % vs 19,863 % (0,137 pt) ; sans effet quand les greeks Alpaca sont utilisés. Horizon RV 20 j de bourse ≈ 28 j calendaires vs IV ~30 j : cohérent.
- **Look-ahead** : `current_vol`, `current_percentile`, `regime`, `iv_*` sont lus dans `series_df` avant la construction d'`analysis` ; `analysis['df']` n'alimente que les deux graphes de régression et le journal.
- **Comparabilité des deux percentiles** : `rank(pct=True, method='average')` (valeur courante incluse) vs `percentile_within` mi-rang : écart exact 1/(2n) = 0,198 pt pour n = 252 (49,21 % vs 49,01 %) ; ex æquo totaux 0,502 vs 0,5 → NORMALE/NEUTRE.
- **Seuils legacy** : `> 10` strict par régime, buckets > 0,8/0,6/0,4/0,2, signaux > 0,8 / < 0,2, `low_mask = ~high_mask` ≡ `<= intersection` — identiques à `option_trading_dashboard.py` L398-422, L487, L495.
- **`linregress` aux bords** : 2 points → slope 1 / p 0 ; y constant → slope 0 / p 1 ; 1 point, NaN, inf → inatteignables (MIN_ANALYSIS_POINTS = 30, > 10 par régime, dropna, RV finie).
- **Déviations legacy déclarées — toutes vérifiées** : (1) pas de `×√252` sur l'IV (analytics n'annualise que la RV, L55) ; (2) `min_periods=60` (L73, premier percentile non-NaN au 60e point, jour 61 d'une série croissante → 1,0) ; (3) `dropna(subset=...)` ne droppe que current/forward (L182 ; le legacy droppait aussi le warm-up) ; (4) `len < 30 → return` remplacé par `raise ValueError` (même seuil) ; (5) gardes `isfinite` ajoutées (bénin) ; (6) insight « pente ≥ 1 » au lieu de « slope > 1 » (plus exact). Aucune déviation non déclarée sur les formules elles-mêmes — sauf la sémantique du signal IV (M2).
- **XSS / sinks HTML** : symbole hostile `<img src=x onerror=alert(1)>` injecté dans le résultat et soumis via le formulaire → n'atteint que `st.caption` (allow_html=False), `st.code` (journal) et `st.error` ; aucun des 5 `st.markdown(unsafe_allow_html=True)` (`_chip`) ne contient de chaîne utilisateur (libellés constants, ints, sorties de `_fmt_pct`) ; `render_page_header` : chaînes statiques. `_render_log` passe par `st.code` → pas de sink HTML.
- **`_decode_opra`** : 13 cas (orchestrateur) + 19 cas (finder) — SPY, SPXW 5000, XSP, BRKB, AAPL1 ajusté, NDX 12345,5, minuscule, tronqués, garbage, vide, None, int, mois 13, strike 7 chiffres — tous corrects ; seule anomalie : une lettre non C/P est décodée en `put` (Alpaca n'émet que C/P).
- **Enums alpaca-py 0.12.0** : `Adjustment.SPLIT` et `DataFeed('iex')` existent ; `StockBarsRequest` accepte `adjustment` et `feed` ; `adjustment=split` est effectivement envoyé ; `end` tz-aware converti en UTC naïf / RFC3339 ; `OptionSnapshotRequest` n'existe pas dans cette version (le code n'en dépend pas : REST direct).
- **Signatures inter-modules** : `implied_vol_call(price, S0, K, t, r, q)`, `fetch_ohlc_history(symbol, *, period, interval)`, `fetch_spot_price(symbol)`, `download_options_alpaca(symbol, *, feed, max_pages, ...)` — bind OK, aucun `TypeError` avalé.
- **Secrets** : clés uniquement via `get_secret` (L44-45) ; headers jamais journalisés ; le texte des `HTTPError` contient l'URL et ses params (feed/limit/dates) mais pas la clé ; `scan_secrets.py` OK.
- **Pagination bornée** (pas de boucle infinie sur un `next_page_token` répété) ; un filtre ignoré par le serveur échoue **bruyamment** (3000 contrats d'échéances proches → « Aucun contrat entre 15 et 60 jours »), pas avec un chiffre faux (mock B).
- **Cache IV** : upsert même jour → 1 ligne ([0.19] conservé) ; dtypes au rechargement `date=datetime64[ns]`, `iv=float64` ; 8 écrivains concurrents même date → 1 ligne (threads ; race multi-process non reproduite). Le modèle n'écrit que sous `CACHE_IV_HISTORY_DIR` ; `cache/IVHistory` ignoré par `.gitignore:158` ; README à jour.
- **Sondes d'ordre** : `bridge → analytics` (27 passed) et `bridge → render` (11 passed) — aucune dépendance d'ordre ; le driver subprocess isole bien le stub Streamlit de `controller_bridge`.
- **Rationale du subprocess exacte** (clean_checks §4.5) : mesuré en un seul process — AppTest sur un script trivial avant `import app.vue.components.options.controller_bridge` rend 1 metric / 1 info / 1 markdown ; après l'import, le même AppTest lève `"'types.SimpleNamespace' object does not support the context manager protocol"` sur `with c1:` et rend 0 élément (`st.columns` remplacé par une lambda, `st.session_state` par `FakeState`, `st._codex_fake_streamlit=True` ; le garde de `controller_bridge.py:118` ne s'active que s'il existe un `ScriptRunContext`). `PYTHONIOENCODING=utf-8` est nécessaire sous Windows (l'enfant meurt en `UnicodeEncodeError` sur l'emoji sans lui). Protocole marqueur stdout robuste (marqueur doublé, dans une valeur JSON, CRLF, newline dans une exception).
- **`width="stretch"` valide en 1.51** (clean_checks §4.4) : signature installée `streamlit/elements/form.py` L239-252 (`width: Width = "content"`, accepte `"content" | "stretch" | int`) ; cohérent avec 20+ usages dans `app/vue` ; `use_container_width` absent de la vue.
- **Architecture / MVC** : `check_mvc_integrity.py` → `[MVC] OK` ; vue n'importe que `iv_dashboard_controller` + `page_utils` (motif des 10 autres onglets) ; contrôleur 58 lignes sans logique métier ; aucun cycle d'import (service→logic, logic→service, market_data→service, analytics seul, contrôleur seul, en sous-process frais sans clés) ; `autodiscover_tabs()` découvre l'onglet, label `'🌡️ Vol Implicite'` dans `TAB_GROUPS['🧩 Modèles']`, position 8/8 ; clés d'état et clés Plotly sans collision ; pyflakes 0 warning.
- **Vue** : traces fantômes à un point (`mode='lines'`, `hoverinfo='skip'`) sans segment ni hover, autoscale y inchangé ; construction des 3 figures pour 1 260 points en ~125 ms ; contraste thème sombre ≥ 6,83:1 sur tous les tons ; états vides/partiels gérés ; formatage des nombres cohérent côté métriques.
- **Tests** : marqueurs `unit`/`integration` conformes à `--strict-markers` ; `test_analyze_forward_vol_mean_reverting_series` déterministe (slope1 0,2016 sur 20 runs, marge 0,80 à 1) ; le test contrôleur patche correctement `ctrl._svc.get_iv_dashboard_data` ; socket bloqué avant tout import numpy/pandas/streamlit dans le driver.

---

## Annexe A — Baseline brute avant/après

Sorties brutes avant toute revue (commit `f2d0812`, interpréteur `.venv/Scripts/python.exe`) :

```
== MVC ==      [MVC] OK — no integrity violations detected.                       exit=0
== SECRETS ==  [secret-scan] OK — no secrets found in tracked files.             exit=0
== UNIT+SMOKE ==  pytest -m "unit or smoke" -q
   676 passed, 2 skipped, 50 deselected, 13 warnings in 99.54s (0:01:39)
   SKIPPED tests\quant\test_dqn_checkpoint.py:52 (no shipped checkpoint)
   SKIPPED tests\smoke\test_entry_points_import.py:83 (optional dependency 'torch')
== FULL not slow ==  pytest tests -q -m "not slow"
   696 passed, 2 skipped, 16 deselected, 13 warnings in 181.04s (0:03:01)
== scripts/ ==  pytest scripts -q -m "not slow"  -> 14 passed
   => 696 + 14 = 710 : la claim « 710 passed » de l'auteur est exacte, mais elle incluait `scripts/` (commande cloud : `pytest tests scripts -q -m "not slow"`).
== analytics alone ==  tests/test_iv_dashboard_analytics.py -> 19 passed in 2.25s
== render alone ==     tests/integration/test_iv_dashboard_render.py -> 3 passed in 4.66s
== boot guard ==       tests/integration/test_app_boot.py -> 3 passed in 7.51s
== ORDER PROBE bridge then analytics ==  tests/test_controller_bridge_helpers.py tests/test_iv_dashboard_analytics.py -> 27 passed
== ORDER PROBE bridge then render ==     tests/test_controller_bridge_helpers.py tests/integration/test_iv_dashboard_render.py -> 11 passed
```

**Baseline « après » : identique.** La revue est review-only — `git status --porcelain` ne montre aucun fichier traqué modifié ; seules écritures : `scripts/review_iv_dashboard/` et le présent document. Il n'y a donc pas de second run à présenter. (Un PNG laissé à la racine par un agent, `iv_series.png`, a été déplacé dans `scripts/review_iv_dashboard/p1_view_iv_series.png`.)

---

## Annexe B — Phase 3 live (§4.6) : sautée, motif mesuré

**Motif du skip.** Aucun `.env` ni `secrets.toml` dans le repo `-fix` ni dans `~/.streamlit`. Un `.env` existe dans l'ancien clone `Dev/PaperTradingApp-1` (mtime 2026-03-21, `APCA_API_BASE_URL=https://paper-api.alpaca.markets`, clé `PK…` 26 caractères, secret 44 caractères). Ces clés sont **révoquées** : `GET https://paper-api.alpaca.markets/v2/clock` avec en-têtes → HTTP 401 (page HTML nginx « 401 Authorization Required »), idem `data.alpaca.markets` (bars, snapshots, `feed=opra`, param bogus). Les en-têtes de réponse (`X-Request-ID`, `Access-Control-Allow-Headers: Apca-Api-Key-Id, Apca-Api-Secret-Key`) prouvent que l'edge Alpaca est bien atteint (pas d'interception réseau) ; GitHub et Yahoo répondent normalement depuis la même machine.

Extrait abrégé de `scripts/review_iv_dashboard/orch_live_alpaca.out.txt` (blocs HTML 401 répétés et sondes « unfiltered » / « feed=opra » élidés, marqués « … » ; le fichier complet fait foi) :

```
keys loaded: yes (redacted) | base url is paper: True
today (local): 2026-08-21 | utc now: 2026-08-21T08:04+00:00

=== (a) filtered snapshot fetch: are server-side filters honored? ===
spot via fetch_spot_price: None
HTTP 401 <html>
<head><title>401 Authorization Required</title></head>
<body>
<center><h1>401 Authorization Required</h1></center>
<hr><center>nginx</center>
</body>
</html>

filtered: status=401 pages=0 contracts=0 latency=0.80s
bogus param probe: HTTP 401 (400 => unknown params rejected; 200 => unknown params silently ignored)

=== (b) greeks / IV presence on the indicative feed today ===
snapshot fetch failed: 401 Client Error: Unauthorized for url: https://data.alpaca.markets/v1beta1/options/snapshots/SPY?feed=indicative&limit=1000&expiration_date_gte=2026-09-05&expiration_date_lte=2026-10-20

=== (c) daily bars: default feed vs IEX, end = now - 16 min ===
feed=None : FAIL <html> ... 401 Authorization Required ...
feed=iex  : FAIL <html> ... 401 Authorization Required ...
feed=sip  : FAIL <html> ... 401 Authorization Required ...
no-end param: FAIL <html> ... 401 Authorization Required ...
```

**Conséquences (non mesuré)** : (a) filtres serveur des snapshots → m1 et l'hypothèse « filtres honorés » restent *unverified-live* ; (b) présence des greeks sur le flux `indicative` aujourd'hui (décide de la fréquence réelle du chemin inversion de M3) ; (c) barres daily free-plan / `end − 16 min` ; (d) IV headline vs référence externe ±3 pts.

**Ce que le run a quand même mesuré** — section (d), bout-en-bout via le fallback :

```
=== (d) end-to-end get_iv_dashboard_data('SPY') vs independent ATM IV ===
e2e 6.8s | source=fallback (Stooq/Yahoo) points=501 current RV=0.1312 pct=0.579 regime=NORMALE
current_iv: None
iv_vs_series_percentile: nan | iv_regime: None | iv_minus_rv: None
log:
   - Barres alpaca indisponibles : <html>
<head><title>401 Authorization Required</title></head>
<body>
<center><h1>401 Authorization Required</h1></center>
<hr><center>nginx</center>
</body>
</html>

   - Barres alpaca (iex) indisponibles : <html>
... (8 lignes identiques)
   - 534 barres daily via fallback Stooq/Yahoo.
   - Vol réalisée courante (20 j, annualisée) : 0.1312 (13.12%).
   - Spot indisponible : impossible de sélectionner les strikes ATM.
analysis_error: None
reg_forward slope=0.487 r2=0.354 n=471 intersection=0.1443 n_high=153 n_low=318

cache file: ...\cache\IVHistory\iv_daily_SPY.csv exists: False
```

Soit : 534 barres daily, 501 points affichés, RV courante 20 j = 0,1312 (13,12 %), percentile 0,579 → régime NORMALE, régression forward pente 0,487, R² 0,354, n = 471, intersection 0,1443, n_high = 153 / n_low = 318 ; `current_iv=None` ; les deux blocs HTML de 8 lignes dans le log sont la source de m6 ; aucun fichier cache écrit (pas d'IV).

**Mesure live complémentaire (publique, sans clé)** : Yahoo v8 chart SPY `range=2y/3y/4y/5y/6y` → HTTP 200, `error=null`, 501 / 753 / 1 003 / 1 254 / 1 506 barres → le candidat `yahoo-period-string` est réfuté.

**Rejouer** : `.venv/Scripts/python.exe scripts/review_iv_dashboard/orch_live_alpaca.py <chemin/.env>` avec des clés paper valides — lecture seule, jamais d'ordre, clés chargées en mémoire uniquement, sorties expurgées ; mesure (a)–(d).

---

## Annexe C — Orchestration et scripts de mesure

**Statistiques.**
- Phase 1 : 1 workflow, 5 finders (un par §4.1–4.5), 645 k tokens, 216 appels d'outils → 33 candidats (0 C / 9 M / 24 m) ; 29 après fusion de 4 doublons inter-dimensions ; + 1 candidat orchestrateur (`html-exception-text-in-log`) = 30 soumis au panel.
- Phase 2 : 3 sondes de l'orchestrateur (`orch_probe_math.py`, `orch_probe_iv_bias.py`, `orch_probe_plumbing.py`) écrites indépendamment des agents ; 26 scripts `p1_*.py` des finders.
- Phase 3 : skip (Annexe B).
- Phase 4 : 1 workflow, 27 sceptiques = 9 groupes thématiques × 3 angles (relecture du code / reproduction indépendante / impact-sévérité), 2,58 M tokens, 601 appels d'outils, 60 sondes déclarées (58 fichiers `p4_*.py` + 1 sortie `.out.txt` + 1 ré-exécution d'un script `p1_`).

**Règle de survie / sévérité** : un finding survit si ≥ 2 des 3 sceptiques ne le réfutent pas ; sévérité finale = vote majoritaire (en cas d'égalité, la plus haute). Résultat : 29/30 survivent au panel (1 tué 3/3) ; 1 survivant supplémentaire tué ensuite par une mesure live de l'orchestrateur (Yahoo) ; **final 28 findings — 0 C, 5 M, 23 m**.

**Candidats tués (2).**
1. `hardcoded-dark-palette-light-theme` (`tab_iv_dashboard.py:31`) — tué 3/3. L'arithmétique de contraste se reproduit (sur #FFFFFF : #e5e7eb 1,24, #fbbf24 1,67… ; sur #0E1117 tous ≥ 6,83) mais la prémisse « le menu Réglages permet de choisir Light » est fausse pour Streamlit 1.51.0 : `.streamlit/config.toml` définit `[theme]` (base='dark') sans `[theme.light]/[theme.dark]` ; `processThemeInput()` appelle `addThemes(N, {keepPresetThemes: false})` et le selectbox du dialogue Réglages est rendu `disabled` pour un thème custom. De plus l'app est dark-only au niveau repo (`main_app.py:17` `alt.themes.enable('dark')`, `theme_animated.css` force le fond) : un thème clair casserait tous les onglets, pas celui-ci.
2. `yahoo-period-string` (`service.py:152`) — tué par mesure de l'orchestrateur : Yahoo accepte `range=3y/4y/6y` (HTTP 200, 753/1 003/1 506 barres). Deux sceptiques n'avaient pas pu réfuter hors ligne ; le troisième l'a réfuté depuis les artefacts du run live (534 barres servies avec `period='3y'` alors que Stooq était en panne).

**Doublons fusionnés (4)** : VRP ×2 (`iv-signal-vrp-bias` + `iv-signal-chip-structurally-biased` → M2) ; fenêtre percentile ×2 (`percentile-label-vs-effective-window` + `percentile-window-not-in-lookback` → m9) ; inventaire smoke ×2 (`smoke-tab-count-and-inventory` + `smoke-controllers-list-not-updated` → m20) ; service non testé ×2 (`service-zero-unit-tests` + `service-layer-untested-payload-contract-duplicated` → M5).

**Inventaire des scripts** (`scripts/review_iv_dashboard/`).

*Orchestrateur (`orch_*`)* : `orch_probe_math.py` (A–I : série constante, 2 niveaux, sous-ensembles dégénérés, `shift(-fw)`, percentiles, G/H, intersection hors plage), `orch_probe_iv_bias.py` (parité r/q, ask-only, croisés, T intraday, 252/365), `orch_probe_plumbing.py` (OPRA, enums alpaca-py, signature fallback, cache, XSS), `orch_live_alpaca.py` + `orch_live_alpaca.out.txt`, `orch_context_notes.md`, `orch_dossier_brut.md`.

*Finders Phase 1 (`p1_*`, 26)* :
- §4.1 maths : `p1_math_degenerate_regressions.py` (m8), `p1_math_regime_subset_degenerate.py` (m7), `p1_math_service_constant_series.py` (m7), `p1_math_percentile_forward.py` (non-findings shift/percentile), `p1_math_effective_window.py` (m9), `p1_math_rv_edges_epistemics.py` (M2, m10, m11).
- §4.2 plomberie : `p1_alpaca_plumbing_opra_decode.py`, `p1_alpaca_plumbing_iv_bias.py` + `_mix.py` (M3, m3), `p1_alpaca_plumbing_snapshot_mock.py` (M1, m1, m2), `p1_alpaca_plumbing_cache.py` (M4, m4), `p1_alpaca_plumbing_bars_request.py` (enums), `p1_alpaca_plumbing_warmup.py` (m9).
- §4.3 architecture : `p1_arch_import_cycles.py` (m18), `p1_arch_tab_wiring.py`, `p1_arch_controller_and_cache.py` (m5, m19, M5).
- §4.4 vue : `p1_view_sinks_and_state.py` (M2, m12, m13, XSS), `p1_view_contrast_bias_ranges.py` (M2, m14), `p1_view_render_charts.py` (m15, m16), `p1_view_iv_series.png`.
- §4.5 tests : `p1_tests_service_sketches.py` (M5, 24 passed), `p1_tests_cache_dir_patch_target.py` (M5), `p1_tests_view_branch_coverage.py` (m21), `p1_tests_form_submit_feasibility.py` (m21), `p1_tests_analytics_assertion_strength.py` (m23), `p1_tests_driver_protocol.py`, `p1_tests_bridge_stub_apptest.py`.

*Panel Phase 4 (`p4_*`, 58 fichiers `.py`)* :
- Par finding (`p4_<id>_repro.py`) : `stale-chain-cache-as-current-iv`, `iv-signal-vrp-bias`, `parity-r-q-zero-bias` (+`_repro2`), `local-date-vs-exchange-date`, `service-zero-unit-tests`, `snapshot-page-cap-silent`, `fallback-chain-cannot-reach-30dte-and-masks-root-cause`, `ask-only-and-crossed-mids`, `iv-cache-corrupt-file-silent-loss`, `iv-history-filename-unsanitized-symbol`, `html-exception-text-in-log`, `linreg-scipy-valueerror-coincidence`, `regime-split-out-of-range`, `percentile-label-vs-effective-window`, `duplicate-date-crash`, `rv-bad-close-silent-drop`, `iv-disabled-says-alpaca-inaccessible`, `stale-result-under-error-and-params-drift`, `iv-history-overlay-unbounded-x-range`, `chart-title-overprinted-by-legend`, `split-annotation-units-mismatch`, `copy-en-dessous-hyphen`, `forbid-streamlit-script-trips-on-docstring`, `controller-bounds-duplicated-silent-clamp`, `smoke-tab-count-and-inventory`, `render-guard-happy-path-only`, `render-guard-not-in-ci-gate`, `weak-realized-vol-oracle`, `hardcoded-dark-palette-light-theme`, `yahoo-period-string`.
- Par groupe thématique : G1 maths `p4_g1_math_code_reading.py`, `p4_impact_math.py`, `p4_regime_split_param_sweep.py` ; G2 épistémique `p4_g2_epistemics_probe.py`, `p4_impact_g2_epistemics.py`, `p4_impact_g2_smooth_iv.py` ; G3 méthode IV `p4_g3_ivmethod_skeptic.py` + `_b.py`, `p4_g3_mid_fallbacks.py`, `p4_g3_page_cap_e2e.py`, `p4_g3_parity_median.py` ; G4 fallback `p4_fallback_probe.py`, `p4_fallback_impact.py`, `p4_yahoo_range_evidence.py` ; G5 cache `p4_cache_group_code_reading.py`, `p4_g5_cache_impact.py`, `p4_real_cache_scan.py` ; G6 vue A `p4_g6_viewA_probe.py`, `p4_g6_viewA_impact.py` ; G7 vue B `p4_g7_viewB_code_reading.py`, `p4_g7_viewB_title_legend_overlap.py` ; G8 archi `p4_g8_arch_probe.py`, `p4_g8_arch_impact_probe.py` ; G9 tests `p4_g9_tests_probe.py` (+`.out.txt`), `p4_g9_tests_oracle_probe.py`, `p4_g9_tests_ci_gate_probe.py`, `p4_g9_tests_render_degraded_probe.py`.

*Données machine* : `phase1_findings.json`, `phase1_merged.json`, `phase4_verdicts.json`, `report_numbering.json`.

---

## Annexe D — Ce qui n'a PAS été couvert

- Tout le §4.6 live (Annexe B) : acceptation réelle par `GET /v1beta1/options/snapshots/{underlying}` des paramètres `expiration_date_gte/lte`, `strike_price_gte/lte`, `limit=1000`, `feed` (alpaca-py 0.12.0 n'a aucun modèle de requête options pour servir de proxy ; les noms viennent de mémoire des releases ultérieures) ; présence des greeks/`impliedVolatility` sur `indicative` ; code 403 OPRA ; `end − 16 min` sur barres daily free-plan (le code garde la barre en cours si renvoyée, comme le legacy IB `endDateTime=''`) ; 429 sur l'appel filtré (pas de retry, chute sur le fallback) ; IV headline vs référence externe.
- Ordre de tri réel des snapshots Alpaca (l'hypothèse « OPRA ascendant » rend la troncature m1 inoffensive).
- Fréquence empirique IV > RV (VRP) sur SPY/QQQ réels — M2 repose sur AAPL (RV réelle, IV proxy) et sur des séries synthétiques ; SPY non mesurable hors ligne (pas de clôtures SPY en cache).
- Splits/dividendes : `Adjustment.SPLIT` est envoyé sur le chemin Alpaca ; ajustement des historiques Stooq/Yahoo du fallback non vérifié (un split 4:1 non ajusté ferait passer RV20 de 0,115 à 4,963).
- Existence réelle de dates dupliquées (doublon live Yahoo en séance) ou de clôtures ≤ 0 dans les sources — 0 cas sur 10 415 lignes Stooq en cache, rien de plus.
- Rendu visuel réel dans un navigateur en session Streamlit : m15 mesuré en Chrome headless sur le SVG avec le template Streamlit ré-appliqué, pas dans le frontend ; `streamlit run` réel non exécuté (AppTest + sonde d'autodiscovery seulement).
- Bascule effective vers le thème clair dans le frontend 1.51 avec thème custom (raisonnée depuis le bundle JS, non cliquée).
- Concurrence réelle de deux sessions Streamlit (multi-process) écrivant le même CSV IV — seuls des threads testés ; read-modify-write sans verrou.
- Exécution sous Linux/CI : encodage, AppTest + plotly en CI — mesuré sous Windows uniquement.
- `_fetch_closes_alpaca` avec un faux `StockHistoricalDataClient` (feed défaut → retry IEX, mise en forme du DataFrame) : contourné par monkeypatch du wrapper, non testé unitairement.
- Run complet de la suite avec ordre aléatoire / xdist (pas de pytest-randomly disponible) ; durée de `test_app_boot.py` non mesurée par le finder §4.5 (mesurée par l'orchestrateur : 3 passed in 7,51 s, Annexe A ; par le panel : 4,3–6 s).
- Perf : import lazy de `alpaca.data.*` et d'`app.model.options.logic` (~1 500 lignes) au premier clic non chronométré ; latence de l'appel Alpaca bloquant le rendu pendant le spinner non mesurée.
- Aucune cassette réseau : les fixtures des sketches suivent les noms de champs du code, pas un enregistrement réel.

---

## Annexe E — Hors périmètre (préexistant, une ligne chacun)

- `app/model/options/logic.py::download_options_alpaca` n'envoie jamais de `limit` (page serveur 100) — c'est ce qui rend le fallback `max_pages=3` inutile (m2).
- `download_options_alpaca` renvoie un `options_alpaca_{SYM}.csv` de n'importe quel âge sur échec — comportement hérité par le nouveau fallback (M1).
- `options/logic.py:1280` construit aussi un nom de cache depuis le symbole non sanitisé — même faiblesse que m5.
- `market_data.fetch_spot_price` utilise le dernier trade IEX par défaut (peut dévier du SIP sur les noms minces) ; cache Stooq permanent = seconde source périmée sur le chemin M1.
- `market_data._fetch_yahoo_ohlc` passe `period` tel quel en `range` Yahoo sans validation.
- `market_data.fetch_ohlc_history` (fallback) : déduplication/tri des dates non vérifiés.
- `app/model/calibration/implied_vol.py::implied_vol_call` : précision/bornes de l'inversion non auditées.
- alpaca-py épinglé à 0.12.0 sans `alpaca.data.historical.option` / `OptionChainRequest` — l'approche REST brute est forcée.
- `scripts/precommit_forbid_streamlit.py` n'est branché ni en CI ni en hook git — gate orphelin (seul `check_mvc_integrity.py` l'est).
- `docs/architecture_mvc.md:66-72` énumère des contrôleurs disparus (portfolio, buy_sell, backtest, hedger, dashboard) — doc déjà obsolète.
- Les env vars de feed non secrètes (`ALPACA_STOCK_DATA_FEED`, `ALPACA_OPTION_CHAIN_CACHE_TTL_SEC`) ne sont pas documentées dans `.env.example` — la nouvelle `ALPACA_OPTION_DATA_FEED` prolonge la lacune.
- Le sous-process Python sort en cp1252 sous Windows : les labels emoji font planter tout `print` sans `PYTHONIOENCODING=utf-8` ; le warning de `record_iv_observation` s'affiche en mojibake sur console cp1252.
- La palette sombre codée en dur est partagée par les autres onglets (« aligned with the other tabs ») ; le thème clair casserait toute l'app, pas cet onglet.
- Les hovertemplates `%{x|%d %b %Y}` affichent les mois en anglais (locale Plotly) dans une UI française — cosmétique, non retenu.
- Le gate CI excluant tout test `integration` est une politique préexistante ; `test_app_boot.py` est dans la même situation que le render guard (m22).
- `tests/integration/test_app_boot.py` embarque la même fixture subprocess/marqueur que le nouveau driver (~35 lignes dupliquées) ; un helper partagé serait possible.
- Le driver de rendu asserte `n_metrics >= 3` (3 `st.metric` attendus exactement) — assertion lâche, notée non reportée.
- `fetch_current_atm_iv` appelle `fetch_spot_price` hors de tout `try` (`service.py:352`) ; `fetch_spot_price` est lui-même défensif (None sur échec), aucun chemin de crash trouvé.
- Outillage : un dossier `.playwright-mcp/` créé dans la worktree par le MCP Playwright pendant les captures a été déplacé vers le scratchpad ; `git status` propre.
