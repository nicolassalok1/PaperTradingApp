# Notes de l'orchestrateur pour la rédaction du livrable

Livrable : `docs/review-2026-08-iv-dashboard-alpaca.md`, en français (convention maison — voir le style de `docs/review-2026-08-rough-heston-qrhplus.md` du repo principal si besoin d'un modèle, mais la structure imposée est celle du §6 du prompt).

## 1. Récit (à reprendre dans l'en-tête / contexte)

- Commande : prompt `prompt-review-iv-dashboard.md` (ultracode, review-only) visant le commit `cd278ec` — « Add 🌡️ Vol Implicite tab: IV regime dashboard ported from IB/Tkinter to Alpaca » (12 fichiers, +1887/−2).
- `cd278ec` n'existait ni en local ni sur `origin` : il a été créé dans une session Claude cloud (`cse_018a8jdADUpTXJJCTHoKTB5b`) dont le `git push` a été refusé deux fois (403, repo hors des sources autorisées du proxy git). La session avait livré `iv-dashboard-alpaca.patch` (73 170 octets, `git diff --staged`) et un zip.
- Récupération : worktree `.claude/worktrees/feature+iv-dashboard-alpaca` créé depuis `origin/main` (`ed657c6`, merge PR #14), `git apply --check` OK, patch appliqué, commit local **`f2d0812`** : 12 fichiers, 1887 insertions, 2 suppressions — empreinte strictement identique à `cd278ec`. Tous les ancrages de lignes du §4 du prompt ont été re-vérifiés dans le working copy et correspondent à la ligne près (aucune divergence, pas de ré-ancrage nécessaire).
- Interpréteur : `.venv/Scripts/python.exe` du repo principal — Python 3.11.15, numpy 1.26.4, scipy 1.11.4, pandas 2.1.4, streamlit 1.51.0, alpaca-py 0.12.0 (versions épinglées confirmées).
- Review-only respecté : `git status --porcelain` ne montre aucun fichier traqué modifié ; seules écritures : `scripts/review_iv_dashboard/` (+ le doc final). Un PNG laissé à la racine par un agent (`iv_series.png`) a été déplacé dans `scripts/review_iv_dashboard/p1_view_iv_series.png`.

## 2. Baseline §5 (sorties brutes, avant toute revue)

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
Aucune sonde d'ordre n'a révélé de dépendance (le driver subprocess isole bien le stub Streamlit de `controller_bridge`).

Baseline « après » : identique — la revue n'a modifié aucun fichier traqué (review-only), donc pas de second run à présenter ; le dire explicitement.

## 3. Phase 3 live (§4.6) — SKIP, motif mesuré

- Aucun `.env` ni `secrets.toml` dans le repo `-fix` ni dans `~/.streamlit`. Un `.env` existe dans l'ancien clone `Dev/PaperTradingApp-1` (mtime 2026-03-21, `APCA_API_BASE_URL=https://paper-api.alpaca.markets`, clé `PK…` 26 car., secret 44 car.).
- Ces clés sont **révoquées** : `GET https://paper-api.alpaca.markets/v2/clock` avec en-têtes → HTTP 401 (page HTML nginx « 401 Authorization Required »), idem `data.alpaca.markets` (bars, snapshots, feed=opra, param bogus). Les en-têtes de réponse (`X-Request-ID`, `Access-Control-Allow-Headers: Apca-Api-Key-Id, Apca-Api-Secret-Key`) prouvent que l'edge Alpaca est bien atteint (pas d'interception réseau). GitHub et Yahoo répondent normalement depuis la même machine.
- Conséquence : (a) filtres serveur des snapshots **non vérifiés en live** → le finding m1 (page cap) et l'hypothèse « filtres honorés » restent *unverified-live* ; (b) présence des greeks sur le flux `indicative` aujourd'hui : non mesuré ; (c) barres daily free-plan / `end − 16 min` : non mesuré ; (d) IV headline vs référence externe ±3 pts : non mesuré.
- Ce que le run live a quand même produit (`orch_live_alpaca.out.txt`) : `get_iv_dashboard_data('SPY', years=2)` en 6,8 s via le **fallback Stooq/Yahoo** (534 barres daily, 501 points affichés), RV courante 20 j = 0,1312 (13,12 %), percentile 0,579 → régime NORMALE, régression forward pente 0,487, R² 0,354, n=471, intersection 0,1443, n_high=153 / n_low=318 ; `current_iv=None` (« Spot indisponible : impossible de sélectionner les strikes ATM. ») ; le log contient deux blocs HTML bruts de 8 lignes (`<html>…<center>nginx</center>…`) — c'est la source du finding m6. Aucun fichier cache écrit (pas d'IV).
- Le script `scripts/review_iv_dashboard/orch_live_alpaca.py <chemin/.env>` est prêt à rejouer les 4 mesures (a)–(d) avec des clés paper valides : lecture seule, jamais d'ordre, clés chargées en mémoire uniquement, sorties expurgées.
- Mesure live complémentaire (publique, sans clé) : Yahoo v8 chart SPY `range=2y/3y/4y/5y/6y` → HTTP 200, error=null, 501/753/1003/1254/1506 barres → le candidat `yahoo-period-string` est réfuté.

## 4. Orchestration réellement exécutée (à décrire en annexe)

- Phase 1 : 1 workflow, 5 finders (un par §4.1–4.5), 645 k tokens, 216 appels d'outils → 33 candidats (0 C / 9 M / 24 m), 30 après fusion des doublons inter-dimensions (VRP ×2, fenêtre percentile ×2, inventaire smoke ×2, service non testé ×2) + 1 candidat orchestrateur (`html-exception-text-in-log`).
- Phase 2 : sondes de l'orchestrateur (`orch_probe_math.py`, `orch_probe_iv_bias.py`, `orch_probe_plumbing.py`) écrites indépendamment des agents ; 26 scripts `p1_*.py` des finders.
- Phase 3 : skip (ci-dessus).
- Phase 4 : 1 workflow, 27 sceptiques = 9 groupes thématiques × 3 angles indépendants (relecture du code / reproduction indépendante / impact-sévérité), 2,58 M tokens, 601 appels d'outils, 60 scripts `p4_*.py`. Règle : un finding survit si ≥ 2 des 3 sceptiques ne le réfutent pas ; sévérité finale = vote majoritaire (en cas d'égalité, la plus haute).
- Résultat : 29/30 survivent au panel ; `hardcoded-dark-palette-light-theme` tué 3/3 (l'app est dark-only : `.streamlit/config.toml` base='dark', Streamlit 1.51 retire les presets Light/Dark quand un thème custom est défini ; prémisse fausse) ; `yahoo-period-string` tué par mesure de l'orchestrateur (Yahoo accepte 3y). **Final : 28 findings — 0 C, 5 M, 23 m.**
- Scripts de mesure : tout est sous `scripts/review_iv_dashboard/` (lister les familles orch_*, p1_*, p4_*, les JSON `phase1_findings.json`, `phase1_merged.json`, `phase4_verdicts.json`, `report_numbering.json`, et `orch_live_alpaca.out.txt`).

## 5. Résultats des sondes de l'orchestrateur (chiffres à réutiliser tels quels)

`orch_probe_math.py` :
- A. Série constante (120 pts) → `ValueError: Cannot calculate a linear regression if all x values are identical` ; `isinstance(exc, ValueError)` → True → attrapée par service.py L586.
- B. Clôtures alternant 1,00/1,01 tous les 7 jours sur 300 j → RV à 2 valeurs distinctes (n=280) → la régression globale passe, la régression **par régime** lève la même ValueError → analysis=None (reg_forward/reg_diff pourtant calculables sont perdus).
- C/D. Sous-ensemble de régime à x identiques mais tombant à n=0 ou régime à 2 niveaux → pas d'exception.
- E. Forward `rolling(fw, min_periods=1).mean().shift(-fw)` : **0 ligne à moyenne partielle** parmi les lignes conservées (les moyennes partielles sont expulsées par le shift) → la crainte du prompt (« moyennes partielles sur les fw−1 premières lignes ») est un **non-finding** ; forward[t] = moyenne de v[t+1..t+fw] exactement.
- F. Percentile roulant `rank(pct=True)` (dernier = 0,5040) vs `percentile_within` (0,5020 incl. / 0,5020 excl.) : conventions comparables à ~0,2 pt près ; ex æquo totaux : 0,502 vs 0,5.
- G. IV 17 % dans une RV uniforme 12–18 % → percentile 0,857 → « VOL ÉLEVÉE / MEAN REVERSION ↓ ATTENDUE » ; IV 19 % → 1,000.
- H. min_periods=60 : premier percentile non-NaN au 60e point ; jour 61 d'une série croissante → 1,0.
- I. Intersection hors plage : intersection 3,2427 pour des données [0,0999 ; 0,2007] → n_high=0, reg_high=None.

`orch_probe_iv_bias.py` (SPY S=640, r=4 %, q=1,3 %, σ vraie 16 %, dte=30, T=0,0822) :
- Biais par type (pipeline exact du code, parité r=0/q=0, spot) : K/S=0,95 call +271 bp / put −48 ; 0,97 : +168 / −62 ; 0,99 : +112 / −84 ; **1,00 : +95 / −98** ; 1,01 : +82 / −118 ; 1,03 : +62 / −187 ; 1,05 : +49 / −361.
- Médiane calls+puts sur la bande ±5 % (14 contrats) : **+1 bp** ; |K/S−1|<1,5 % (6 contrats) : −1 bp → annulation par symétrie, fragile (déséquilibre de comptage ou un seul côté → ~1 pt de vol).
- Sensibilité ATM put : r=0 % → +46 bp ; r=2 % → −27 ; r=4 % → −98 ; r=5 % → −133. q=0 → put −142 / call +145 ; q=3 % → put −40 / call +31.
- Ask-only (bid absent/0 → ask) : spread 0,02 $ → +1 bp ; 0,10 $ → +7 ; 0,30 $ → +20 ; 1,00 $ → +68 (mid ATM ≈ 12,41 $).
- `_snapshot_mid` accepte une cotation croisée (bid 5,50 / ask 5,00 → 5,25) ; bid 0 / ask 5,20 → 5,20 ; bid 0 / ask 0 / trade 4,00 → 4,00.
- `T = max(dte,1)/365` ignore l'intraday : −7 bp (dte 30), −15 bp (dte 15) à la clôture.
- 252 vs 365 : **non-finding** (RV annualisée « par an » et T en années calendaires sont la même unité).

`orch_probe_plumbing.py` :
- `_decode_opra` : 13 cas (SPY, SPXW 5000, XSP, BRKB, AAPL1 ajusté, NDX strike 12345,5, minuscule, tronqués, garbage, vide) → tous corrects ; seule anomalie : un type inconnu `X` est décodé silencieusement en `put` (sans impact réel).
- alpaca-py 0.12.0 : `Adjustment.SPLIT` et `DataFeed('iex')` existent ; `StockBarsRequest` accepte `adjustment` et `feed` ; `OptionSnapshotRequest` n'existe pas dans cette version (le code n'en dépend pas : appel REST direct).
- `download_options_alpaca(sym, feed=..., max_pages=3)` : signature valide (bind OK).
- Cache : upsert même jour OK ([0.19]) ; rechargement dtypes date=datetime64[ns], iv=float64 ; fichier vide → « No columns to parse from file » → warning, observation perdue ; fichier sans colonne `date` → « Unalignable boolean Series provided as indexer » → le garde `df.get('date', Series vide)` est **mort**, observation perdue ; `DataFrame.get` renvoie bien une Series.
- `_render_log` rend le journal via `st.code(...)` → pas de sink HTML → **XSS non-finding** (le symbole hostile `<img src=x onerror=alert(1)>` ressort en texte dans RuntimeError / st.error, échappé par Streamlit).

## 6. Verdicts par dimension (proposition, à reprendre dans la table)

| Dimension | Verdict | Justification courte |
|---|---|---|
| §4.1 Maths | acceptable avec réserves | aucune erreur de formule ; déviations legacy déclarées tiennent ; `shift(-fw)` et 252/365 = non-findings ; M2 (épistémique VRP) naît ici mais s'affiche dans la vue ; m7–m11 |
| §4.2 Plomberie Alpaca | corrections requises | M1, M3, M4 touchent le chiffre affiché ou persisté ; m1–m6 ; contrat API **unverified-live** |
| §4.3 Architecture/MVC | acceptable en l'état | gates verts, vue → contrôleur seul, pas de Streamlit dans modèle/contrôleur, pas de cycle d'import ; m18–m20 |
| §4.4 Vue | acceptable avec réserves | pas de sink HTML ; m12–m17 (+ chip M2) |
| §4.5 Tests | corrections requises | M5 (service sans test direct, 8 % de couverture sous le gate CI) ; m21–m23 |
| §4.6 Live | non vérifié | clés révoquées ; script prêt |

**Verdict global : corrections requises** (aucun bloquant : aucun crash de l'onglet, aucune perte de données dans le chemin nominal, aucun chiffre affiché faux dans les conditions par défaut avec Alpaca joignable ; mais M1–M4 rendent l'IV affichée/persistée fausse ou trompeuse dans des conditions réalistes — API indisponible, flux sans greeks, utilisateur hors fuseau US — et M5 laisse tout cela sans filet).

## 7. Déviations legacy déclarées — vérifiées

- Pas de `×√252` sur l'IV : confirmé (analytics.py n'annualise que la RV, L55).
- `min_periods=60` : confirmé (L73, via DEFAULT_PERCENTILE_MIN_PERIODS=60) ; effet mesuré (section 5.H) ; cf. m9 pour l'étiquetage.
- Échantillon de régression ne droppe que current/forward : confirmé (L182 dropna subset) ; le legacy droppait aussi le warm-up du percentile.
- Formules identiques au legacy (`option_trading_dashboard.py` L331, L446, L467–500) : rank(pct) 252, rolling(30,min_periods=1).mean().shift(−30), linregress, intersection intercept/(1−slope) avec fallback médiane, split `>`/`<=`, régimes `> 10` — aucune déviation non déclarée trouvée.

## 8. Ce qui n'a PAS été couvert (à lister honnêtement)

- Tout le §4.6 (live) — voir §3.
- Comportement réel d'Alpaca sur : ordre de tri des snapshots (hypothèse « OPRA ascendant » des sondes m1), présence de greeks sur `indicative`, code 403 OPRA, `end − 16 min` sur barres daily, splits (`Adjustment.SPLIT`) vs fallback Stooq/Yahoo non ajusté — raisonnés, non mesurés.
- Fréquence empirique IV > RV sur SPY (M2) : non mesurée (pas de données IV).
- Rendu visuel réel dans un navigateur (m15 mesuré en Chrome headless sur le SVG, pas en session Streamlit).
- Concurrence réelle de deux sessions écrivant le même CSV (raisonné : read-modify-write sans verrou ; non mesuré).
- Exécution sous Linux/CI (encodage mesuré sur Windows uniquement).
- `_fetch_closes_alpaca` avec un faux `StockHistoricalDataClient` (mise en forme du DataFrame) : non testé unitairement.
- Plus les `not_covered` de chaque finder (dans le dossier brut).
