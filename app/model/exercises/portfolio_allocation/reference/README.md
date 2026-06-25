# Exercices › Portfolio Allocation (SPX / VIX)

Module d'exercice quant à brancher dans **PaperTradingApp** : construit un
portefeuille à deux actifs **long SPX (moteur de rendement) + long VIX
(couverture convexe)**, pondéré en inverse-vol, mis à l'échelle sur une **cible
de vol ex-ante** puis projeté sur les caps **gross / par-instrument / VaR**.
Point-in-time, rebalancé quotidiennement. Le user choisit la source de
données : **Yahoo Finance** (live) ou **fichier CSV**.

## Contenu

```
portfolio-allocation/
├─ engine/
│  ├─ types.ts                  # types + config par défaut (= mandat du take-home)
│  ├─ portfolioBacktest.ts      # MOTEUR pur TS, sans dépendance (validé)
│  ├─ csv.ts                    # parseur CSV (Date,SPX,VIX)
│  └─ yahoo.ts                  # fetch ^GSPC/^VIX côté serveur (pas de CORS)
├─ ui/
│  └─ PortfolioAllocation.tsx   # composant React de référence (recharts)
├─ api/
│  └─ yahoo.route.example.ts    # exemple de route Next.js (App Router)
├─ engine.py                    # moteur Python CLI (secours / stacks non-JS)
├─ spx_vix_daily.csv            # jeu d'exemple fourni (1990 → 2023)
└─ _validate/                   # harnais de validation TS ↔ Python
```

## Architecture & flux de données

Le **calcul est le même** quelle que soit la source — seul le chargement diffère :

- **CSV** : le fichier est parsé et backtesté **dans le navigateur** (`parseCsvText`
  → `runBacktest`). Aucun backend requis.
- **Yahoo** : une **route serveur** (`api/yahoo.route.example.ts` → `fetchYahooPrices`)
  récupère `^GSPC` et `^VIX` (impossible côté client à cause du CORS), renvoie
  les prix, puis **le même moteur TS tourne dans le navigateur**. Conséquence :
  modifier un paramètre relance le backtest instantanément, sans re-fetch.

Le backtest sur ~9 000 jours s'exécute en < 100 ms en JS, donc tout le compute
peut rester côté client une fois les prix chargés.

## Validation (le moteur reproduit le take-home au point de base près)

`engine/portfolioBacktest.ts` réplique les conventions NumPy (covariance
`ddof=1`, quantiles en interpolation linéaire, rendements simples). Confronté au
CSV fourni, le moteur TS est **numériquement identique** au moteur Python :

| Métrique | Référence (CSV 1990→2023) |
|---|---|
| Rendement annualisé (CAGR) | **14.5 %** |
| Vol réalisée | 9.6 % (cible 10 % ±1 %) |
| Sharpe (rf=0) | 1.46 |
| Drawdown max | −20.3 % |
| VaR 95 % 1j réalisée | 0.80 % (≤ 2.5 %) |
| Gross moyen | 99 % (≤ 150 %) |
| Poids max / instrument | 100 % (≤ 100 %) |
| Turnover | 3.1 ×/an · coût 3.6 bp/an |
| Benchmark SPX seul (Sharpe) | 0.61 |

Rejouer la validation : `tsx _validate/validate.ts` (toutes les lignes `ok`).
Sur Yahoo (1990→aujourd'hui) on retrouve CAGR ≈ 15.8 % / Sharpe ≈ 1.57.

## Intégration manuelle (si tu ne passes pas par le prompt CLI)

1. Copier `engine/`, `ui/` et le contenu de `api/` dans ton arbo (ex.
   `src/exercises/portfolio-allocation/`). Ajuster les chemins d'import relatifs.
2. `npm i recharts` (ou adapter le composant à ta lib de graphes).
3. Créer la route Yahoo serveur depuis `api/yahoo.route.example.ts`
   (runtime **nodejs**, pas edge). Faire pointer `YAHOO_ROUTE` dans le composant
   vers son chemin réel.
4. Enregistrer un onglet **Exercices** dans ta navigation, avec un sous-onglet
   **Portfolio Allocation** qui rend `<PortfolioAllocation />`.
5. Re-styler le composant avec ton design system (il est volontairement neutre).

## Déploiement

- La route Yahoo doit tourner en **runtime Node** (elle utilise `fetch` serveur).
  Sur Vercel : une route serverless Node convient. En edge runtime, l'appel
  externe peut être bloqué.
- La voie **CSV** ne nécessite aucun backend (tout est client-side).

## Moteur Python (secours)

Pour un backend Python, ou appelé depuis Node via `child_process` :

```bash
python engine.py --csv spx_vix_daily.csv                 # JSON metrics → stdout
python engine.py --csv spx_vix_daily.csv --series --sample 5
python engine.py --source yahoo --fig nav_and_weights.png
python engine.py --csv spx_vix_daily.csv --format table  # lisible
```

Même schéma de sortie (clés camelCase) que le moteur TS : le composant
fonctionne indifféremment avec l'un ou l'autre.

## Mise en garde (à conserver dans l'UI)

Le backtest capte le rendement de **l'indice VIX sans friction**. En réel
(futures VIX / VXX), le roll en contango détruit une grande part de la jambe VIX :
le Sharpe affiché est une **borne haute**, pas une espérance. Le benchmark
SPX-seul (Sharpe 0.61) est plus proche d'un plancher atteignable. La série SPX
est l'**indice prix** (hors dividendes), ce qui sous-estime la jambe actions
d'environ 2 %/an.
