# Prompt CLI — intégration « Exercices › Portfolio Allocation »

> À coller dans Claude Code, **à la racine de `PaperTradingApp`**, après avoir
> déposé le dossier `portfolio-allocation/` quelque part dans le repo (idéalement
> à la racine, ou dans `src/`). Le moteur de calcul est déjà écrit et **validé** ;
> ton boulot est le branchement UI/données, pas les maths.

---

Tu intègres un nouvel exercice quant dans cette app. Les sources sont déjà dans
le repo, dans le dossier `portfolio-allocation/` (engine TS + composant React +
route Yahoo d'exemple + moteur Python de secours + jeu CSV + README). Lis
d'abord `portfolio-allocation/README.md` en entier.

## Règles strictes

1. **Ne modifie PAS la logique numérique** de `portfolio-allocation/engine/portfolioBacktest.ts`
   ni de `engine.py`. Ils sont validés au point de base près contre une référence
   Python. Tu peux corriger des imports/chemins, jamais les calculs.
2. **Commence par un repérage** avant d'écrire quoi que ce soit (voir Étape 0).
   Adapte-toi aux conventions RÉELLES du repo (routeur, structure de dossiers,
   nommage, design system). N'invente pas une stack.
3. Tout doit **builder** et **typecheck** à la fin. Pas de `any` gratuit ajouté
   au-delà de ce qui existe déjà dans les sources fournies.
4. Travaille par petits commits logiques.

## Étape 0 — Repérage (et rends-moi un court compte-rendu)

Inspecte et résume :
- Framework et **type de routeur** : Next.js App Router (`app/`) ou Pages
  (`pages/`) ? Autre (Vite/React, Remix, Electron, Tauri…) ?
- TypeScript ? Version de React ? Gestionnaire de paquets (npm/pnpm/yarn) ?
- Comment la **navigation par onglets** est faite aujourd'hui (composant de tabs
  maison, lib UI, routing par URL ?). Où vivent les pages/écrans.
- **Lib de graphes** déjà présente (`recharts`, `chart.js`, `visx`,
  `lightweight-charts`/TradingView, `d3`…) et **design system** (Tailwind ?
  composants UI maison ? shadcn ? tokens de couleur ?).
- Où placer proprement un module d'exercice (ex. `src/exercises/…`,
  `app/(exercices)/…`, `src/features/…`).

Propose-moi le plan d'intégration concret AVANT de coder, puis exécute.

## Étape 1 — Placer les sources

Déplace `portfolio-allocation/engine`, `…/ui`, `…/api`, `engine.py`,
`spx_vix_daily.csv` à l'emplacement qui colle aux conventions du repo
(ex. `src/exercises/portfolio-allocation/`). Corrige les chemins d'import
relatifs en conséquence. Garde le dossier `_validate/` (tests de non-régression).

## Étape 2 — Onglet « Exercices » + sous-onglet « Portfolio Allocation »

- Crée un onglet de **premier niveau « Exercices »** dans la navigation
  existante, **conçu pour être extensible** (un registre/liste de sous-exercices,
  pas un hardcode d'un seul). Modèle attendu : un index « Exercices » qui liste
  les exercices, le premier (et seul pour l'instant) étant « Portfolio
  Allocation ».
- Crée le **sous-onglet « Portfolio Allocation »** qui rend `<PortfolioAllocation />`.
- Suis le mécanisme d'onglets/route déjà en place dans l'app (URL ou state),
  pas un nouveau système parallèle.

## Étape 3 — Route données Yahoo (serveur)

- Si Next.js : crée la route serveur à partir de
  `portfolio-allocation/api/yahoo.route.example.ts` (runtime **nodejs**, pas
  edge ; `dynamic = 'force-dynamic'`). Mets le chemin réel de la route dans la
  constante `YAHOO_ROUTE` du composant.
- Si **pas** de backend HTTP (Vite/React pur, Electron…) : route le fetch Yahoo
  par le moyen idoine (process main Electron, petit serveur, ou proxy de dev) en
  réutilisant `engine/yahoo.ts`. La voie **CSV reste 100 % client** et doit
  fonctionner sans backend.

## Étape 4 — Monter et re-styler le composant

- `<PortfolioAllocation />` est un composant de **référence** volontairement
  neutre (styles inline). **Re-style-le avec le design system de l'app** :
  remplace les styles inline par les composants/utilities maison (Tailwind,
  tokens, boutons/inputs/cards existants), garde la même structure et les mêmes
  libellés FR.
- **Graphes** : le composant utilise `recharts`. Si l'app a déjà une autre lib,
  **remplace les blocs `<LineChart>`** par cette lib (NAV en échelle log +
  trajectoire des poids avec lignes de référence aux caps). La forme de sortie du
  moteur (`series: {date,nav,wSpx,wVix,gross}[]`, `metrics`) est agnostique.
- Si tu gardes recharts : installe-le (`npm i recharts` / équivalent).

## Étape 5 — Vérification (obligatoire)

1. `build` + typecheck OK.
2. Lance l'app, ouvre Exercices › Portfolio Allocation.
3. Source **CSV** : charge `spx_vix_daily.csv`, clique « Lancer le backtest ».
   Tu DOIS retrouver ces chiffres (sinon un import/chemin a cassé le moteur) :
   - CAGR **14.5 %**, vol réalisée **9.6 %**, Sharpe **1.46**, DD max **−20.3 %**,
     VaR 95 % 1j **0.80 %**, gross moyen **99 %**, benchmark SPX seul **0.61**.
   - Les badges de conformité (vol dans la bande, VaR/gross/poids sous les caps)
     doivent être verts.
4. Source **Yahoo** : « Charger depuis Yahoo », puis backtest → doit retourner
   un résultat (≈ CAGR 15–16 %, fenêtre 1990→aujourd'hui).
5. (Bonus) `tsx <chemin>/_validate/validate.ts` → toutes les lignes `ok`.

## Critères d'acceptation

- [ ] Onglet « Exercices » extensible + sous-onglet « Portfolio Allocation ».
- [ ] Bascule de source Yahoo / CSV fonctionnelle ; CSV sans backend.
- [ ] Le backtest sur le CSV fourni reproduit les chiffres de référence ci-dessus.
- [ ] Tableau de métriques + graphes NAV (log) et poids (avec caps) rendus.
- [ ] Composant re-stylé au design system ; graphes via la lib du repo.
- [ ] Build/typecheck verts ; moteur (`portfolioBacktest.ts`, `engine.py`) intact.
- [ ] La mise en garde « VIX spot non investable » reste visible dans l'UI.

Commence par l'Étape 0 et montre-moi le plan avant de coder.
