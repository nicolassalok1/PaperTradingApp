# Audit & cahier des charges — onglet `Bots` (intégration de `QG/`)

## 1) Contexte & contraintes

- Objectif: intégrer les idées/outils contenus dans `QG/` **dans un onglet Streamlit unique `Bots`**, sans ajouter d’onglets top-level supplémentaires.
- Contraintes:
  - **Interdiction d’intégrer `ibapi`** (pas de dépendance, pas d’appel IB, pas de code IB dans le flux de l’app).
  - Utiliser **uniquement les APIs déjà présentes dans le projet** (Alpaca via `alpaca-py`, market data via Stooq/Yahoo/Alpaca data, OpenAI optionnel).
  - Respect **MVC strict**: pas de Streamlit dans `app/model/`, contrôleurs fins, UI dans `app/vue/`.
  - Déploiement Streamlit Community Cloud via `streamlit_app.py` + `requirements.txt`.

## 2) Audit de `QG/` (synthèse des dossiers)

### `QG/AI_Trading_Bot-main/`
- Nature: **GUI desktop Tkinter** (non Streamlit) + dépendance legacy `alpaca_trade_api` + OpenAI.
- Pertinent à récupérer:
  - l’idée “copilot portfolio” (prompt + snapshot positions/ordres),
  - l’idée “bot d’exécution” basé sur **niveaux (grid/DCA)** et gestion simple de configuration (universe JSON).
- Non pertinent / à exclure:
  - Tkinter (UI) et threading GUI,
  - clés hardcodées,
  - usage direct `alpaca_trade_api` (on reste sur les services Alpaca du projet).

### `QG/earnings_tradings/`
- Nature: **Tkinter** + **Interactive Brokers (`ibapi`)** + analyses “earnings / IV crush”.
- Pertinent à récupérer:
  - la logique “scenario” pré/post earnings (spot + IV) et PnL de structures simples (ex: straddle),
  - la présentation d’un “workflow” event-driven.
- Non pertinent / à exclure:
  - IB connection / requêtes IB,
  - Tkinter + matplotlib embedded (UI).

### `QG/vol_analyzer/` et `QG/vol_dashboard/`
- Nature: **Tkinter** + `ibapi` + dashboards iVOL/régimes/mean-reversion.
- Pertinent à récupérer:
  - les heuristiques de **régime** (percentiles) et “mean reversion”,
  - le “crush analyzer” (pré/post IV, greeks, PnL).
- Non pertinent / à exclure:
  - tout le bloc IB connection / data fetch IB,
  - UI Tkinter.

### `QG/Markov_chain/`
- Nature: script orienté “regime model” mais majoritairement IB/Tkinter; la classe Markov est incomplète.
- Pertinent à récupérer:
  - l’intention “regime model” -> à implémenter via **données OHLC** (Stooq/Yahoo) déjà présentes.
- Non pertinent / à exclure:
  - IB/TWS, code de streaming ticks.

## 3) Proposition de regroupement dans `Bots`

`Bots` doit rester un onglet “outils bot + analytics bot”, sans dupliquer la suite complète `Options`.

Sous-onglets recommandés (dans `Bots`) :
1. **Assistant**: snapshot (compte/positions/ordres) + Q/A (OpenAI optionnel).
2. **Exécution (Grid/DCA)**: configuration, simulation, et **soumission explicitement protégée**.
3. **Volatilité & Régimes**:
   - Straddle (pricing + greeks via moteurs existants),
   - IV crush (pré/post) orienté earnings,
   - régime realized vol (percentiles) + **mean reversion** + **Markov transition** (sur OHLC).

## 4) Architecture cible (MVC)

- Model (`app/model/bots/`):
  - `assistant.py`: construit un snapshot et génère une réponse via `app.model.ai.chatgpt`.
  - `grid_bot.py`: calcule un plan de limites BUY, déduplique les ordres existants, et soumet via `AlpacaOrdersService` si autorisé.
  - `volatility.py`: outils de vol/régimes (realized vol, straddle, crush, mean-reversion, markov).
  - `storage.py`: persistance JSON des configs.
- Controller (`app/controller/bots_controller.py`):
  - wrappers “UI-friendly”, sérialisation dict/JSON, pas de logique lourde.
- View (`app/vue/tabs/tab_bots.py`):
  - UI Streamlit, sous-onglets, widgets et affichage.

## 5) Sécurité / garde-fous (exécution)

- Default **dry-run** (aucun ordre n’est soumis).
- Le bouton d’envoi exige:
  - `allow_submit=True` + config `enabled=True` + `dry_run=False`.
- Refus “live” par défaut:
  - si `APCA_API_BASE_URL` ne contient pas `paper`, la soumission est bloquée sauf `allow_live=True`.
- Aucune clé hardcodée; uniquement via `.streamlit/secrets.toml` / env.

## 6) Périmètre MVP (ce qui est intégré maintenant)

- Assistant portfolio (snapshot + copilot).
- Bot d’exécution Grid/DCA (simulation + soumission protégée).
- Outils vol:
  - Straddle snapshot + greeks (Black-Scholes),
  - IV crush (pré/post),
  - Régime de realized vol (percentiles).
- **Sans** `ibapi`, sans Tkinter, sans duplication “Options full”.

## 7) Roadmap (itérations recommandées)

- V1 (proche QG, sans IB):
  - mean-reversion analyzer sur realized vol (régressions + split régimes),
  - matrice de transition Markov (régimes discrets sur realized vol),
  - import optionnel d’un “universe bots” depuis JSON (format projet).
- V2:
  - scheduling (run périodique) + logs/executions,
  - backtests de signaux (si aligné avec `app/model/backtesting/`),
  - alerting (ex: vol regime HIGH -> warning).

## 8) Critères d’acceptation

- Top-level tabs visibles **uniquement**: Dashboard, Portefeuille & Risque, Trading, Hedging Systems, Yield Curve, Calibration avancée, Options, Bots.
- Le projet s’exécute en local et sur Streamlit Cloud sans dépendances IB.
- `Bots` fonctionne sans clés (dégradation gracieuse) et sans crash.
- Toute exécution d’ordres est explicitement “opt-in” et protégée.

## 9) Déploiement Streamlit Cloud (points de contrôle)

- `streamlit_app.py` doit être l’entrypoint de l’app sur Cloud.
- `requirements.txt` = dépendances “lean”.
- Secrets via `Streamlit Community Cloud -> Settings -> Secrets`.
- Si l’UI web affiche `TypeError: Failed to fetch dynamically imported module .../static/js/index...js`,
  vérifier que l’app est **publique** (sinon les assets JS peuvent être redirigés vers une page d’auth et casser le chargement).

