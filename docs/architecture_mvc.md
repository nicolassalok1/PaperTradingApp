# Architecture MVC du projet PaperTradingApp

## 1. Vue d'ensemble

Le projet est structuré selon une architecture MVC stricte :

- `app/model/` : logique métier pure (pricing, hedging, backtesting, yield curves, etc.).
- `app/controller/` : couche de coordination entre la vue (Streamlit) et le modèle.
- `app/vue/` : couche UI (Streamlit), 100 % orientée présentation.
- `app/utils/` : utilitaires génériques (I/O, maths, chemins), sans logique métier.

Objectif : isoler la logique de pricing/hedging des détails d’implémentation Streamlit, pour obtenir un code testable, maintenable et extensible.

---

## 2. `app/model/` — Domaine métier

### 2.1. Options

Le sous-module Options est organisé en sous-domaines :

- `app/model/options/core/`  
  Logique de base des options :
  - `payoff.py` : payoffs standards.
  - `greeks.py` : calculs de grecs.
  - `iv.py` : volatilité implicite.
  - `surfaces.py` : surfaces de volatilité.
  - `trees.py` : structures d’arbres (binomial, CRR).
  - `shared.py`, `defaults.py`, `pnl.py`, `book.py` : fonctions de support.

- `app/model/options/exotic/`  
  Options exotiques organisées par famille :
  - `american/` : américaines.
  - `lookback/` : lookback.
  - `basket/`, `quanto/`, etc.

- `app/model/options/engines/`  
  Implémentations des moteurs de pricing déterministes / MC :
  - Black–Scholes.
  - CRR / binomial.
  - Engines de grid/pricing déterministe.

- `app/model/options/ui_gateway/`  
  - `pricing_ui.py` : façade métier exposant des helpers de pricing « UI-friendly »
    (grilles de payoff, objets prêts à tracer), **sans aucune dépendance Streamlit**.

### 2.2. Autres domaines

- `app/model/trading/` : exécution, systèmes de trading, logs.
- `app/model/portfolio/` : positions, valorisation, statistiques, forwards, règlements.
- `app/model/yieldcurve/` : construction de courbe, engine de taux, utils de taux, services de courbe.
- `app/model/hedger/` : agents de couverture (delta-hedger, DQN, modèles de hedging), services d’orchestration de la couverture.
- `app/model/dashboard/` : cache et services d’agrégation pour le dashboard.
- `app/model/market_data/` : accès marché, données temps réel, taux, rafraîchissement de caches.
- `app/model/backtesting/` : moteur de backtest et signaux.
- `app/model/ai/chatgpt.py` : intégration AI pour diagnostics / texte.

Tous ces modules sont **dépourvus de Streamlit** : ils peuvent être testés indépendamment de l’UI.

---

## 3. `app/controller/` — Couche de coordination

Les contrôleurs sont de minces adaptateurs entre Vue et Modèle :

- `options_controller.py`
- `portfolio_controller.py`
- `buy_sell_controller.py`
- `backtest_controller.py`
- `hedger_controller.py`
- `dashboard_controller.py`
- `yieldcurve_controller.py`
- etc.

Rôles typiques :

- récupérer les inputs venant de la Vue (sous forme de types Python simples),
- appeler les fonctions du modèle (pricing, hedging, backtest, yield curve),
- formater les résultats pour la Vue (DataFrame, séries, dictionnaires).

Contraintes :

- pas de logique métier lourde ;
- pas d’accès direct à Streamlit ;
- pas d’import de `app.vue.*`.

---

## 4. `app/vue/` — Couche UI (Streamlit)

Le code Streamlit vit dans :

- `app/vue/main_app.py` (ou équivalent) : point d’entrée de l’application.
- `app/vue/tabs/` : onglets logiques (Options, Portfolio, Backtest, Yield Curve, etc.).
- `app/vue/components/` : composants réutilisables (panels, graphes, layouts).

### 4.1. Bridge Options

Pour les Options, un pont explicite est utilisé :

- `app/vue/components/options/controller_bridge.py`  
  Ce bridge expose au code UI les fonctions du contrôleur Options et des helpers de pricing, sans permettre à la Vue d’appeler directement le Modèle.

### 4.2. Helpers UI & state

- `app/vue/state/options_context.py` : accès à `st.session_state` et construction du contexte Options côté UI.
- `app/vue/components/options/ui_helpers.py` : helpers Streamlit (widgets, contrôles, interactions) spécifiques à la vue Options.

La Vue ne parle JAMAIS directement au Modèle : elle passe par les contrôleurs et/ou le bridge.

---

## 5. `app/utils/` — Utilitaires génériques

Ce module contient uniquement des helpers non métier :

- `io.py` : fonctions d’entrée/sortie génériques.
- `math_utils.py` : fonctions mathématiques simples.
- `paths.py` : gestion centralisée des chemins (JSON, data, etc.).

Aucune logique de pricing, hedging ou marché ne doit vivre ici.

---

## 6. Contrôles automatiques

Deux scripts permettent de garantir la pérennité de l’architecture :

- `scripts/check_mvc_integrity.py`  
  Vérifie les règles suivantes :
  - Model ne dépend ni de Vue ni de Controller, ni de Streamlit.
  - Vue ne dépend ni de Model ni de Utils.
  - Controller ne dépend pas de Vue ni de Streamlit.
  - Utils ne dépend d’aucune couche métier ni de Streamlit.
  - Aucun module `app/model/options/*` ne contient de référence à `streamlit`.

- `scripts/scan_imports.py`  
  Construit un graphe de dépendances `app.*` et liste les modules potentiellement orphelins (non importés), pour faciliter le nettoyage du code mort.

Ces scripts peuvent être utilisés en local ou intégrés dans un pipeline CI.
