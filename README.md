# Volatility & RL Dashboards

Une app Streamlit qui regroupe trois modules :
- **Volatility Tools** : analyse IV, pricing straddles et earnings crush (Yahoo Finance).
- **RL & Finance Lab** : notebooks interactifs et démos RL/quant (pricing, hedging, signals, stratégies, NLP sentiment).
- **AI Trading Bot** : tableau de bord trading/options (Alpaca, pricing avancé, persistance JSON locale).

## Arborescence
- `streamlit_app.py` : point d’entrée qui assemble les modules en onglets.
- `scripts/` : app Volatility Tools.
- `streamlit_appVol.py` et `database/RLtab/` : app RL & Finance Lab + datasets.
- `streamlit_appGPT.py` et `scripts/scriptsGPT/` : app AI Trading Bot + scripts/pricing.
- `database/GPTab/` : datasets de l’onglet Volatility/Trading.
- `database/GPTab/jsons/` : stockage local (portefeuille, options, forwards…).
- `tests/` : scripts de vérification/fixtures.

## Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Lancement
```bash
streamlit run streamlit_app.py
```
Onglets disponibles : Volatility Tools / RL & Finance Lab / AI Trading Bot.

## Variables d’environnement utiles
- `OPENAI_API_KEY` : pour les appels ChatGPT dans l’onglet AI Trading Bot.
- `APCA_API_KEY_ID` et `APCA_API_SECRET_KEY` : clés Alpaca (paper ou live) si tu actives les calls API.
- `HTTP_PROXY` / `HTTPS_PROXY` : si nécessaire derrière un proxy.

## Données locales
- Les portefeuilles/options/forwards sont stockés dans `database/GPTab/jsons/`. Ils sont créés/écrasés par l’UI.
- Les jeux de données RL/quant sont dans `database/RLtab/` (sous-dossiers data). Les notebooks ont été déplacés sous `notebooks/`.
- Les datasets Trading/IV sont dans `database/GPTab/`.

## Notes
- L’app charge des dépendances lourdes (torch, tensorflow, plotly…). Prévois quelques minutes d’installation.
- Si tu ne veux pas installer Alpaca/OpenAI, garde les variables non définies : l’UI gère les erreurs mais les fonctions liées ne marcheront pas.
