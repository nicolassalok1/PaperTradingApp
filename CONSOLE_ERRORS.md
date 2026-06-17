# CONSOLE_ERRORS.md — Runtime console-error hunt

> Suivi de la traque des erreurs/warnings au runtime de l'app Streamlit `PaperTradingApp`.
>
> **Capture locale (Phase 0)** : Streamlit headless dans l'env conda `papertrading` avec
> `PYTHONWARNINGS=default` (`scripts/capture_console.ps1`), sortie → `logs/console_runtime.log`.
> Navigation + console navigateur + boîtes rouges via Playwright. Les 8 onglets top-level
> utilisent `st.tabs`, qui **rend le code de tous les onglets à chaque exécution** : un seul
> chargement exerce tout le code de rendu (Dashboard, Portefeuille & Risque, Trading + sous-
> onglets Spot/Ordres/Options, Hedging, Yield Curve, Calibration, Options ×6 modèles ×6
> panneaux stratégie, Bots).
>
> **Capture prod** : log de déploiement Streamlit Cloud (`logs-...-2026-06-17T06_31_32.285Z.txt`).
>
> **Résultat** : prod KO au boot (E00). En local : 0 traceback serveur, 1 boîte rouge (Dashboard,
> E02→FIXED), + warnings (E03 thème, E04 matplotlib, E05 conda).

## État temporaire à restaurer en fin de traque
- [x] `.streamlit/config.toml` `[logger] level` = `"error"` (déjà restauré par l'utilisateur).
- [ ] `scripts/capture_console.ps1` + `CONSOLE_ERRORS.md` : garder (utiles) ou retirer selon décision.

---

## E00 — [PROD DOWN] Le déploiement Streamlit Cloud ne démarre plus (cascade KeyError d'import)
- **Statut** : NEEDS ACTION (reboot) — **pas un bug du code commité**
- **Sévérité** : BLOCKER (prod KO, aucun onglet ne charge)
- **Catégorie** : déploiement / runtime cloud (hot-reload)
- **Où** : `alpaca-paper-trading.streamlit.app` (branche `main`, commit `68e04df`).
- **Message** : 36× `ImportError: cannot import name 'main' from 'app.vue.main_app'`
  + cascade `KeyError: 'app.utils' / 'app.model' / 'app.vue' / 'app.model.options.core.greeks'
  / 'app.model.options.core.shared' / 'app.utils.trading_guard'` levés dans
  `importlib._bootstrap._load_unlocked:701` (aucune frame de code app).
- **Repro** : sur le cloud seulement, déclenché juste après `🔄 Updated app!` (05:54).
  **Ne se reproduit PAS en local** : import à froid de `68e04df` = `IMPORT OK`.
- **Cause racine** : hot-reload Streamlit Cloud — l'app tournait un **ancien commit
  pré-fiabilisation** ; le pull du nouveau code (qui déplace/ajoute/supprime des modules :
  `options/core/{greeks,shared}`, `utils/trading_guard`, split options) s'est fait **sans
  redémarrer le process Python**. `sys.modules` mélange ancien layout caché + nouvelle
  structure → `KeyError` sur les modules déplacés → `app.vue.main_app` n'achève pas son
  import → `main` jamais défini. Confirmé : **0 manip `sys.modules`/`importlib.reload`**,
  **0 `ModuleNotFoundError`** (deps cloud complètes), import à froid local OK.
- **Fix** : **USER** — rebooter l'app (Streamlit Cloud → *Manage app → Reboot*, idéalement
  *Clear cache* puis *Reboot*). Prévention : après un refacto qui déplace des modules, rebooter
  plutôt que se fier au hot-reload.
- **Réserve** : non reproductible en py3.11 localement (indispo). Si après reboot propre l'app
  replante, monter un env py3.11 et traquer un vrai import circulaire. Le reboot tranche.
- **Vérif** : après reboot, l'app charge ; seule subsiste la bannière Alpaca `unauthorized` (E01).

## E01 — Alpaca API « unauthorized » sur tous les appels
- **Statut** : NEEDS CONFIG
- **Sévérité** : ERROR (bloque fonctionnellement les features Alpaca)
- **Catégorie** : config/secrets
- **Où** : bannière de statut + onglets Dashboard, Portefeuille & Risque, Trading, Hedging,
  Alpaca Spot/Ordres/Options.
- **Message** : `APIError: {"message": "unauthorized."}`
- **Cause racine** : `.env` contient `APCA_API_KEY_ID` (26 c) + `APCA_API_SECRET_KEY` (44 c)
  **présents mais rejetés** → clé révoquée/invalide (rotation Alpaca en attente, cf. commit
  `68e04df`). Pas un bug code : le code charge bien les clés, la clé est invalide.
- **Fix** : USER — régénérer une paire de clés paper Alpaca, remplacer dans `.env`. Pas de code.

## E02 — L'onglet Dashboard affiche un traceback brut (boîte rouge) sur échec Alpaca
- **Statut** : FIXED
- **Sévérité** : ERROR
- **Catégorie** : code-bug (robustesse / gestion d'erreur incohérente)
- **Où** : `app/vue/tabs/tab_dashboard_v2.py` → `render_tab()`.
- **Message** : `APIError unauthorized` rendu en boîte rouge + `st.error("Onglet '📊 Dashboard'
  non rendu : ...")`.
- **Cause racine** : `render_tab` appelait `get_account_summary()` / `get_drawdowns()` /
  `get_live_risk_snapshot()` **sans `try/except`**, alors que tous les autres onglets Alpaca
  attrapent l'erreur et affichent `st.error("Unable to load ...: {exc}")`. L'`offline fallback`
  de l'engine ne se déclenche que sur clé vide/"dummy", pas sur clé **révoquée** → chemin LIVE
  → `APIError` remonte au handler défensif de `main_app._render_tab` → `st.exception` (rouge).
- **Fix** : wrapper les 3 appels eager dans `try/except Exception` → `st.error("Unable to load
  account: {exc}")` + `return`, par parité avec `tab_alpaca_spot` / `tab_alpaca_orders`.
  Engine non touché (surfacer l'erreur d'auth est le comportement honnête établi).
- **Vérif** : test de régression `tests/test_dashboard_v2_graceful.py` (`@pytest.mark.unit`) —
  monkeypatch `get_account_summary` → raise ; asserte `render_tab` ne lève pas + `st.error`
  contient « Unable to load account » + `st.exception` jamais appelée. 22 tests verts (1 nouveau
  + 21 smoke). E2E : recharger l'app avec clé révoquée → message propre, **plus de boîte rouge**.

## E03 — Console navigateur : couleurs de thème sidebar invalides (vides)
- **Statut** : OPEN
- **Sévérité** : WARNING
- **Catégorie** : config/dépendance (thème frontend Streamlit 1.51)
- **Message** (3 distincts, ré-émis à chaque rerun) :
  `Invalid color passed for widgetBackgroundColor / widgetBorderColor / skeletonBackgroundColor
  in theme.sidebar: ""`
- **Cause racine** : (à confirmer) `config.toml` définit `[theme]` mais aucun token de couleur
  sidebar ; Streamlit 1.51 dérive les couleurs widget sidebar et reçoit des chaînes vides.
- **Fix** : (Phase 2) définir les tokens sidebar manquants dans `config.toml` (vérifier doc thème
  Streamlit 1.51 via context7) OU confirmer défaut upstream à justifier.

## E04 — DeprecationWarning matplotlib (API pyparsing) ×3, à l'import
- **Statut** : OPEN (décision deps)
- **Sévérité** : WARNING
- **Catégorie** : dépendance/déprécation
- **Message** : `matplotlib/_fontconfig_pattern.py:88 'parseString' deprecated`,
  `:92 'resetCache' deprecated`, `_mathtext.py:45 'enablePackrat' deprecated`.
- **Cause racine** : matplotlib (utilisé dans `model/options/engines/{pricing,tree}.py`,
  `model/yieldcurve/engine.py` + vues) appelle l'ancienne API pyparsing. **Code app non
  concerné** — interne à matplotlib.
- **Fix** : (Phase 2) décision USER — bump matplotlib / pin pyparsing / `warnings.filterwarnings`
  ciblé. Candidat `WONTFIX (env)` si on ne touche pas les deps.

## E05 — PendingDeprecationWarning conda (bruit de `conda run`)
- **Statut** : WONTFIX (env)
- **Sévérité** : WARNING (cosmétique)
- **Catégorie** : env/outillage
- **Cause racine** : émis par `conda run` (plugins conda), pas l'app. Disparaît dans un env activé.

## E06 — `OPENAI_API_KEY` vide (latent, pas une erreur runtime actuelle)
- **Statut** : NEEDS CONFIG (si Bots/ChatGPT utilisés)
- **Sévérité** : WARNING (latent)
- **Catégorie** : config/secrets
- **Où** : onglet Bots (gate OpenAI derrière des actions ; rendu sans boîte rouge).
- **Fix** : renseigner la clé si bots voulus ; sinon ignorer.

---

## Ordre de traitement (Phase 2)

1. ✅ **E02** (code-bug) — FIXED.
2. **E00** — USER : reboot de l'app Streamlit Cloud (prod down).
3. **E03** (warning thème) — compléter les couleurs sidebar dans `config.toml`.
4. **E04** (déprécation matplotlib) — décision deps avec le user.
5. **E01 / E06** — actions USER (rotation clé Alpaca / clé OpenAI).
6. **E05** — rien (bruit env).
