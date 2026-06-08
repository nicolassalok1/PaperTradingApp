# Plan: Reprise de PaperTradingApp — Programme de fiabilisation (local)
_Locked via grill — by Claude + nicolas. Hardened via Codex (rounds 1-2)._

## Goal
Remettre PaperTradingApp sur des fondations saines **avant toute nouvelle feature**. MVP Streamlit MVC fonctionnel (boot OK, 11/11 tabs importent, dégradation offline gracieuse) mais sans chaîne qualité : gate MVC rouge, 8 tests non exécutables, zéro CI, zéro couverture du cœur quant (154 fichiers `app/model/`), clé Alpaca en clair, god-object `controller_bridge` (fan-in 55), DQN qui s'entraîne au runtime. Cible = **programme complet de fiabilisation en local** (pas de déploiement) : poser des filets (smoke offline + snapshot de comportement) **avant** de toucher au code, sécuriser, rendre les tests exécutables + CI **réseau-bloquée**, couvrir le cœur quant avec des **oracles indépendants**, sortir l'entraînement DQN du runtime (checkpoint versionné), puis refactorer le god-object **sous protection des tests, derrière une façade stable**.

## Approach
Ordre strict — chaque étape est un gate vérifiable. **Le smoke test offline (étape 0) tourne après CHAQUE étape.**

### 0. Filets AVANT toute modification → vérif : smoke vert sur l'état actuel, snapshot bridge capturé
- **Bootstrap « deps-only »** autorisé avant toute modif du code applicatif (créer un venv + installer les deps nécessaires à l'import), afin de pouvoir exécuter le smoke/snapshot sur l'état courant.
- **Smoke test offline créé en premier** : importe les 11 tabs + modèles + controllers **sans clés, sans réseau, sans entraînement, sans client externe** ; assert aucun effet de bord à l'import. Sert de garde de non-régression **dès maintenant** et est relancé après chaque étape suivante.
- **Snapshot de caractérisation minimal du comportement public de `controller_bridge`** capturé **avant** toute modif qui le touche (y compris la migration ConfigProvider).

### 1. Sécurité & hygiène → vérif : aucun secret versionné/loggé/dans prompts/dans cassettes ; lecture locale via env/st.secrets gitignoré, jamais affichée ; gitleaks clean
- Régénérer la clé Alpaca paper (compromise car lue pendant l'audit).
- **`ConfigProvider` indépendant de Streamlit** (utils/config pur) ; la **vue injecte** la valeur. Pas de `st.secrets` dans `utils`.
- `.env.example` sans secret ; `.env` + `.streamlit/secrets.toml` + `logs/` + artefacts debug gitignorés.
- **Logging structuré rédigé minimal** introduit dès cette couche config/adapters (jamais de secret en clair dans les logs).
- Scan `gitleaks`/`trufflehog` sur tree + historique + `logs/` + `tests/fixtures` (clé jamais commitée d'après audit ; net one-time). **Allowlist explicite des stores locaux gitignorés volontaires** (`.env`, `.streamlit/secrets.toml`) pour ne pas faire échouer le scan sur le mécanisme de secrets choisi, tout en gardant l'anti-secret sur les fichiers tracked/logs/fixtures. Documenter rotation.

### 2. Règle d'archi MVC (matrice complète) + fix gate → vérif : `check_mvc_integrity` exit 0, matrice d'imports vérifiée statiquement
- Écrire la **matrice d'imports autorisés complète** `view → controller → model → utils` (et interdits), pas seulement « utils ≠ vue ». Vérification statique des imports.
- **Retirer totalement `streamlit` de `app/utils`** (pas de lazy-import ni whitelist). `secrets.py:50` → config injectée par la vue via `ConfigProvider`.
- Réécrire `test_mvc_rules.py` pour encoder la matrice (le code converge vers la règle, pas l'inverse).

### 3. Env reproductible + deps propres + CI réseau-bloquée → vérif : `pytest -m "unit or smoke"` vert local+CI, gate en CI, réseau bloqué en test
- Deps : **ne PAS geler l'env accidentel** ; résoudre depuis un **clean install compatible** (venv neuf), générer un **lockfile**, pin Python.
- **Split de groupes** : `runtime`, **`runtime-ml` (inférence DQN : torch/onnxruntime requis à l'exécution)**, `test`, `train` (entraînement RL), `dev`. Si les deps `runtime-ml` manquent → l'app affiche "model unavailable" (pas de crash). Supprimer tensorflow/sklearn de `runtime` seulement après preuve d'absence d'import ; les garder dans `train`/`dev` si utiles.
- **Bloqueur réseau en test** (`pytest-socket` ou équiv.) : réseau interdit pendant les tests, autorisé seulement à l'install des deps.
- Markers `unit`/`smoke`/`slow` + **policy `unmarked = fail`** + `testpaths` : **les 8 tests existants sont migrés/marqués explicitement** (aucun test silencieusement exclu).
- `.github/workflows/tests.yml` : venv depuis lockfile + `pytest -m "unit or smoke"` + `check_mvc_integrity`, réseau bloqué.

### 4. Garde-fous trading + protection boot + couverture quant (pricing & calibration d'abord) → vérif : fail-closed paper, idempotence rerun, oracles indépendants, seuils tenus
- **Idempotence Streamlit rerun (remontée ici, AVANT tout test de comportement trading et avant refacto)** : actions Alpaca / calibration / chargement modèle protégées (session_state / transaction unique) — un rerun ne re-soumet pas d'ordre ni ne relance d'entraînement.
- **Paper fail-closed** (pas seulement testé) : live impossible sans flag explicite + endpoint live séparé + confirmation runtime ; tests contractuels que l'endpoint paper est le défaut et que le live est bloqué sans opt-in.
- **Adapters mockés** Yahoo/Alpaca : **fixtures rédigées, déterministes** (pas de headers/tokens/données instables) + **test anti-secret sur `tests/fixtures`** ; timezone exchange + stale-data checks.
- **Pricing** : BSM, CRR, Monte-Carlo. Oracles = valeurs analytiques + invariants (put-call parity, bornes) ; MC **seeds fixes + tolérances**, convergence en `slow`.
- **Calibration Heston** : **benchmarks indépendants** (pas le même code) + invariants ; cas limites bornes/Feller, quotes manquantes, IV aberrantes, illiquidité, minima locaux.
- **Exotiques** : cas limites connus (barrier éloignée ≈ vanilla, asian 1 fixing ≈ vanilla, rebates, monitoring discret, digitals près du strike), puis portfolio/hedging.
- **Observabilité** étendue aux événements clés (source data/fallback offline, échec calibration, checkpoint chargé, ordre soumis/refusé paper).
- Seuils `coverage.py` progressifs sur modules critiques.

### 5. DQN : sortir l'entraînement du runtime + checkpoint versionné → vérif : app charge un poids ou "model unavailable" (jamais d'entraînement bloquant), CI smoke indépendante du checkpoint
- **Supprimer le fallback training runtime** de `load_or_train_dqn_model()` : checkpoint absent **ou deps `runtime-ml` absentes** → "model unavailable" + script CLI. L'app ne s'entraîne jamais.
- **Checkpoint** : `safetensors`/weights-only/ONNX, Git LFS (ou release artifact), checksum + metadata version. **La CI smoke ne dépend JAMAIS du checkpoint** ; un job `slow` vérifie téléchargement/checksum si l'artifact est configuré.
- Repro : versionner dataset, env RL, reward, seed, hyperparams, versions libs ; éval vs delta-hedge de référence.

### 6. Refacto `controller_bridge` sous protection des tests, façade stable → vérif : snapshot caractérisation + 55 importeurs verts, cycle `import *` éliminé, API publique préservée
- Extraire derrière l'API existante (façade stable), supprimer le `import *` (cycle), migrer les importeurs **par lots**.
- Mesurer cycles / API publique / import graph (pas que fan-in). Tests de compat sur les 55 importeurs.

### 7. Nettoyage code mort confirmé → vérif : suppression sans casser gate/tests/smoke
- Supprimer **seulement après preuve import-graph + tests runtime** : scaffold calib v1 (`logic.py`/`submit()`), UI orpheline `vue/components/shared.py`. Respecter les faux positifs.
- Rafraîchir `docs/architecture_mvc.md`.

### 8. Hygiène bots ChatGPT (scopée) → vérif : tests bots verts offline
- Bots : **mock OpenAI**, **redaction** (jamais clés/secrets/ordres dans les prompts → test dédié), garde **coût/rate-limit**. (Hardening prompt-injection approfondi = hors scope, feature.)

## Key decisions & tradeoffs
- Fiabiliser avant features. Programme complet (jours). Cible locale, pas de déploiement (deps lourdes gardées en `train`/`dev`).
- **Filets d'abord** : smoke offline + snapshot bridge créés en étape 0, relancés après chaque étape.
- Deps : clean install + lockfile + split `runtime`/`runtime-ml`/`test`/`train`/`dev` (pas de gel accidentel ni de suppression sur heuristique).
- MVC : matrice d'imports complète vérifiée statiquement ; Streamlit hors de `utils`.
- Trading : **paper fail-closed** + idempotence rerun en amont des tests de comportement.
- CI : **réseau bloqué** (`pytest-socket`) + fixtures rédigées ; `unmarked = fail` (les 8 tests migrés) ; `unit/smoke` en CI, `slow` manuel ; **smoke indépendante du checkpoint DQN**.
- DQN : zéro training runtime ; inférence = deps `runtime-ml` (sinon "model unavailable") ; checkpoint LFS/safetensors/checksum.
- Ordre : tests AVANT refacto ; refacto derrière façade stable, migration par lots.
- Couverture : pricing & calibration d'abord, **oracles indépendants** (anti auto-référence).

## Risks / open questions
- Gate déjà branché en CI ailleurs ou purement local ? (vérif début étape 2).
- Deps 2.x cassant des appels supposés 1.x (numpy/pandas) → détecté par les tests.
- 55 importeurs = large surface de régression (façade stable + lots + snapshot limitent).
- Checkpoint DQN stale si feature-space change → politique de ré-entraînement à définir.
- Dispo `gitleaks`/`trufflehog`, Git LFS, `pytest-socket` sur la machine (installer si absent).
- Existence de benchmarks Heston indépendants (sinon : invariants + valeurs de littérature).

## Out of scope
- Tout déploiement. Nouvelles features fonctionnelles. Hardening prompt-injection approfondi des bots. Perf non liée à la fiabilité.
- Suppression de legacy actif (faux positifs : `dashboard/` v1, `calibrator_legacy.py`/`heston_v1`, `train_heston_surface_net.py`).
