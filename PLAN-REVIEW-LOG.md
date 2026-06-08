# Plan Review Log: Reprise de PaperTradingApp — Programme de fiabilisation
Act 1 (grill) complete — plan locked with the user. MAX_ROUNDS=5.

## Round 1 — Codex (bloqué, environnement)
Codex n'a pas pu lire `PLAN.md` : le sandbox read-only Windows échoue au spawn (`windows sandbox: spawn setup refresh`), le runtime Node échoue aussi, et le fallback GitHub ne trouve pas `PLAN.md` (jamais poussé). Verdict REVISE non-substantiel (faute d'accès au plan).

### Claude's response
Cause = environnement (sandbox FS indisponible sur cette machine), pas le plan. Fix : relancer une session fraîche en embarquant le contenu intégral de `PLAN.md` + contexte d'audit directement dans le prompt, en gardant `-s read-only` (Codex critique depuis le texte, n'a plus besoin du FS). On repart sur un round 1 substantiel.

## Round 1 (substantiel) — Codex
Thread `019ea664-aaee-7100-8556-6214616153f2`. Plan embarqué dans le prompt, read-only. 24 findings, VERDICT: REVISE. Critique :
- Secret handling contradictoire (`st.secrets`+fallback `.env` = plaintext local) → `ConfigProvider` indépendant de Streamlit, `.env.example`, gitignore, rotation+redaction.
- Aucun scan de fuite après clé compromise → gitleaks/trufflehog sur tree+historique+logs.
- Bots ChatGPT oubliés → tests redaction, mock OpenAI, garde coût/rate, no-secrets-in-prompt.
- Risque Streamlit rerun (ordre/calibration/DQN relancés) → idempotence via session_state/locks/transaction unique.
- `load_or_train_dqn_model()` ne doit jamais entraîner depuis l'app → retirer fallback training, "model unavailable" + CLI.
- Checkpoint sans décision LFS/sécurité → LFS/release artifact, checksum, metadata, safetensors/ONNX.
- Reproductibilité DQN sous-spécifiée → versionner dataset/env/reward/seed/hyperparams/libs.
- CI trop vague pour deps lourdes → pin Python, lockfile clean, markers unit/slow, network-free.
- "Geler l'install réelle" consacre la dérive → résoudre depuis clean install compatible.
- `requirements_full` "jamais importé" = preuve faible → splitter runtime/test/train/dev avant suppression.
- Boot offline non protégé → smoke test CI "11/11 tabs sans clés/réseau".
- Tests Yahoo/Alpaca absents → adapters mockés/cassettes, zéro réseau CI.
- MC flaky → seeds fixes, tolérances, slow.
- Heston CF auto-référentiel → benchmarks indépendants + invariants.
- Calibration ignore contraintes/données sales → bornes/Feller/quotes manquantes/IV aberrantes/illiquidité/minima locaux.
- Exotiques vagues → cas limites connus (barrier=vanilla, asian 1 fixing=vanilla, rebates, digitals).
- Couverture sans seuil → coverage.py seuils progressifs.
- Refacto controller_bridge → façade stable, migration par lots, mesurer cycles/API publique (pas que fan-in).
- Nettoyage code mort dangereux → import-graph + tests runtime avant suppression ; respecter faux positifs.
- MVC fix autorise lazy/whitelist → retirer totalement Streamlit de utils, vue injecte config.
- "Aligner test_mvc_rules" affaiblit la règle → écrire la règle d'abord.
- Observabilité absente → logs structurés rédigés (source data/fallback/calibration/checkpoint/ordre).
- Pas de garde paper-vs-live → tests contractuels endpoint paper par défaut.
- Effets de bord à l'import non traités → test import sans réseau/secrets/entraînement.

### Claude's response
ACCEPTÉ (majorité — rend le plan plus concret et sûr) : ConfigProvider + retrait total de Streamlit hors utils (#3,#23), gitleaks (#4), smoke test offline + test effets de bord (#13,#27), adapters mockés zéro-réseau + garde paper-vs-live (#14,#26), MC seeds/tolérances/slow (#15), Heston benchmarks indépendants + cas limites (#16,#17), exotiques cas limites (#18), seuils coverage (#19), DQN no-runtime-training + LFS/safetensors/checksum + repro spec (#7,#8,#9), CI lockfile/markers/network-free + split deps runtime/test/train/dev + résolution clean-install (#8→#12), façade stable + migration par lots (#20,#21), suppression code mort gated par import-graph (#22), écrire la règle d'archi d'abord (#24), idempotence rerun + observabilité (#5-rerun, #25), bots scopés mock/redaction/coût (#5-partiel).
REJETÉ/RÉDUIT avec raison : hardening prompt-injection complet des bots → hors scope (concerne une feature, pas la fiabilité du socle) ; scan historique pour CETTE clé → la clé n'a jamais été commitée (établi par l'audit), le scan reste comme filet une fois mais n'est pas un blocker.
→ PLAN.md révisé (8 étapes au lieu de 7, décisions durcies). Resume du même thread pour round 2.

## Round 2 — Codex
Resume thread, plan révisé embarqué. Codex valide la reprise de la majorité des points round 1. 12 raffinements, VERDICT: REVISE. Critique :
- Gate "aucun secret en clair lu par l'app" contradictoire (l'app doit lire une clé locale) → reformuler "aucun secret versionné/loggé/prompts/cassettes ; lecture locale gitignorée, jamais affichée".
- Boot offline protégé trop tard (étapes 1-2 peuvent casser les imports) → smoke test dès le début, relancé après chaque étape.
- Caractérisation controller_bridge après migration ConfigProvider → capturer le comportement public AVANT toute modif qui le touche.
- "CI sans réseau" non enforceable avec mocks seuls → bloqueur réseau (pytest-socket), réseau autorisé seulement à l'install.
- Cassettes peuvent fuiter headers/tokens → fixtures rédigées déterministes + test anti-secret sur tests/fixtures.
- torch en `train` mais inférence DQN peut nécessiter torch/onnxruntime au runtime → groupe `runtime-ml`/inference distinct, "model unavailable" si absent.
- Idempotence rerun en étape 8 trop tard → remonter étape 4, avant refacto et tests de comportement trading.
- Observabilité trop tardive → logging rédigé minimal dès adapters/config/DQN.
- Règle MVC partielle → matrice complète view/controller/model/utils vérifiée statiquement.
- `-m "unit or smoke"` exclut les non-marqués (les 8 tests) → marquer/migrer tous + testpaths + policy unmarked=fail.
- Checkpoint LFS/artifact sans comportement quand absent → CI smoke jamais dépendante du checkpoint ; job slow vérifie download/checksum.
- Paper "par défaut" doit être fail-closed → live impossible sans flag explicite + endpoint live séparé + confirmation runtime testée.

### Claude's response
ACCEPTÉ EN TOTALITÉ — aucun désaccord de fond, ce sont des durcissements de séquençage et d'enforcement, tous légitimes. Changements : nouvelle étape 0 (smoke + snapshot bridge AVANT toute modif, relancé après chaque étape) ; gate sécurité reformulé ; matrice d'imports MVC complète + vérif statique ; pytest-socket + fixtures rédigées + anti-secret ; split deps `runtime-ml` distinct de `train` ; idempotence rerun + paper fail-closed remontés en étape 4 ; observabilité dès étape 1 ; policy unmarked=fail + migration des 8 tests ; CI smoke indépendante du checkpoint.
→ PLAN.md révisé (étape 0 ajoutée). Resume pour round 3.

## Round 3 — Codex
Resume thread, plan v3 embarqué. VERDICT: APPROVED. Codex confirme les 12 points round 2 adressés et la structure saine (filets avant modif, smoke à chaque étape, réseau bloqué, DQN hors runtime, runtime-ml séparé, paper fail-closed, MVC statique complet, fixtures rédigées, CI indépendante du checkpoint). Deux précisions NON-bloquantes :
- `gitleaks clean` doit allowlister les stores locaux gitignorés volontaires (.env/secrets.toml) tout en gardant l'anti-secret sur tracked/logs/fixtures.
- Étape 0 dépend d'un env capable d'importer l'app → autoriser un bootstrap "deps only" avant toute modif de code, puis capturer smoke/snapshot.

### Claude's response
Les deux précisions intégrées au PLAN.md (bootstrap deps-only en étape 0 ; allowlist gitleaks + cible tests/fixtures en étape 1). Plan APPROUVÉ — convergence atteinte au round 3/5.

---
**CONVERGENCE : APPROVED au round 3 sur 5.** Plan durci par 1 round de grill (7 décisions verrouillées avec l'utilisateur) + 3 rounds Codex (36 findings traités, ~6 réduits/rejetés avec raison).

