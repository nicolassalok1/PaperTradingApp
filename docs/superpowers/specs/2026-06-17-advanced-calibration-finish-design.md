# Spec — Terminer les onglets de la Calibration avancée

_Date : 2026-06-17 · Branche : `feat/finish-advanced-calibration` · Statut : design validé, prêt pour plan_

## Contexte & problème

L'onglet **🧪 Calibration avancée** (`app/vue/tabs/tab_advanced_calibration.py`) expose 6 sous-onglets
modèles. Seul **Heston** (`heston_v1`) est accessible. Les 5 autres sont court-circuités par une garde UI
`in_progress_models` (`tab_advanced_calibration.py:401`, court-circuit lignes 408-410) qui affiche
« Cette fonctionnalité est en cours d'implémentation » et `continue`.

Un audit du code + une vérification d'exécution headless (env conda `papertrading`, scipy présent, surface
synthétique 5 moneyness × 4 maturités) ont établi :

- **Le backend des 5 modèles est complet et câblé.** `CalibrationController.run_advanced_surface_calibration`
  (`app/controller/calibration_controller.py:845-978`) dispatche déjà via `calibrator_map` vers des
  calibrateurs existants ; les 5 sont enregistrés dans `calibrator_map` et dans
  `model_comparison._MODEL_LABELS`. Le corps de l'onglet (lignes 411-689) est entièrement générique.
- **4 modèles sur 5 sont réellement fonctionnels** end-to-end (succès, métriques finies, surface IV 2D
  bien formée, temps < 1 s même avec les configs MC par défaut) :
  - `sabr` (impl. de référence canonique), `merton_jump_diffusion`, `rbergomi`, `volterra`.
- **`rheston` est cassé** : son CF Markovien overflow dans l'intégration Riccati explicite-Euler
  (`app/model/volatility_models/rheston/cf_markovian.py:84-89`, terme `0.5·ξ²·S²` → `exp(A)` inf/NaN),
  ce qui propage des NaN dans toute la surface via la FFT (`app/model/volatility_models/common/fft.py`).
  Le wrapper LS garde les paramètres d'ancrage et **rapporte quand même `success=True`** → l'onglet
  afficherait des heatmaps vides en prétendant avoir réussi.
- **La crainte de performance est levée** : les simulateurs MC (rBergomi, Volterra) sont vectorisés et
  terminent en < 1 s sur une grille réaliste, même avec leurs configs par défaut (4000 paths).
- **Zéro couverture de tests** sur ces 5 calibrateurs, et **aucune validation numérique**.

Périmètre retenu (décision utilisateur) : **les 5 onglets fonctionnels + suite de tests quant**.
Décision rHeston (utilisateur) : **fix pragmatique ciblé** (pas de réécriture d'intégrateur sauf échec).

## Objectif

Rendre les 5 sous-onglets de Calibration avancée réellement utilisables et honnêtes, et verrouiller la
non-régression par des tests quant.

Critères de succès vérifiables :
1. Les 5 sous-onglets (`sabr`, `merton_jump_diffusion`, `rheston`, `rbergomi`, `volterra`) calibrent sans
   exception et affichent des métriques finies + heatmaps non vides sur une surface marché normale.
2. `rheston` ne renvoie **plus jamais** `success=True` avec une surface 100 % NaN ; il produit un fit fini
   et raisonnable sur une surface normale (tolérance lâche) ou échoue explicitement.
3. `pytest tests/quant` vert sur les nouveaux tests, 100 % offline.

## Conception

### Unité 1 — Déverrouillage UI

- **Quoi** : retirer les 5 clés de `in_progress_models` (vider le set, ou supprimer la garde 408-410).
- **Dépendances** : aucune logique nouvelle — le corps générique de l'onglet (profil « Normal » fixe pour
  les non-Heston, bouton Calibrer, métriques, diagnostics, surfaces 3D, comparaison, « Envoyer vers
  Options ») fonctionne déjà pour n'importe quel `model_key`.
- **Caption ETA honnête** : la caption ETA (`tab_advanced_calibration.py:466-471`) se base sur `max_nfev`,
  mais Volterra ignore `max_nfev`/`n_starts` (budget piloté par sa config MC interne) et rBergomi
  partiellement. Rendre la caption non trompeuse pour les modèles MC (`rbergomi`, `volterra`) — sans
  re-câbler les sliders (c'est leur design intentionnel).
- **Interface inchangée** : aucune signature publique modifiée.

### Unité 2 — Fix numérique rHeston (systematic-debugging)

- **Cause racine** : overflow dans l'intégration Riccati explicite-Euler du CF Markovien
  (`cf_markovian.py:84-89`). `exp(A)` devient inf/NaN, propagé par `carr_madan_fft_call_prices`.
- **Démarche** :
  1. **Reproduire** : test capturant le all-NaN sur la surface standard (avant fix → rouge).
  2. **Corriger (ciblé)** : garde anti-overflow dans la récursion (détection/clamp du non-fini, borne sur
     la partie réelle de l'exposant) + intégration plus fine (`steps_per_year` ↑) pour la stabilité. Si un
     point `(u, params)` diverge malgré tout → NaN **local** propre, jamais un inf qui contamine tout.
  3. **Vérifier** : round-trip sur surface rHeston synthétique → mae/rmse finis sous tolérance lâche, et
     surface non-NaN.
- **Acceptation** : fit fini et raisonnable sur surfaces normales ; jamais de surface NaN silencieuse.
- **Escalade** : intégrateur robuste (RK4 / implicite / pas adaptatif) **seulement si** le fix ciblé ne
  converge pas (jugé peu probable).
- **Contrainte chirurgicale** : ne toucher que le chemin d'intégration du CF rHeston ; ne pas modifier les
  pricers/CF des autres modèles.

### Unité 3 — Filet anti-faux-succès (générique, transverse)

- **Quoi** : dans `CalibrationController.run_advanced_surface_calibration`, après le retour du calibrateur,
  valider le résultat : si `iv_model` est entièrement NaN **ou** `metrics` (mae/rmse) non finies →
  forcer `success=False` + message explicite (« Calibration dégénérée : surface modèle non finie (NaN). »).
- **Pourquoi centralisé (controller) plutôt que par calibrateur** : le controller construit déjà le
  `result_dict` consommé par l'UI/diagnostics ; un seul point protège les 6 modèles + tout futur modèle ;
  c'est exactement le mode de défaillance observé sur rHeston.
- **Portée** : s'applique au chemin `calibrator_map` (sabr, merton, rheston, rbergomi, volterra,
  heston legacy). Le chemin spécial `heston_v1` (`run_heston_global_calibration`) garde son comportement
  existant (déjà finite-safe) — la garde ne doit pas casser son fallback NN-only.

### Unité 4 — Suite de tests quant (`tests/quant/`)

100 % offline, déterministe (seeds fixes), alignée sur la structure/markers existants
(`tests/quant/test_heston_pricing.py` comme référence ; markers confirmés au moment du plan).

- **Round-trip par modèle** : générer une surface IV *depuis le modèle* avec des paramètres connus (via son
  propre pricer/CF/simulateur), calibrer, asserter :
  - `success=True`, `metrics` finies, `iv_model`/`iv_error` 2D de shape `(len(t_grid), len(m_grid))`,
    params dans les bornes, `mae` < tolérance.
  - Tolérances : **SABR** serrée (analytique) ; **Merton** modérée ; **rHeston** lâche (post-fix) +
    assertion explicite **non-NaN** ; **rBergomi/Volterra** `mc_cfg` minuscule + seed fixe → lâche (bruit
    MC), focus succès + finitude + shapes + bornes.
- **Test anti-faux-succès** : surface IV dégénérée (all-NaN) → assert `success=False` (valide l'Unité 3).
- **Régression rHeston** : la surface qui produisait all-NaN produit maintenant une surface finie
  (verrouille l'Unité 2).

## Hors périmètre (explicite)

- Bug LS `heston_v1` (broadcast `(21,)→(20,)`, fallback NN-only constaté à la vérification) → **follow-up
  séparé**, non corrigé ici. `heston_v1` reste fonctionnel via son fallback.
- Optimisation perf (vectorisation des boucles SABR, cache du kernel rHeston), réduction de variance MC
  (antithétique, common random numbers), calibration jointe de β SABR, étape de polissage Nelder-Mead pour
  les MC → non nécessaires (perf et fonctionnement vérifiés OK).
- Tout déploiement / nouvelle source de surface marché.

## Risques / questions ouvertes

- Le fix rHeston ciblé pourrait ne pas suffire sur des surfaces très « rough » (H petit) → escalade
  intégrateur prévue en repli.
- Round-trip MC (rBergomi/Volterra) intrinsèquement bruité → tolérances lâches + seeds fixes ; on teste la
  robustesse/forme, pas une récupération exacte des paramètres.
- Convention exacte des markers de tests (`unit`/`smoke`/`slow`) et blocage réseau en CI → à confirmer en
  lisant `tests/` au moment du plan.
- La garde anti-faux-succès (Unité 3) ne doit pas requalifier en échec un succès légitime partiel (ex.
  SABR avec NaN sur des maturités non observées) : le critère est « **entièrement** NaN » ou « métriques
  non finies », pas « contient des NaN ».
