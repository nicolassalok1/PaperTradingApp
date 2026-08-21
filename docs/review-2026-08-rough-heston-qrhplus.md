# Review — rough Heston (Calibration avancée) à la lumière du papier QRH+ (SSRN 6072928)

**Date** : 21 août 2026 · **Papier** : Bourgey, Noble, Petursson, Rosenbaum, Szymanski, *The Quadratic Rough Heston+ Model for Short-Dated Options* (13 juillet 2026) · **Périmètre code** : `app/model/volatility_models/rheston/` + `common/fft.py` + `calibrator_fft.py` + wiring `tab_advanced_calibration.py` / `calibration_controller.py` · **Méthode** : chaque affirmation chiffrée est reproduite par script (annexe), avec deux oracles indépendants (Heston fermé d'Albrecher pour H→0,5 ; schéma d'Adams fractionnaire d'El Euch–Rosenbaum pour le régime rough).

---

## Résumé exécutif

Le papier ne décrit pas notre modèle : il introduit le **QRH+** (Quadratic Rough Heston + terme de boost), calibré en Monte-Carlo GPU sur des options SPX très courtes (0DTE). Notre brique est le **rough Heston classique** (El Euch–Rosenbaum : κ, θ, ξ, ρ, v0, H) pricé par approximation markovienne multi-facteurs + Riccati + FFT Carr-Madan. Les deux partagent exactement les mêmes fondations numériques — noyau fractionnaire approché en somme d'exponentielles, facteurs OU, schéma implicite-explicite — et c'est là que la comparaison mord.

**Verdict : la brique rHeston est actuellement hors service, et le papier contient précisément le schéma qui la répare.** Le système Riccati est intégré par Euler *explicite* alors que 5 des 12 facteurs du noyau sont raides (x·dt jusqu'à 558, la stabilité exige x·dt < 2). Résultat mesuré : la fonction caractéristique vaut ~0 pour tout u dès T ≥ 0,1 an (120/121 colonnes gelées), ou des valeurs jusqu'à 7,5·10⁵⁶ en très court terme ; la surface IV rendue est faite de NaN et de fausses vols de 1–2 % ; **H n'a strictement aucun effet sur la sortie** (H=0,1 et H=0,49 rendent la même surface). Le calibrateur, lui, s'arrête après **1 évaluation** sur un coût plat, renvoie tel quel son point de départ heuristique, et affiche `success=True — "OK (rHeston approx, expensive)"` avec 15 points de vol de MAE. Les trois tests dédiés passent en 0,75 s sur ce code cassé (assertions trop faibles), et sont de toute façon désélectionnés du gate CI (`slow`).

La bonne nouvelle : le schéma du papier (IMEX sur les facteurs OU, éq. §3.1 — identique à celui du lifted Heston d'Abi Jaber–El Euch) transposé à notre Riccati, plus deux corrections (facteur constant du noyau, pas de temps minimum par maturité), **remet la brique d'équerre à ~1–5 bp des deux oracles**, pour ~0,12 s par maturité en numpy pur. Le correctif de référence est écrit et validé (annexe).

| Mesure | Code actuel | Corrigé (validé) |
|---|---|---|
| CF vs Heston fermé, H→0,5, smiles T=0,1/0,25/1 | écart 363–513 bp (et sans aucun effet de H) | **0,7–1,7 bp** |
| CF vs Adams fractionnaire, H=0,07, T=1 sem | NaN / \|φ\| jusqu'à 10³⁰⁴ | **≈ 5 bp** |
| Surface modèle (grille des tests repo) | 9/15 NaN + fausses IV 1–2 % | finie et exacte |
| Calibration sur surface propre | `success=True`, nfev=1, params = x0, MAE 0,152 | fit réel, MAE 0,012 |

---

## 1. Ce que dit le papier (et ce qui est transposable)

**Le modèle.** QRH+ : dS = S√V(ρdW + √(1−ρ²)dW⊥), X_t = ∫K(t−s)√V_s dW_s avec K(s)=s^α/Γ(1+α), et V = a₀ + a₁(X−a₂)² + b₀·1_{X<x₀}(x₀−X)^p. Le terme b₀ (« boost ») est la seule nouveauté vs le QRH de Gatheral–Jusselin–Rosenbaum : il génère l'accélération du put-wing des smiles 0DTE. b₀=0 redonne le QRH. Huit paramètres dont trois fixés en pratique (p=1,5 ; a₀≈0 ; et en calibration jointe ρ=0,75, a₁=0,25, α=−0,4275 soit **H=0,0725**) — il ne reste que {a₂, x₀, b₀} par échéance.

**La simulation (§3.1) — le morceau directement réutilisable.** Schéma hybride multi-facteurs de Rømer : noyau scindé en partie singulière (traitée analytiquement sur le dernier pas) + somme d'exponentielles K_m(t)=Σc_j e^{−γ_j t} (coefficients Beylkin–Monzón). Les facteurs OU sont discrétisés **implicite-explicite** :

```
U^j_{t_k} = ( U^j_{t_{k−1}} + √V_{t_{k−1}} · δW_{t_k} ) / (1 + γ_j Δ)
```

C'est exactement l'ingrédient qui manque à notre code (finding C1). Ils utilisent **N=200 pas par maturité, y compris 0DTE** (pas un nombre de pas proportionnel à T), des trajectoires antithétiques, et précalculent les constantes d'approximation sur une grille fine en α.

**La calibration (§3.3).** Objectif en **espace vol**, erreur par strike normalisée par le spread de vol moyen de l'échéance, **poids vega avec plancher** (pour que les ailes et le call-wing comptent quand même), MSE pondérée, échéances équipondérées. Optimisation **sans gradient** (bruit MC) : 1000 candidats Halton → 30 meilleurs → 125 itérations de Differential Evolution best/2/bin → raffinement des 3 meilleurs. Recalibration intraday **amorcée par le fit précédent** (gain de temps « dramatique »). ~20 s sur H100 à Q=1M chemins.

**Les résultats.** Fits dans le bid-ask de 0 jour à 3 semaines avec une poignée de paramètres, y compris le 5 août 2024 (VIX ≈ 66) ; Heston classique incapable sur ces jours ; QRH sans b₀ rate le raidissement du put-wing. Validation dynamique par le **Skew-Stickiness Ratio** : SSR modèle ∈ [1,7, 1,9] vs empirique [1,3, 1,9]. À noter aussi la Remark 1 : a₂ joue le rôle d'une courbe de vol forward, il est *normal* qu'il dépende de la maturité.

**Transposable chez nous, par ordre de valeur** : (1) le schéma IMEX des facteurs → répare notre CF ; (2) N pas par maturité, pas ∝T ; (3) l'objectif de calibration en unités de spread avec vega planché ; (4) multi-start quasi-aléatoire + optimiseur global ; (5) fixer H (≈0,07) et les paramètres peu identifiés ; (6) warm start intraday ; (7) plus tard, le modèle QRH+ lui-même en MC (§6).

---

## 2. Ce qu'on a dans l'app

`RHestonFFTMarkovianModel` / `RHestonFFTMarkovianCalibrator` (« rHeston (Markovian approx) via FFT », onglet Calibration avancée, profil Normal fixé : max_nfev=80, n_starts=2). Pipeline : noyau fractionnaire ≈ Σwᵢe^{−xᵢt} par quadrature géométrique (`markovian_kernel.py`) → système de Riccati multi-facteurs pour la CF du log-rendement (`cf_markovian.py`) → prix calls par FFT Carr-Madan (`common/fft.py`, α=1,5, n=2048, η=0,25) → IV par Brent → moindres carrés `least_squares` sur erreurs de **prix relatives** (`calibrator_fft.py`).

La structure mathématique de la CF est **juste** : G reproduit bien F(u,x)=−½(u²+iu)+(iuρξ−κ)x+½ξ²x², la décomposition h=Σbᵢ est l'approximation multi-facteurs standard de la Riccati fractionnaire, et l'astuce V₀·I^{1−α}h = V₀·I¹F (α=H+½, l'ordre fractionnaire de la Riccati) est correcte. L'enchaînement multi-maturités en une seule intégration est même une bonne idée (que le calibrateur n'exploite pas — il appelle maturité par maturité). Tout le problème est numérique et méthodologique.

---

## 3. Findings

### C1 — Euler explicite sur un système raide par construction : la CF est du bruit

**`cf_markovian.py:98`** — `B = B + dt*(−rates·B + weights·G)` avec, ligne 52, `x_max = x_max_mult · steps_per_year` = 1000·120 = 120 000. Le pas étant dt ≈ 1/120, le produit x·dt du facteur le plus rapide vaut **~558** (et 5 facteurs sur 12 dépassent la limite de stabilité x·dt < 2). C'est structurel : augmenter `steps_per_year` ne change rien puisque x_max croît d'autant. Chaque colonne u≠0 diverge géométriquement (~×557 par pas), le feedback quadratique ½ξ²S² accélère en doublement d'exposant, et le garde-fou anti-overflow (lignes 104–109) gèle alors la colonne à φ≈0 — ou, pire, le clip `Re(A) ≤ +700` (ligne 113) laisse passer des φ ~ e⁷⁰⁰ finis.

Mesuré (params des tests repo, H=0,1, κ=2, θ=0,04, ξ=0,6, ρ=−0,5, v0=0,04) :

| T | comportement de φ(u), u ∈ [0, 60] |
|---|---|
| 1/52 | max\|φ\| = 7,5·10⁵⁶ ; 107/121 colonnes avec \|φ\|>1 (impossible pour une CF) |
| 0,05 | max\|φ\| = 1,0·10³⁰⁴ |
| 0,1 / 0,25 / 1,0 | 120/121 colonnes gelées : φ(u) ≈ 10⁻³⁰⁵ ∀u≠0 |

Conséquence sur la surface (grille des tests, m=0,9–1,1, T=0,25–1) : **9/15 NaN** (strikes ITM : prix FFT ≈ 0 < intrinsèque) et, sur l'aile OTM, des prix-bruit minuscules inversés en **fausses IV de 0,4 % à 2,2 %** — des nombres finis, plausibles à l'œil, entièrement artefactuels. Et le test décisif : **la surface à H=0,49 est identique bit à bit à celle à H=0,1**. Le paramètre qui définit le modèle n'a aucun effet.

**Correctif** (validé, annexe) : le schéma du papier. IMEX sur la partie linéaire — `B ← (B + dt·w·G)/(1 + x·dt)` — **plus** la prise en compte de la raideur du terme quadratique lui-même, qui impose dt ≲ 2/(ξ·u_max) pour les grands u de la grille FFT. Deux variantes validées :

- *(i)* IMEX + règle de pas `n_steps = max(spy·T, ξ·u_max·T/1,5)` ;
- *(ii)* **schéma totalement implicite** (recommandé) : à chaque pas, S_{n+1} résout la quadratique dtWγS² + (dtWβ−1)S + (P − dtW·half) = 0 (racine stable par formule de Citardauq), avec P = ΣBᵢ/(1+xᵢdt), W = Σwᵢ/(1+xᵢdt) — inconditionnellement stable, 200 pas fixes suffisent.

| Validation du correctif (ii) | écart max |
|---|---|
| vs Heston fermé (Albrecher), H=0,499, T=0,1/0,25/1 | 1,7 / 0,7 / 1,7 bp |
| vs Adams fractionnaire, H=0,07, ξ=0,3, T=1/52 | 5,0 bp |
| vs Adams, H=0,07, T=0,05 (CF elle-même) | \|Δφ\| ≤ 6,4·10⁻⁴ |
| coût (2048 u, 21 facteurs, 200 pas, numpy) | ~0,12 s / maturité |

(L'oracle Adams est lui-même validé contre Heston fermé : max\|Δφ\| = 3,3·10⁻⁴ à H=0,499.)

### C2 — Le calibrateur renvoie son point de départ avec `success=True`

Conséquence directe de C1, mais le défaut de conception est indépendant : avec des prix modèle ≈ 0 partout, le résidu vaut −1 en chaque point, le coût est **plat**, `least_squares` s'arrête après **nfev=1** avec `converged=True`, et `calibrator_fft.py:332` renvoie inconditionnellement `success=True, "OK (rHeston approx, expensive)"`. Reproduit sur une surface synthétique propre :

```
success=True  message='OK (rHeston approx, expensive)'
params = {H: 0.1, kappa: 2.0, theta: 0.0395, xi: 0.6, rho: -0.5, v0: 0.0395}   # = x0 heuristique
metrics = {mae: 0.152, rmse: 0.155, max_abs: 0.195}                            # 15 pts de vol !
run0: converged=True  cost=13.7  nfev=1
```

`apply_degeneracy_guard` ne bascule pas en échec car la surface n'est pas *entièrement* NaN (les fausses IV d'aile comptent comme finies). L'UI affiche donc un succès vert avec des métriques absurdes. Même famille que le finding M8 (SABR) de l'audit du 13 août : **`success` doit être conditionné** — convergence d'au moins un run *et* RMSE(IV) sous un seuil *et* fraction de points modèle finis ≥ 80 % sur le masque. Renforcer aussi le garde de dégénérescence (fraction finie, pas seulement « au moins une »).

### C3 — Résidus : un prix modèle manquant compte comme un fit parfait

**`calibrator_fft.py:249`** — `res = np.where(np.isfinite(res), res, 0.0)`. Tout point où le modèle échoue (NaN d'interpolation, colonne explosée…) contribue **zéro** au coût : l'optimiseur est activement récompensé quand il pousse les paramètres vers des régions où le pricer se casse. À remplacer par une pénalité explicite (résidu = constante élevée, p.ex. 10× le pire résidu fini) et un rejet de l'évaluation au-delà de ~10 % de points manquants.

### M1 — Troncature du noyau : la masse « lente » manque

La quadrature géométrique de `markovian_kernel.py` intègre la représentation de Laplace sur [x_min, x_max] et **jette la masse de [0, x_min)** — la composante lente du noyau. Elle se récupère presque gratuitement par un facteur constant w∞ = c·x_min^{1/2−H}/(1/2−H) de taux 0 (e^{−xt}≈1 sur [0,x_min] dès que x_min·T ≤ 0,1, ce que garantit x_min = 0,1/T_max). Mesuré à H=0,499 : sans w∞ le smile est plat à 20,0 % (**439 bp** d'écart au vrai smile Heston 16,8–24,4 %) ; avec w∞, ≤ 2 bp. À H≈0,1 le biais est moindre mais réel (~10–30 % de la valeur du noyau aux t moyens). Même approximation en somme d'exponentielles que le papier (Beylkin–Monzón) — notre quadrature géométrique est un substitut acceptable *une fois complétée du terme constant*.

### M2 — Pas de temps proportionnel à T : 2 pas pour une échéance d'une semaine

**`cf_markovian.py:84`** — `n_steps = max(1, round(steps_per_year·dt_total))` → T=1/52 reçoit **2 pas** d'intégration, T=0,02 (présent dans la grille par défaut `T_GRID_DEFAULT`) en reçoit 2. Pour une dynamique à noyau singulier, c'est le régime où il faut le plus de soin — le papier utilise **N=200 pas par maturité y compris 0DTE**. Correctif : `n_steps = max(200, round(spy·T))` (le coût reste ~0,12 s/maturité).

### M3 — Bornes ξ jusqu'à 5,0 : l'optimiseur visite des régions où le prix n'existe pas

Avec le damping Carr-Madan α=1,5, le pricer évalue la CF en u − 2,5i, ce qui exige E[S^{2,5}] < ∞. Mesuré à H=0,07, κ=2, T=0,25 (oracle Adams) : ξ=0,6 → 0 colonne explosée ; **ξ=1,5 → 17/121 ; ξ=3 → 72/121 ; ξ=5 → 89/121**. Les bornes actuelles (1e-3, 5,0) garantissent que `least_squares` traverse ces régions. Trois mesures : resserrer ξ à ≤ 1,0 (les calibrations publiées du rough Heston classique sur SPX donnent ξ de l'ordre de 0,3–0,4) ; garde à la frontière FFT : colonne non finie → 0 **avec compteur** exposé dans `details` (si > quelques %, évaluation rejetée — c'est la version saine du gel actuel) ; et plafonner Re(A) à ~+50, pas +700 (une valeur dampée légitime reste O(10) ; e⁷⁰⁰ est toujours un artefact).

### M4 — L'objectif de calibration contredit celui du papier

Résidus = erreurs de **prix relatives**, `scale = max(1e-4, px_mkt)` (`calibrator_fft.py:139`). Un point d'aile à 0,008 € mal pricé de 50 % pèse ~50× plus qu'une erreur de 1 % de vol ATM — exactement l'écueil que le papier évite : erreurs en **espace vol**, normalisées par le spread typique de l'échéance, **vega planché** pour que les ailes contribuent sans dominer, échéances équipondérées. L'app calcule déjà tout le nécessaire (`compute_bs_vega_grid`, `iv_error_metrics_weighted`) mais ne s'en sert que pour l'affichage. À déplacer dans l'objectif. (Le pipeline rBergomi en cours — spec §4.10 — fait précisément ce choix ; autant partager le module de pondération.)

### M5 — Identifiabilité : même corrigé, le problème est multi-modal

Démonstration involontaire de notre round-trip corrigé : surface générée à (H=0,12, κ=1,5, ξ=0,35, ρ=−0,65), calibrée à MAE 121 bp… sur (H=0,49, κ=8,0, ρ=−0,999) — deux paramètres aux bornes, un κ énorme qui mime la rugosité sur ces maturités (≥ 1 mois). C'est le point du papier : « it can be difficult to distinguish between certain parameter combinations », d'où leur protocole Halton→DE *et* le gel des paramètres peu identifiés (ils fixent p, a₀, puis ρ, a₁, α en calibration jointe). Transposition : **fixer H par défaut** (0,07–0,10, exposé comme contrainte — le mécanisme `constraints` le permet déjà), envisager κ fixe ou borné bas, utiliser `latin_hypercube_samples` (déjà dans `optimizers.py`) pour les starts, signaler dans `details` tout paramètre collé à une borne, et amorcer les recalibrations par le dernier fit. Rappel aussi : la rugosité ne s'identifie que sur les **maturités courtes** — la grille par défaut en contient (0,02, 0,05, 0,1), il faut des données dessus, pas seulement des slots.

### M6 — Des tests qui passent sur un modèle mort

`tests/quant/test_rheston_calibration.py` : vérifié, **3 passed en 0,75 s sur le code cassé**. Les assertions — `isfinite(iv_model).any()`, `isfinite(iv).sum() > 0`, round-trip mae < 5e-2 — sont satisfaites par les fausses IV d'aile, et le round-trip est tautologique : la « surface marché » générée est le même bruit que celui que la calibration reproduit, et x0 = paramètres générateurs. De plus le module est `slow` → jamais exécuté par le gate CI (`-m "unit or smoke"`). À remplacer par des tests-oracle rapides (la CF corrigée coûte des ms) sur le modèle de `test_mc_pricing.py` : (a) \|φ(u)\| ≤ 1 sur u réels, ∀T ; (b) H=0,499 vs Heston fermé ≤ 10 bp (l'oracle tient en 15 lignes, il est dans l'annexe) ; (c) sensibilité : la surface à H=0,1 doit différer de H=0,4 ; (d) en `slow`, H=0,07 vs Adams ≤ 15 bp et round-trip à H fixé.

### Mineurs

| # | Sujet |
|---|---|
| m1 | Le calibrateur appelle la CF **maturité par maturité** (`uniq_T` loop) alors que `rheston_log_return_cf_markovian` sait enchaîner les maturités en une intégration — ×2–4 gratuit une fois le schéma corrigé. Idem `model_fft.py`. |
| m2 | Défauts FFT incohérents : `FFTConfig()` par défaut n=2048 (`fft.py:15`) mais le parsing des contraintes retombe sur n=1024 (`calibrator_fft.py:157`). |
| m3 | `model_fft.py` duplique les bornes du calibrateur (risque de dérive) ; `param_bounds` n'est pas consommé par le calibrateur. |
| m4 | ETA UI (`tab_advanced_calibration.py:496-501`) : 0,05 s×5/éval → ~40 s affichés pour le profil Normal ; le schéma corrigé coûte ~0,5 s/éval, Jacobien FD compris → plutôt ~15 min (mesuré : 305 s pour 50 nfev × 1 start). À recaler (ou réduire nfev par défaut une fois l'objectif en vol-space, mieux conditionné). |
| m5 | `theta0 = iv_atm²` lit `iv_mkt[0, atm_j]` : si la première ligne de la grille est un slot non observé (NaN), l'heuristique retombe silencieusement sur 0,04. Prendre la première ligne *observée* du masque. |
| m6 | Surface marché : calls uniquement, pas de bid/ask (`market_surface.py`) — la normalisation par spread du papier n'est pas encore possible. Le pipeline rBergomi (CleanChain, §4.1–4.2 de la spec) apportera mid/spread/OTM-par-parité : brancher rHeston dessus plutôt que de dupliquer. |

---

## 4. Le correctif de référence (validé)

Implémentation complète dans `verif4_final.py` (annexe) — à porter dans `cf_markovian.py` avec ses tests. Les trois changements essentiels tiennent en peu de lignes :

```python
# 1) noyau : ajouter le facteur constant (masse [0, x_min))
w_inf = c * x_min**(0.5-H) / (0.5-H)      # taux 0, à concaténer à rates/weights

# 2) pas de temps : minimum par maturité, pas seulement ∝ T
n_steps = max(200, round(steps_per_year * T))

# 3) schéma : totalement implicite (Citardauq) — inconditionnellement stable
P = (B / (1 + rates*dt)).sum(axis=0);  W = (weights / (1 + rates*dt)).sum()
a2, b1, c0 = dt*W*gam, dt*W*beta - 1.0, P - dt*W*half
disc = np.sqrt(b1*b1 - 4*a2*c0)
qq   = -0.5*(b1 + np.where(np.real(np.conj(b1)*disc) >= 0, disc, -disc))
S    = c0 / qq                             # racine stable (S→P quand dt→0)
G    = -half + beta*S + gam*S*S
A   += dt*(iu*(r-q) + kappa*theta*0.5*(S_prev+S) + v0*0.5*(G_prev+G))
B    = (B + dt*weights*G) / (1 + rates*dt)
```

plus la garde de sortie : colonne non finie ou Re(A) > 50 → φ=0 **et** compteur remonté dans `details` (rejet de l'évaluation au-delà de ~10 %). La variante minimale IMEX (`B ← (B+dt·w·G)/(1+x·dt)` + règle de pas ∝ ξ·u_max·T) est validée aussi (0–5,4 bp) si l'on préfère rester au plus près du code actuel — mais l'implicite est plus robuste au même coût.

Ordre de grandeur après correctif : surface 6 maturités ≈ 0,7 s ; calibration profil Normal (80 nfev × 2 starts, Jacobien FD) ≈ 15 min en l'état — mesuré 305 s pour 50 nfev × 1 start sur 4 maturités —, ramenée à ~4–5 min avec le batch multi-maturités (m1) et H fixé (6 paramètres → 5, mieux conditionné).

---

## 5. Méthodologie de calibration : ce que le papier ferait de notre onglet

En reprenant §3.3 du papier point par point : objectif en espace vol normalisé par spread avec vega planché et échéances équipondérées (M4, m6) ; initialisation quasi-aléatoire multi-start (Halton/LHS — `latin_hypercube_samples` existe) au lieu d'un x0 unique + starts log-uniformes ; paramètres peu identifiés gelés par défaut (H≈0,07 exposé en contrainte, comme leur α=−0,4275) ; warm start du fit précédent pour les recalibrations ; et un critère de succès qui mérite son nom (C2). Le DE complet du papier n'est nécessaire que si l'on passe au pricing MC (bruit) — pour une CF/FFT déterministe et désormais lisse, `least_squares` multi-start est adapté.

Deux validations à leur manière, faciles chez nous : le **test du 5 août 2024** (jeu de données de stress à archiver dans `tests/fixtures/`) et, plus tard, le **SSR** comme diagnostic de dynamique (ils le calculent précisément via l'approximation en facteurs OU que nous partageons — bump des U^j initiaux, m≈20).

## 6. Aller plus loin : QRH+ comme nouveau modèle de l'onglet

Le papier est une recette d'implémentation complète, et l'app a déjà l'infrastructure : simulateur Volterra MC (`volterra/simulator.py`), conventions MC de la spec rBergomi (CRN, antithétiques, grille exacte par maturité, estimateur conditionnel), et bientôt courbe forward/ξ₀ propre. Un `qrh_plus/simulator_mc.py` suivrait §3.1 à la lettre : mêmes facteurs OU IMEX que notre CF corrigée, incréments δX ~ N(0, Δ^{2α+1}/((2α+1)Γ(1+α)²)), Cov(δX, δW) = Δ^{α+1}/Γ(α+2), V=V(X) avec le boost (et le cap x^p→ constante au-delà de \|x\|>M pour la martingalité, note 2 du papier), log-spot Euler. Le papier chiffre 445 ms pour 100k trajectoires ×200 pas sur un cœur CPU en C++ — en numpy vectorisé, quelques secondes par évaluation : pricing/visualisation de smiles 0DTE accessibles tout de suite ; la calibration DE complète restera lente sur CPU (à réserver à un usage batch, ou Q réduit + warm start). C'est le modèle que le papier démontre supérieur précisément là où l'app veut aller (très court terme) ; nouvelle clé de registre (`"qrh_plus"`), les trois édits d'enregistrement documentés dans la spec rBergomi.

## 7. Plan d'action proposé

Étape A — remettre la brique en service *(bloquant, indépendant du chantier rBergomi)* :
1. C1+M1+M2 : porter le schéma implicite + w∞ + n_steps min 200 dans `cf_markovian.py`, garde φ non finie → 0 avec compteur. *(moyen — le code de référence validé existe)*
2. C3 : pénalité sur résidus manquants + rejet d'évaluation > 10 % manquants. *(petit)*
3. C2 : `success` conditionné (convergence ∧ RMSE ∧ fraction finie), garde de dégénérescence par fraction. *(petit, à partager avec SABR/M8 de l'audit)*
4. M6 : tests-oracle unit (Heston fermé H→0,5, \|φ\|≤1, sensibilité en H) + slow (Adams, round-trip H fixé). *(moyen)*
5. En attendant A1–A4 : bandeau UI « modèle en réparation » sur l'onglet rHeston — des résultats passés « OK » ont pu être pris au sérieux.

Étape B — calibration à la hauteur du papier : 6. M4/m6 : objectif vol-space pondéré (module partagé avec la spec rBergomi §4.10). 7. M5 : H fixé par défaut + LHS multi-start + flag params aux bornes + warm start. 8. m1/m2/m4 : batch multi-maturités, défauts FFT unifiés, ETA recalée.

Étape C — extension : 9. Fixture 5 août 2024 + diagnostic par échéance courte. 10. QRH+ MC (§6) après la Phase 3–4 du pipeline rBergomi, en réutilisant ses briques de données. 11. SSR comme métrique de validation commune rBergomi/rHeston/QRH+.

---

## Annexe — reproduction

Scripts exécutés dans la session (copie du package `rheston` isolée, numpy 2.4.4 / scipy 1.17.1) :

| script | couvre |
|---|---|
| `verif1_diagnose.py` | C1 : \|φ\| par maturité, colonnes gelées, surface 9/15 NaN + fausses IV, insensibilité à H |
| `verif3_fix2.py` | oracles (Heston fermé Albrecher ; Adams fractionnaire validé à 3,3·10⁻⁴ du fermé), IMEX simple + w∞, explosion de moment vs ξ |
| `verif4_final.py` | schémas finaux (i) IMEX+règle de pas et (ii) implicite Citardauq ; validations 0–5 bp ; timings |
| `verif5_calibrator.py` | C2 avant/après : success=True/nfev=1/params=x0/MAE 0,152 → fit réel MAE 0,012 ; démonstration d'identifiabilité (M5) |
| `pytest tests/test_rheston_calibration.py` | M6 : 3 passed en 0,75 s sur le code cassé |

Rappels de correspondance papier ↔ code : α_papier = H − ½ (leur α=−0,4275 ⇔ H=0,0725) ; leur K(s)=s^α/Γ(1+α) ⇔ notre K(t)=t^{H−1/2}/Γ(H+½) ; leurs (c_j, γ_j) Beylkin–Monzón ⇔ nos (wᵢ, xᵢ) par quadrature géométrique (+ w∞ après correctif).
