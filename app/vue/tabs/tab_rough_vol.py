"""
🌊 Rough Vol (H) — pipeline de volatilité rugueuse rBergomi (spec 4.1 → 4.11).

Deux boutons, jamais un seul :

1. **Préparer (4.1 → 4.9)** — nettoyage des chaînes, courbe forward par parité,
   surface OTM, strikes de swap de variance K_var, courbe de variance forward ξ₀,
   estimation asymptotique H0 par le skew ATM, puis point de départ (H₀, η₀, ρ₀).
   Tout est en forme fermée ou en quadrature : aucune simulation, aucune attente.
2. **Calibrer (4.10 / 4.11)** — l'ajustement joint (H, η, ρ) à ξ₀ **figé**, en
   Monte-Carlo. C'est le seul calcul coûteux et il est verrouillé derrière le
   devis affiché à l'étape 1 : nombre d'évaluations, nombre de trajectoires, et
   pourquoi l'heuristique « max_nfev × n_starts » de l'onglet *Calibration
   avancée* le sous-estime.

Deux garde-fous que cet onglet fait respecter à l'écran :

* **ξ₀ est une donnée**, jamais un paramètre : la courbe construite en 4.4 est
  passée telle quelle au calibrateur, qui est structurellement incapable de la
  bouger. La clé ``xi0`` est retirée des contraintes par le contrôleur.
* **``success=False`` est un verdict informatif**, pas une erreur à avaler : il
  signifie « cette surface n'identifie pas H ». Dans ce cas le triplet renvoyé
  ne mesure rien, il est affiché comme diagnostic (replié, barré d'un
  avertissement) et **jamais** comme un résultat de calibration.

Vue MVC : n'importe QUE des contrôleurs (jamais ``app.model`` / ``app.utils``).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

from app.controller.calibration_controller import CalibrationController
from app.vue.components.page_utils import render_page_header

TAB_LABEL = "🌊 Rough Vol (H)"

_PREPARE_KEY = "rough_vol_prepare_result"
_CALIB_KEY = "rough_vol_calibration_result"
_PAYLOAD_KEY = "rough_vol_prepare_payload"

#: Monte-Carlo presets. "Exploration" is cheap ON PURPOSE and will very often
#: report `success=False` — that is the calibrator saying it does not have the
#: budget to identify H, not a bug. The label says so.
_MC_PROFILES: Dict[str, Dict[str, Any]] = {
    "Exploration (rapide, peut ne pas identifier H)": {
        "n_design": 8,
        "stage1_paths": 2_000,
        "stage2_paths": 4_000,
        "profile_paths": 4_000,
        "final_paths": 20_000,
        "profile_points": 5,
        "valley_points": 5,
        "noise_replicates": 2,
        "refinement_check": False,
        "local_nfev_per_param": 20,
        "grid_n_max": 192,
    },
    "Référence (défauts spec 4.10)": {},
}


# ---------------------------------------------------------------------------
# Small view-side formatters (no numerics, no model import)
# ---------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 4, suffix: str = "") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if f != f:  # NaN
        return "—"
    return f"{f:.{digits}f}{suffix}"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def _render_steps(result: Dict[str, Any]) -> None:
    steps = result.get("steps") or []
    if not steps:
        return
    rows = [
        {
            "Étape": str(s.get("label_fr") or s.get("step") or ""),
            "État": "✅" if s.get("ok") else "⛔",
            "Détail": str(s.get("message_fr") or ""),
        }
        for s in steps
        if isinstance(s, dict)
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_cost(cost: Dict[str, Any]) -> None:
    """The price tag. Shown BEFORE the expensive button is even enabled."""
    if not isinstance(cost, dict) or not cost.get("success"):
        st.warning(str((cost or {}).get("message") or "Devis Monte-Carlo indisponible."))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Évaluations MC", _fmt_int(cost.get("n_evaluations")))
    c2.metric("Trajectoires simulées", _fmt_int(cost.get("n_paths_total")))
    c3.metric("Paramètres libres", _fmt_int(cost.get("n_free_parameters")))
    c4.metric("Grille (n_max)", _fmt_int(cost.get("grid_n_max")))

    st.warning(str(cost.get("message_fr") or ""))
    st.caption(str(cost.get("wall_time_fr") or ""))

    # The Phase-4 note: `max_nfev == 80` cannot be told apart from "not passed".
    if cost.get("max_nfev_is_ambiguous"):
        st.info(str(cost.get("max_nfev_ambiguity_fr") or ""))
    else:
        st.caption(str(cost.get("max_nfev_ambiguity_fr") or ""))

    with st.expander("Détail du devis par étape", expanded=False):
        st.caption(
            "L'onglet « Calibration avancée » chiffre un run comme "
            "`per_eval × max_nfev × n_starts`, c'est-à-dire la seule ligne "
            "« recherche locale » ci-dessous — "
            f"{_fmt_int(cost.get('heuristic_evaluations'))} évaluations avec les "
            f"réglages demandés ici, contre {_fmt_int(cost.get('n_evaluations'))} "
            f"réellement exécutées (facteur ≈ {_fmt(cost.get('ratio_vs_heuristic'), 1)}). "
            "Toutes les autres lignes lui échappent, et son `per_eval` est "
            "calibré sur des modèles FFT, pas sur du Monte-Carlo."
        )
        stages = cost.get("stages") or []
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Étape": str(s.get("label_fr") or s.get("stage") or ""),
                        "Évaluations": int(s.get("n_evaluations") or 0),
                        "Trajectoires / évaluation": int(s.get("n_paths_per_evaluation") or 0),
                        "Trajectoires": int(s.get("n_paths_total") or 0),
                    }
                    for s in stages
                    if isinstance(s, dict)
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _render_preparation(result: Dict[str, Any]) -> None:
    """Everything 4.1 → 4.9 produced. Numbers only, no verdict on H."""
    forward = result.get("forward_curve") or {}
    variance = result.get("variance_swap") or {}
    xi0 = result.get("forward_variance") or {}
    hurst = result.get("hurst") or {}
    initializer = result.get("initializer") or {}
    cleaning = result.get("cleaning") or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Échéances retenues", _fmt_int(result.get("n_maturities")))
    c2.metric("Cotations OTM", _fmt_int(result.get("n_quotes")))
    c3.metric("Spot S₀", _fmt(result.get("S0"), 2))
    c4.metric("K_var exploitables", _fmt_int(variance.get("n_points")))

    tab_x, tab_h, tab_i, tab_f, tab_c = st.tabs(
        [
            "ξ₀ (4.4)",
            "H₀ par le skew (4.5)",
            "Point de départ (4.9)",
            "Forward & K_var (4.2/4.3)",
            "Nettoyage (4.1)",
        ]
    )

    with tab_x:
        st.caption(
            "ξ₀ est **une donnée** reconstruite des swaps de variance : elle est "
            "figée pendant l'ajustement joint et l'optimiseur ne peut pas la modifier."
        )
        knots = xi0.get("T_knots") or []
        levels = xi0.get("levels") or []
        vols = xi0.get("xi0_vol") or []
        if knots:
            df_xi = pd.DataFrame(
                {
                    "T (ans)": [float(t) for t in knots],
                    "ξ₀ (variance)": [float(v) for v in levels],
                    "√ξ₀ (vol)": [float(v) for v in vols] if vols else [float("nan")] * len(knots),
                }
            )
            st.dataframe(df_xi, width="stretch", hide_index=True)
            st.line_chart(df_xi.set_index("T (ans)")[["√ξ₀ (vol)"]])
        st.caption(
            f"Méthode : {xi0.get('method', '—')} · extrapolation : "
            f"{xi0.get('extrapolation_policy', '—')} · erreur de reconstruction max : "
            f"{_fmt(xi0.get('max_abs_reconstruction_error'), 12)}"
        )

    with tab_h:
        st.metric("H₀ (estimation asymptotique)", _fmt(hurst.get("H0")))
        ci = hurst.get("ci95") or []
        if len(ci) == 2:
            st.caption(
                f"IC95 [{_fmt(ci[0])}, {_fmt(ci[1])}] · R² = {_fmt(hurst.get('r2'), 4)} · "
                f"{_fmt_int(hurst.get('n_expiries'))} échéance(s)"
            )
        st.info(str(hurst.get("message_fr") or ""))
        if hurst.get("warning_fr"):
            st.warning(str(hurst["warning_fr"]))
        if hurst.get("unstable"):
            st.error(
                "Régression instable : H₀ est une valeur de repli, utilisable "
                "seulement comme point de départ."
            )
        for reason in hurst.get("rejection_reasons_fr") or []:
            st.caption(f"• {reason}")

    with tab_i:
        p0 = result.get("initial_params") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("H₀", _fmt(p0.get("H")))
        c2.metric("η₀", _fmt(p0.get("eta")))
        c3.metric("ρ₀", _fmt(p0.get("rho")))
        st.info(str(initializer.get("message_fr") or ""))
        if initializer.get("warning_fr"):
            st.warning(str(initializer["warning_fr"]))
        if initializer.get("rho_eta_degeneracy_fr"):
            st.caption(str(initializer["rho_eta_degeneracy_fr"]))
        for warn in initializer.get("warnings_fr") or []:
            st.caption(f"• {warn}")

    with tab_f:
        points = forward.get("points") or []
        if points:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "T (ans)": float(p.get("T", float("nan"))),
                            "F": float(p.get("F", float("nan"))),
                            "D": float(p.get("D", float("nan"))),
                            "r": float(p.get("r", float("nan"))),
                            "q implicite": float(p.get("q_implied", float("nan"))),
                            "source": str(p.get("discount_source", "")),
                        }
                        for p in points
                        if isinstance(p, dict)
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        mats = variance.get("maturities") or []
        kvars = variance.get("k_var") or []
        if mats and len(mats) == len(kvars):
            st.dataframe(
                pd.DataFrame({"T (ans)": [float(t) for t in mats],
                              "K_var": [float(v) for v in kvars]}),
                width="stretch",
                hide_index=True,
            )
        for failure in variance.get("failures") or []:
            if isinstance(failure, dict):
                st.caption(f"⛔ T = {_fmt(failure.get('T'), 4)} : {failure.get('message_fr', '')}")
        flags = variance.get("flags") or []
        if flags:
            st.caption(
                "Signalements 4.3 : " + ", ".join(str(f) for f in flags) +
                " — K_var porte le biais de discrétisation CBOE, quantifié et signalé, "
                "jamais corrigé en silence."
            )

    with tab_c:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cotations conservées", _fmt_int(cleaning.get("n_quotes_kept")))
        c2.metric("Cotations retirées", _fmt_int(cleaning.get("n_quotes_removed")))
        c3.metric("Échéances utilisables (skew)", _fmt_int(cleaning.get("n_usable_for_skew")))
        reasons = cleaning.get("removals_by_reason") or {}
        labels = cleaning.get("removal_labels_fr") or {}
        if reasons:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Motif": str(labels.get(k, k)), "Nombre": int(v)}
                        for k, v in reasons.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        if cleaning.get("exercise_style_caveat"):
            st.caption(str(cleaning["exercise_style_caveat"]))


def _render_calibration(result: Dict[str, Any]) -> None:
    """
    The verdict. ``success=False`` is displayed as a REASON, never swallowed, and
    the parameter triple that comes with it is never presented as a result.
    """
    report = result.get("calibration") or {}
    params = result.get("params") or {}
    usable = bool(result.get("params_usable"))

    if usable:
        st.success(str(result.get("message") or "Calibration jointe réussie."))
        c1, c2, c3 = st.columns(3)
        c1.metric("H", _fmt(params.get("H")))
        c2.metric("η", _fmt(params.get("eta")))
        c3.metric("ρ", _fmt(params.get("rho")))
    else:
        st.error(str(result.get("message") or "Calibration en échec."))
        st.markdown(
            "**Aucun (H, η, ρ) n'est rendu comme résultat.** Le verdict "
            "`success=False` signifie que cette surface n'identifie pas H : le "
            "triplet où l'optimiseur s'est arrêté ne mesure rien."
        )

    # Flags, blocking first, each with its French sentence.
    details = result.get("flag_details") or []
    if details:
        blocking = [d for d in details if isinstance(d, dict) and d.get("blocking")]
        advisory = [d for d in details if isinstance(d, dict) and not d.get("blocking")]
        for d in blocking:
            st.error(f"⛔ **{d.get('flag')}** — {d.get('label_fr')}")
        for d in advisory:
            st.warning(f"⚠️ **{d.get('flag')}** — {d.get('label_fr')}")
    else:
        for warn in result.get("warnings_fr") or []:
            st.warning(str(warn))

    if report.get("identification_fr"):
        st.caption(str(report["identification_fr"]))
    if report.get("h_comparison_fr"):
        st.caption(str(report["h_comparison_fr"]))
    if report.get("grid_bias_message_fr"):
        st.caption(str(report["grid_bias_message_fr"]))

    metrics = result.get("metrics") or {}
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (vol)", _fmt(metrics.get("mae"), 5))
        c2.metric("RMSE (vol)", _fmt(metrics.get("rmse"), 5))
        c3.metric("|err| max (vol)", _fmt(metrics.get("max_abs"), 5))

    if not usable and params:
        with st.expander(
            "Diagnostic uniquement — le triplet où l'optimiseur s'est arrêté "
            "(À NE PAS CONSOMMER)",
            expanded=False,
        ):
            st.caption(
                "Affiché pour comprendre où la recherche s'est arrêtée sur une "
                "surface plate. Ce n'est pas une calibration."
            )
            st.json({k: params.get(k) for k in ("H", "eta", "rho")})

    with st.expander("Rapport 4.11 complet (identifiabilité, bruit MC, biais de grille)"):
        # `mean_evaluation_seconds` is MEASURED by the run that just happened —
        # the only honest per-evaluation duration this app ever shows.
        mean_eval = report.get("mean_evaluation_seconds")
        if mean_eval is not None:
            st.caption(
                f"Durée moyenne mesurée par évaluation : {_fmt(mean_eval, 3)} s sur "
                f"{_fmt_int(report.get('n_objective_evaluations'))} évaluations."
            )
        st.json(report)


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------
def render_tab() -> None:
    ctrl = CalibrationController()

    render_page_header(
        "Rough Vol (H)",
        "rBergomi — H calibré conjointement à (η, ρ), ξ₀ figé comme donnée de marché",
        icon="🌊",
        badge="Spec 4.1 → 4.11",
    )

    st.caption(
        "Chaîne complète : nettoyage 4.1 → forward 4.2 → K_var 4.3 → ξ₀ 4.4 → "
        "H₀ 4.5 → initialisation 4.9 → ajustement joint 4.10/4.11. "
        "H n'est **jamais** une constante de ce code : soit il est calibré et le "
        "rapport 4.11 dit à quel point la surface l'identifie, soit la calibration "
        "est déclarée en échec et rien n'est rendu."
    )

    # -- inputs ------------------------------------------------------------
    st.markdown("### Surface de marché")
    c1, c2, c3, c4 = st.columns([1.4, 1.0, 1.0, 1.0])
    with c1:
        ticker = st.text_input("Ticker (Yahoo)", value="SPY", key="rv_ticker").strip().upper()
    with c2:
        max_years = float(
            st.number_input("Maturité max (ans)", min_value=0.05, max_value=5.0,
                            value=2.0, step=0.25, key="rv_max_years")
        )
    with c3:
        max_expiries = int(
            st.number_input("Échéances max", min_value=2, max_value=30, value=12,
                            step=1, key="rv_max_expiries")
        )
    with c4:
        use_cache = st.toggle("Cache disque", value=True, key="rv_use_cache")

    c5, c6, c7 = st.columns([1.2, 1.2, 1.4])
    with c5:
        pin_r = st.toggle(
            "Fixer r", value=False, key="rv_pin_r",
            help="Épingle l'actualisation 4.2 à un taux constant (run reproductible "
                 "hors ligne). Sans cela, la courbe de taux du dépôt est utilisée.",
        )
        r_override = (
            float(st.number_input("r", min_value=-0.05, max_value=0.25, value=0.02,
                                  step=0.005, format="%.4f", key="rv_r"))
            if pin_r
            else None
        )
    with c6:
        window_lo = float(
            st.number_input("Fenêtre skew — T min (ans)", min_value=0.0, max_value=1.0,
                            value=0.0, step=0.01, format="%.4f", key="rv_win_lo")
        )
        window_hi = float(
            st.number_input("Fenêtre skew — T max (ans)", min_value=0.0, max_value=2.0,
                            value=0.0, step=0.01, format="%.4f", key="rv_win_hi")
        )
    with c7:
        profile_label = st.selectbox(
            "Profil Monte-Carlo (4.10)", list(_MC_PROFILES), index=1, key="rv_mc_profile",
            help="« Exploration » réduit fortement le budget : le calibrateur "
                 "répondra très probablement qu'il n'identifie pas H. C'est un "
                 "verdict honnête, pas une panne.",
        )
        seed_raw = int(
            st.number_input("Graine (0 = aucune)", min_value=0, max_value=2**31 - 2,
                            value=0, step=1, key="rv_seed")
        )

    window = [window_lo, window_hi] if window_hi > window_lo > 0.0 else None
    mc_cfg = dict(_MC_PROFILES.get(profile_label) or {})

    payload: Dict[str, Any] = {
        "ticker": ticker,
        "max_maturity_years": max_years,
        "max_expiries": max_expiries,
        "use_cache": bool(use_cache),
        "seed": seed_raw or None,
    }
    if r_override is not None:
        payload["r"] = r_override
    if window is not None:
        payload["short_maturity_window"] = window
    if mc_cfg:
        payload["mc_cfg"] = mc_cfg

    # -- step 1: cheap preparation ----------------------------------------
    st.markdown("### 1 — Préparation (4.1 → 4.9) · sans Monte-Carlo")
    if st.button(
        "Préparer la surface", type="primary", width="stretch",
        disabled=not ticker, key="rv_prepare_btn",
    ):
        with st.spinner("Nettoyage, forward, K_var, ξ₀, H₀, initialisation…"):
            prepared = ctrl.run_rbergomi_hurst_pipeline({**payload, "stage": "prepare"})
        st.session_state[_PREPARE_KEY] = prepared
        st.session_state[_PAYLOAD_KEY] = payload
        st.session_state.pop(_CALIB_KEY, None)

    prepared = st.session_state.get(_PREPARE_KEY)
    if not isinstance(prepared, dict):
        st.info("Lance la préparation pour voir la surface, ξ₀ et le devis de la calibration.")
        return

    _render_steps(prepared)
    if not prepared.get("success"):
        st.error(str(prepared.get("message") or "Préparation en échec."))
        failed = prepared.get("failed_step")
        if failed:
            st.caption(f"Étape en cause : {failed}")
        return

    _render_preparation(prepared)

    # -- step 2: the expensive fit, behind its price tag -------------------
    st.markdown("### 2 — Calibration jointe (4.10 / 4.11) · Monte-Carlo coûteux")
    st.caption(
        "ξ₀ construite ci-dessus est passée **par référence et en lecture seule** : "
        "elle n'entre pas dans le vecteur de paramètres. Seuls (H, η, ρ) sont ajustés."
    )
    cost = prepared.get("cost") or {}
    _render_cost(cost)

    acknowledged = st.checkbox(
        "J'ai lu le devis ci-dessus et j'accepte de lancer la simulation Monte-Carlo.",
        value=False,
        key="rv_ack_cost",
    )
    if st.button(
        f"Calibrer (H, η, ρ) — {_fmt_int(cost.get('n_paths_total'))} trajectoires",
        type="primary",
        width="stretch",
        disabled=not acknowledged,
        key="rv_calibrate_btn",
    ):
        run_payload = dict(st.session_state.get(_PAYLOAD_KEY) or payload)
        run_payload["stage"] = "full"
        with st.spinner(
            "Calibration jointe en cours — durée inconnue à l'avance, "
            "voir le devis ci-dessus."
        ):
            st.session_state[_CALIB_KEY] = ctrl.run_rbergomi_hurst_pipeline(run_payload)

    calibrated = st.session_state.get(_CALIB_KEY)
    if isinstance(calibrated, dict):
        st.markdown("### Verdict")
        _render_steps(calibrated)
        _render_calibration(calibrated)


__all__ = ["TAB_LABEL", "render_tab"]
