"""Orchestrator — apply the fact-checkers' wording/arithmetic fixes to the review doc (idempotent)."""
from __future__ import annotations

import io
from pathlib import Path

DOC = Path(__file__).resolve().parents[2] / "docs" / "review-2026-08-iv-dashboard-alpaca.md"

REPS = [
    ("Phase 2 : sondes de l'orchestrateur écrites indépendamment des agents ; Phase 4 : 27 sceptiques",
     "Phase 2 : sondes de l'orchestrateur écrites indépendamment des agents ; Phase 3 (live Alpaca) : sautée, clés révoquées — Annexe B ; Phase 4 : 27 sceptiques"),
    ("| §4.6 Live | non vérifié | clés révoquées ; script prêt | — |",
     "| §4.6 Live | non vérifié | clés révoquées ; script prêt | — (Phase 3 sautée — Annexe B) |"),
    ("-            chain = download_options_alpaca(sym, feed=feed, max_pages=_SNAPSHOT_MAX_PAGES)\n",
     "-            chain = download_options_alpaca(sym, feed=feed_val, max_pages=_SNAPSHOT_MAX_PAGES)\n"),
    ("+            chain = download_options_alpaca(sym, feed=feed, max_pages=_SNAPSHOT_MAX_PAGES,",
     "+            chain = download_options_alpaca(sym, feed=feed_val, max_pages=_SNAPSHOT_MAX_PAGES,"),
    ("-        chain = download_options_alpaca(sym, feed=feed, max_pages=_SNAPSHOT_MAX_PAGES)",
     "-        chain = download_options_alpaca(sym, feed=feed_val, max_pages=_SNAPSHOT_MAX_PAGES)"),
    ('-        df = df[df.get("date", pd.Series(dtype=str)) != row["date"]]',
     '-        df = df[df.get("date", pd.Series(dtype=str)).astype(str) != today]'),
    ('-    result["generated_at"] = dt.datetime.now().strftime("%H:%M:%S")\n+    result["generated_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")',
     '-    result["generated_at"] = pd.Timestamp.now().strftime("%H:%M:%S")\n+    result["generated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")'),
    ("les deux autres préfèrent le `pop` (supprime aussi la variante C).",
     "le sceptique reproduction tranche pour le `pop` (supprime aussi la variante C) ; le sceptique relecture accepte les deux options."),
    ("Extrait brut de `scripts/review_iv_dashboard/orch_live_alpaca.out.txt` :",
     "Extrait abrégé de `scripts/review_iv_dashboard/orch_live_alpaca.out.txt` (blocs HTML 401 répétés et sondes « unfiltered » / « feed=opra » élidés, marqués « … » ; le fichier complet fait foi) :"),
    ("→ 33 candidats (0 C / 9 M / 24 m), 30 après fusion des doublons inter-dimensions + 1 candidat orchestrateur (`html-exception-text-in-log`).",
     "→ 33 candidats (0 C / 9 M / 24 m) ; 29 après fusion de 4 doublons inter-dimensions ; + 1 candidat orchestrateur (`html-exception-text-in-log`) = 30 soumis au panel."),
    ("601 appels d'outils, 60 scripts `p4_*.py`.",
     "601 appels d'outils, 60 sondes déclarées (58 fichiers `p4_*.py` + 1 sortie `.out.txt` + 1 ré-exécution d'un script `p1_`)."),
    ("Résultat : 29/30 survivent au panel ; **final 28 findings — 0 C, 5 M, 23 m**.",
     "Résultat : 29/30 survivent au panel (1 tué 3/3) ; 1 survivant supplémentaire tué ensuite par une mesure live de l'orchestrateur (Yahoo) ; **final 28 findings — 0 C, 5 M, 23 m**."),
    ("*Panel Phase 4 (`p4_*`, 60)* :", "*Panel Phase 4 (`p4_*`, 58 fichiers `.py`)* :"),
    ("durée de `test_app_boot.py` mesurée seulement par le panel (4,3–6 s).",
     "durée de `test_app_boot.py` non mesurée par le finder §4.5 (mesurée par l'orchestrateur : 3 passed in 7,51 s, Annexe A ; par le panel : 4,3–6 s)."),
]


def main() -> None:
    s = io.open(DOC, encoding="utf-8").read()
    applied, missing = 0, []
    for a, b in REPS:
        if a in s:
            s = s.replace(a, b)
            applied += 1
        elif b in s:
            applied += 1  # already applied
        else:
            missing.append(a[:80])
    io.open(DOC, "w", encoding="utf-8", newline="\n").write(s)
    print(f"applied {applied}/{len(REPS)}")
    for m in missing:
        print("MISSING:", m)


if __name__ == "__main__":
    main()
