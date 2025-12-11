# Contributing to PaperTradingApp (Options Module)

## Coding Rules
- Use Black/Isort style (see `pyproject.toml`).
- Import order example:
  - `import streamlit as st`
  - `from app.utils.options.layout import option_panel, params_expander, compute_button, render_crr_payoff_surface`
  - domain helpers from `app.model.*` or `app.utils.options.pricing`
  - third-party libs (numpy/pandas/etc.).
- Keep pricing logic in `app/vue/components/options/pricing.py` or the relevant model module (e.g., `app/model/heston/*`). Avoid heavy computation in UI code.
- UI panels: only orchestration + widgets; no heavy computation inline.

## UI Glossary (French)
- Sous-jacent (S0)
- Strike (K)
- Maturité T (années)
- Volatilité (sigma)
- Taux sans risque (r)
- Dividend yield (q)
- Nombre d'étapes (n_steps)
- Type d'option (Call / Put)

## Panel Structure
1. `option_panel(title, subtitle=None)`
2. `with params_expander():` inputs grouped
3. `if compute_button():` perform pricing and display result
4. Optional diagnostics: CRR surface in an expander

## Tests
- Run: `pytest scripts/test_options_*.py`
- CI runs full test suite.

## CI
- GitHub Actions workflow in `.github/workflows/tests.yml` installs deps and runs pytest on push/PR.

