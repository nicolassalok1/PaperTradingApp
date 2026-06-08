# MVC import matrix (canonical architecture rule)

Single source of truth = `scripts/check_mvc_integrity.py` (enforced by the CI gate
and by `scripts/test_mvc_rules.py`, which delegates to it). This document is the
human-readable statement of the same rule — write/agree the rule here first, then
the code and the test converge to it.

Layers live under `app/`: `model/`, `controller/`, `vue/` (view), `utils/`.

## Allowed / forbidden imports

| From ↓ \ May import → | streamlit | app.model | app.controller | app.vue | app.utils |
|---|:---:|:---:|:---:|:---:|:---:|
| **model**      | ❌ | ✅ (self) | ❌ | ❌ | ✅ |
| **controller** | ❌ | ✅ | ✅ (self) | ❌ | ✅ |
| **view (vue)** | ✅ | ❌ | ✅ | ✅ (self) | ❌ |
| **utils**      | ❌ | ❌ | ❌ | ❌ | ✅ (self) |

Additional hard rule:
- `app/model/options/**` must contain **no** `streamlit` reference at all (text scan),
  not even in strings/comments.

## Rationale
- **utils is the leaf**: pure helpers, no dependency on view/model/controller/streamlit.
  Streamlit's `st.secrets` is injected from the composition root (`streamlit_app.py`,
  which lives outside `app/` and is not bound by this matrix) via
  `app.utils.secrets.set_secret_source`.
- **view never reaches model/utils directly**: it talks to controllers only, so business
  logic and helpers stay swappable and testable headless.
- **model/controller are Streamlit-free**: they must run in plain Python (tests, CLI,
  batch) with no UI runtime.

## Composition root exception
`streamlit_app.py` and `conftest.py` (repo root, outside `app/`) wire layers together
and may import both Streamlit and `app.utils`. They are intentionally out of scope of
the gate.
