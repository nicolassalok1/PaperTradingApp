"""
Streamlit Community Cloud entrypoint.

This is the composition root: it lives OUTSIDE the layered `app/` package and is
therefore not subject to the MVC import rules. It is the one place allowed to wire
the view (Streamlit) to the pure ConfigProvider in `app.utils.secrets`, then delegate
to the real app entrypoint in `app/vue/main_app.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _wire_streamlit_secrets() -> None:
    """Inject `st.secrets` as a secondary secret source (after OS env vars)."""
    from app.utils.secrets import set_secret_source

    def _source(key: str):
        try:
            import streamlit as st  # imported here, at the composition root only

            return st.secrets.get(key)  # type: ignore[attr-defined]
        except Exception:
            return None

    set_secret_source(_source)


_wire_streamlit_secrets()


from app.vue.main_app import main


main()
