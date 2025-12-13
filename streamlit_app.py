"""
Streamlit Community Cloud entrypoint.

This delegates to the existing app entrypoint in `app/vue/main_app.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from app.vue.main_app import main


main()

