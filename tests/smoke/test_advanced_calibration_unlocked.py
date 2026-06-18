"""Smoke guard: no advanced-calibration model is gated behind the
'en cours d'implémentation' placeholder.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def test_no_advanced_calibration_models_are_gated():
    from app.vue.tabs import tab_advanced_calibration as t

    assert hasattr(t, "_IN_PROGRESS_MODELS")
    assert t._IN_PROGRESS_MODELS == set()
