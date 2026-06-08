"""The shipped DQN checkpoint loads without training, and its checksum matches.

Closes PLAN step 5's 'versioned checkpoint' deliverable: a fresh checkout has a
working model (no runtime training). No network.
"""

from __future__ import annotations

import hashlib

import pytest

from app.model.hedger_v2 import dqn_hedger as dh

pytestmark = pytest.mark.unit

_NPZ = dh._TRACKED_WEIGHTS_DIR / f"{dh.DQN_HEDGER_VERSION}.npz"
_SHA = dh._TRACKED_WEIGHTS_DIR / f"{dh.DQN_HEDGER_VERSION}.npz.sha256"


def test_shipped_checkpoint_present_and_loads_without_training():
    assert _NPZ.exists(), f"shipped DQN checkpoint missing at {_NPZ}"
    # Clear in-memory cache; use the REAL paths (tracked checkpoint present).
    dh._CACHED_AGENT = None
    dh._CACHED_META = None
    meta = dh.load_or_train_dqn_model(force_retrain=False)
    assert meta["available"] is True
    assert str(_NPZ) == str(dh._weights_path())  # tracked path is preferred


def test_shipped_checkpoint_checksum_matches():
    assert _SHA.exists(), f"checksum file missing at {_SHA}"
    expected = _SHA.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(_NPZ.read_bytes()).hexdigest()
    assert actual == expected, "DQN checkpoint corrupted (checksum mismatch)"


@pytest.mark.slow
def test_training_does_not_clobber_shipped_checkpoint():
    # Training writes to the gitignored cache, never the tracked shipped checkpoint.
    before = hashlib.sha256(_NPZ.read_bytes()).hexdigest()
    base = dh.DQNConfig()
    cfg = dh.DQNConfig(
        **{**base.__dict__, "train_steps": 100, "warmup_steps": 30, "eval_episodes": 1}
    )
    dh.train_dqn_model(config=cfg)
    after = hashlib.sha256(_NPZ.read_bytes()).hexdigest()
    assert before == after, "training overwrote the tracked shipped checkpoint"
    assert dh._cache_weights_path().exists()
