"""DQN checkpoint resolution & no-clobber guarantees. No network.

Under the historical-training version, a checkpoint cannot be generated offline
(training pulls Alpaca bars), so the shipped-checkpoint check skips when none is
present. The resolution/no-clobber mechanism is always tested.
"""

from __future__ import annotations

import hashlib

import pytest

from app.model.hedger_v2 import dqn_hedger as dh

pytestmark = pytest.mark.unit

_NPZ = dh._TRACKED_WEIGHTS_DIR / f"{dh.DQN_HEDGER_VERSION}.npz"
_SHA = dh._TRACKED_WEIGHTS_DIR / f"{dh.DQN_HEDGER_VERSION}.npz.sha256"


def test_write_target_is_cache_never_tracked():
    # Training writes to the cache dir; reads may prefer the tracked dir. They are
    # distinct, so training can never clobber a shipped (tracked) checkpoint.
    tracked_npz = dh._TRACKED_WEIGHTS_DIR / f"{dh.DQN_HEDGER_VERSION}.npz"
    assert dh._cache_weights_path() != tracked_npz
    assert dh._cache_weights_path().parent == dh._model_dir()


def test_read_prefers_complete_tracked_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(dh, "_TRACKED_WEIGHTS_DIR", tmp_path / "tracked")
    monkeypatch.setattr(dh, "_model_dir", lambda: tmp_path / "cache")
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)

    # No tracked files -> read resolves to cache.
    assert dh._read_dir() == tmp_path / "cache"

    # A COMPLETE tracked pair -> read resolves to tracked.
    td = tmp_path / "tracked"
    td.mkdir(parents=True, exist_ok=True)
    (td / f"{dh.DQN_HEDGER_VERSION}.npz").write_bytes(b"x")
    (td / f"{dh.DQN_HEDGER_VERSION}.json").write_text("{}", encoding="utf-8")
    assert dh._read_dir() == td

    # Incomplete tracked (only .npz) -> fall back to cache (never a mismatched pair).
    (td / f"{dh.DQN_HEDGER_VERSION}.json").unlink()
    assert dh._read_dir() == tmp_path / "cache"


def test_shipped_checkpoint_loads_and_checksum_if_present():
    if not _NPZ.exists():
        pytest.skip("no shipped checkpoint for this version (trained offline via Alpaca)")
    if _SHA.exists():
        expected = _SHA.read_text(encoding="utf-8").split()[0]
        assert hashlib.sha256(_NPZ.read_bytes()).hexdigest() == expected, "checkpoint corrupted"
    dh._CACHED_AGENT = None
    dh._CACHED_META = None
    meta = dh.load_or_train_dqn_model(force_retrain=False)
    assert meta["available"] is True
