"""DQN: no training on the app's render path — PLAN.md step 5."""

from __future__ import annotations

import pytest

from app.model.hedger_v2 import dqn_hedger as dh

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_model(monkeypatch, tmp_path):
    # Hermetic: cache dir = empty tmp; tracked dir = nonexistent (no shipped pair),
    # so reads resolve to the empty cache and writes go to the empty cache too.
    monkeypatch.setattr(dh, "_CACHED_AGENT", None, raising=False)
    monkeypatch.setattr(dh, "_CACHED_META", None, raising=False)
    monkeypatch.setattr(dh, "_model_dir", lambda: tmp_path)
    monkeypatch.setattr(dh, "_TRACKED_WEIGHTS_DIR", tmp_path / "no_tracked")
    yield


def test_load_returns_unavailable_and_never_trains(tmp_path):
    meta = dh.load_or_train_dqn_model(force_retrain=False)
    assert meta["available"] is False
    assert meta["status"] == "model_unavailable"
    # Critical: no checkpoint was written -> the app did not train at runtime.
    assert not dh._cache_weights_path().exists()


def test_get_cached_agent_raises_typed_unavailable():
    with pytest.raises(dh.DQNModelUnavailable):
        dh._get_cached_agent()


def test_suggest_hedge_action_degrades_gracefully():
    # client=None is fine: the unavailable branch returns before touching the client.
    out = dh.suggest_hedge_action(client=None, underlying_symbol="aapl")
    assert out["available"] is False
    assert out["side"] == "none"
    assert out["delta_qty"] == 0.0
    assert out["underlying"] == "AAPL"


@pytest.mark.slow
def test_train_then_load_roundtrip(tmp_path, monkeypatch):
    import numpy as np

    # Training now pulls historical Alpaca bars; feed deterministic synthetic prices
    # so the train -> cache -> load roundtrip runs offline (no network).
    prices = (100.0 + np.cumsum(np.random.default_rng(0).normal(0, 1, 300))).astype(np.float32)
    monkeypatch.setattr(dh, "_load_historical_prices", lambda cfg: prices)

    base = dh.DQNConfig()
    cfg = dh.DQNConfig(
        **{
            **base.__dict__,
            "train_steps": 200,
            "warmup_steps": 30,
            "eval_episodes": 1,
            "historical_episode_length": 32,
        }
    )
    meta = dh.train_dqn_model(config=cfg)
    assert meta["available"] is True
    assert dh._cache_weights_path().exists()

    # Fresh process-like state: clear cache, then a plain load must succeed.
    dh._CACHED_AGENT = None
    dh._CACHED_META = None
    meta2 = dh.load_or_train_dqn_model(force_retrain=False)
    assert meta2["available"] is True
