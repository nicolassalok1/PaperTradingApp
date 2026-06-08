"""
Deep Q-Network (DQN) hedger (numpy-only).

Goal: suggest a *discrete* hedge adjustment (hold / buy / sell / flatten) for an
underlying stock, given a simplified state (net option delta proxy + equity hedge).

The model is trained offline on historical price paths (Alpaca bars) and persisted under:
`cache/HedgerDQN/`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np

from app.utils.paths import CACHE_CSV_DIR

from .alpaca_client import AlpacaHedgerClient
from .env import HedgingEnv, HistoricalHedgingEnv


DQN_HEDGER_VERSION = "dqn-v3-historical"


@dataclass(frozen=True)
class DQNConfig:
    # State encoding
    state_keys: Tuple[str, ...] = ("net_delta_proxy", "equity_position")
    delta_scale: float = 10.0
    position_scale: float = 10.0

    # Network
    hidden_sizes: Tuple[int, ...] = (64, 64)

    # DQN hyperparams
    gamma: float = 0.98
    learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 25_000
    warmup_steps: int = 800
    target_update_interval: int = 250

    # Exploration schedule (training)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 4_000

    # Training loop
    train_steps: int = 7_500
    eval_episodes: int = 25

    # Historical env (only mode kept)
    train_mode: str = "historical"  # enforced historical-only training
    env_position_scale: float = 1.0
    historical_symbol: str = "SPY"
    historical_timeframe: str = "1Day"
    historical_lookback_days: int = 180
    historical_episode_length: int = 64
    historical_delta_scale: float = 80.0
    historical_transaction_cost: float = 0.02

    # Repro
    seed: int = 42


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Versioned checkpoint shipped with the repo (tracked). Loaded in preference to a
# locally-trained one in the (gitignored) cache dir, so a fresh checkout/CI/deploy
# has a working model without training. Training writes to the cache dir.
_TRACKED_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def _model_dir() -> Path:
    path = CACHE_CSV_DIR / "HedgerDQN"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_weights_path() -> Path:
    return _model_dir() / f"{DQN_HEDGER_VERSION}.npz"


def _cache_meta_path() -> Path:
    return _model_dir() / f"{DQN_HEDGER_VERSION}.json"


def _read_dir() -> Path:
    """
    READ resolution: prefer the tracked shipped checkpoint, but only as a CONSISTENT
    PAIR (both .npz and .json present) so weights and metadata never come from
    different dirs. Otherwise fall back to the (gitignored) cache dir.
    """
    t_npz = _TRACKED_WEIGHTS_DIR / f"{DQN_HEDGER_VERSION}.npz"
    t_json = _TRACKED_WEIGHTS_DIR / f"{DQN_HEDGER_VERSION}.json"
    if t_npz.exists() and t_json.exists():
        return _TRACKED_WEIGHTS_DIR
    return _model_dir()


def _weights_path() -> Path:
    return _read_dir() / f"{DQN_HEDGER_VERSION}.npz"


def _meta_path() -> Path:
    return _read_dir() / f"{DQN_HEDGER_VERSION}.json"


def _load_historical_prices(cfg: DQNConfig) -> np.ndarray:
    client = AlpacaHedgerClient()
    if not client.is_ready():
        raise RuntimeError("Alpaca client offline; impossible de récupérer les données historiques.")

    df = client.get_stock_bars_df(
        cfg.historical_symbol,
        timeframe=cfg.historical_timeframe,
        lookback_days=cfg.historical_lookback_days,
    )
    if df is None or df.empty:
        raise RuntimeError("Aucune donnée historique Alpaca récupérée pour l'entraînement.")
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    price_col = None
    for col in ("close", "Close", "c"):
        if col in df.columns:
            price_col = col
            break
    if price_col is None:
        for col in df.columns[::-1]:
            try:
                df[col].astype(float)
                price_col = col
                break
            except Exception:
                continue
    if price_col is None:
        raise RuntimeError("Impossible de trouver une colonne de prix dans les barres Alpaca.")

    prices = np.asarray(df[price_col].astype(float).to_numpy(), dtype=np.float32)
    prices = prices[np.isfinite(prices)]
    if prices.size < 3:
        raise RuntimeError("Données Alpaca insuffisantes pour entraîner le DQN (moins de 3 points).")
    return prices


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int | None = None) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self._rng = np.random.default_rng(seed)

        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)

        self._idx = 0
        self._size = 0

    @property
    def size(self) -> int:
        return int(self._size)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self._states[i] = state
        self._next_states[i] = next_state
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._dones[i] = 1.0 if bool(done) else 0.0

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self._size <= 0:
            raise ValueError("Cannot sample from empty buffer.")
        n = min(int(batch_size), self._size)
        idx = self._rng.integers(0, self._size, size=n)
        return {
            "states": self._states[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_states": self._next_states[idx],
            "dones": self._dones[idx],
        }


class Adam:
    def __init__(
        self,
        params: Iterable[np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        self._t = 0
        self._m = [np.zeros_like(p, dtype=np.float32) for p in params]
        self._v = [np.zeros_like(p, dtype=np.float32) for p in params]

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self._t += 1
        t = self._t
        for i, (p, g) in enumerate(zip(params, grads)):
            g = g.astype(np.float32, copy=False)
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / (1.0 - self.beta1**t)
            v_hat = self._v[i] / (1.0 - self.beta2**t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class NumpyMLP:
    def __init__(self, layer_sizes: Tuple[int, ...], seed: int | None = None) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output sizes.")
        self.layer_sizes = tuple(int(s) for s in layer_sizes)
        rng = np.random.default_rng(seed)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = np.sqrt(2.0 / max(in_dim, 1))
            w = (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)
            b = np.zeros((out_dim,), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def copy(self) -> "NumpyMLP":
        clone = NumpyMLP(self.layer_sizes, seed=0)
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        return clone

    def predict(self, x: np.ndarray) -> np.ndarray:
        out, _ = self.forward(x, retain_cache=False)
        return out

    def forward(self, x: np.ndarray, retain_cache: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
        x2 = np.asarray(x, dtype=np.float32)
        if x2.ndim == 1:
            x2 = x2[None, :]

        activations: list[np.ndarray] = [x2]
        preacts: list[np.ndarray] = []

        h = x2
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            preacts.append(z)
            if i != last:
                h = np.maximum(z, 0.0)
            else:
                h = z
            activations.append(h)

        cache = {"activations": activations, "preacts": preacts} if retain_cache else {}
        return h, cache

    def backward(self, dout: np.ndarray, cache: Dict[str, Any]) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        activations: list[np.ndarray] = cache["activations"]
        preacts: list[np.ndarray] = cache["preacts"]

        dW: list[np.ndarray] = []
        db: list[np.ndarray] = []

        dh = dout.astype(np.float32, copy=False)
        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            z = preacts[i]

            if i != len(self.weights) - 1:
                dh = dh * (z > 0.0)

            grad_w = a_prev.T @ dh
            grad_b = dh.sum(axis=0)
            dW.insert(0, grad_w)
            db.insert(0, grad_b)

            dh = dh @ self.weights[i].T

        return dW, db

    def params(self) -> list[np.ndarray]:
        # weights + biases interleaved
        out: list[np.ndarray] = []
        for w, b in zip(self.weights, self.biases):
            out.append(w)
            out.append(b)
        return out

    def set_params(self, params: list[np.ndarray]) -> None:
        if len(params) != 2 * len(self.weights):
            raise ValueError("Invalid params length for this network.")
        for i in range(len(self.weights)):
            self.weights[i][...] = params[2 * i]
            self.biases[i][...] = params[2 * i + 1]


class DQNAgent:
    """
    Minimal DQN implementation (experience replay + target network).

    Notes:
    - Discrete action space fixed to 4 actions: hold / buy / sell / flatten.
    - Numpy only, designed to run quickly inside Streamlit without heavy deps.
    """

    def __init__(self, config: DQNConfig, n_actions: int = 4) -> None:
        self.config = config
        self.n_actions = int(n_actions)
        self.state_dim = len(self.config.state_keys)
        self._rng = np.random.default_rng(self.config.seed)

        layer_sizes = (self.state_dim, *self.config.hidden_sizes, self.n_actions)
        self.q_net = NumpyMLP(layer_sizes, seed=self.config.seed)
        self.target_net = self.q_net.copy()
        self.optimizer = Adam(self.q_net.params(), lr=self.config.learning_rate)

        self._train_updates = 0

    def encode_state(self, state: Dict[str, float] | np.ndarray) -> np.ndarray:
        if isinstance(state, dict):
            vals = np.array([float(state.get(k, 0.0) or 0.0) for k in self.config.state_keys], dtype=np.float32)
        else:
            vals = np.asarray(state, dtype=np.float32).reshape(-1)
            if vals.shape[0] != self.state_dim:
                raise ValueError(f"State dim mismatch: got {vals.shape[0]} expected {self.state_dim}.")

        # Smooth bounded normalization (robust to live values > training range)
        if self.state_dim >= 1:
            vals[0] = np.tanh(vals[0] / max(self.config.delta_scale, 1e-6))
        if self.state_dim >= 2:
            vals[1] = np.tanh(vals[1] / max(self.config.position_scale, 1e-6))
        return vals.astype(np.float32, copy=False)

    def q_values(self, state: Dict[str, float] | np.ndarray) -> np.ndarray:
        s = self.encode_state(state)
        q = self.q_net.predict(s)
        return q.reshape(-1).astype(np.float32, copy=False)

    def select_action(self, state: Dict[str, float] | np.ndarray, epsilon: float) -> int:
        if self._rng.random() < float(epsilon):
            return int(self._rng.integers(0, self.n_actions))
        q = self.q_values(state)
        return int(np.argmax(q))

    def update_target_hard(self) -> None:
        self.target_net.set_params([p.copy() for p in self.q_net.params()])

    def train_on_batch(self, batch: Dict[str, np.ndarray]) -> float:
        states = np.asarray(batch["states"], dtype=np.float32)
        actions = np.asarray(batch["actions"], dtype=np.int64)
        rewards = np.asarray(batch["rewards"], dtype=np.float32)
        next_states = np.asarray(batch["next_states"], dtype=np.float32)
        dones = np.asarray(batch["dones"], dtype=np.float32)

        # Q(s, a)
        q_pred_all, cache = self.q_net.forward(states, retain_cache=True)
        q_pred = q_pred_all[np.arange(actions.shape[0]), actions]

        # target: r + gamma * max_a' Q_target(s', a')
        q_next = self.target_net.predict(next_states)
        max_q_next = np.max(q_next, axis=1)
        q_target = rewards + self.config.gamma * (1.0 - dones) * max_q_next

        td = (q_pred - q_target).astype(np.float32)
        loss = float(np.mean(td * td))

        # dLoss/dQ(s,a): 2*(q_pred-q_target)/N
        dQ = np.zeros_like(q_pred_all, dtype=np.float32)
        dQ[np.arange(actions.shape[0]), actions] = (2.0 / actions.shape[0]) * td

        dW, db = self.q_net.backward(dQ, cache)

        # Adam step on weights + biases (interleaved)
        params = self.q_net.params()
        grads: list[np.ndarray] = []
        for gw, gb in zip(dW, db):
            grads.append(gw)
            grads.append(gb)
        self.optimizer.step(params, grads)

        self._train_updates += 1
        return loss

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(self.q_net.weights, self.q_net.biases)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        np.savez_compressed(path, **arrays)

    def load(self, path: Path) -> None:
        data = np.load(path)
        params: list[np.ndarray] = []
        for i in range(len(self.q_net.weights)):
            params.append(np.asarray(data[f"W{i}"], dtype=np.float32))
            params.append(np.asarray(data[f"b{i}"], dtype=np.float32))
        self.q_net.set_params(params)
        self.update_target_hard()


def _epsilon_at_step(cfg: DQNConfig, step: int) -> float:
    if step <= 0:
        return float(cfg.epsilon_start)
    if cfg.epsilon_decay_steps <= 0:
        return float(cfg.epsilon_end)
    frac = min(max(step / float(cfg.epsilon_decay_steps), 0.0), 1.0)
    return float(cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start))


def _make_historical_env(cfg: DQNConfig, prices: np.ndarray, seed: int | None = None) -> HistoricalHedgingEnv:
    return HistoricalHedgingEnv(
        prices=tuple(float(x) for x in prices),
        position_scale=cfg.env_position_scale,
        delta_scale=cfg.historical_delta_scale,
        transaction_cost=cfg.historical_transaction_cost,
        max_episode_length=cfg.historical_episode_length,
        seed=seed,
    )


def _eval_agent_generic(agent: DQNAgent, env_factory: Callable[[], Any], episodes: int) -> Dict[str, float]:
    rewards: list[float] = []
    abs_residual: list[float] = []
    for _ in range(int(episodes)):
        env = env_factory()
        state = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, info = env.step(action)
            ep_r += float(reward)
            abs_residual.append(abs(float(info.get("total_delta", 0.0))))
        rewards.append(ep_r)
    return {
        "eval_avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "eval_avg_abs_total_delta": float(np.mean(abs_residual)) if abs_residual else 0.0,
    }


def get_dqn_model_info() -> Dict[str, Any]:
    weights_path = _weights_path()
    meta_path = _meta_path()
    info: Dict[str, Any] = {
        "version": DQN_HEDGER_VERSION,
        "weights_path": str(weights_path),
        "meta_path": str(meta_path),
        "weights_exists": weights_path.exists(),
        "meta_exists": meta_path.exists(),
    }
    if meta_path.exists():
        try:
            info["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            info["meta"] = {}
    return info


_CACHED_AGENT: DQNAgent | None = None
_CACHED_META: Dict[str, Any] | None = None


class DQNModelUnavailable(RuntimeError):
    """Raised when no usable DQN checkpoint is present (training is offline-only)."""


_UNAVAILABLE_MSG = (
    "No DQN checkpoint found. Train one offline via: python scripts/train_dqn_hedger.py"
)


def _unavailable_meta(message: str = _UNAVAILABLE_MSG) -> Dict[str, Any]:
    return {
        "version": DQN_HEDGER_VERSION,
        "available": False,
        "status": "model_unavailable",
        "message": message,
    }


def load_or_train_dqn_model(
    *,
    config: DQNConfig | None = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Load the persisted DQN model if present.

    The app NEVER trains on its render path: training is blocking and must run
    offline. With ``force_retrain=False`` (the default, used by the UI), this loads
    an existing checkpoint or returns an "unavailable" meta. ``force_retrain=True``
    (CLI / explicit) delegates to :func:`train_dqn_model`.
    """

    global _CACHED_AGENT, _CACHED_META
    cfg = config or DQNConfig()
    requested_train_mode = "historical"

    if not force_retrain and _CACHED_AGENT is not None and _CACHED_META is not None:
        return dict(_CACHED_META)

    if force_retrain:
        return train_dqn_model(config=cfg)

    weights_path = _weights_path()
    meta_path = _meta_path()

    if not weights_path.exists():
        return _unavailable_meta()

    agent = DQNAgent(cfg, n_actions=4)
    try:
        agent.load(weights_path)
    except Exception as exc:
        # Corrupted/incompatible checkpoint: stay unavailable, never train at runtime.
        return _unavailable_meta(
            f"DQN checkpoint unreadable ({exc}). Retrain via: python scripts/train_dqn_hedger.py"
        )

    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta = {
        "version": DQN_HEDGER_VERSION,
        "available": True,
        "loaded_utc": _utc_iso_now(),
        "config": asdict(cfg),
        **(meta or {}),
    }
    # Reflect the (historical) training mode in the loaded meta + config (from main).
    meta.setdefault("train_mode", requested_train_mode)
    cfg_dict = meta.get("config", {}) or {}
    cfg_dict["train_mode"] = requested_train_mode
    meta["config"] = cfg_dict
    _CACHED_AGENT = agent
    _CACHED_META = meta
    return dict(meta)


def train_dqn_model(*, config: DQNConfig | None = None) -> Dict[str, Any]:
    """
    Train the DQN on the synthetic env and persist a checkpoint.

    OFFLINE/CLI ONLY — this is blocking and is never invoked on the app's render
    path. Caches the trained agent in-memory and returns its metadata.
    """
    global _CACHED_AGENT, _CACHED_META
    cfg = config or DQNConfig()
    # Always write to the (gitignored) CACHE dir — training never clobbers the
    # tracked, version-controlled shipped checkpoint. Shipping a new checkpoint is
    # a deliberate step (copy cache -> weights/ + checksum, then commit).
    weights_path = _cache_weights_path()
    meta_path = _cache_meta_path()

    agent = DQNAgent(cfg, n_actions=4)

    # Train on historical prices only
    train_mode = "historical"  # enforced historical-only training
    prices = _load_historical_prices(cfg)
    env = _make_historical_env(cfg, prices, seed=cfg.seed)
    eval_env_factory = lambda: _make_historical_env(cfg, prices, seed=cfg.seed + 1)

    buffer = ReplayBuffer(cfg.buffer_capacity, state_dim=agent.state_dim, seed=cfg.seed)
    state = env.reset(seed=cfg.seed)
    done = False

    losses: list[float] = []
    rolling_abs_residual: list[float] = []
    for step in range(int(cfg.train_steps)):
        if done:
            state = env.reset()
            done = False

        eps = _epsilon_at_step(cfg, step)
        action = agent.select_action(state, epsilon=eps)
        next_state, reward, done, info = env.step(action)

        s_vec = agent.encode_state(state)
        ns_vec = agent.encode_state(next_state)
        buffer.add(s_vec, action, reward, ns_vec, done)

        state = next_state
        rolling_abs_residual.append(abs(float(info.get("total_delta", 0.0))))

        if buffer.size >= int(cfg.warmup_steps):
            batch = buffer.sample(cfg.batch_size)
            loss = agent.train_on_batch(batch)
            losses.append(loss)

        if (step + 1) % int(cfg.target_update_interval) == 0:
            agent.update_target_hard()

    eval_metrics = _eval_agent_generic(agent, eval_env_factory, episodes=cfg.eval_episodes)
    meta = {
        "version": DQN_HEDGER_VERSION,
        "available": True,
        "trained_utc": _utc_iso_now(),
        "config": asdict(cfg),
        "train_steps": int(cfg.train_steps),
        "train_updates": int(agent._train_updates),
        "train_avg_loss": float(np.mean(losses)) if losses else None,
        "train_avg_abs_total_delta": float(np.mean(rolling_abs_residual)) if rolling_abs_residual else None,
        **eval_metrics,
    }
    meta.update(
        {
            "train_mode": "historical",
            "historical_symbol": cfg.historical_symbol,
            "historical_timeframe": cfg.historical_timeframe,
            "historical_lookback_days": int(cfg.historical_lookback_days),
            "historical_price_points": int(len(prices) if prices is not None else 0),
        }
    )

    agent.save(weights_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _CACHED_AGENT = agent
    _CACHED_META = meta
    return dict(meta)


def _get_cached_agent(config: DQNConfig | None = None) -> Tuple[DQNAgent, Dict[str, Any]]:
    global _CACHED_AGENT, _CACHED_META
    if _CACHED_AGENT is None or _CACHED_META is None:
        meta = load_or_train_dqn_model(config=config, force_retrain=False)
        if _CACHED_AGENT is None:
            raise DQNModelUnavailable(meta.get("message", _UNAVAILABLE_MSG))
        _CACHED_META = meta
    return _CACHED_AGENT, dict(_CACHED_META or {})


def suggest_hedge_action(client: AlpacaHedgerClient, underlying_symbol: str) -> dict:
    """
    Returns a hedge suggestion for the *underlying* (equity leg).

    Action space:
    - 0: hold
    - 1: buy +1 * position_scale
    - 2: sell -1 * position_scale
    - 3: flatten current equity hedge (trade -equity_position)
    """

    sym = (underlying_symbol or "").strip().upper()

    try:
        agent, meta = _get_cached_agent()
    except DQNModelUnavailable as exc:
        # Graceful degradation: never train at render time, never emit a trade.
        return {
            "underlying": sym,
            "available": False,
            "status": "model_unavailable",
            "message": str(exc),
            "action": 0,
            "side": "none",
            "delta_qty": 0.0,
            "model_version": DQN_HEDGER_VERSION,
        }

    env = HedgingEnv(client=client, underlying_symbol=sym, position_scale=1.0)
    state = env.reset()

    q_vals = agent.q_values(state)
    action = int(np.argmax(q_vals))

    equity_pos = float(state.get("equity_position", 0.0) or 0.0)
    if action == 1:
        delta_qty = +1.0
        side = "buy"
    elif action == 2:
        delta_qty = -1.0
        side = "sell"
    elif action == 3:
        delta_qty = -equity_pos
        side = "flatten"
    else:
        delta_qty = 0.0
        side = "none"

    return {
        "underlying": sym,
        "available": True,
        "action": action,
        "side": side,
        "delta_qty": float(delta_qty),
        "q_values": [float(x) for x in q_vals.tolist()],
        "model_version": DQN_HEDGER_VERSION,
        "model_trained_utc": meta.get("trained_utc") or meta.get("loaded_utc"),
        "comment": "DQN hedge suggestion (replay buffer + target network; historical Alpaca price training)",
        "state": state,
    }


__all__ = [
    "DQNConfig",
    "DQNAgent",
    "DQNModelUnavailable",
    "get_dqn_model_info",
    "load_or_train_dqn_model",
    "train_dqn_model",
    "suggest_hedge_action",
]
