"""
Offline trainer for the DQN hedger — PLAN.md step 5.

The app NEVER trains on its render path (load_or_train_dqn_model only LOADS).
Use this CLI to (re)train and persist a checkpoint:

    python scripts/train_dqn_hedger.py                 # default config
    python scripts/train_dqn_hedger.py --steps 50000 --seed 7

The resulting .npz checkpoint + meta.json are written under the hedger model dir
(cache/HedgerDQN/). Commit the checkpoint (Git LFS recommended) so the app loads it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.model.hedger_v2.dqn_hedger import DQNConfig, train_dqn_model, _weights_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist the DQN hedger checkpoint.")
    parser.add_argument("--steps", type=int, default=None, help="training steps (override)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (override)")
    args = parser.parse_args()

    base = DQNConfig()
    overrides = {}
    if args.steps is not None:
        overrides["train_steps"] = int(args.steps)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    cfg = DQNConfig(**{**base.__dict__, **overrides}) if overrides else base

    print(f"[train-dqn] training (steps={cfg.train_steps}, seed={cfg.seed}) ...")
    meta = train_dqn_model(config=cfg)
    print(f"[train-dqn] checkpoint written: {_weights_path()}")
    print(json.dumps({k: v for k, v in meta.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
