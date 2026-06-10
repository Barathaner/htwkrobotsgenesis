"""Evaluate a trained K1 walking policy in the Genesis viewer.

Usage (from repo root):
  .venv/bin/python walk/K1_eval.py -e k1-walking --ckpt 100
  .venv/bin/python walk/K1_eval.py -e mein-run --ckpt 2000
"""

import argparse
import os
import pickle
import sys
from importlib import metadata

import genesis as gs
import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

WALK_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    if WALK_DIR not in sys.path:
        sys.path.insert(0, WALK_DIR)
    from K1_env import K1Env

    parser = argparse.ArgumentParser(description="Evaluate K1 walking policy")
    parser.add_argument("-e", "--exp_name", type=str, default="k1-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = os.path.join("logs", args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        cfgs = pickle.load(f)
    if len(cfgs) == 6:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, _video_opts = cfgs
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfgs
    reward_cfg["reward_scales"] = {}

    env = K1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
