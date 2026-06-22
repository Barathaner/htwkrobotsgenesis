"""PPO training for Booster K1 locomotion (22-DOF, rsl-rl + wandb).

Usage (from repo root):
  .venv/bin/python walk/K1_train.py -e k1-walking -B 2048 --max_iterations 500
  .venv/bin/wandb login   # once, before first run

Checkpoints: logs/<exp>/model_<iter>.pt every save_interval (default 250)
Videos:      logs/<exp>/videos/train_<iter>.mp4 → uploaded to wandb
"""

import argparse
import os
import shutil
import sys
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

WALK_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    if WALK_DIR not in sys.path:
        sys.path.insert(0, WALK_DIR)
    from k1_train_session import (
        build_train_cfg,
        create_k1_env,
        create_train_runner,
        init_genesis,
        load_all_cfgs,
        run_training,
        save_session_cfgs,
    )

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_yaml = load_all_cfgs()

    parser = argparse.ArgumentParser(description="Train K1 walking policy with PPO")
    parser.add_argument("-e", "--exp_name", type=str, default=train_cfg_yaml.get("run_name", "k1-walking"))
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=train_cfg_yaml.get("wandb_project", "k1-locomotion"),
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a model_*.pt checkpoint to resume from (skips log dir wipe).",
    )
    args = parser.parse_args()

    log_dir = os.path.join("logs", args.exp_name)
    train_cfg, video_opts = build_train_cfg(
        train_cfg_yaml, args.exp_name, wandb_project=args.wandb_project
    )

    if args.checkpoint is None and os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    save_session_cfgs(log_dir, env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, video_opts)
    init_genesis(args.seed)

    env = create_k1_env(
        env_cfg, obs_cfg, reward_cfg, command_cfg, num_envs=args.num_envs
    )

    print(
        f"K1 train: obs_dim={env.obs_dim}  num_actions={env.num_actions}  "
        f"num_envs={args.num_envs}  save_interval={train_cfg['save_interval']}  "
        f"video_interval={video_opts['video_interval']}  wandb={args.wandb_project}"
    )

    runner = create_train_runner(
        env,
        train_cfg,
        log_dir,
        (env_cfg, obs_cfg, reward_cfg, command_cfg),
        video_opts,
        enable_video=True,
    )

    if args.checkpoint is not None:
        runner.load(args.checkpoint)
        print(f"Resumed from {args.checkpoint} at iteration {runner.current_learning_iteration}")
        # runner.load() restores the adaptive velocity-curriculum level from the checkpoint
        # (it is performance-driven, so it can't be recomputed from the iteration number).

    run_training(runner, args.max_iterations)


if __name__ == "__main__":
    main()
