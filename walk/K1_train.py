"""PPO training for Booster K1 locomotion (22-DOF, rsl-rl + wandb).

Usage (from repo root):
  .venv/bin/python walk/K1_train.py -e k1-walking -B 2048 --max_iterations 500
  .venv/bin/wandb login   # once, before first run

Checkpoints: logs/<exp>/model_<iter>.pt every save_interval (default 250)
Videos:      logs/<exp>/videos/train_<iter>.mp4 → uploaded to wandb
"""

import argparse
import copy
import os
import pickle
import shutil
import sys
from importlib import metadata

import genesis as gs
import yaml

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

WALK_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(WALK_DIR, "config", "k1_env.yaml")


def load_cfgs() -> tuple[dict, dict, dict, dict, dict]:
    with open(CONFIG_PATH, "r") as f:
        all_cfg = yaml.safe_load(f)
    return (
        all_cfg["env_cfg"],
        all_cfg["obs_cfg"],
        all_cfg["reward_cfg"],
        all_cfg["command_cfg"],
        all_cfg["train_cfg"],
    )


def build_train_cfg(train_cfg: dict, exp_name: str, wandb_project: str) -> tuple[dict, dict]:
    cfg = copy.deepcopy(train_cfg)
    cfg["run_name"] = exp_name
    cfg["logger"] = copy.deepcopy(cfg["logger"])
    cfg["logger"]["project_name"] = wandb_project

    video_opts = {
        "video_interval": cfg.pop("video_interval", 250),
        "video_steps": cfg.pop("video_steps", 300),
        "video_fps": cfg.pop("video_fps", 50),
    }
    cfg.pop("wandb_project", None)
    return cfg, video_opts


def main() -> None:
    if WALK_DIR not in sys.path:
        sys.path.insert(0, WALK_DIR)
    from K1_env import K1Env
    from k1_train_runner import K1TrainRunner

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_yaml = load_cfgs()

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
    args = parser.parse_args()

    log_dir = os.path.join("logs", args.exp_name)
    train_cfg, video_opts = build_train_cfg(train_cfg_yaml, args.exp_name, args.wandb_project)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, video_opts], f)

    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )

    env = K1Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        num_steps_per_env=train_cfg_yaml["num_steps_per_env"],  # für Style-Curriculum (Iterationen)
    )

    print(
        f"K1 train: obs_dim={env.obs_dim}  num_actions={env.num_actions}  "
        f"num_envs={args.num_envs}  save_interval={train_cfg['save_interval']}  "
        f"video_interval={video_opts['video_interval']}  wandb={args.wandb_project}"
    )

    runner = K1TrainRunner(
        env,
        train_cfg,
        log_dir,
        device=gs.device,
        video_interval=video_opts["video_interval"],
        video_steps=video_opts["video_steps"],
        video_fps=video_opts["video_fps"],
        video_env_cfgs=(env_cfg, obs_cfg, reward_cfg, command_cfg),
    )
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
