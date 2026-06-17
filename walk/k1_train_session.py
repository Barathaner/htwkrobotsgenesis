"""Shared training session helpers for K1_train.py and Optuna tuning."""

from __future__ import annotations

import copy
import os
import pickle
from collections.abc import Callable
from typing import Any

import genesis as gs
import yaml

WALK_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(WALK_DIR, "config", "k1_env.yaml")


def load_all_cfgs(config_path: str = CONFIG_PATH) -> tuple[dict, dict, dict, dict, dict]:
    with open(config_path, "r") as f:
        all_cfg = yaml.safe_load(f)
    return (
        all_cfg["env_cfg"],
        all_cfg["obs_cfg"],
        all_cfg["reward_cfg"],
        all_cfg["command_cfg"],
        all_cfg["train_cfg"],
    )


def apply_reward_scales(reward_cfg: dict, scales: dict[str, float]) -> dict:
    """Deepcopy reward_cfg and override reward_scales."""
    cfg = copy.deepcopy(reward_cfg)
    cfg["reward_scales"] = {**cfg["reward_scales"], **scales}
    return cfg


def build_train_cfg(
    train_cfg: dict,
    exp_name: str,
    *,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    logger_class: str = "WandbLogWriter",
) -> tuple[dict, dict]:
    cfg = copy.deepcopy(train_cfg)
    cfg["run_name"] = exp_name
    cfg["logger"] = copy.deepcopy(cfg.get("logger", {}))
    cfg["logger"]["class_name"] = logger_class
    if wandb_project is not None:
        cfg["logger"]["project_name"] = wandb_project
    if wandb_group is not None:
        cfg["logger"]["group"] = wandb_group
    if wandb_tags is not None:
        cfg["logger"]["tags"] = wandb_tags

    video_opts = {
        "video_interval": cfg.pop("video_interval", 250),
        "video_steps": cfg.pop("video_steps", 300),
        "video_fps": cfg.pop("video_fps", 50),
    }
    cfg.pop("wandb_project", None)
    return cfg, video_opts


def init_genesis(seed: int) -> None:
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=seed,
        performance_mode=True,
    )


def create_k1_env(
    env_cfg: dict,
    obs_cfg: dict,
    reward_cfg: dict,
    command_cfg: dict,
    *,
    num_envs: int,
    show_viewer: bool = False,
):
    from K1_env import K1Env

    return K1Env(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )


def create_train_runner(
    env,
    train_cfg: dict,
    log_dir: str,
    env_cfgs: tuple[dict, dict, dict, dict],
    video_opts: dict,
    *,
    enable_video: bool = True,
    on_iteration_end: Callable[[int, Any], None] | None = None,
):
    from k1_train_runner import K1TrainRunner

    return K1TrainRunner(
        env,
        train_cfg,
        log_dir,
        device=gs.device,
        video_interval=video_opts["video_interval"],
        video_steps=video_opts["video_steps"],
        video_fps=video_opts["video_fps"],
        video_env_cfgs=env_cfgs,
        enable_video=enable_video,
        on_iteration_end=on_iteration_end,
    )


def save_session_cfgs(
    log_dir: str,
    env_cfg: dict,
    obs_cfg: dict,
    reward_cfg: dict,
    command_cfg: dict,
    train_cfg: dict,
    video_opts: dict,
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, video_opts], f)


def run_training(
    runner,
    max_iterations: int,
    *,
    init_at_random_ep_len: bool = True,
) -> None:
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=init_at_random_ep_len)
