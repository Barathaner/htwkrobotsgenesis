"""Distill the privileged K1 teacher into an onboard-only student (rsl-rl Distillation).

The teacher (trained with K1_train.py / AMP_PPO) sees the full 68-dim privileged obs
(env "policy" group, incl. true base velocity/height). The student is a recurrent (LSTM)
policy that sees ONLY the onboard "proprio" group — IMU + joint encoders + last action +
commands + gait clock — i.e. exactly what the real K1 measures. rsl-rl's DistillationRunner
runs DAgger: the student acts in the env, the teacher labels each visited state, and the
student is trained with an MSE behavior-cloning loss (BPTT through the LSTM).

Usage (from repo root):
  .venv/bin/python walk/k1_distill.py --teacher logs/ampslownocurr/model_14500.pt -B 2048 \
      --max_iterations 2000
  .venv/bin/python walk/k1_distill.py --teacher logs/ampslownocurr --no_wandb   # auto-pick latest

The teacher's env obs layout MUST match the checkpoint (same num_actions → same 68-dim obs).
Checkpoints: logs/<exp>/model_<iter>.pt  (student_state_dict + teacher_state_dict).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

WALK_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_teacher(path: str) -> str:
    """Accept a .pt file or a log dir (auto-picks the highest-numbered model_*.pt)."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        cands = sorted(
            glob.glob(os.path.join(path, "model_*.pt")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
        )
        if not cands:
            raise FileNotFoundError(f"No model_*.pt found in {path}")
        print(f"Auto-selected teacher checkpoint: {cands[-1]}")
        return cands[-1]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    return path


def main() -> None:
    if WALK_DIR not in sys.path:
        sys.path.insert(0, WALK_DIR)

    import yaml
    from k1_train_session import (
        CONFIG_PATH,
        create_k1_env,
        init_genesis,
        load_all_cfgs,
        save_session_cfgs,
    )

    env_cfg, obs_cfg, reward_cfg, command_cfg, _train_cfg = load_all_cfgs()
    with open(CONFIG_PATH, "r") as f:
        distill_cfg = yaml.safe_load(f)["distill_cfg"]

    parser = argparse.ArgumentParser(description="Distill K1 teacher → onboard student")
    parser.add_argument("--teacher", required=True, help="Teacher .pt checkpoint (or a log dir)")
    parser.add_argument("-e", "--exp_name", type=str, default=distill_cfg.get("run_name", "k1-distill"))
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default=distill_cfg.get("wandb_project", "k1-locomotion"))
    parser.add_argument("--no_wandb", action="store_true", help="Log to TensorBoard instead of wandb")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a distillation model_*.pt")
    args = parser.parse_args()

    teacher_path = _resolve_teacher(args.teacher)

    log_dir = os.path.join("logs", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # Build the runner train_cfg (DistillationRunner pops student/teacher/obs_groups/algorithm).
    cfg = dict(distill_cfg)
    cfg["run_name"] = args.exp_name
    if args.no_wandb:
        cfg["logger"] = "tensorboard"
    else:
        cfg["logger"] = {"class_name": "WandbLogWriter", "project_name": args.wandb_project}
    cfg.pop("wandb_project", None)

    init_genesis(args.seed)

    import genesis as gs
    from rsl_rl.runners import DistillationRunner

    env = create_k1_env(
        env_cfg, obs_cfg, reward_cfg, command_cfg, num_envs=args.num_envs, show_viewer=False
    )

    # Persist cfgs next to the checkpoints BEFORE the runner mutates cfg (construct_algorithm
    # pops student/teacher/obs_groups). Lets k1_record_video.py / eval rebuild the env later.
    import copy
    save_session_cfgs(log_dir, env_cfg, obs_cfg, reward_cfg, command_cfg, copy.deepcopy(cfg), {})

    runner = DistillationRunner(env, cfg, log_dir, device=gs.device)

    # Load the teacher weights from the RL checkpoint (falls back to actor_state_dict).
    runner.load(teacher_path, load_cfg={"teacher": True, "iteration": False})
    if args.resume:
        runner.load(args.resume, load_cfg={"student": True, "optimizer": True, "iteration": True})
        print(f"Resumed student from {args.resume}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
