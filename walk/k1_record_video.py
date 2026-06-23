"""Record a rollout video from a trained K1 .pt checkpoint.

Usage (from repo root):
  .venv/bin/python walk/k1_record_video.py --pt logs/k1-walking/model_500.pt
  .venv/bin/python walk/k1_record_video.py --pt /path/to/model.pt --out /tmp/run.mp4 --steps 600
  .venv/bin/python walk/k1_record_video.py --pt model.pt --cfg logs/k1-walking/cfgs.pkl
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys

import imageio.v2 as imageio
import torch

WALK_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    if WALK_DIR not in sys.path:
        sys.path.insert(0, WALK_DIR)

    parser = argparse.ArgumentParser(description="Record K1 rollout video from a .pt checkpoint")
    parser.add_argument("--pt", required=True, help="Path to .pt checkpoint file")
    parser.add_argument(
        "--cfg",
        default=None,
        help="Path to cfgs.pkl (default: looks in the same directory as --pt)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .mp4 path (default: <pt_stem>_video.mp4 next to the checkpoint)",
    )
    parser.add_argument("--steps", type=int, default=300, help="Number of rollout steps")
    parser.add_argument("--fps", type=int, default=50, help="Video frames per second")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sweep",
        nargs="*",
        type=float,
        default=None,
        help="Forward vx commands [m/s] to step through, held in equal segments over the "
        "rollout (overrides random commands). Pass with no values to use a default "
        "slow→jog sweep covering the command range.",
    )
    args = parser.parse_args()

    pt_path = os.path.abspath(args.pt)

    # If --pt is a directory, find the highest-numbered model_*.pt inside it
    if os.path.isdir(pt_path):
        import glob
        candidates = sorted(
            glob.glob(os.path.join(pt_path, "model_*.pt")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
        )
        if not candidates:
            raise FileNotFoundError(f"No model_*.pt files found in {pt_path}")
        pt_path = candidates[-1]
        print(f"Auto-selected checkpoint: {pt_path}")

    pt_dir = os.path.dirname(pt_path)
    pt_stem = os.path.splitext(pt_path)[0]

    cfg_path = args.cfg or os.path.join(pt_dir, "cfgs.pkl")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"cfgs.pkl not found at {cfg_path}. "
            "Pass --cfg <path> to specify it explicitly."
        )

    # If --out is a directory (or ends with /), place the file inside it
    if args.out is not None and (os.path.isdir(args.out) or args.out.endswith(os.sep)):
        ckpt_name = os.path.splitext(os.path.basename(pt_path))[0]
        out_path = os.path.join(args.out, f"{ckpt_name}_video.mp4")
    else:
        out_path = args.out or f"{pt_stem}_video.mp4"

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    with open(cfg_path, "rb") as f:
        cfgs = pickle.load(f)
    if len(cfgs) == 6:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, _video_opts = cfgs
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfgs

    # Disable reward shaping for clean rollout
    reward_cfg = copy.deepcopy(reward_cfg)
    reward_cfg["reward_scales"] = {}

    import genesis as gs

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)

    from K1_env import K1Env
    from k1_train_runner import K1TrainRunner
    from k1_video_overlay import render_annotated_frame, update_camera_centroid

    video_num_envs = env_cfg.get("video", {}).get("num_envs", 1)
    env = K1Env(
        num_envs=video_num_envs,
        env_cfg=copy.deepcopy(env_cfg),
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        record_camera=True,
    )
    assert env.cam is not None, "K1Env must have a camera when record_camera=True"

    # Build runner to get the right actor architecture, then load weights
    import tempfile
    tmp_log = tempfile.mkdtemp(prefix="k1_video_")
    runner = K1TrainRunner(
        env,
        train_cfg,
        tmp_log,
        device=gs.device,
        enable_video=False,
        video_env_cfgs=(env_cfg, obs_cfg, reward_cfg, command_cfg),
    )
    runner.load(
        pt_path,
        load_cfg={"actor": True, "critic": False, "discriminator": False, "optimizer": False},
    )
    policy = runner.get_inference_policy(device=gs.device)

    # Scripted command sweep: hold each vx for an equal segment of the rollout so the
    # video clearly walks through slow → jog. command_target is re-set every step so it
    # overrides the env's random resampling; env.commands smooths toward it (command_smooth_s).
    sweep_speeds = None
    if args.sweep is not None:
        sweep_speeds = args.sweep if len(args.sweep) > 0 else [0.3, 0.6, 0.9, 1.2]
        lo, hi = command_cfg["lin_vel_x_range"]
        sweep_speeds = [max(lo, min(hi, s)) for s in sweep_speeds]
        print(f"Command sweep (vx m/s): {sweep_speeds}")

    frames: list = []
    with torch.inference_mode():
        obs = env.reset()
        for step in range(args.steps):
            if sweep_speeds is not None:
                seg = min(step * len(sweep_speeds) // args.steps, len(sweep_speeds) - 1)
                env.command_target[:, 0] = sweep_speeds[seg]
                env.command_target[:, 1:] = 0.0

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            if env.hero_ghost is not None:
                env.hero_ghost.set_frame(env)
            if env.color_shadow is not None:
                env.color_shadow.update(env)
            if env.cam._followed_entity is not None:
                env.cam.update_following()
            update_camera_centroid(env)
            frames.append(render_annotated_frame(env))

            if (step + 1) % 50 == 0:
                print(f"  {step + 1}/{args.steps} steps recorded")

    imageio.mimsave(
        out_path,
        frames,
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-crf", "18"],
    )
    print(f"Video saved: {out_path}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
