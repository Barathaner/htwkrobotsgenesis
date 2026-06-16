#!/usr/bin/env python3
"""Debug: NPZ reference motion auf Booster K1 in GENESIS abspielen (Viewer).

Spielt eine mit walk/build_motion_reference.py erzeugte Referenz-Motion KINEMATISCH
ab — der Roboter ist ein KinematicEntity (gs.materials.Kinematic), folgt also exakt
den per Frame gesetzten qpos und fällt NICHT unter Schwerkraft zusammen (ein normaler
Rigid-Roboter würde bei scene.step() umkippen). Ideal, um die Retarget-Qualität
(Fuß-Aufsatz, Schwung, Pose) visuell zu bewerten.

Usage (repo root):
  python debugnpzgenesis.py --npz walk/data/motions/k1_jogging_motion.npz
  python debugnpzgenesis.py --npz ... --loop --speed 0.5
  python debugnpzgenesis.py --npz ... --print-amp-every 50 --ghost
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "walk"))
from build_motion_reference import JOINT_NAMES, build_amp_obs  # noqa: E402

DEFAULT_URDF = os.path.join(REPO_ROOT, "models", "K1", "K1_22dof.urdf")


def load_motion(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    for k in ("root_pos", "root_quat", "dof_pos", "fps"):
        if k not in data:
            raise KeyError(f"{path} missing '{k}'. Keys: {list(data.files)}")
    return {k: data[k] for k in data.files}


def npz_dof_to_genesis(dof_npz: np.ndarray, gen_joint_names: list[str]) -> np.ndarray:
    """NPZ columns (JOINT_NAMES order) → Genesis URDF joint order."""
    if dof_npz.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f"expected {len(JOINT_NAMES)} dof, got {dof_npz.shape[-1]}")
    col = [JOINT_NAMES.index(n) for n in gen_joint_names]
    return dof_npz[..., col]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="motion NPZ from walk/build_motion_reference.py")
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--ghost", action="store_true", help="translucent ghost look")
    ap.add_argument("--no-follow", action="store_true", help="don't let the camera follow the robot")
    ap.add_argument("--print-amp-every", type=int, default=0, help="print AMP obs every N frames (0=off)")
    args = ap.parse_args()

    motion = load_motion(args.npz)
    root_pos = motion["root_pos"].astype(np.float32)
    root_quat = motion["root_quat"].astype(np.float32)   # wxyz
    dof_pos = motion["dof_pos"].astype(np.float32)
    fps = float(motion["fps"])
    M = root_pos.shape[0]
    proj_g = motion.get("projected_gravity")
    ang_b = motion.get("root_ang_vel_body")
    dof_vel = motion.get("dof_vel")
    foot_clear = motion.get("foot_clear")

    import genesis as gs
    import torch

    gs.init(backend=getattr(gs, args.backend), logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / fps, substeps=1),
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -2.0, 1.5), camera_lookat=(0.0, 0.0, 0.5), camera_fov=30,
        ),
    )
    scene.add_entity(gs.morphs.Plane())

    surface = gs.surfaces.Default(color=(0.7, 0.85, 1.0), opacity=0.55) if args.ghost else None
    robot = scene.add_entity(
        gs.morphs.URDF(file=os.path.abspath(args.urdf), pos=(0.0, 0.0, 0.56)),
        material=gs.materials.Kinematic(),   # driven by set_qpos, ignored by dynamics → no fall
        surface=surface,
    )
    scene.build()

    gen_joint_names = [j.name for j in robot.joints[1:]]
    dof_gen = npz_dof_to_genesis(dof_pos, gen_joint_names)

    if not args.no_follow:
        scene.viewer.follow_entity(robot, fixed_axis=(None, None, 1.2), smoothing=0.9)

    print(f"[motion] {args.npz}: {M} frames @ {fps} fps, speed={args.speed}")
    print(f"[robot]  kinematic K1, {len(gen_joint_names)} joints | ghost={args.ghost}")

    dt = (1.0 / fps) / max(args.speed, 1e-6)
    frame = 0
    while True:
        i = frame % M
        qpos = np.concatenate([root_pos[i], root_quat[i], dof_gen[i]], axis=0)
        robot.set_qpos(torch.as_tensor(qpos, dtype=gs.tc_float, device=gs.device), zero_velocity=True)

        if args.print_amp_every and i % args.print_amp_every == 0 and all(
            x is not None for x in (proj_g, ang_b, dof_vel, foot_clear)
        ):
            amp = build_amp_obs(root_pos[i:i+1, 2], proj_g[i:i+1], ang_b[i:i+1],
                                dof_pos[i:i+1], dof_vel[i:i+1], foot_clear[i:i+1])
            print(f"  frame {i:4d}  root_z={root_pos[i,2]:.3f}  "
                  f"foot_clear L/R={foot_clear[i,0]:.3f}/{foot_clear[i,1]:.3f}  amp_dim={amp.shape[1]}")

        scene.step()
        if not args.no_follow:
            scene.viewer.update_following()

        if not args.loop and frame >= M - 1:
            break
        frame += 1
        time.sleep(dt)


if __name__ == "__main__":
    main()
