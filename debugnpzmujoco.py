#!/usr/bin/env python3
"""Debug: NPZ reference motion auf Booster K1 in MUJOCO abspielen (Viewer).

MuJoCo-Pendant zu debugnpzgenesis.py. Lädt models/K1/K1_22dof.urdf via MjSpec,
ergänzt ein Free-Joint an der Wurzel (der URDF-Base ist sonst fix) und spielt die
Referenz-Motion KINEMATISCH ab: pro Frame qpos setzen + mj_forward (KEIN mj_step →
keine Dynamik, der Roboter folgt exakt der Referenz und fällt nicht). Passiver
mujoco.viewer.

Usage (repo root):
  python debugnpzmujoco.py --npz walk/data/motions/k1_jogging_motion.npz
  python debugnpzmujoco.py --npz ... --loop --speed 0.5
  python debugnpzmujoco.py --npz ... --print-amp-every 50
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="motion NPZ from walk/build_motion_reference.py")
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--no-follow", action="store_true", help="don't let the camera track the robot")
    ap.add_argument("--print-amp-every", type=int, default=0, help="print AMP obs every N frames (0=off)")
    args = ap.parse_args()

    motion = load_motion(args.npz)
    root_pos = motion["root_pos"].astype(np.float64)
    root_quat = motion["root_quat"].astype(np.float64)   # wxyz (MuJoCo convention)
    dof_pos = motion["dof_pos"].astype(np.float64)
    fps = float(motion["fps"])
    M = root_pos.shape[0]
    proj_g = motion.get("projected_gravity")
    ang_b = motion.get("root_ang_vel_body")
    dof_vel = motion.get("dof_vel")
    foot_clear = motion.get("foot_clear")

    import mujoco
    import mujoco.viewer

    spec = mujoco.MjSpec.from_file(os.path.abspath(args.urdf))
    spec.worldbody.bodies[0].add_freejoint()   # URDF base is fixed → make it floating
    floor = spec.worldbody.add_geom()          # visual ground reference for judging foot contact
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10.0, 10.0, 0.1]
    floor.rgba = [0.55, 0.55, 0.6, 1.0]
    m = spec.compile()
    d = mujoco.MjData(m)

    for nm in JOINT_NAMES:
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm) < 0:
            raise ValueError(f"joint '{nm}' not in {args.urdf}")
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in JOINT_NAMES]
    base_q = m.jnt_qposadr[0]   # free joint: pos3 + quat4 (wxyz)

    print(f"[motion] {args.npz}: {M} frames @ {fps} fps, speed={args.speed}")
    print(f"[robot]  MuJoCo K1 (freejoint), {len(JOINT_NAMES)} joints")

    dt = (1.0 / fps) / max(args.speed, 1e-6)

    def apply(i: int) -> None:
        d.qpos[base_q:base_q + 3] = root_pos[i]
        d.qpos[base_q + 3:base_q + 7] = root_quat[i]
        for k, a in enumerate(qadr):
            d.qpos[a] = dof_pos[i, k]
        mujoco.mj_forward(m, d)   # kinematics only, no integration → no fall

    with mujoco.viewer.launch_passive(m, d) as viewer:
        frame = 0
        while viewer.is_running():
            i = frame % M
            apply(i)
            if not args.no_follow:
                viewer.cam.lookat[:] = root_pos[i]

            if args.print_amp_every and i % args.print_amp_every == 0 and all(
                x is not None for x in (proj_g, ang_b, dof_vel, foot_clear)
            ):
                amp = build_amp_obs(root_pos[i:i+1, 2], proj_g[i:i+1], ang_b[i:i+1],
                                    dof_pos[i:i+1], dof_vel[i:i+1], foot_clear[i:i+1])
                print(f"  frame {i:4d}  root_z={root_pos[i,2]:.3f}  "
                      f"foot_clear L/R={foot_clear[i,0]:.3f}/{foot_clear[i,1]:.3f}  amp_dim={amp.shape[1]}")

            viewer.sync()
            if not args.loop and frame >= M - 1:
                break
            frame += 1
            time.sleep(dt)


if __name__ == "__main__":
    main()
