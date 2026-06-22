"""build_motion_reference_mujoco.py — GMR retargeted-CSV → K1 motion NPZ, MuJoCo FK.

Same output as ``build_motion_reference.py`` (the Genesis version) but computes
the body world poses / per-foot clearance with **MuJoCo** forward kinematics
instead of Genesis. Useful as an independent cross-check (the two backends agree
to mm) and when you'd rather not spin up a Genesis/GPU scene just for FK.

Self-contained to this repo: loads ``models/K1/K1_22dof.urdf`` directly via
MuJoCo's ``MjSpec`` and adds a floating (free) joint to the root so the root
position/orientation can be driven — the URDF imports with a FIXED base
otherwise. No MJCF and no mesh copying required. Depends only on ``mujoco`` +
numpy (+ the shared math helpers in ``build_motion_reference.py``).

CSV / NPZ schema and all processing (xyzw→wxyz, 30→50 Hz lerp+slerp, finite-diff
velocities, per-foot clearance = z − per-foot-min, optional 53-dim AMP
transitions) are identical to the Genesis script — only the FK backend differs.

USAGE (from repo root):
    python walk/build_motion_reference_mujoco.py \
        --csv k1_jogging.csv \
        --out walk/data/motions/k1_jogging_motion.npz --emit-transitions
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Reuse the shared, backend-agnostic helpers so the two scripts never diverge.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_motion_reference import (  # noqa: E402
    AMP_OBS_DIM,
    DEFAULT_OUT,
    DEFAULT_URDF,
    FOOT_LINKS,
    JOINT_NAMES,
    N_DOF,
    body_frame_velocity,
    build_amp_obs,
    lerp_resample,
    projected_gravity,
    quat_ang_vel_world,
    slerp_resample,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="GMR beyondmimic CSV (root_pos3, root_rot xyzw4, dof22)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output NPZ (default {DEFAULT_OUT})")
    ap.add_argument("--urdf", default=DEFAULT_URDF, help="K1 URDF for MuJoCo FK")
    ap.add_argument("--src-fps", type=float, default=30.0)
    ap.add_argument("--out-fps", type=float, default=50.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--emit-transitions", action="store_true",
                    help="also write 53-dim AMP transitions (M-1,106) next to --out")
    args = ap.parse_args()

    # ── 1. load + crop ──
    raw = np.loadtxt(args.csv, delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 + 4 + N_DOF:
        raise ValueError(f"{args.csv}: expected (N,{3 + 4 + N_DOF}) cols, got {raw.shape}")
    end = raw.shape[0] if args.end < 0 else args.end
    raw = raw[args.start:end]
    N = raw.shape[0]
    root_pos, quat_xyzw, dof = raw[:, :3], raw[:, 3:7], raw[:, 7:]
    print(f"[load] {args.csv}: {N} frames @ {args.src_fps} fps")

    quat = quat_xyzw[:, [3, 0, 1, 2]].copy()                 # xyzw → wxyz
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    for i in range(1, N):
        if np.dot(quat[i], quat[i - 1]) < 0.0:
            quat[i] = -quat[i]

    # ── 2. resample ──
    dt = 1.0 / args.out_fps
    t_src = np.arange(N) / args.src_fps
    t_new = np.arange(0.0, t_src[-1] + 1e-9, dt)
    root_pos = lerp_resample(root_pos, t_src, t_new)
    dof = lerp_resample(dof, t_src, t_new)
    quat = slerp_resample(quat, t_src, t_new)
    M = len(t_new)
    print(f"[resample] {N}→{M} frames @ {args.out_fps} fps (dt={dt:.4f})")

    # ── 3. velocities ──
    dof_vel = np.zeros_like(dof); dof_vel[:-1] = (dof[1:] - dof[:-1]) / dt; dof_vel[-1] = dof_vel[-2]
    root_lin_vel = np.zeros_like(root_pos)
    root_lin_vel[:-1] = (root_pos[1:] - root_pos[:-1]) / dt; root_lin_vel[-1] = root_lin_vel[-2]
    ang_world = np.zeros((M, 3))
    ang_world[:-1] = quat_ang_vel_world(quat[:-1], quat[1:], dt); ang_world[-1] = ang_world[-2]
    root_ang_vel_body = body_frame_velocity(quat.astype(np.float32), ang_world.astype(np.float32))
    proj_g = projected_gravity(quat.astype(np.float32))

    # ── 4. MuJoCo FK (URDF + added free joint) ──
    import mujoco
    spec = mujoco.MjSpec.from_file(os.path.abspath(args.urdf))
    spec.worldbody.bodies[0].add_freejoint()   # URDF base is fixed → make it floating
    m = spec.compile()
    data = mujoco.MjData(m)

    for nm in JOINT_NAMES:
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm) < 0:
            raise ValueError(f"joint '{nm}' not in {args.urdf}")
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in JOINT_NAMES]
    base_q = m.jnt_qposadr[0]                   # free joint qpos: pos3 + quat4(wxyz)
    body_names = [m.body(i).name for i in range(m.nbody)]
    foot_bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in FOOT_LINKS]

    body_pos_w = np.zeros((M, m.nbody, 3), dtype=np.float32)
    body_quat_w = np.zeros((M, m.nbody, 4), dtype=np.float32)
    for i in range(M):
        data.qpos[base_q:base_q + 3] = root_pos[i]
        data.qpos[base_q + 3:base_q + 7] = quat[i]
        for k, a in enumerate(qadr):
            data.qpos[a] = dof[i, k]
        mujoco.mj_forward(m, data)
        body_pos_w[i] = data.xpos
        body_quat_w[i] = data.xquat

    foot_z = body_pos_w[:, foot_bid, 2]
    ground = foot_z.min(0)
    foot_clear = np.clip(foot_z - ground, 0.0, 0.5).astype(np.float32)
    swing = (foot_clear > 0.03).mean(0)
    print(f"[fk] {m.nbody} bodies | foot ground z={np.round(ground, 3)} "
          f"clearance max={np.round(foot_clear.max(0), 3)} swing frac(>0.03)={np.round(swing, 2)}")

    # ── 5. save per-frame NPZ ──
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        fps=np.float32(args.out_fps),
        joint_names=np.array(JOINT_NAMES),
        body_names=np.array(body_names),
        root_pos=root_pos.astype(np.float32),
        root_quat=quat.astype(np.float32),
        root_lin_vel=root_lin_vel.astype(np.float32),
        root_ang_vel_body=root_ang_vel_body.astype(np.float32),
        dof_pos=dof.astype(np.float32),
        dof_vel=dof_vel.astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        foot_clear=foot_clear,
        projected_gravity=proj_g.astype(np.float32),
    )
    print(f"[save] {args.out}  ({M} frames)")

    # ── 6. optional AMP transitions ──
    if args.emit_transitions:
        amp = build_amp_obs(root_pos[:, 2], proj_g, root_ang_vel_body,
                            dof.astype(np.float32), dof_vel.astype(np.float32), foot_clear)
        trans = np.concatenate([amp[:-1], amp[1:]], axis=1).astype(np.float32)
        tout = args.out.replace(".npz", "_transitions.npz")
        if tout == args.out:
            tout = args.out + ".transitions.npz"
        np.savez(tout, transitions=trans)
        print(f"[save] {tout}  transitions={trans.shape} (AMP_OBS_DIM={AMP_OBS_DIM})")


if __name__ == "__main__":
    main()
