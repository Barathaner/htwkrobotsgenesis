"""build_motion_reference.py — GMR retargeted-CSV → Genesis-native K1 motion NPZ.

Converts a GMR "beyondmimic" CSV export (e.g. ``k1_jogging.csv`` from a
GVHMR→GMR retarget onto the Booster K1) into a per-frame reference-motion
``.npz`` for later use as an Adversarial Motion Priors (AMP) + style-reward
reference in this Genesis RL pipeline.

Self-contained: depends only on ``genesis`` + numpy + torch. Forward kinematics
(body world poses + per-foot ground clearance) is computed with **Genesis**
``RigidEntity.forward_kinematics`` on this repo's ``models/K1/K1_22dof.urdf`` —
no MuJoCo, no external repo.

CSV format (no header): each row is
    root_pos(3) | root_rot xyzw(4) | dof_pos(22)
at 30 fps (GMR downsamples >30→30). The 22 dof columns are in the K1_22dof
joint order (``JOINT_NAMES`` below); they are remapped onto the Genesis robot's
own joint order BY NAME, so a differing URDF joint order is handled correctly.

What it does (the Genesis equivalent of Booster's IsaacSim playback script):
  1. load CSV, optional frame-range crop, xyzw→wxyz quat, hemisphere-align;
  2. resample 30→50 Hz (lerp root_pos/dof, SLERP the trunk quaternion);
  3. finite-difference velocities (dof_vel, root_lin_vel, body-frame ang vel);
  4. Genesis FK per frame → all link world poses + per-foot ground clearance;
  5. save a SCHEMA-AGNOSTIC per-frame NPZ. Optionally also emit a 53-dim AMP
     ``transitions`` (M-1, 106) file (``--emit-transitions``).

USAGE (run from the repo root, with the Genesis python):
    python walk/build_motion_reference.py \
        --csv k1_jogging.csv \
        --out walk/data/motions/k1_jogging_motion.npz --emit-transitions
    # options: --src-fps 30 --out-fps 50 --start 0 --end -1 --backend gpu
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# CSV dof-column order = Booster K1_22dof joint order (== GMR/robocup cfg order).
# Used only to label the CSV columns; mapping to the Genesis robot is by NAME.
JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
N_DOF = len(JOINT_NAMES)
FOOT_LINKS = ("left_foot_link", "right_foot_link")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.join(REPO_ROOT, "models", "K1", "K1_22dof.urdf")
DEFAULT_OUT = os.path.join(REPO_ROOT, "walk", "data", "motions", "k1_jogging_motion.npz")

AMP_OBS_DIM = 1 + 3 + 3 + N_DOF + N_DOF + 2  # root_h, proj_g, ang_b, dof, dof_vel, foot_clear = 53


# ─── inlined math helpers (no external deps) ─────────────────────────────────


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity (0,0,-1) in body frame from (w,x,y,z) world→body quats. (N,3)."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    gx = -2.0 * (x * z + w * y)
    gy = -2.0 * (y * z - w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return np.stack([gx, gy, gz], axis=-1).astype(np.float32)


def body_frame_velocity(quat_wxyz: np.ndarray, vel_world: np.ndarray) -> np.ndarray:
    """Rotate a world-frame velocity into the body frame using YAW only
    (locomotion policies are yaw-invariant; pitch/roll live in proj-gravity)."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    vx = c * vel_world[:, 0] - s * vel_world[:, 1]
    vy = s * vel_world[:, 0] + c * vel_world[:, 1]
    return np.stack([vx, vy, vel_world[:, 2]], axis=-1).astype(np.float32)


def slerp_resample(quat_src: np.ndarray, t_src: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """SLERP (w,x,y,z) quats onto new times. quat_src (N,4) hemisphere-aligned."""
    i1 = np.clip(np.searchsorted(t_src, t_new, side="right"), 1, len(t_src) - 1)
    i0 = i1 - 1
    span = t_src[i1] - t_src[i0]
    alpha = np.where(span > 0, (t_new - t_src[i0]) / span, 0.0)
    q0, q1 = quat_src[i0], quat_src[i1].copy()
    dot = np.sum(q0 * q1, axis=1)
    flip = dot < 0.0
    q1[flip] = -q1[flip]
    dot = np.abs(np.clip(dot, -1.0, 1.0))
    out = np.empty((len(t_new), 4), dtype=np.float64)
    near = dot > 0.9995
    out[near] = q0[near] + alpha[near, None] * (q1[near] - q0[near])
    far = ~near
    if np.any(far):
        th = np.arccos(dot[far]); sn = np.sin(th); a = alpha[far]
        out[far] = (np.sin((1.0 - a) * th) / sn)[:, None] * q0[far] + (np.sin(a * th) / sn)[:, None] * q1[far]
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def quat_ang_vel_world(qc: np.ndarray, qn: np.ndarray, dt: float) -> np.ndarray:
    """WORLD-frame angular velocity between consecutive (w,x,y,z) quats."""
    w0, x0, y0, z0 = qc.T
    w1, x1, y1, z1 = qn.T
    cw, cx, cy, cz = w0, -x0, -y0, -z0
    dw = w1 * cw - x1 * cx - y1 * cy - z1 * cz
    dx = w1 * cx + x1 * cw + y1 * cz - z1 * cy
    dy = w1 * cy - x1 * cz + y1 * cw + z1 * cx
    dz = w1 * cz + x1 * cy - y1 * cx + z1 * cw
    s = np.sign(dw); s[s == 0] = 1.0
    return 2.0 * np.stack([dx * s, dy * s, dz * s], axis=1) / dt


def lerp_resample(col: np.ndarray, t_src: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_new, t_src, col[:, k]) for k in range(col.shape[1])], axis=1)


def build_amp_obs(root_h, proj_g, ang_b, dof_pos, dof_vel, foot_clear) -> np.ndarray:
    """53-dim AMP feature (matches robocup training.algorithms.amp.build_amp_obs)."""
    return np.concatenate([
        np.asarray(root_h, np.float32).reshape(-1, 1),
        np.asarray(proj_g, np.float32), np.asarray(ang_b, np.float32),
        np.asarray(dof_pos, np.float32), np.asarray(dof_vel, np.float32),
        np.asarray(foot_clear, np.float32).reshape(-1, 2),
    ], axis=1).astype(np.float32)


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="GMR beyondmimic CSV (root_pos3, root_rot xyzw4, dof22)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output NPZ (default {DEFAULT_OUT})")
    ap.add_argument("--urdf", default=DEFAULT_URDF, help="K1 URDF for Genesis FK")
    ap.add_argument("--src-fps", type=float, default=30.0)
    ap.add_argument("--out-fps", type=float, default=50.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
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
    for i in range(1, N):                                    # hemisphere-align
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

    # ── 4. Genesis forward kinematics ──
    import genesis as gs
    gs.init(backend=getattr(gs, args.backend), logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    robot = scene.add_entity(gs.morphs.URDF(file=os.path.abspath(args.urdf)))
    scene.build()

    gen_joint_names = [j.name for j in robot.joints[1:]]      # Genesis dof order (skip base)
    if sorted(gen_joint_names) != sorted(JOINT_NAMES):
        raise ValueError(f"URDF joints {gen_joint_names} != expected {JOINT_NAMES}")
    csv_col = [JOINT_NAMES.index(n) for n in gen_joint_names]  # CSV→Genesis column remap
    dof_gen = dof[:, csv_col]

    link_names = [l.name for l in robot.links]
    foot_idx = [robot.get_link(n).idx_local for n in FOOT_LINKS]

    import torch
    qpos = np.concatenate([root_pos, quat, dof_gen], axis=1)  # (M, 7+22) wxyz base
    body_pos_w = np.zeros((M, len(link_names), 3), dtype=np.float32)
    body_quat_w = np.zeros((M, len(link_names), 4), dtype=np.float32)
    for i in range(M):
        lp, lq = robot.forward_kinematics(torch.as_tensor(qpos[i], dtype=gs.tc_float, device=gs.device))
        body_pos_w[i] = lp.detach().cpu().numpy()
        body_quat_w[i] = lq.detach().cpu().numpy()

    foot_z = body_pos_w[:, foot_idx, 2]                       # (M,2)
    ground = foot_z.min(0)
    foot_clear = np.clip(foot_z - ground, 0.0, 0.5).astype(np.float32)
    swing = (foot_clear > 0.03).mean(0)
    print(f"[fk] {len(link_names)} links | foot ground z={np.round(ground, 3)} "
          f"clearance max={np.round(foot_clear.max(0), 3)} swing frac(>0.03)={np.round(swing, 2)}")

    # ── 5. save per-frame NPZ ──
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        fps=np.float32(args.out_fps),
        joint_names=np.array(JOINT_NAMES),
        body_names=np.array(link_names),
        root_pos=root_pos.astype(np.float32),
        root_quat=quat.astype(np.float32),                   # wxyz
        root_lin_vel=root_lin_vel.astype(np.float32),        # world
        root_ang_vel_body=root_ang_vel_body.astype(np.float32),
        dof_pos=dof.astype(np.float32),                      # JOINT_NAMES order
        dof_vel=dof_vel.astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,                             # wxyz
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
