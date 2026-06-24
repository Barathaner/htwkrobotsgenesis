"""Re-center a K1 motion clip so it has no NET lateral drift or yaw turn (fix #1 for the
deploy "drift right" bug).

The slow.npz mocap walk carries a baked-in mean yaw rate (~-0.10 rad/s) and a small mean
heading-frame lateral velocity. The AMP discriminator reads exactly two root-drift signals —
heading-frame lateral velocity (vy) and yaw rate (wz) — so with a high style weight the policy
imitates that turn even at zero command → continuous drift.

This script zeroes the MEAN of those two signals while preserving their oscillation (a straight
walk has ~0 mean lateral/yaw over a cycle; this clip didn't). Only root_lin_vel and
root_ang_vel_body[:,2] are touched — exactly the fields the discriminator / style reward derive
the drift features from (height, projected_gravity, dof_*, foot_clear are unaffected; root_quat
is left intact so RSI spawn orientation and the heading transform are unchanged).

Usage (from repo root):
  .venv/bin/python walk/recenter_slow_motion.py walk/data/motions/slow.npz
A backup <name>_original.npz is written once; re-running is idempotent.
"""

from __future__ import annotations

import os
import sys
import numpy as np


def _yaw_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Yaw angle from a wxyz quaternion (matches K1Env / k1_amp_loader heading extraction)."""
    q = quat_wxyz.astype(np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _heading_vel(quat_wxyz: np.ndarray, vel_world: np.ndarray) -> np.ndarray:
    """World linear velocity -> heading frame [vx, vy] (yaw-only)."""
    yaw = _yaw_from_quat(quat_wxyz)
    c, s = np.cos(-yaw), np.sin(-yaw)
    vx = c * vel_world[:, 0] - s * vel_world[:, 1]
    vy = s * vel_world[:, 0] + c * vel_world[:, 1]
    return np.stack([vx, vy], axis=-1)


def recenter(path: str) -> None:
    data = dict(np.load(path, allow_pickle=True))
    quat = data["root_quat"]
    vw = data["root_lin_vel"].astype(np.float64)
    wz = data["root_ang_vel_body"][:, 2].astype(np.float64)

    vh = _heading_vel(quat, vw)
    print(f"[before] mean heading vel = [{vh[:,0].mean():+.4f}, {vh[:,1].mean():+.4f}]  "
          f"mean yaw_rate = {wz.mean():+.4f}")

    # 1) zero mean heading-frame lateral velocity (vy), preserving vx and the oscillation
    yaw = _yaw_from_quat(quat)
    c, s = np.cos(-yaw), np.sin(-yaw)
    vxh = c * vw[:, 0] - s * vw[:, 1]
    vyh = s * vw[:, 0] + c * vw[:, 1]
    vyh = vyh - vyh.mean()
    # rotate heading-frame velocity back into the world frame (inverse yaw)
    ci, si = np.cos(yaw), np.sin(yaw)
    vw_new = np.stack([ci * vxh - si * vyh, si * vxh + ci * vyh, vw[:, 2]], axis=-1)
    data["root_lin_vel"] = vw_new.astype(np.float32)

    # 2) zero mean yaw rate (net turn), preserving the per-step oscillation
    w_full = data["root_ang_vel_body"].astype(np.float64)
    w_full[:, 2] = w_full[:, 2] - w_full[:, 2].mean()
    data["root_ang_vel_body"] = w_full.astype(np.float32)

    # backup once, then overwrite
    backup = path.replace(".npz", "_original.npz")
    if not os.path.exists(backup):
        np.savez(backup, **dict(np.load(path, allow_pickle=True)))
        print(f"[backup] wrote {backup}")
    np.savez(path, **data)

    # verify
    chk = np.load(path, allow_pickle=True)
    vh2 = _heading_vel(chk["root_quat"], chk["root_lin_vel"].astype(np.float64))
    wz2 = chk["root_ang_vel_body"][:, 2]
    print(f"[after ] mean heading vel = [{vh2[:,0].mean():+.4f}, {vh2[:,1].mean():+.4f}]  "
          f"mean yaw_rate = {wz2.mean():+.4f}")
    print(f"[saved ] {path}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "walk/data/motions/slow.npz"
    if not os.path.isabs(p):
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)
    recenter(p)
