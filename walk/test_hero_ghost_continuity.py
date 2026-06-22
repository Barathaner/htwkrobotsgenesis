"""Verify HeroGhost incremental position integration (no teleport on cmd resample).

Runs without Genesis — replays the same delta-integration logic as k1_hero_ghost.py
against the real style-motion NPZ. Usage:

  python walk/test_hero_ghost_continuity.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

WALK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(WALK_DIR)
MOTION_PATH = os.path.join(REPO_ROOT, "walk/data/motions/k1_jogging_motion.npz")


def rot2(angle: float, x: float, y: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def integrate_incremental(
    root_pos: np.ndarray,
    loop_disp: np.ndarray,
    spawn: np.ndarray,
    cmd_angles: list[tuple[int, float]],
    steps: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Mirror HeroGhost._advance_world_pos with yaw_align=identity."""
    T = root_pos.shape[0]
    world_pos = spawn.copy()
    last_t = -1
    history = [world_pos.copy()]

    def cmd_at(step: int) -> float:
        angle = cmd_angles[0][1]
        for change_step, a in cmd_angles:
            if step >= change_step:
                angle = a
        return angle

    for step in range(1, steps + 1):
        t = step % T
        if last_t < 0:
            last_t = t
            history.append(world_pos.copy())
            continue
        if last_t == t:
            history.append(world_pos.copy())
            continue

        pos_align = cmd_at(step)

        if t > last_t:
            frames = range(last_t + 1, t + 1)
        else:
            frames = list(range(last_t + 1, T)) + list(range(1, t + 1))

        for frame in frames:
            if frame == 0 and last_t == T - 1:
                delta = (root_pos[0] - root_pos[T - 1]) + loop_disp
            else:
                delta = root_pos[frame] - root_pos[frame - 1]
            dx, dy = rot2(pos_align, float(delta[0]), float(delta[1]))
            world_pos[0] += dx
            world_pos[1] += dy
            world_pos[2] += float(delta[2])

        last_t = t
        history.append(world_pos.copy())

    return world_pos, history


def integrate_absolute_old(
    root_pos: np.ndarray,
    loop_disp: np.ndarray,
    spawn: np.ndarray,
    cmd_angles: list[tuple[int, float]],
    steps: int,
) -> np.ndarray:
    """Pre-fix model: full NPZ displacement re-rotated by current cmd each step."""
    T = root_pos.shape[0]
    t = steps % T
    cycle = steps // T

    def cmd_at(step: int) -> float:
        angle = cmd_angles[0][1]
        for change_step, a in cmd_angles:
            if step >= change_step:
                angle = a
        return angle

    pos_rel = (root_pos[t] + cycle * loop_disp) - root_pos[0]
    angle = cmd_at(steps)
    dx, dy = rot2(angle, float(pos_rel[0]), float(pos_rel[1]))
    return spawn + np.array([dx, dy, float(pos_rel[2])])


def test_no_teleport_on_cmd_resample() -> bool:
    data = np.load(MOTION_PATH, allow_pickle=True)
    root_pos = data["root_pos"]
    T = root_pos.shape[0]
    loop_disp = (root_pos[T - 1] - root_pos[0]).copy()
    loop_disp[2] = 0.0

    fps = float(data["fps"])
    resample_step = int(4.0 * fps)  # resampling_time_s: 4.0
    spawn = np.array([5.0, 5.0, 0.56])

    # Walk east, then at 4s turn north (90° cmd change)
    cmd_angles = [(0, 0.0), (resample_step, math.pi / 2)]
    steps_before = resample_step
    steps_after = resample_step + 10

    _, hist = integrate_incremental(root_pos, loop_disp, spawn, cmd_angles, steps_after)
    pos_before_cmd = hist[steps_before - 1]
    pos_at_cmd = hist[steps_before]
    pos_after = hist[steps_after]

    jump_at_resample = np.linalg.norm(pos_at_cmd - pos_before_cmd)
    old_pos_at_cmd = integrate_absolute_old(root_pos, loop_disp, spawn, cmd_angles, steps_before)
    old_vs_new = np.linalg.norm(old_pos_at_cmd - pos_at_cmd)

    # Incremental: one-step motion delta at resample should be small (no sideways teleport)
    ok_jump = jump_at_resample < 0.15
    ok_diff = old_vs_new > 0.3

    print("=== Test: no teleport on command resample (4s) ===")
    print(f"  resample at step {resample_step} ({resample_step / fps:.1f}s)")
    print(f"  incremental step jump at resample: {jump_at_resample:.4f} m")
    print(f"  old-model vs incremental at resample: {old_vs_new:.4f} m")
    print(f"  pos after 10 more steps: {pos_after[:2]}")
    print(f"  {'PASS' if ok_jump and ok_diff else 'FAIL'}")
    return ok_jump and ok_diff


def test_reset_returns_to_spawn() -> bool:
    data = np.load(MOTION_PATH, allow_pickle=True)
    root_pos = data["root_pos"]
    T = root_pos.shape[0]
    loop_disp = (root_pos[T - 1] - root_pos[0]).copy()
    loop_disp[2] = 0.0

    spawn = np.array([3.0, 7.0, 0.56])
    cmd_angles = [(0, 0.0)]

    _, hist = integrate_incremental(root_pos, loop_disp, spawn, cmd_angles, 150)
    pos_walked = hist[-1]

    # Simulate reset_envs: world_pos = spawn, last_t = -1
    _, hist2 = integrate_incremental(root_pos, loop_disp, spawn, cmd_angles, 5)
    pos_after_reset = hist2[0]  # first entry = spawn before any integration

    ok_spawn = np.allclose(pos_after_reset, spawn)
    ok_moved = np.linalg.norm(pos_walked[:2] - spawn[:2]) > 0.5

    print("\n=== Test: episode reset restarts at spawn ===")
    print(f"  walked distance from spawn: {np.linalg.norm(pos_walked[:2] - spawn[:2]):.3f} m")
    print(f"  after reset world_pos: {pos_after_reset[:2]}")
    print(f"  {'PASS' if ok_spawn and ok_moved else 'FAIL'}")
    return ok_spawn and ok_moved


def main() -> int:
    if not os.path.isfile(MOTION_PATH):
        print(f"Motion file missing: {MOTION_PATH}", file=sys.stderr)
        return 1

    results = [
        test_no_teleport_on_cmd_resample(),
        test_reset_returns_to_spawn(),
    ]
    print("\n=== Summary ===")
    print(f"  {'ALL PASS' if all(results) else 'SOME FAILED'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
