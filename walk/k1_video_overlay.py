"""Gemeinsames HUD + Command-Pfeile für Hero-Test und Trainings-Rollout-Videos."""

from __future__ import annotations

import cv2
import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import inv_quat, transform_by_quat

# Per-env colors (BGR for OpenCV) — must stay in sync with k1_hero_ghost.ENV_GHOST_COLORS
ENV_COLORS_BGR = [
    (60,  255,  60),   # env 0: green
    (  0, 140, 255),   # env 1: orange
    (220, 210,   0),   # env 2: cyan
    (220,   0, 200),   # env 3: purple
]


def project_to_pixels(world_pts: np.ndarray, cam) -> tuple[np.ndarray, np.ndarray]:
    """Welt-Punkte (N,3) → Pixel (uv) und Kamera-tiefe z."""
    T = np.asarray(cam.transform, dtype=np.float64)
    if T.ndim == 3:
        T = T[0]
    Tf = T.copy()
    Tf[:3, 1:3] *= -1
    E = np.linalg.inv(Tf)
    hom = np.concatenate([np.asarray(world_pts, np.float64), np.ones((len(world_pts), 1))], axis=1)
    cam_pts = (E @ hom.T).T[:, :3]
    z = cam_pts[:, 2]
    zc = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = np.stack([cam.f * cam_pts[:, 0] / zc + cam.cx, cam.f * cam_pts[:, 1] / zc + cam.cy], axis=1)
    return uv, z


def _cmd_info_for_env(env, env_idx: int) -> tuple[torch.Tensor, float, float, float]:
    """Command direction in world frame + scalars for one env."""
    heading_quat = inv_quat(env.inv_heading_quat[env_idx : env_idx + 1])
    cmd_vx = float(env.commands[env_idx, 0])
    cmd_vy = float(env.commands[env_idx, 1])
    speed = float(torch.norm(env.commands[env_idx, :2]).item())
    cmd_dir = torch.tensor([cmd_vx, cmd_vy, 0.0], dtype=gs.tc_float, device=env.device)
    unit = transform_by_quat((cmd_dir / max(speed, 1e-6)).unsqueeze(0), heading_quat)[0]
    return unit, cmd_vx, cmd_vy, speed


def draw_env_markers(
    rgb: np.ndarray,
    env,
    env_idx: int,
    cmd_unit_world: torch.Tensor,
    speed: float,
    color_bgr: tuple[int, int, int],
) -> None:
    """Draw a command arrow above the robot."""
    base = env.base_pos[env_idx].detach().cpu().numpy()

    # ── command arrow ─────────────────────────────────────────────────────────
    horiz = cmd_unit_world.clone()
    horiz[2] = 0.0
    n = torch.norm(horiz).clamp(min=1e-6)
    horiz = horiz / n
    length = 0.30 + 0.50 * speed
    p0 = base + np.array([0.0, 0.0, 0.60])
    p1 = p0 + horiz.detach().cpu().numpy() * length

    uv, z = project_to_pixels(np.stack([p0, p1]), env.cam)
    if z[0] <= 0 or z[1] <= 0:
        return
    a = (int(round(uv[0, 0])), int(round(uv[0, 1])))
    b = (int(round(uv[1, 0])), int(round(uv[1, 1])))
    cv2.arrowedLine(rgb, a, b, (0, 0, 0), 7, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(rgb, a, b, color_bgr, 4, cv2.LINE_AA, tipLength=0.3)


def draw_all_hud(rgb: np.ndarray, env_infos: list[tuple[float, float, float, tuple]]) -> np.ndarray:
    """Top-left HUD: one compact line per env with per-env color.

    env_infos: list of (cmd_vx, cmd_vy, speed, color_bgr) per env.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    line_h = 18
    x0 = 8
    y0 = 18
    for i, (vx, vy, spd, color) in enumerate(env_infos):
        text = f"E{i}: spd={spd:.2f}  vx={vx:+.2f}  vy={vy:+.2f}"
        org = (x0, y0 + i * line_h)
        cv2.putText(rgb, text, org, font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(rgb, text, org, font, scale, color, thickness, cv2.LINE_AA)
    return rgb


def render_annotated_frame(env, *, force_render: bool = False) -> np.ndarray:
    """Genesis-Render + per-env Command-Pfeile + HUD."""
    assert env.cam is not None
    out = env.cam.render(force_render=force_render)
    rgb = out[0] if isinstance(out, (tuple, list)) else out
    rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3]).astype(np.uint8)

    n = env.num_envs
    env_infos: list[tuple[float, float, float, tuple]] = []
    for i in range(n):
        color = ENV_COLORS_BGR[i % len(ENV_COLORS_BGR)]
        unit, vx, vy, spd = _cmd_info_for_env(env, i)
        draw_env_markers(rgb, env, i, unit, spd, color)
        env_infos.append((vx, vy, spd, color))

    return draw_all_hud(rgb, env_infos)
