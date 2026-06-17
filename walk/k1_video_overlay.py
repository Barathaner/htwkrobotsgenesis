"""Gemeinsames HUD + Command-Pfeil für Hero-Test und Trainings-Rollout-Videos."""

from __future__ import annotations

import cv2
import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import inv_quat, transform_by_quat


def draw_hud(rgb: np.ndarray, cmd_vx: float, cmd_vy: float, speed: float) -> np.ndarray:
    """Command-Geschwindigkeit als Text (Richtung = 3D-Pfeil über dem Kopf)."""
    lines = [
        f"cmd speed: {speed:0.2f} m/s",
        f"cmd vx/vy: {cmd_vx:+0.2f} / {cmd_vy:+0.2f} m/s",
    ]
    for i, s in enumerate(lines):
        org = (14, 30 + 28 * i)
        cv2.putText(rgb, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(rgb, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 60), 1, cv2.LINE_AA)
    return rgb


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


def draw_command_arrow(rgb: np.ndarray, env, cmd_unit_world: torch.Tensor, speed: float) -> None:
    """2D-Projektion eines horizontalen Command-Pfeils über dem Kopf."""
    horiz = cmd_unit_world.clone()
    horiz[2] = 0.0
    n = torch.norm(horiz).clamp(min=1e-6)
    horiz = horiz / n
    length = 0.35 + 0.55 * speed
    base = env.base_pos[0].detach().cpu().numpy()
    p0 = base + np.array([0.0, 0.0, 0.65])
    p1 = p0 + horiz.detach().cpu().numpy() * length

    uv, z = project_to_pixels(np.stack([p0, p1]), env.cam)
    if z[0] <= 0 or z[1] <= 0:
        return
    a = (int(round(uv[0, 0])), int(round(uv[0, 1])))
    b = (int(round(uv[1, 0])), int(round(uv[1, 1])))
    cv2.arrowedLine(rgb, a, b, (0, 0, 0), 7, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(rgb, a, b, (60, 255, 60), 4, cv2.LINE_AA, tipLength=0.3)


def cmd_overlay_from_env(env) -> tuple[torch.Tensor, float, float, float]:
    """Command-Richtung (Welt) + vx/vy/speed aus env.commands (Heading-Frame)."""
    heading_quat = inv_quat(env.inv_heading_quat)
    cmd_vx = float(env.commands[0, 0])
    cmd_vy = float(env.commands[0, 1])
    speed = float(torch.norm(env.commands[0, :2]).item())
    cmd_dir = torch.tensor([cmd_vx, cmd_vy, 0.0], dtype=gs.tc_float, device=env.device)
    unit = transform_by_quat((cmd_dir / max(speed, 1e-6)).unsqueeze(0), heading_quat)[0]
    return unit, cmd_vx, cmd_vy, speed


def render_annotated_frame(env, *, force_render: bool = False) -> np.ndarray:
    """Genesis-Render + Command-Pfeil + HUD (Hero + Trainings-Videos)."""
    assert env.cam is not None
    out = env.cam.render(force_render=force_render)
    rgb = out[0] if isinstance(out, (tuple, list)) else out
    rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3]).astype(np.uint8)
    unit, cmd_vx, cmd_vy, speed = cmd_overlay_from_env(env)
    draw_command_arrow(rgb, env, unit, speed)
    return draw_hud(rgb, cmd_vx, cmd_vy, speed)
