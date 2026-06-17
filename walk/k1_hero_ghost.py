"""Semi-transparenter Hero-Ghost (NPZ-Referenz) für Trainings-Rollout-Videos.

Die Referenz-NPZ läuft in ihrer eigenen Welt-Richtung (typ. −x). Training nutzt
Commands im Spawn-Heading-Frame (+vx vorwärts). Der Ghost wird per Yaw-Rotation
in dieselbe Richtung/Orientierung gebracht und startet am Trainings-Spawn.
"""

from __future__ import annotations

import os

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import inv_quat, transform_by_quat, transform_quat_by_quat, xyz_to_quat


class HeroGhost:
    """Kinematic K1-URDF: Referenzpose im Trainings-Heading — nur Visualisierung."""

    def __init__(self, scene, env_cfg: dict, reward_cfg: dict) -> None:
        vcfg = env_cfg.get("video", {})
        color = vcfg.get("hero_ghost_color", [0.55, 1.0, 0.55])
        opacity = float(vcfg.get("hero_ghost_opacity", 0.42))
        self.entity = scene.add_entity(
            gs.morphs.URDF(
                file="models/K1/K1_22dof.urdf",
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            ),
            material=gs.materials.Kinematic(),
            surface=gs.surfaces.Default(
                color=tuple(float(c) for c in color),
                opacity=opacity,
            ),
        )
        path = reward_cfg["style_motion_file"]
        if not os.path.isabs(path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(repo_root, path)
        self._motion_path = path
        self._ready = False
        self.T = 0
        self.yaw_align_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)

    def attach_after_build(self, env) -> None:
        """Motion laden und Yaw-Offset NPZ → Trainings-Heading berechnen."""
        if self._ready:
            return
        data = np.load(self._motion_path, allow_pickle=True)
        npz_jn = [str(x) for x in data["joint_names"]]
        robot_joint_names = [j.name for j in self.entity.joints[1:]]
        col_full = [npz_jn.index(n) for n in robot_joint_names]

        dev = gs.device
        self.T = int(data["root_pos"].shape[0])
        self.root_pos = torch.tensor(data["root_pos"], dtype=gs.tc_float, device=dev)
        self.root_quat = torch.tensor(data["root_quat"], dtype=gs.tc_float, device=dev)
        self.root_lin_vel = torch.tensor(data["root_lin_vel"], dtype=gs.tc_float, device=dev)
        self.root_ang_vel_body = torch.tensor(data["root_ang_vel_body"], dtype=gs.tc_float, device=dev)
        self.dof_full = torch.tensor(data["dof_pos"][:, col_full], dtype=gs.tc_float, device=dev)
        self.dof_vel_full = torch.tensor(data["dof_vel"][:, col_full], dtype=gs.tc_float, device=dev)

        loop_disp = (self.root_pos[self.T - 1] - self.root_pos[0]).clone()
        loop_disp[2] = 0.0
        self.loop_disp = loop_disp

        self.yaw_align_quat = self._compute_yaw_align(env)
        self._ready = True

    def _compute_yaw_align(self, env) -> torch.Tensor:
        """Dreht NPZ-Vorwärtsrichtung auf +x im Spawn-Heading-Frame (wie Training-Command)."""
        dev = gs.device
        n = min(50, self.T - 1)
        delta = self.root_pos[n] - self.root_pos[0]
        fwd_npz = delta[:2]
        if torch.norm(fwd_npz) < 1e-4:
            fwd_npz = self.root_lin_vel[:n, :2].mean(dim=0)
        fwd_npz = fwd_npz / torch.norm(fwd_npz).clamp(min=1e-6)

        # +x im Heading-Frame → Welt (Spawn-Yaw)
        heading_quat = inv_quat(env.inv_heading_quat[0:1])
        fwd_train = transform_by_quat(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=gs.tc_float, device=dev),
            heading_quat,
        )[0, :2]
        fwd_train = fwd_train / torch.norm(fwd_train).clamp(min=1e-6)

        yaw_npz = torch.atan2(fwd_npz[1], fwd_npz[0])
        yaw_train = torch.atan2(fwd_train[1], fwd_train[0])
        yaw_delta = yaw_train - yaw_npz
        spawn_rpy = torch.zeros(1, 3, dtype=gs.tc_float, device=dev)
        spawn_rpy[0, 2] = yaw_delta
        return xyz_to_quat(spawn_rpy)[0]

    def _align_pos(self, pos_rel: torch.Tensor) -> torch.Tensor:
        return transform_by_quat(pos_rel.unsqueeze(0), self.yaw_align_quat.unsqueeze(0))[0]

    def set_frame(self, step: int, env) -> None:
        """Referenzpose + Geschwindigkeit im Trainings-Heading, Start = env.init_base_pos."""
        if not self._ready:
            self.attach_after_build(env)

        t = step % self.T
        cycle = step // self.T
        pos_ref = self.root_pos[t] + cycle * self.loop_disp
        pos_rel = pos_ref - self.root_pos[0]
        ghost_pos = env.init_base_pos + self._align_pos(pos_rel)

        ghost_quat = transform_quat_by_quat(
            self.yaw_align_quat.unsqueeze(0),
            self.root_quat[t].unsqueeze(0),
        )[0]

        qpos = torch.cat([ghost_pos, ghost_quat, self.dof_full[t]]).unsqueeze(0)
        self.entity.set_qpos(qpos, zero_velocity=False, skip_forward=False)

        wlv = transform_by_quat(
            self.root_lin_vel[t].unsqueeze(0),
            self.yaw_align_quat.unsqueeze(0),
        )[0]
        ang_world = transform_by_quat(
            self.root_ang_vel_body[t].unsqueeze(0),
            ghost_quat.unsqueeze(0),
        )[0]
        vel_full = torch.cat([wlv, ang_world, self.dof_vel_full[t]]).unsqueeze(0)
        try:
            self.entity.set_dofs_velocity(vel_full, skip_forward=False)
        except Exception:
            pass
