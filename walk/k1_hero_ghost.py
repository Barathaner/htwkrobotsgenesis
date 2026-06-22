"""Semi-transparenter Hero-Ghost (NPZ-Referenz) für Trainings-Rollout-Videos.

Die Referenz-NPZ läuft in ihrer eigenen Welt-Richtung (typ. −x). Training nutzt
Commands im Spawn-Heading-Frame (+vx vorwärts). Der Ghost wird per Yaw-Rotation
in dieselbe Richtung/Orientierung gebracht und startet am Trainings-Spawn.

Position wird inkrementell integriert (nur der aktuelle NPZ-Step-Delta wird mit
der aktuellen Command-Richtung rotiert). Command-Wechsel ohne Episode-Reset
dreht den Ghost, ohne die bisherige Weltposition neu vom Spawn zu spiegeln.
"""

from __future__ import annotations

import os

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import inv_quat, transform_by_quat, transform_quat_by_quat, xyz_to_quat

# Per-env ghost colors (RGB, 0–1) — index matches env index
ENV_GHOST_COLORS = [
    (0.24, 1.00, 0.24),   # env 0: green
    (1.00, 0.55, 0.00),   # env 1: orange
    (0.00, 0.85, 1.00),   # env 2: cyan
    (0.85, 0.00, 1.00),   # env 3: purple
]


class HeroGhost:
    """One kinematic K1-URDF per env, each a different colour, following that env's command."""

    def __init__(self, scene, env_cfg: dict, reward_cfg: dict, num_envs: int = 1) -> None:
        vcfg = env_cfg.get("video", {})
        opacity = float(vcfg.get("hero_ghost_opacity", 0.42))
        urdf_pos = env_cfg["base_init_pos"]
        urdf_quat = env_cfg["base_init_quat"]

        self.num_envs = num_envs
        self.entities: list = []
        colors = vcfg.get("ghost_colors", ENV_GHOST_COLORS)
        for i in range(num_envs):
            color = tuple(float(c) for c in colors[i % len(colors)])
            entity = scene.add_entity(
                gs.morphs.URDF(
                    file="models/K1/K1_22dof.urdf",
                    pos=urdf_pos,
                    quat=urdf_quat,
                ),
                material=gs.materials.Kinematic(),
                surface=gs.surfaces.Default(color=color, opacity=opacity),
            )
            self.entities.append(entity)

        path = reward_cfg["style_motion_file"]
        if not os.path.isabs(path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(repo_root, path)
        self._motion_path = path
        self._ready = False
        self.T = 0
        self.yaw_align_quats: list[torch.Tensor] = []
        self._world_pos: torch.Tensor | None = None   # (num_envs, 3) integrated root position
        self._last_t: torch.Tensor | None = None      # (num_envs,) previous NPZ frame; -1 after reset
        self._hidden_qpos: torch.Tensor | None = None

    def attach_after_build(self, env) -> None:
        """Motion laden, Yaw-Offset berechnen, Spawn-Positionen jedes Envs speichern."""
        if self._ready:
            return
        data = np.load(self._motion_path, allow_pickle=True)
        npz_jn = [str(x) for x in data["joint_names"]]
        robot_joint_names = [j.name for j in self.entities[0].joints[1:]]
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

        self.yaw_align_quats = [self._compute_yaw_align(env, i) for i in range(self.num_envs)]

        self._world_pos = env.robot.get_pos().clone().detach()
        self._last_t = torch.full((self.num_envs,), -1, dtype=torch.long, device=dev)

        hidden_pos = torch.tensor([0.0, 0.0, -20.0], dtype=gs.tc_float, device=dev)
        hidden_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=dev)
        self._hidden_qpos = torch.cat([hidden_pos, hidden_quat, self.dof_full[0]])

        self._ready = True

    def reset_envs(self, env, envs_idx=None) -> None:
        """Sync ghost lifecycle with env episode reset — only place that restarts world position."""
        if not self._ready:
            return
        spawn = env.robot.get_pos()
        if envs_idx is None:
            self._world_pos.copy_(spawn)
            self._last_t.fill_(-1)
            for i in range(self.num_envs):
                self.yaw_align_quats[i] = self._compute_yaw_align(env, i)
        else:
            reset_indices = envs_idx.nonzero(as_tuple=True)[0]
            for i in reset_indices.tolist():
                self._world_pos[i] = spawn[i]
                self._last_t[i] = -1
                self.yaw_align_quats[i] = self._compute_yaw_align(env, i)

    def _compute_yaw_align(self, env, env_idx: int = 0) -> torch.Tensor:
        """Rotates NPZ forward direction to match env env_idx's spawn heading in world frame."""
        dev = gs.device
        n = min(50, self.T - 1)
        delta = self.root_pos[n] - self.root_pos[0]
        fwd_npz = delta[:2]
        if torch.norm(fwd_npz) < 1e-4:
            fwd_npz = self.root_lin_vel[:n, :2].mean(dim=0)
        fwd_npz = fwd_npz / torch.norm(fwd_npz).clamp(min=1e-6)

        heading_quat = inv_quat(env.inv_heading_quat[env_idx : env_idx + 1])
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

    def _get_cmd_rot(self, env, env_idx: int) -> torch.Tensor | None:
        """World-Z rotation (1,4) für die commanded velocity direction von env env_idx, or None."""
        speed = torch.norm(env.commands[env_idx, :2]).item()
        if speed < 0.05:
            return None
        cmd_angle = torch.atan2(env.commands[env_idx, 1], env.commands[env_idx, 0])
        cmd_rpy = torch.zeros(1, 3, dtype=gs.tc_float, device=gs.device)
        cmd_rpy[0, 2] = cmd_angle
        return xyz_to_quat(cmd_rpy)

    def _pos_align_quat(self, env, env_idx: int) -> torch.Tensor:
        """Combined yaw_align @ cmd_rot for translating NPZ deltas into world frame."""
        base = self.yaw_align_quats[env_idx].unsqueeze(0)
        cmd_rot = self._get_cmd_rot(env, env_idx)
        return transform_quat_by_quat(base, cmd_rot) if cmd_rot is not None else base

    def _integrate_npz_delta(self, env_idx: int, delta_npz: torch.Tensor, pos_align: torch.Tensor) -> None:
        self._world_pos[env_idx] = self._world_pos[env_idx] + transform_by_quat(
            delta_npz.unsqueeze(0), pos_align
        )[0]

    def _advance_world_pos(self, env, env_idx: int, t: int) -> None:
        """Integrate NPZ root displacement from last_t to t using current command direction."""
        last = int(self._last_t[env_idx].item())
        if last < 0:
            self._last_t[env_idx] = t
            return
        if last == t:
            return

        pos_align = self._pos_align_quat(env, env_idx)

        if t > last:
            for frame in range(last + 1, t + 1):
                delta_npz = self.root_pos[frame] - self.root_pos[frame - 1]
                self._integrate_npz_delta(env_idx, delta_npz, pos_align)
        else:
            # NPZ loop wrap: last → T-1, then T-1 → 0 (+ loop_disp), then 0 → t
            for frame in range(last + 1, self.T):
                delta_npz = self.root_pos[frame] - self.root_pos[frame - 1]
                self._integrate_npz_delta(env_idx, delta_npz, pos_align)
            delta_npz = (self.root_pos[0] - self.root_pos[self.T - 1]) + self.loop_disp
            self._integrate_npz_delta(env_idx, delta_npz, pos_align)
            for frame in range(1, t + 1):
                delta_npz = self.root_pos[frame] - self.root_pos[frame - 1]
                self._integrate_npz_delta(env_idx, delta_npz, pos_align)

        self._last_t[env_idx] = t

    def _ghost_qpos_for_env(self, t: int, env, env_idx: int) -> torch.Tensor:
        """Compute the qpos (29,) for ghost entity env_idx at NPZ frame t."""
        base = self.yaw_align_quats[env_idx].unsqueeze(0)
        cmd_rot = self._get_cmd_rot(env, env_idx)

        ghost_pos = self._world_pos[env_idx]

        ghost_quat = transform_quat_by_quat(base, self.root_quat[t].unsqueeze(0))
        if cmd_rot is not None:
            ghost_quat = transform_quat_by_quat(ghost_quat, cmd_rot)
        ghost_quat = ghost_quat[0]

        return torch.cat([ghost_pos, ghost_quat, self.dof_full[t]])

    def set_frame(self, env) -> None:
        """Update each ghost entity: env i's ghost follows env i's command and spawn position."""
        if not self._ready:
            self.attach_after_build(env)

        n = self.num_envs

        for i, entity in enumerate(self.entities):
            env_step = int(env.episode_length_buf[i].item())
            t = env_step % self.T

            self._advance_world_pos(env, i, t)

            qpos = self._hidden_qpos.unsqueeze(0).expand(n, -1).contiguous().clone()
            qpos[i] = self._ghost_qpos_for_env(t, env, i)
            entity.set_qpos(qpos, zero_velocity=False, skip_forward=False)

            pos_align = self._pos_align_quat(env, i)
            ghost_quat = qpos[i, 3:7]

            wlv = transform_by_quat(self.root_lin_vel[t].unsqueeze(0), pos_align)[0]
            ang_world = transform_by_quat(
                self.root_ang_vel_body[t].unsqueeze(0), ghost_quat.unsqueeze(0)
            )[0]
            vel_full = torch.cat([wlv, ang_world, self.dof_vel_full[t]])
            vel_batch = torch.zeros((n, vel_full.shape[0]), dtype=gs.tc_float, device=gs.device)
            vel_batch[i] = vel_full
            try:
                entity.set_dofs_velocity(vel_batch, skip_forward=False)
            except Exception:
                pass
