"""Colored kinematic shadow robots — one per env, each in that env's HUD color.

Each entity mirrors the training robot's exact qpos for its env and parks all
other env slots underground, so the renderer sees exactly one colored robot per
training env without touching the physics robot.
"""

from __future__ import annotations

import genesis as gs
import torch
from k1_hero_ghost import ENV_GHOST_COLORS


class ColorShadow:
    """One kinematic K1-URDF per env colored to match the HUD and command arrow."""

    def __init__(self, scene, env_cfg: dict, num_envs: int) -> None:
        vcfg = env_cfg.get("video", {})
        urdf_pos = env_cfg["base_init_pos"]
        urdf_quat = env_cfg["base_init_quat"]
        colors = vcfg.get("ghost_colors", ENV_GHOST_COLORS)

        self.num_envs = num_envs
        self.entities: list = []
        for i in range(num_envs):
            color = tuple(float(c) for c in colors[i % len(colors)])
            entity = scene.add_entity(
                gs.morphs.URDF(
                    file="models/K1/K1_22dof.urdf",
                    pos=urdf_pos,
                    quat=urdf_quat,
                ),
                material=gs.materials.Kinematic(),
                surface=gs.surfaces.Default(color=color, opacity=1.0),
            )
            self.entities.append(entity)

        self._hidden_qpos: torch.Tensor | None = None

    def _build_hidden_qpos(self, env) -> torch.Tensor:
        hidden_pos = torch.tensor([0.0, 0.0, -20.0], dtype=gs.tc_float, device=gs.device)
        hidden_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
        return torch.cat([hidden_pos, hidden_quat, env.init_dof_pos])

    def update(self, env) -> None:
        if self._hidden_qpos is None:
            self._hidden_qpos = self._build_hidden_qpos(env)

        n = self.num_envs
        full_qpos = env.robot.get_qpos()  # (n_envs, n_qs): [pos(3), quat(4), dofs...]

        for i, entity in enumerate(self.entities):
            qpos = self._hidden_qpos.unsqueeze(0).expand(n, -1).contiguous().clone()
            qpos[i] = full_qpos[i]
            entity.set_qpos(qpos, zero_velocity=True, skip_forward=False)
