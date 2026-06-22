"""Expert motion data loader for AMP — bridges K1 .npz files to AMP_PPO's feed_forward_generator API.

Features match K1Env._build_amp_obs():
  [root_height(1), projected_gravity(3), dof_pos(n), dof_vel(n), foot_clear(2)]
Same per-group normalization + sqrt-weight as _setup_style_reference.
"""

from __future__ import annotations

import os
import numpy as np
import torch


class K1AMPLoader:
    """Wraps pre-computed .npz motion reference data as an AMP expert dataset.

    Produces (state, next_state) pairs whose features and normalization exactly
    match K1Env._build_amp_obs(), so the discriminator sees consistent inputs.
    """

    def __init__(self, paths: list[str], joint_names: list[str], device: str) -> None:
        self.device = device

        obs_list: list[torch.Tensor] = []
        next_obs_list: list[torch.Tensor] = []

        for path in paths:
            if not os.path.isabs(path):
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                path = os.path.join(repo_root, path)

            data = np.load(path, allow_pickle=True)
            ref_jn = [str(n) for n in data["joint_names"]]
            col = [ref_jn.index(n) for n in joint_names]
            n = len(joint_names)

            ref = np.concatenate(
                [
                    data["root_pos"][:, 2:3],
                    data["projected_gravity"],
                    data["dof_pos"][:, col],
                    data["dof_vel"][:, col],
                    data["foot_clear"],
                ],
                axis=1,
            ).astype(np.float32)

            mean = ref.mean(0)
            std = ref.std(0) + 1e-6
            w = np.concatenate(
                [
                    np.full(1, 0.5, np.float32),
                    np.full(3, 2.0, np.float32),
                    np.full(n, 1.0, np.float32),
                    np.full(n, 0.25, np.float32),
                    np.full(2, 2.0, np.float32),
                ]
            )
            sqrt_w = np.sqrt(w)
            ref_norm = (ref - mean) / std * sqrt_w

            s = torch.tensor(ref_norm[:-1], dtype=torch.float32, device=device)
            sn = torch.tensor(ref_norm[1:], dtype=torch.float32, device=device)
            obs_list.append(s)
            next_obs_list.append(sn)

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)
        print(f"[K1AMPLoader] {len(self.all_obs)} expert transitions from {len(paths)} file(s)")

    def __len__(self) -> int:
        return len(self.all_obs)

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ):
        """Yield (state, next_state) mini-batches sampled uniformly with replacement."""
        N = len(self.all_obs)
        for _ in range(num_mini_batch):
            idx = torch.randint(N, (mini_batch_size,), device=self.device)
            yield self.all_obs[idx], self.all_next_obs[idx]
