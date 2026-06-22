"""Expert motion data loader for AMP — bridges K1 .npz files to AMP_PPO's feed_forward_generator API.

Features match K1Env._build_amp_obs():
  [root_height(1), projected_gravity(3), dof_pos(n), dof_vel(n), foot_clear(2), commands(3)]
Same per-group normalization + sqrt-weight as _setup_style_reference, with raw commands appended.

When multiple paths are provided, mean/std are computed from ALL motion features combined so that
every file is normalized in the same feature space.  Per-file cmd_ranges control which command
values are randomly sampled and appended to expert transitions — this gives the discriminator
the commanded speed context it needs to score motion appropriately.
"""

from __future__ import annotations

import os
import numpy as np
import torch


class K1AMPLoader:
    """Wraps pre-computed .npz motion reference data as an AMP expert dataset.

    Produces (state, next_state) pairs whose features and normalization exactly
    match K1Env._build_amp_obs(): normalized motion features (38-dim) followed by
    raw velocity commands (3-dim) sampled from per-file cmd_ranges.

    With multiple files the combined mean/std are used for motion features across
    all files.  self.mean / .std / .sqrt_w expose the 38-dim motion stats so callers
    can push them into the env's _build_amp_obs().

    Args:
        paths: list of .npz motion reference file paths.
        joint_names: ordered policy joint names (must match env_cfg["joint_names"]).
        device: torch device string.
        cmd_ranges: per-file command ranges as list of (lo, hi) numpy arrays of shape (3,)
            representing [lin_vel_x, lin_vel_y, ang_vel_yaw] limits.  If None or shorter
            than paths, missing files get zero commands appended.
    """

    def __init__(
        self,
        paths: list[str],
        joint_names: list[str],
        device: str,
        cmd_ranges: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.device = device
        n = len(joint_names)

        # ── per-group sqrt-weights (same as _setup_style_reference) ──────────
        w = np.concatenate([
            np.full(1,  0.5,  np.float32),   # root_height
            np.full(3,  2.0,  np.float32),   # projected_gravity
            np.full(n,  1.0,  np.float32),   # dof_pos
            np.full(n,  0.25, np.float32),   # dof_vel
            np.full(2,  2.0,  np.float32),   # foot_clear
        ])
        sqrt_w = np.sqrt(w)

        # ── first pass: load raw (un-normalised) motion arrays ────────────────
        raw_arrays: list[np.ndarray] = []
        for path in paths:
            if not os.path.isabs(path):
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                path = os.path.join(repo_root, path)

            data = np.load(path, allow_pickle=True)
            ref_jn = [str(n_) for n_ in data["joint_names"]]
            col = [ref_jn.index(name) for name in joint_names]

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
            raw_arrays.append(ref)

        # ── combined motion stats from ALL files ──────────────────────────────
        combined = np.concatenate(raw_arrays, axis=0)
        mean = combined.mean(0)
        std  = combined.std(0) + 1e-6

        # ── second pass: normalise motion features + append sampled commands ──
        obs_list: list[torch.Tensor] = []
        next_obs_list: list[torch.Tensor] = []
        total_transitions = 0
        for i, ref in enumerate(raw_arrays):
            ref_norm = (ref - mean) / std * sqrt_w  # (M, 38)

            M = len(ref)
            # Sample one command per transition (same cmd for state and next_state so
            # the discriminator sees a stable command context, matching the env where
            # commands are held constant for many steps).
            if cmd_ranges is not None and i < len(cmd_ranges):
                lo, hi = cmd_ranges[i]
                cmds = (hi - lo) * np.random.rand(M - 1, len(lo)).astype(np.float32) + lo
            else:
                cmds = np.zeros((M - 1, 3), dtype=np.float32)

            s  = np.concatenate([ref_norm[:-1], cmds], axis=1)  # (M-1, 38+3)
            sn = np.concatenate([ref_norm[1:],  cmds], axis=1)  # same cmd for next

            obs_list.append(torch.tensor(s,  dtype=torch.float32, device=device))
            next_obs_list.append(torch.tensor(sn, dtype=torch.float32, device=device))
            total_transitions += M - 1

        self.all_obs      = torch.cat(obs_list,      dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)

        # Expose 38-dim motion stats so the runner can push them into env._build_amp_obs().
        # Commands are appended raw (not normalized) in both policy and expert obs.
        self.mean   = torch.tensor(mean,   dtype=torch.float32, device=device)
        self.std    = torch.tensor(std,    dtype=torch.float32, device=device)
        self.sqrt_w = torch.tensor(sqrt_w, dtype=torch.float32, device=device)

        feat_dim = self.all_obs.shape[1]
        print(f"[K1AMPLoader] {total_transitions} expert transitions "
              f"from {len(paths)} file(s)  feat_dim={feat_dim} "
              f"(motion={combined.shape[1]} + cmd={feat_dim - combined.shape[1]})")

    def __len__(self) -> int:
        return len(self.all_obs)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Yield (state, next_state) mini-batches sampled uniformly with replacement."""
        N = len(self.all_obs)
        for _ in range(num_mini_batch):
            idx = torch.randint(N, (mini_batch_size,), device=self.device)
            yield self.all_obs[idx], self.all_next_obs[idx]
