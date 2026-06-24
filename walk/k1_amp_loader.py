"""Expert motion data loader for AMP — bridges K1 .npz files to AMP_PPO's feed_forward_generator API.

Features match K1Env._build_amp_obs():
  [root_height(1), projected_gravity(3), root_lin_vel_xy(2), root_ang_vel_z(1),
   dof_pos(n), dof_vel(n), foot_clear(2), commands(3)]
Same per-group normalization + sqrt-weight as _setup_style_reference, with raw commands appended.

When multiple paths are provided, mean/std are computed from ALL motion features combined so that
every file is normalized in the same feature space.

Conditioning (the command channel) is NOT random.  Each expert transition is labelled with the
clip's OWN heading-frame root velocity [vx, vy, yaw_rate], EMA-smoothed with the same time constant
the env uses for command tracking (tracking_ema_window_s).  This gives the discriminator a command
label that genuinely correlates with the kinematics, so it can learn p_expert(motion | velocity)
instead of p_expert(motion) · uniform(command).  Per-file cmd_ranges, when given, only clamp the
derived label to the valid command range for that motion style.
"""

from __future__ import annotations

import os
import numpy as np
import torch


def _heading_frame_vel_xy(quat_wxyz: np.ndarray, vel_world: np.ndarray) -> np.ndarray:
    """Yaw-only rotation of a world-frame linear velocity into the heading frame → [vx, vy].

    Matches env.base_lin_vel_heading and build_motion_reference.body_frame_velocity (locomotion
    is yaw-invariant; pitch/roll live in projected_gravity)."""
    q = quat_wxyz.astype(np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    vx = c * vel_world[:, 0] - s * vel_world[:, 1]
    vy = s * vel_world[:, 0] + c * vel_world[:, 1]
    return np.stack([vx, vy], axis=-1).astype(np.float32)


def _build_joint_mirror(joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """L<->R permutation + sign flip for a sagittal (left/right) mirror of the policy joints.

    Partner = the joint name with Left<->Right swapped; sign = -1 for Roll/Yaw joints (lateral
    DOFs flip across the sagittal plane), +1 for Pitch joints. Matches K1Env._symm_mirror."""
    perm, sgn = [], []
    for name in joint_names:
        if "Left" in name:
            partner = name.replace("Left", "Right")
        elif "Right" in name:
            partner = name.replace("Right", "Left")
        else:
            partner = name
        perm.append(joint_names.index(partner))
        sgn.append(-1.0 if ("Roll" in name or "Yaw" in name) else 1.0)
    return np.asarray(perm, dtype=np.int64), np.asarray(sgn, dtype=np.float32)


def _mirror_motion(ref: np.ndarray, n: int, perm: np.ndarray, sgn: np.ndarray) -> np.ndarray:
    """Sagittal mirror of a motion-feature array. Layout (cols):
       [root_height(1) | projected_gravity(3) | vel_heading(2) | yaw_rate(1) |
        dof_pos(n) | dof_vel(n) | foot_clear(2)].
    Negates the lateral/antisymmetric channels and swaps left/right legs+feet, so adding the
    mirrored clip makes the expert set L/R-symmetric (zero net lateral velocity and yaw)."""
    m = ref.copy()
    m[:, 2] *= -1.0   # projected_gravity y (lateral lean)
    m[:, 5] *= -1.0   # heading lateral velocity vy
    m[:, 6] *= -1.0   # yaw rate
    m[:, 7:7 + n]         = ref[:, 7:7 + n][:, perm] * sgn          # dof_pos: swap L/R + sign flip
    m[:, 7 + n:7 + 2 * n] = ref[:, 7 + n:7 + 2 * n][:, perm] * sgn  # dof_vel: same map
    m[:, 7 + 2 * n:7 + 2 * n + 2] = ref[:, 7 + 2 * n:7 + 2 * n + 2][:, [1, 0]]  # foot_clear L<->R
    return m


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA along axis 0: y[t] = alpha·y[t-1] + (1-alpha)·x[t], y[0] = x[0]."""
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * y[t - 1] + (1.0 - alpha) * x[t]
    return y


def _time_warp_ref(
    ref: np.ndarray, vel_cols: np.ndarray, dt: float, k: float, ema_alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Retime a motion-feature array to play k× faster (k>1) / slower (k<1).

    A motion played at rate k is P(k·t) at real time t, so its velocity is d/dt P(k·t) =
    k·P'(k·t).  Therefore: pose columns are linearly resampled at clip-times k·t, and velocity
    columns are resampled AND scaled by k — keeping poses, joint velocities, root velocity and
    the conditioning label internally consistent at the new speed.  k=1 reproduces the input.

    Frames stay dt-spaced in real time (= one env step), so transitions remain env-consistent.

    Returns (ref_warped, cmd_warped) where cmd_warped is the EMA of the warped heading-frame
    velocity (cols 4:7 = [vx, vy, yaw_rate])."""
    M = len(ref)
    tau = np.arange(M, dtype=np.float64) * dt          # original clip times
    t_max = float(tau[-1])
    n_new = max(2, int(np.floor((t_max / k) / dt)) + 1)
    t = np.arange(n_new, dtype=np.float64) * dt        # new real-time grid (dt-spaced)
    sigma = np.clip(k * t, 0.0, t_max)                 # clip times to sample
    out = np.empty((n_new, ref.shape[1]), dtype=np.float32)
    for j in range(ref.shape[1]):
        out[:, j] = np.interp(sigma, tau, ref[:, j]).astype(np.float32)
    out[:, vel_cols] *= np.float32(k)                  # velocities scale with playback rate
    cmd = _ema(out[:, 4:7].copy(), ema_alpha)          # label from the warped (scaled) velocity
    return out, cmd


class K1AMPLoader:
    """Wraps pre-computed .npz motion reference data as an AMP expert dataset.

    Produces (state, next_state) pairs whose features and normalization exactly
    match K1Env._build_amp_obs(): normalized motion features (41-dim) followed by
    the velocity-derived command label (3-dim).

    With multiple files the combined mean/std are used for motion features across
    all files.  self.mean / .std / .sqrt_w expose the 41-dim motion stats so callers
    can push them into the env's _build_amp_obs().

    Time-warp augmentation: a single fixed-speed clip only demonstrates one velocity, so the
    discriminator can only bind that one speed.  warp_factors synthesizes retimed copies of a
    clip (poses resampled, velocities + command label scaled by the factor) so the discriminator
    sees the style across a velocity band.  ~±35% is physically safe; large factors compress
    contact timing past what the robot can reproduce.

    Args:
        paths: list of .npz motion reference file paths.
        joint_names: ordered policy joint names (must match env_cfg["joint_names"]).
        device: torch device string.
        cmd_ranges: per-file command ranges as list of (lo, hi) numpy arrays of shape (3,)
            representing [lin_vel_x, lin_vel_y, ang_vel_yaw] limits.  Used only to CLAMP the
            velocity-derived command label to the valid command range for that motion style.
            If None or shorter than paths, the label is left unclamped.
        ema_window_s: time constant [s] for the causal EMA applied to the expert velocity when
            forming the command label.  Should match the env's tracking_ema_window_s so the
            expert label and the policy command represent the same "mean velocity".
        warp_factors: per-file list of playback-rate factors to synthesize (e.g. [0.85, 1.0,
            1.15, 1.3]).  None or a missing entry means [1.0] (the native clip only).
    """

    def __init__(
        self,
        paths: list[str],
        joint_names: list[str],
        device: str,
        cmd_ranges: list[tuple[np.ndarray, np.ndarray]] | None = None,
        ema_window_s: float = 0.5,
        warp_factors: list[list[float]] | None = None,
        mirror_augment: bool = True,
    ) -> None:
        self.device = device
        n = len(joint_names)
        # velocity-feature column mask in the 41-dim motion vector: vel_heading(2)+yaw_rate(1)
        # at [4:7] and dof_vel(n) at [7+n:7+2n]. Used by the time-warp to scale velocities.
        motion_dim = 7 + 2 * n + 2
        vel_cols = np.zeros(motion_dim, dtype=bool)
        vel_cols[4:7] = True
        vel_cols[7 + n:7 + 2 * n] = True

        # ── per-group sqrt-weights (same as _setup_style_reference) ──────────
        w = np.concatenate([
            np.full(1,  0.5,  np.float32),   # root_height
            np.full(3,  2.0,  np.float32),   # projected_gravity
            np.full(2,  2.0,  np.float32),   # root_lin_vel_xy (heading frame)
            np.full(1,  1.0,  np.float32),   # root_ang_vel_z  (yaw rate)
            np.full(n,  1.0,  np.float32),   # dof_pos
            np.full(n,  0.25, np.float32),   # dof_vel
            np.full(2,  2.0,  np.float32),   # foot_clear
        ])
        sqrt_w = np.sqrt(w)

        # ── first pass: load raw (un-normalised) 41-dim motion arrays per file ──
        file_refs: list[np.ndarray] = []
        file_alpha: list[float] = []
        file_dt: list[float] = []
        for path in paths:
            if not os.path.isabs(path):
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                path = os.path.join(repo_root, path)

            data = np.load(path, allow_pickle=True)
            ref_jn = [str(n_) for n_ in data["joint_names"]]
            col = [ref_jn.index(name) for name in joint_names]

            # Heading-frame (yaw-only) root velocity — same quantity the env tracks commands against.
            vel_heading = _heading_frame_vel_xy(data["root_quat"], data["root_lin_vel"])  # (M,2)
            yaw_rate = data["root_ang_vel_body"][:, 2:3].astype(np.float32)               # (M,1)

            ref = np.concatenate(
                [
                    data["root_pos"][:, 2:3],
                    data["projected_gravity"],
                    vel_heading,
                    yaw_rate,
                    data["dof_pos"][:, col],
                    data["dof_vel"][:, col],
                    data["foot_clear"],
                ],
                axis=1,
            ).astype(np.float32)
            file_refs.append(ref)

            fps = float(data["fps"]) if "fps" in data.files else 50.0
            file_dt.append(1.0 / fps)
            file_alpha.append(float(np.exp(-(1.0 / fps) / max(ema_window_s, 1e-6))))

        # ── mirror augmentation: add a sagittally-mirrored copy of each clip ───
        # A single mocap clip can carry a net lateral/yaw drift (e.g. slow.npz turns slightly);
        # the discriminator then rewards that bias and the policy drifts. Adding the L/R-mirrored
        # clip makes the expert distribution symmetric (zero mean lateral velocity / yaw), so no
        # turn direction is preferred. Each base clip keeps its file index (src) for cmd_ranges +
        # warp_factors. See K1Env._symm_mirror for the same joint mirror map.
        perm, sgn = _build_joint_mirror(joint_names)
        base_refs: list[np.ndarray] = []
        base_src: list[int] = []
        base_dt: list[float] = []
        base_alpha: list[float] = []
        for i, ref in enumerate(file_refs):
            base_refs.append(ref); base_src.append(i)
            base_dt.append(file_dt[i]); base_alpha.append(file_alpha[i])
            if mirror_augment:
                base_refs.append(_mirror_motion(ref, n, perm, sgn)); base_src.append(i)
                base_dt.append(file_dt[i]); base_alpha.append(file_alpha[i])

        # ── expansion: synthesize time-warped copies of each (base) clip ───────
        # Each (clip, warp factor) produces one virtual clip with its own velocity band and a
        # matching EMA command label.  The command label is the conditioning signal the
        # discriminator binds the (warped) motion to; mean/std below cover the whole band.
        warped_refs: list[np.ndarray] = []
        warped_cmds: list[np.ndarray] = []
        warped_src: list[int] = []          # originating file index (for cmd_ranges clamp)
        warp_log: list[tuple[int, float, float]] = []  # (file_idx, k, mean |vx|) for logging
        for ref, src, dt_i, alpha_i in zip(base_refs, base_src, base_dt, base_alpha):
            wf = (warp_factors[src] if warp_factors is not None and src < len(warp_factors)
                  and warp_factors[src] else [1.0])
            for k in wf:
                ref_w, cmd_w = _time_warp_ref(ref, vel_cols, dt_i, float(k), alpha_i)
                warped_refs.append(ref_w)
                warped_cmds.append(cmd_w)
                warped_src.append(src)
                warp_log.append((src, float(k), float(np.abs(cmd_w[:, 0]).mean())))

        # ── combined motion stats from ALL (warped) clips ──────────────────────
        combined = np.concatenate(warped_refs, axis=0)
        mean = combined.mean(0)
        std  = combined.std(0) + 1e-6

        # ── normalise motion features + append velocity-derived command labels ──
        obs_list: list[torch.Tensor] = []
        next_obs_list: list[torch.Tensor] = []
        total_transitions = 0
        for ref_w, cmd_w, src in zip(warped_refs, warped_cmds, warped_src):
            ref_norm = (ref_w - mean) / std * sqrt_w  # (M, 41)

            # Command label per transition = the state's EMA velocity (held across the
            # transition, matching the env where commands are constant for many steps).
            cmds = cmd_w[:-1].copy()  # (M-1, 3)  label attached to state s
            if cmd_ranges is not None and src < len(cmd_ranges):
                lo, hi = cmd_ranges[src]
                cmds = np.clip(cmds, lo, hi).astype(np.float32)

            s  = np.concatenate([ref_norm[:-1], cmds], axis=1)  # (M-1, 41+3)
            sn = np.concatenate([ref_norm[1:],  cmds], axis=1)  # same cmd for next

            obs_list.append(torch.tensor(s,  dtype=torch.float32, device=device))
            next_obs_list.append(torch.tensor(sn, dtype=torch.float32, device=device))
            total_transitions += len(ref_w) - 1

        self.all_obs      = torch.cat(obs_list,      dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)

        # Expose motion-feature stats so the runner can push them into env._build_amp_obs().
        # Commands are appended raw (not normalized) in both policy and expert obs.
        self.mean   = torch.tensor(mean,   dtype=torch.float32, device=device)
        self.std    = torch.tensor(std,    dtype=torch.float32, device=device)
        self.sqrt_w = torch.tensor(sqrt_w, dtype=torch.float32, device=device)

        feat_dim = self.all_obs.shape[1]
        print(f"[K1AMPLoader] {total_transitions} expert transitions from {len(paths)} file(s), "
              f"{len(warped_refs)} clips (incl. time-warps)  feat_dim={feat_dim} "
              f"(motion={combined.shape[1]} + cmd={feat_dim - combined.shape[1]})")
        for src, k, vmean in warp_log:
            print(f"           file[{src}] warp ×{k:.2f} → mean |cmd vx|={vmean:.3f} m/s")

    def __len__(self) -> int:
        return len(self.all_obs)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Yield (state, next_state) mini-batches sampled uniformly with replacement."""
        N = len(self.all_obs)
        for _ in range(num_mini_batch):
            idx = torch.randint(N, (mini_batch_size,), device=self.device)
            yield self.all_obs[idx], self.all_next_obs[idx]
