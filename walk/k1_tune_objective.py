"""Eval rollout + hero-style composite score for Optuna reward-scale tuning."""

from __future__ import annotations

import torch

from k1_reward_log import build_step_reward_row, reward_names

DEFAULT_OBJECTIVE_WEIGHTS: dict[str, float] = {
    # Task
    "tracking_lin_vel": 3.0,
    "tracking_ang_vel": 1.5,
    "style": 2.0,
    # Gait quality (penalty terms: higher raw = worse → negative weight)
    "feet_air_time": 0.5,
    "foot_clearance": 1.0,
    "swing_clearance": -1.0,
    "foot_air_excess": -0.5,
    # Penalties
    "feet_slip": -0.5,
    "orientation": -0.3,
    "base_height": -0.3,
    "lin_vel_z": -0.2,
}


def eval_raw_means(env, policy, n_steps: int = 1000) -> dict[str, float]:
    """Rollout policy and return mean raw reward per term (scale-independent, ∈[0,1])."""
    names = reward_names(env)
    raw_sums = {n: 0.0 for n in names}
    obs = env.reset()
    with torch.inference_mode():
        for _ in range(n_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            row = build_step_reward_row(env, names)
            for n in names:
                raw_sums[n] += row[f"reward_raw/{n}"]
    return {n: s / n_steps for n, s in raw_sums.items()}


def hero_composite_score(
    raw_means: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted sum of raw_mean terms. Penalties use negative weights."""
    w = DEFAULT_OBJECTIVE_WEIGHTS if weights is None else weights
    score = 0.0
    for name, weight in w.items():
        if name in raw_means:
            score += weight * raw_means[name]
    return score


def intermediate_training_metric(runner) -> float:
    """Pruning proxy: mean episode reward if available, else episode length."""
    logger = runner.logger
    for attr in ("rewbuffer", "rew_buffer", "reward_buffer", "episode_reward_buffer"):
        buf = getattr(logger, attr, None)
        if buf is not None and len(buf) > 0:
            return float(sum(buf) / len(buf))
    for attr in ("lenbuffer", "episode_length_buffer", "episode_lengths"):
        buf = getattr(logger, attr, None)
        if buf is not None and len(buf) > 0:
            return float(sum(buf) / len(buf))
    return float(runner.env.episode_length_buf.float().mean().item())
