"""Reward-Logging pro Step (Hero-Test + Trainings-Video-Rollout)."""

from __future__ import annotations


def reward_names(env) -> list[str]:
    return sorted(n[len("_reward_"):] for n in dir(env) if n.startswith("_reward_"))


def build_step_reward_row(env, names: list[str] | None = None) -> dict:
    """Rohe Rewards ∈[0,1], skalierte Step-Beiträge (raw×scale×dt) und cmd/actual — wie Hero."""
    if names is None:
        names = reward_names(env)
    row: dict[str, float] = {}
    total = 0.0
    for n in names:
        raw = float(getattr(env, "_reward_" + n)().mean().item())
        row[f"reward_raw/{n}"] = raw
        scale = env.reward_scales.get(n)
        if scale is not None:
            contrib = raw * scale
            row[f"reward_step/{n}"] = contrib
            total += contrib
    row["reward_step/total"] = total
    row["cmd/vx"] = float(env.commands[0, 0])
    row["cmd/vy"] = float(env.commands[0, 1])
    row["cmd/yaw"] = float(env.commands[0, 2])
    row["cmd/speed"] = float((env.commands[0, 0] ** 2 + env.commands[0, 1] ** 2) ** 0.5)
    row["actual/vx"] = float(env.base_lin_vel_heading[0, 0])
    row["actual/vy"] = float(env.base_lin_vel_heading[0, 1])
    row["actual/speed"] = float((env.base_lin_vel_heading[0, 0] ** 2 + env.base_lin_vel_heading[0, 1] ** 2) ** 0.5)
    return row


def prefix_row(row: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{k}": v for k, v in row.items()}


def episode_summary(
    raw_sums: dict[str, float],
    episode_sums: dict[str, float],
    n_steps: int,
    duration_s: float,
    prefix: str = "video_rollout/episode",
) -> dict[str, float]:
    summary = {f"{prefix}/return_total": sum(episode_sums.values())}
    for n, s in episode_sums.items():
        summary[f"{prefix}/rew_{n}"] = s / duration_s
    for n, s in raw_sums.items():
        summary[f"{prefix}/raw_mean_{n}"] = s / n_steps
    return summary
