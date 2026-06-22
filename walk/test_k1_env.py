"""Smoke tests for K1Env before training.

Usage:
  .venv/bin/python walk/test_k1_env.py --test a
  .venv/bin/python walk/test_k1_env.py --test b
  .venv/bin/python walk/test_k1_env.py --test c --reward action_rate
  .venv/bin/python walk/test_k1_env.py --test d --num-envs 64
  .venv/bin/python walk/test_k1_env.py --test all
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import genesis as gs
import torch
import yaml
from K1_env import K1Env

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "k1_env.yaml")
ALL_REWARDS = [
    "lin_vel_z",
    "action_rate",
    "tracking_lin_vel",
    "tracking_ang_vel",
    "feet_air_time",
    "feet_slip",
    "leg_symmetry",
]
REWARD_SCALE_DEFAULTS = {
    "lin_vel_z": -1.0,
    "action_rate": -0.005,
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.2,
    "feet_air_time": 0.5,
    "feet_slip": -1.0,
    "leg_symmetry": -2.0,
}


def load_cfg() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def make_env(num_envs: int = 1, reward_names: list[str] | None = None) -> K1Env:
    all_cfg = load_cfg()
    if reward_names is not None:
        all_cfg["reward_cfg"] = copy.deepcopy(all_cfg["reward_cfg"])
        all_cfg["reward_cfg"]["reward_scales"] = {
            name: REWARD_SCALE_DEFAULTS[name] for name in reward_names
        }
    return K1Env(
        num_envs=num_envs,
        env_cfg=all_cfg["env_cfg"],
        obs_cfg=all_cfg["obs_cfg"],
        reward_cfg=all_cfg["reward_cfg"],
        command_cfg=all_cfg["command_cfg"],
        show_viewer=False,
    )


def zero_actions(env: K1Env) -> torch.Tensor:
    return torch.zeros(env.num_envs, env.num_actions, device=gs.device)


def small_random_actions(env: K1Env, scale: float = 0.1) -> torch.Tensor:
    return scale * torch.randn(env.num_envs, env.num_actions, device=gs.device)


def has_nan(env: K1Env) -> bool:
    tensors = [
        env.base_pos,
        env.base_quat,
        env.base_lin_vel,
        env.base_ang_vel,
        env.dof_pos,
        env.dof_vel,
        env.obs_buf,
        env.rew_buf,
    ]
    return any(torch.isnan(t).any().item() for t in tensors)


def print_reward_breakdown(env: K1Env, step_i: int, rew_total: float) -> None:
    lines = [f"  step {step_i:4d}  rew_total={rew_total:.6f}"]
    for name, func in env.reward_functions.items():
        raw = float(func()[0])
        scaled = raw * env.reward_scales[name]
        lines.append(f"    {name:22s} raw={raw:10.6f}  contrib={scaled:10.6f}")
    z = float(env.base_pos[0, 2])
    vz = float(env.base_lin_vel[0, 2])
    vx, vy = float(env.base_lin_vel[0, 0]), float(env.base_lin_vel[0, 1])
    roll, pitch = float(env.base_euler[0, 0]), float(env.base_euler[0, 1])
    lines.append(
        f"    state: z={z:.4f}  vz={vz:.4f}  vx={vx:.4f}  vy={vy:.4f}  "
        f"roll={roll:.1f}° pitch={pitch:.1f}°"
    )
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Test A — Reset + Stand (zero actions)
# ---------------------------------------------------------------------------
def test_a(steps: int = 500) -> bool:
    print("\n=== Test A: Reset + Stand (zero actions) ===")
    print("Was passiert: Roboter wird zurückgesetzt, dann 500× Null-Aktionen.")
    print("PD hält Default-Pose. Wir messen Höhe z und ob NaNs auftreten.\n")

    env = make_env(num_envs=1)
    env.reset()
    z_history: list[float] = []
    done_at: int | None = None

    for step_i in range(steps):
        _, rew, done, _ = env.step(zero_actions(env))
        z = float(env.base_pos[0, 2])
        z_history.append(z)

        if has_nan(env):
            print(f"FAIL: NaN detected at step {step_i}")
            return False

        if step_i % 50 == 0:
            print(f"  step {step_i:3d}  z={z:.4f}  rew={float(rew[0]):.6f}  done={bool(done[0])}")

        if done[0] and done_at is None:
            done_at = step_i

    z_min, z_max = min(z_history), max(z_history)
    z_final = z_history[-1]
    target = env.env_cfg["base_init_pos"][2]

    print("\nErgebnis:")
    print(f"  z: min={z_min:.4f}  max={z_max:.4f}  final={z_final:.4f}  init={target}")
    print(f"  Episode-Ende (Umfallen/Timeout): step {done_at if done_at is not None else 'keins'}")
    print("  NaNs: keine")

    # Pass: no NaN. Standing 500 steps is a bonus, not required for humanoid without training.
    ok = not has_nan(env)
    if done_at is None:
        print("  PASS: Kein Abbruch, z blieb im Bereich (gut für PD).")
    elif done_at > 100:
        print(f"  OK:  Stabil >{done_at} Steps — für Null-Aktionen akzeptabel.")
    else:
        print(f"  HINWEIS: Umgekippt bei Step {done_at} — für Humanoid ohne Training normal.")
        print("           Wichtig: keine NaNs, Physik läuft. Rewards/Training kommen später.")

    return ok


# ---------------------------------------------------------------------------
# Test B — Zero vs small random actions
# ---------------------------------------------------------------------------
def test_b(steps_each: int = 300) -> bool:
    print("\n=== Test B: Null-Aktionen vs. kleine Random-Aktionen ===")
    print("Was passiert:")
    print("  Phase 1: actions=0     → PD hält exakt Default-Pose")
    print("  Phase 2: actions~N(0,0.1) → max ±0.025 rad Gelenk-Offset (0.1×0.25)\n")

    env = make_env(num_envs=1)

    print("--- Phase 1: ZERO ---")
    z_zero: list[float] = []
    for step_i in range(steps_each):
        _, _, done, _ = env.step(zero_actions(env))
        z_zero.append(float(env.base_pos[0, 2]))
        if has_nan(env):
            print(f"FAIL: NaN in zero phase at step {step_i}")
            return False
        if done[0]:
            print(f"  Episode reset at step {step_i} (zero)")
            env.reset()

    print(f"  z: min={min(z_zero):.4f}  max={max(z_zero):.4f}  mean={sum(z_zero)/len(z_zero):.4f}")

    print("--- Phase 2: SMALL RANDOM (scale=0.1) ---")
    z_rand: list[float] = []
    for step_i in range(steps_each):
        _, _, done, _ = env.step(small_random_actions(env, scale=0.1))
        z_rand.append(float(env.base_pos[0, 2]))
        if has_nan(env):
            print(f"FAIL: NaN in random phase at step {step_i}")
            return False
        if step_i % 50 == 0:
            print(f"  step {step_i:3d}  z={z_rand[-1]:.4f}")
        if done[0]:
            print(f"  Episode reset at step {step_i} (random)")

    print(f"  z: min={min(z_rand):.4f}  max={max(z_rand):.4f}  mean={sum(z_rand)/len(z_rand):.4f}")

    print("\nErwartung:")
    print("  Zero:   z nahe 0.53–0.56, pose_dev klein")
    print("  Random: z kann stärker schwanken — prüfen ob PD nicht sofort explodiert")
    print("  PASS wenn: keine NaNs in beiden Phasen")
    return True


# ---------------------------------------------------------------------------
# Test C — Each reward in isolation
# ---------------------------------------------------------------------------
def test_c(reward_name: str | None = None, steps: int = 200) -> bool:
    names = [reward_name] if reward_name else ALL_REWARDS
    print("\n=== Test C: Rewards einzeln ===")
    print("Was passiert: Nur EIN Reward aktiv, wir prüfen ob contrib sinnvoll reagiert.\n")

    action_plan = {
        "lin_vel_z": ("zero", "vz≈0 → contrib≈0; beim Fallen vz groß → contrib negativ"),
        "action_rate": ("random", "jeder Step neue Random-Aktion → contrib < 0; bei zero = 0"),
        "tracking_lin_vel": ("zero", "cmd=[0.5,0], Heading-vx≈0 → raw≈exp(-0.25)≈0.37; Heading-vx=0.5 → raw≈1.0"),
        "tracking_ang_vel": ("zero", "cmd=0, yaw_rate≈0 → raw≈1.0"),
        "feet_air_time": ("random", "Bonus = (air_time − target) beim Aufsetzen; >0 bei langen Schritten, 0 bei cmd≈0"),
        "feet_slip": ("random", "raw>0 wenn Stützfuß im Kontakt horizontal rutscht; 0 bei stillem Stützfuß"),
        "leg_symmetry": ("random", "raw>0 bei both_air/zu langer Doppelstütze; 0 bei sauberer Alternation/cmd≈0"),
    }

    all_ok = True
    for name in names:
        if name not in REWARD_SCALE_DEFAULTS:
            print(f"Unknown reward: {name}")
            all_ok = False
            continue

        print(f"--- Reward: {name} (scale={REWARD_SCALE_DEFAULTS[name]}) ---")
        action_type, hint = action_plan[name]
        print(f"  Erwartung: {hint}")

        env = make_env(num_envs=1, reward_names=[name])
        action_fn = zero_actions if action_type == "zero" else lambda e: small_random_actions(e, 0.3)

        for step_i in range(steps):
            _, rew, done, _ = env.step(action_fn(env))
            if has_nan(env):
                print(f"  FAIL: NaN at step {step_i}")
                all_ok = False
                break
            if step_i in (0, 50, 100) or (done[0] and step_i < 50):
                print_reward_breakdown(env, step_i, float(rew[0]))
            if done[0] and step_i > 0:
                env.reset()
        print()

    return all_ok


# ---------------------------------------------------------------------------
# Test D — Parallel envs (GPU memory / stability)
# ---------------------------------------------------------------------------
def test_d(num_envs: int = 64, steps: int = 50) -> bool:
    print(f"\n=== Test D: {num_envs} parallele Envs ===")
    print("Was passiert: Viele Roboter gleichzeitig auf der GPU.")
    print("Prüfen: kein Crash, kein OOM, keine NaNs.\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e6
    else:
        mem_before = 0.0

    t0 = time.perf_counter()
    try:
        env = make_env(num_envs=num_envs)
        env.reset()
        for step_i in range(steps):
            _, rew, done, _ = env.step(zero_actions(env))
            if has_nan(env):
                print(f"FAIL: NaN at step {step_i}")
                return False
        elapsed = time.perf_counter() - t0
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or "OOM" in str(exc):
            print(f"FAIL: GPU OOM mit num_envs={num_envs}")
            print("  → reduziere: --num-envs 32 oder 16")
            return False
        raise

    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1e6
        mem_peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"  GPU memory: before={mem_before:.0f} MB  after={mem_after:.0f} MB  peak={mem_peak:.0f} MB")
    print(f"  {num_envs} envs × {steps} steps in {elapsed:.2f}s  ({num_envs * steps / elapsed:.0f} steps/s)")
    print(f"  obs shape: {env.obs_buf.shape}  rew shape: {env.rew_buf.shape}")
    print(f"  done count last step: {int(done.sum())}")
    print("  PASS: Kein OOM, keine NaNs.")
    return True


# ---------------------------------------------------------------------------
# Test E — feet_air_time landing reward (deterministisch)
# ---------------------------------------------------------------------------
def test_e() -> bool:
    print("\n=== Test E: feet_air_time (phasen-gekoppelt: On-Beat + Kadenz) ===")
    print("Idee: Touchdown nur wenn _desired_stance() und Luftzeit nahe gait_swing_time.\n")

    env = make_env(num_envs=1, reward_names=["feet_air_time"])
    swing = env.gait_swing_time
    fwd = torch.tensor([0.6, 0.0, 0.0], device=gs.device)

    def land_with(cmd, air_time, phase=0.05, step_quality=1.0):
        env.commands[:] = cmd
        env.gait_phase[:] = phase
        env.foot_in_contact[:] = False
        env.foot_air_time[:] = air_time
        env.foot_step_quality[:] = step_quality
        env._update_foot_contact()
        return float(env._reward_feet_air_time()[0]), int(env.foot_in_contact[0].sum())

    # On-beat bei φ=0.05 (Doppelstütz-Fenster), swing_time≈0.32 s
    rew_good, n = land_with(fwd, swing - env.dt, phase=0.05, step_quality=1.0)
    rew_short, _ = land_with(fwd, 0.05, phase=0.05, step_quality=1.0)
    rew_offbeat, _ = land_with(fwd, swing - env.dt, phase=0.75, step_quality=1.0)
    rew0, _ = land_with(torch.zeros(3, device=gs.device), swing - env.dt, phase=0.05, step_quality=1.0)

    print(f"  on-beat, air≈swing ({swing:.2f}s): raw={rew_good:.4f} (>0)")
    print(f"  on-beat, kurze Luft:              raw={rew_short:.4f} (< good)")
    print(f"  off-beat:                          raw={rew_offbeat:.4f} (=0)")
    print(f"  cmd=0:                             raw={rew0:.4f} (=0)")

    ok = rew_good > rew_short > 0 and rew_offbeat == 0.0 and rew0 == 0.0
    print(f"  {'PASS' if ok else 'FAIL'}: On-Beat-Gating + Gauß-Luftzeit.")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="K1Env smoke tests")
    parser.add_argument("--test", choices=["a", "b", "c", "d", "e", "all"], default="a")
    parser.add_argument("--reward", choices=ALL_REWARDS, help="for test c: single reward")
    parser.add_argument("--num-envs", type=int, default=64, help="for test d")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level="warning")

    results: dict[str, bool] = {}
    if args.test in ("a", "all"):
        results["A"] = test_a(steps=args.steps or 500)
    if args.test in ("b", "all"):
        results["B"] = test_b(steps_each=args.steps or 300)
    if args.test in ("c", "all"):
        results["C"] = test_c(reward_name=args.reward, steps=args.steps or 200)
    if args.test in ("d", "all"):
        results["D"] = test_d(num_envs=args.num_envs, steps=args.steps or 50)
    if args.test in ("e", "all"):
        results["E"] = test_e()

    print("\n=== Zusammenfassung ===")
    for name, ok in results.items():
        print(f"  Test {name}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
