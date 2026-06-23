"""Interactive debug viewer: drive the distilled STUDENT in Genesis with the keyboard.

Spawns ONE K1 in the Genesis viewer, loads a distillation student checkpoint
(walk/k1_distill.py), and runs the student's onboard-only inference (the "proprio" obs
group — exactly what deploy.py feeds the real robot). You steer it live from the VIEWER
window with arrow keys, the current command is shown in the top-right corner, and it
automatically respawns only once it has fallen to the ground.

Usage (from repo root):
  .venv/bin/python walk/k1_distill_debug.py                       # latest model in logs/k1-distill
  .venv/bin/python walk/k1_distill_debug.py -e k1-distill --ckpt 2000
  .venv/bin/python walk/k1_distill_debug.py --accel 1.2           # ramp commands faster

Keys (the GENESIS WINDOW must be focused — these avoid the default viewer shortcuts):
  Up / Down     ramp forward velocity vx up / down  (clamped to command_cfg.lin_vel_x_range)
  Left / Right  ramp yaw rate wz left / right        (lin_vel ... ang_vel_range)
  , / .         ramp strafe velocity vy left / right (lin_vel_y_range)
  space         zero all commands (and stop ramping)
  x             reset the robot now
  q             quit (Ctrl+C in the terminal also works)

Throttle control: HOLD a key to ramp that axis at --accel units/s; it holds its value on
release, so you can dial in any velocity across the full trained range (e.g. vx up to 1.3).
The env smooths self.commands toward the target over env_cfg.command_smooth_s (~1.5 s).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import genesis as gs
import numpy as np
import torch

WALK_DIR = os.path.dirname(os.path.abspath(__file__))


class Drive:
    """Throttle command state mutated by viewer keybind callbacks, integrated by the sim loop.

    Each held key nudges its axis toward the range bound at ``accel`` units/s; the value persists
    when released. Per-key booleans (not a single direction) keep simultaneous presses correct.
    """

    def __init__(self, ranges, accel: float, vx=0.0, vy=0.0, wz=0.0):
        (self.x_lo, self.x_hi), (self.y_lo, self.y_hi), (self.w_lo, self.w_hi) = ranges
        self.accel = accel
        self.vx, self.vy, self.wz = vx, vy, wz
        self.up = self.down = self.left = self.right = self.strafe_l = self.strafe_r = False
        self.do_reset = False
        self.quit = False

    def set_held(self, attr: str, value: bool) -> None:
        setattr(self, attr, value)

    def stop(self) -> None:
        self.vx = self.vy = self.wz = 0.0
        self.up = self.down = self.left = self.right = self.strafe_l = self.strafe_r = False

    def request_reset(self) -> None:
        self.do_reset = True

    def request_quit(self) -> None:
        self.quit = True

    def integrate(self, dt: float) -> None:
        d = self.accel * dt
        self.vx = min(self.x_hi, max(self.x_lo, self.vx + (self.up - self.down) * d))
        self.vy = min(self.y_hi, max(self.y_lo, self.vy + (self.strafe_l - self.strafe_r) * d))
        self.wz = min(self.w_hi, max(self.w_lo, self.wz + (self.left - self.right) * d))


def _make_keybinds(d: Drive):
    """Arrow keys + ,/./space/x/q — none of which the default-controls plugin uses."""
    from genesis.vis.keybindings import Key, KeyAction, Keybind

    P, R = KeyAction.PRESS, KeyAction.RELEASE
    held = {  # name: (key, drive attribute toggled while held)
        "fwd": (Key.UP, "up"),
        "back": (Key.DOWN, "down"),
        "left": (Key.LEFT, "left"),
        "right": (Key.RIGHT, "right"),
        "strafe_l": (Key.COMMA, "strafe_l"),
        "strafe_r": (Key.PERIOD, "strafe_r"),
    }
    binds = []
    for name, (key, attr) in held.items():
        binds.append(Keybind(f"dbg_{name}", key, P, callback=d.set_held, args=(attr, True)))
        binds.append(Keybind(f"dbg_{name}_rel", key, R, callback=d.set_held, args=(attr, False)))
    binds += [
        Keybind("dbg_stop", Key.SPACE, P, callback=d.stop),
        Keybind("dbg_reset", Key.X, P, callback=d.request_reset),
        Keybind("dbg_quit", Key.Q, P, callback=d.request_quit),
    ]
    return binds


def _latest_ckpt(log_dir: str) -> int:
    cands = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not cands:
        raise FileNotFoundError(f"No model_*.pt in {log_dir}")
    return max(int(os.path.splitext(os.path.basename(p))[0].split("_")[1]) for p in cands)


def main() -> None:
    if WALK_DIR not in sys.path:
        sys.path.insert(0, WALK_DIR)
    import pickle

    from genesis.ext.pyrender.constants import TextAlign
    from K1_env import K1Env
    from rsl_rl.runners import DistillationRunner

    parser = argparse.ArgumentParser(description="Interactive K1 student debug viewer")
    parser.add_argument("-e", "--exp_name", type=str, default="k1-distill")
    parser.add_argument("--ckpt", type=int, default=None, help="model_<ckpt>.pt (default: latest)")
    parser.add_argument("--vx", type=float, default=0.0, help="initial forward command")
    parser.add_argument("--vy", type=float, default=0.0, help="initial strafe command")
    parser.add_argument("--wz", type=float, default=0.0, help="initial yaw command")
    parser.add_argument("--accel", type=float, default=0.8, help="command ramp rate while a key is held [units/s]")
    parser.add_argument("--fall_height", type=float, default=0.25,
                        help="base height [m] below which it counts as fallen (standing ~0.56)")
    parser.add_argument("--no_realtime", action="store_true", help="run as fast as possible")
    args = parser.parse_args()

    log_dir = os.path.join("logs", args.exp_name)
    ckpt = args.ckpt if args.ckpt is not None else _latest_ckpt(log_dir)
    ckpt_path = os.path.join(log_dir, f"model_{ckpt}.pt")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        cfgs = pickle.load(f)
    env_cfg, obs_cfg, reward_cfg, command_cfg, distill_cfg, _video_opts = cfgs

    # No rewards needed for inference. Disable the timer-based command resample and the episode
    # timeout so it only resets on an actual fall. Turn OFF domain randomization, random pushes
    # and RSI so a clean teleop session isn't shoved over or respawned in random poses.
    reward_cfg["reward_scales"] = {}
    env_cfg["resampling_time_s"] = 1.0e9
    env_cfg["episode_length_s"] = 1.0e9
    env_cfg["randomization"] = {}
    env_cfg["rsi_prob"] = 0.0

    env = K1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = DistillationRunner(env, distill_cfg, log_dir, device=gs.device)
    runner.load(ckpt_path, load_cfg={"student": True, "iteration": False})
    policy = runner.get_inference_policy(device=gs.device)
    print(f"Loaded student {ckpt_path}  (proprio obs dim = {env.proprio_dim})")
    print("Drive from the Genesis window: arrows ramp move/turn, ,/. strafe, space stop, x reset, q quit")

    # Command ranges the student was trained on — clamp the throttle to these.
    ranges = (
        tuple(command_cfg["lin_vel_x_range"]),
        tuple(command_cfg["lin_vel_y_range"]),
        tuple(command_cfg["ang_vel_range"]),
    )
    print(f"Command ranges: vx{ranges[0]}  vy{ranges[1]}  wz{ranges[2]}  |  ramp {args.accel}/s")

    # Wire up keyboard control + the top-right command overlay via Genesis' native viewer APIs.
    drive = Drive(ranges, args.accel, args.vx, args.vy, args.wz)

    # The env's _reset_idx() calls _resample_commands(), which normally writes a RANDOM command on
    # every reset (incl. the first) — that is what made it spin and re-randomize on each fall.
    # Override it to inject the live keyboard command instead, so a respawn keeps our command.
    def _hold_command(envs_idx=None) -> None:
        env.command_target[:, 0] = drive.vx
        env.command_target[:, 1] = drive.vy
        env.command_target[:, 2] = drive.wz

    env._resample_commands = _hold_command

    # The env's default termination flags a "fall" as soon as the base dips below 0.42 m (standing
    # is ~0.56 m) or tilts >30deg — far too eager for teleop, so a slightly crouched but upright
    # student respawns constantly. Only reset once it has actually gone down to the ground.
    def _fallen_mask() -> torch.Tensor:
        env._term_timeout = env.episode_length_buf > env.max_episode_length
        env._term_pitch = torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)
        env._term_roll = torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)
        env._term_height = env.base_pos[:, 2] < args.fall_height
        env._term_sim_error = env.scene.rigid_solver.get_error_envs_mask()
        return env._term_height | env._term_sim_error

    env._termination_mask = _fallen_mask

    env.scene.viewer.register_keybinds(*_make_keybinds(drive), overwrite=True)
    pyrender_viewer = env.scene.viewer._pyrender_viewer

    def refresh_overlay() -> None:
        pyrender_viewer.viewer_flags["caption"] = [{
            "text": f"vx {drive.vx:+.2f}  vy {drive.vy:+.2f}  wz {drive.wz:+.2f}",
            "location": TextAlign.TOP_RIGHT,
            "font_name": "UbuntuMono-Regular",
            "font_pt": 22,
            "color": np.array([0.25, 0.95, 0.45, 1.0]),
            "scale": 1.0,
        }]

    cmd = torch.zeros((1, 3), dtype=gs.tc_float, device=gs.device)
    obs = env.reset()
    try:
        with torch.no_grad():
            while not drive.quit:
                t0 = time.perf_counter()
                if drive.do_reset:
                    env.reset()
                    drive.do_reset = False
                    print("[reset]")
                drive.integrate(env.dt)  # ramp held axes (sim-time, framerate-independent)
                cmd[0, 0], cmd[0, 1], cmd[0, 2] = drive.vx, drive.vy, drive.wz
                env.command_target.copy_(cmd)  # env.commands smooths toward this inside step()
                refresh_overlay()
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if bool(dones[0]):
                    print("[fell -> respawned]")
                if not args.no_realtime:
                    dt_left = env.dt - (time.perf_counter() - t0)
                    if dt_left > 0:
                        time.sleep(dt_left)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[exit]")


if __name__ == "__main__":
    main()
