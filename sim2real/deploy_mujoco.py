# load model inference

#create mujoco scene
# load robot urdf in mujoco
# get robot data observations
# transform everything into model observation
# send to model
# get actions
# send to robot
# kill switches
# add joint limit termination
import argparse
import math
import os
import time

import mujoco as mj
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import yaml

from mujoco_calibration import DEFAULT_DEPLOY, apply_mujoco_calibration, enable_self_collision

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(DEFAULT_DEPLOY, "r") as f:
    config = yaml.safe_load(f)

class _RNN(nn.Module):
    """Wrapper whose attribute name (`rnn`) makes the LSTM weights load under the checkpoint's
    `rnn.rnn.*` keys (rsl-rl RNNModel nests an RNN module that itself holds the nn.LSTM)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers)


class ActorLSTM(nn.Module):
    """Onboard recurrent actor matching the trained rsl-rl RNNModel actor.

    Flow per step: obs(obs_dim) -> LSTM(obs_dim -> rnn_hidden_dim) carrying hidden state across
    steps -> MLP head -> action(num_actions). The LSTM hidden state IS the policy's memory (it
    replaces the training-time proprio frame-stack), so we feed a single proprio frame each step
    and keep (h, c) between steps. Call reset() once before handing control to the policy.
    """

    def __init__(self, obs_dim: int, rnn_hidden_dim: int, head_dims: tuple[int, int],
                 num_actions: int, num_layers: int = 1) -> None:
        super().__init__()
        self.rnn = _RNN(obs_dim, rnn_hidden_dim, num_layers)
        h0, h1 = head_dims
        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim, h0), nn.ELU(),
            nn.Linear(h0, h1),             nn.ELU(),
            nn.Linear(h1, num_actions),
        )
        self._hidden = None  # (h, c), each (num_layers, 1, rnn_hidden_dim)

    def reset(self) -> None:
        """Clear the LSTM memory (call at bring-up before the policy takes over)."""
        self._hidden = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (obs_dim,) for a single robot -> (seq=1, batch=1, obs_dim)
        x = obs.reshape(1, 1, -1)
        out, self._hidden = self.rnn.rnn(x, self._hidden)
        return self.mlp(out.reshape(-1))


def load_model(checkpoint_path: str) -> ActorLSTM:
    """Load the trained recurrent actor for inference.

    All layer sizes are inferred from the checkpoint weights, so the deploy network always matches
    the trained one regardless of config. Loads `actor_state_dict` (direct RL training); falls back
    to `student_state_dict` / `actor` for older checkpoints.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("actor_state_dict") or ckpt.get("student_state_dict") or ckpt.get("actor")
    if sd is None:
        raise KeyError(f"No actor weights in {checkpoint_path} (keys: {list(ckpt.keys())})")
    if "rnn.rnn.weight_ih_l0" not in sd:
        raise KeyError(
            "Checkpoint has no LSTM weights ('rnn.rnn.weight_ih_l0'); this deploy script expects the "
            "recurrent RNNModel actor. For a feedforward MLP actor, use the older ActorMLP path."
        )
    # Infer dims from the weights: LSTM weight_ih_l0 is (4*hidden, obs_dim); head from mlp.*.weight.
    w_ih = sd["rnn.rnn.weight_ih_l0"]
    obs_dim = w_ih.shape[1]
    rnn_hidden_dim = w_ih.shape[0] // 4
    num_layers = sum(1 for k in sd if k.startswith("rnn.rnn.weight_ih_l"))
    head_dims = (sd["mlp.0.weight"].shape[0], sd["mlp.2.weight"].shape[0])
    num_actions = sd["mlp.4.weight"].shape[0]

    actor = ActorLSTM(obs_dim, rnn_hidden_dim, head_dims, num_actions, num_layers)
    # strict=False: the checkpoint also holds distribution.std_param, unused for deterministic inference.
    missing, unexpected = actor.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("distribution.")]
    if missing or unexpected:
        print(f"[load_model] missing={missing}  unexpected={unexpected}")
    actor.eval()
    print(f"[load_model] recurrent actor: obs_dim={obs_dim} rnn_hidden={rnn_hidden_dim} "
          f"layers={num_layers} head={head_dims} actions={num_actions}")
    return actor


# --- quaternion helpers (wxyz convention, identical math to genesis.utils.geom) --------------------
# MuJoCo and Genesis both store quaternions as [w, x, y, z], so the obs transforms match exactly.
def inv_quat(q: torch.Tensor) -> torch.Tensor:
    """Inverse (conjugate, for a unit quat) of a wxyz quaternion."""
    w, x, y, z = q
    return torch.tensor([w, -x, -y, -z], dtype=q.dtype)


def transform_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by wxyz quaternion q: v' = q * (0,v) * q^-1."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + cross(q_xyz, t)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.tensor([rx, ry, rz], dtype=v.dtype)


policy = load_model(config["policy"]["model"])
print(policy)

model = mj.MjModel.from_xml_path(os.path.abspath(config["policy"]["mujoco_path"]))
data = mj.MjData(model)
print(f"model: {model.njnt} joints, {model.nu} actuators, "
      f"nq={model.nq} (qpos size), nv={model.nv} (qvel size)")

# --- sim2sim parity tuning (Genesis-trained policy → MuJoCo) ---------------------------------------
# Training kp/kv in the MJCF do NOT match Genesis step response. calibrate_pd.py fits per-joint
# MuJoCo gains; apply_mujoco_calibration() loads deploy_mujoco_calib.yaml (path in deploy.yaml).
CONTACT_SOLREF = 0.02   # match MJCF default + Genesis contact_solref_range
_calib_path = config["policy"].get("mujoco_calib", "deploy_mujoco_calib.yaml")
if not os.path.isabs(_calib_path):
    _calib_path = os.path.join(_SCRIPT_DIR, _calib_path)
apply_mujoco_calibration(model, _calib_path)
enable_self_collision(model, solref=(CONTACT_SOLREF, 1.0))

_trunk_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "Trunk")
_vel_buf = np.zeros(6, dtype=np.float64)

p = config["policy"]
joint_names = p["joint_names"]          # 16 policy joints, action order
fixed_names = p["fixed_joint_names"]    # PD-held joints (head, shoulders, elbows)
all_names = joint_names + fixed_names

# Per-joint MuJoCo addressing. Unlike Genesis (control_dofs_position re-sorts targets), MuJoCo
# <position> actuators take a joint target in data.ctrl — one actuator per joint, no argsort needed.
def joint_info(name):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
    return {
        "qadr": int(model.jnt_qposadr[jid]),
        "vadr": int(model.jnt_dofadr[jid]),
        "aid": int(aid),
        "default": float(p["default_joint_angles"][name]),
    }

motor_info = [joint_info(n) for n in joint_names]   # policy-controlled
fixed_info = [joint_info(n) for n in fixed_names]    # PD-held at default

default_motor = torch.tensor([m["default"] for m in motor_info], dtype=torch.float32)
fixed_target = torch.tensor([f["default"] for f in fixed_info], dtype=torch.float32)

action_scale = p["action_scale"]
clip_actions = p["clip_actions"]
obs_scales = p["obs_scales"]
commands_scale = torch.tensor(
    [p["commands_scales"]["lin_vel_x"],
     p["commands_scales"]["lin_vel_y"],
     p["commands_scales"]["ang_vel_yaw"]],
    dtype=torch.float32,
)

# --- control / physics timing (match joint_test.py) -----------------------------------------------
# Same MuJoCo plant as joint_test: MJCF timestep (0.001 s), default Euler integrator, and
# decimation = control_dt / timestep (~20 mj_step per 50 Hz tick). Targets are written once per
# tick; ctrl is held across all substeps. (joint_test additionally pins the floating base.)
control_dt = p["dt"]
decimation = max(1, round(control_dt / model.opt.timestep))
g_world = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
viewer = None  # passive viewer handle; set in main, used for sync during bring-up


def verify_actuator_mapping() -> None:
    """Confirm each policy joint name maps to the actuator with the same name (no Genesis argsort)."""
    mismatches = []
    for i, name in enumerate(joint_names):
        info = motor_info[i]
        act_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, info["aid"])
        jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT,
                                 mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name))
        if act_name != name or jnt_name != name:
            mismatches.append((i, name, act_name, jnt_name))
    if mismatches:
        raise RuntimeError(f"actuator/joint mapping mismatches: {mismatches}")
    print(f"[mapping] verified {len(joint_names)} policy joints: 1:1 name→actuator (no argsort)")


verify_actuator_mapping()
_integrator = ("Euler", "RK4", "implicit", "implicitfast")[int(model.opt.integrator)]
print(f"control: {1/control_dt:.0f} Hz  physics: timestep={model.opt.timestep*1000:.3f}ms "
      f"×{decimation} substeps/control  integrator={_integrator} (joint_test parity)\n")


def apply_position_targets(motor_targets, fixed_targets):
    """Write joint position targets to <position> actuators (kp/kv live in MJCF)."""
    for info, tgt in zip(motor_info, motor_targets):
        data.ctrl[info["aid"]] = float(tgt)
    for info, tgt in zip(fixed_info, fixed_targets):
        data.ctrl[info["aid"]] = float(tgt)


def control_step(motor_targets, fixed_targets):
    """Advance one 50 Hz control tick: set targets once, then physics for control_dt."""
    apply_position_targets(motor_targets, fixed_targets)
    for _ in range(decimation):
        mj.mj_step(model, data)


def physics_substeps(motor_targets, fixed_targets, n=None):
    """Run n control ticks (default 1). Each tick = one policy/PD update at 50 Hz."""
    for _ in range(1 if n is None else n):
        control_step(motor_targets, fixed_targets)


def read_ang_body_vel() -> torch.Tensor:
    """Body-frame angular velocity [rad/s], matching K1_env / deploy_genesis.

    mj_objectVelocity fills [angular(0:3), linear(3:6)] in world frame — so the ANGULAR part is
    [0:3]. (Reading [3:6] here fed the base LINEAR velocity in as fake angular velocity, which let the
    robot stand at rest but diverge and fall as soon as it started moving.)"""
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, _trunk_bid, _vel_buf, 0)
    world_ang = torch.tensor(_vel_buf[0:3].copy(), dtype=torch.float32)
    base_quat = torch.tensor(data.qpos[3:7].copy(), dtype=torch.float32)
    return transform_by_quat(world_ang, inv_quat(base_quat))


def build_obs(step: int, last_actions: torch.Tensor) -> torch.Tensor:
    """Single proprio frame (59-dim), layout = K1_env._update_observation proprio group."""
    base_quat = torch.tensor(data.qpos[3:7].copy(), dtype=torch.float32)
    obs_ang_vel = read_ang_body_vel() * obs_scales["ang_vel"]
    proj_grav = transform_by_quat(g_world, inv_quat(base_quat))

    q = torch.tensor([data.qpos[m["qadr"]] for m in motor_info], dtype=torch.float32)
    dq = torch.tensor([data.qvel[m["vadr"]] for m in motor_info], dtype=torch.float32)
    obs_dof_pos = (q - default_motor) * obs_scales["dof_pos"]
    dof_vel = dq * obs_scales["dof_vel"]

    commands = (
        torch.tensor([cmd["vx"], cmd["vy"], cmd["wz"]], dtype=torch.float32) * commands_scale
    )
    phase = (step % p["gait_period_steps"]) / p["gait_period_steps"]
    clock = torch.tensor(
        [math.sin(phase * 2.0 * math.pi), math.cos(phase * 2.0 * math.pi)], dtype=torch.float32
    )
    return torch.cat(
        [obs_ang_vel, proj_grav, commands, obs_dof_pos, dof_vel, last_actions, clock], dim=0
    )


# --- spawn & bring-up -----------------------------------------------------------------------------
# Bring-up mirrors deploy_genesis.prepare_to_default: spawn standing at the URDF rest pose (all joints
# zero, legs straight, feet on the ground), then smoothly PD-ramp every joint to the training default
# pose so the robot squats into its bent-knee stance — crucially ramping the arms to their non-zero
# defaults too — and finally hold the default pose. Ramping from a planted stand (instead of popping
# into a bent pose and dropping) keeps it balanced through bring-up.
PREPARE_RAMP_STEPS = 150
PREPARE_HOLD_STEPS = 50

_all_info = motor_info + fixed_info       # every joint, for spawning qpos
_ground_for_height = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")


def _spawn_height(joint_angles, clearance=0.01):
    """Base z that puts the lowest geom ~clearance above the ground for the given joint pose."""
    mj.mj_resetData(model, data)
    for info, a in zip(_all_info, joint_angles):
        data.qpos[info["qadr"]] = a
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mj.mj_forward(model, data)
    lowest = min(float(data.geom_xpos[g, 2]) for g in range(model.ngeom) if g != _ground_for_height)
    return 1.0 - lowest + clearance


def prepare_to_default(ramp_steps=PREPARE_RAMP_STEPS, hold_steps=PREPARE_HOLD_STEPS):
    """Spawn like deploy_genesis (base z=0.56, URDF rest joints), ramp to default, then hold."""
    mj.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.56]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)
    print(f"[prepare] spawn at base_z=0.560 (genesis parity), ramping to default over {ramp_steps} steps")

    q_motor = torch.tensor([data.qpos[m["qadr"]] for m in motor_info], dtype=torch.float32)
    q_fixed = torch.tensor([data.qpos[f["qadr"]] for f in fixed_info], dtype=torch.float32)

    for i in range(ramp_steps):
        alpha = (i + 1) / ramp_steps
        physics_substeps((1 - alpha) * q_motor + alpha * default_motor,
                         (1 - alpha) * q_fixed + alpha * fixed_target)
        if viewer is not None:
            viewer.sync()
    for _ in range(hold_steps):
        physics_substeps(default_motor, fixed_target)
        if viewer is not None:
            viewer.sync()
    print(f"[prepare] at default pose: base_z={float(data.qpos[2]):.3f}  contacts={data.ncon}")


# --- RSI spawn (poses from training motion reference, K1_env._setup_rsi_data) ----------------------
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_urdf_joint_names = p["urdf_joint_names"]
_rsi_frames: list[dict] | None = None


def _load_rsi_frames() -> list[dict]:
    """Load reference frames from the same NPZ files training uses for RSI (rsi_prob=0.5)."""
    global _rsi_frames
    if _rsi_frames is not None:
        return _rsi_frames
    paths = [
        os.path.join(_repo_root, "walk/data/motions/slow.npz"),
        os.path.join(_repo_root, "walk/data/motions/k1_jogging_motion.npz"),
    ]
    frames: list[dict] = []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        ref_jn = [str(n) for n in d["joint_names"]]
        col = [ref_jn.index(name) for name in _urdf_joint_names]
        n = int(d["root_pos"].shape[0])
        for i in range(n):
            frames.append({
                "source": os.path.basename(path),
                "index": i,
                "root_pos": d["root_pos"][i].astype(np.float64),
                "root_quat": d["root_quat"][i].astype(np.float64),
                "dof_pos": d["dof_pos"][i, col].astype(np.float64),
                "dof_vel": d["dof_vel"][i, col].astype(np.float64),
            })
    _rsi_frames = frames
    print(f"[rsi] loaded {len(frames)} reference frames from {len(paths)} motion files")
    return frames


def _write_joint_state(dof_pos, dof_vel):
    """Set all 22 URDF joints in MuJoCo qpos/qvel (by name, any list order)."""
    for name, angle, vel in zip(_urdf_joint_names, dof_pos, dof_vel):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        data.qpos[qadr] = float(angle)
        data.qvel[vadr] = float(vel)


def prepare_rsi(frame_idx=0, hold_steps=PREPARE_HOLD_STEPS):
    """Teleport to a training RSI frame (joint pose + velocity), then PD-hold that pose briefly."""
    frames = _load_rsi_frames()
    frame = frames[frame_idx % len(frames)]
    mj.mj_resetData(model, data)
    data.qpos[0:3] = frame["root_pos"]
    data.qpos[3:7] = frame["root_quat"]
    data.qvel[:] = 0.0
    _write_joint_state(frame["dof_pos"], frame["dof_vel"])
    mj.mj_forward(model, data)
    print(f"[prepare/rsi] frame {frame_idx % len(frames)} from {frame['source']}[{frame['index']}]  "
          f"base_z={float(data.qpos[2]):.3f}")

    # Hold current pose (not default) so PD does not snap joints on the first tick.
    hold_motor = torch.tensor([data.qpos[m["qadr"]] for m in motor_info], dtype=torch.float32)
    hold_fixed = torch.tensor([data.qpos[f["qadr"]] for f in fixed_info], dtype=torch.float32)
    for _ in range(hold_steps):
        physics_substeps(hold_motor, hold_fixed)
        if viewer is not None:
            viewer.sync()
    print(f"[prepare/rsi] after hold: base_z={float(data.qpos[2]):.3f}  contacts={data.ncon}")


def prepare_spawn(mode="default", rsi_frame=0):
    if mode == "rsi":
        prepare_rsi(rsi_frame)
    else:
        prepare_to_default()


# --- kill switch ----------------------------------------------------------------------------------
# Cut all actuation so the robot collapses limp instead of twitching. Triggered by a detected fall or
# by pressing 'q' in the viewer; once limp the loop just steps physics so it settles on the ground.
FALL_HEIGHT = 0.20   # base height [m] below which it counts as fallen (standing ~0.56)
FALL_TILT = -0.1     # projected-gravity z above this ⇒ tilted >~65° (upright ≈ -1.0)
killed = {"pending": False, "on": False}


def go_limp():
    """Zero every actuator command so the robot goes fully passive (motors off)."""
    data.ctrl[:] = 0.0


# --- keyboard velocity commands -------------------------------------------------------------------
# Live [lin_vel_x, lin_vel_y, ang_vel_yaw] command, nudged by key presses and fed to the policy each
# step. Starts at zero so the robot stands still until you drive it.
#   Up / Down    : forward / backward  (lin_vel_x)
#   Left / Right : turn left / right    (ang_vel_yaw)
#   J / L        : strafe left / right  (lin_vel_y)
#   X            : zero all commands (stop)
#   Q            : kill switch (go limp)
cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
CMD_STEP = {"vx": 0.1, "vy": 0.1, "wz": 0.1}
CMD_RANGE = {"vx": (-0.5, 1.3), "vy": (-0.5, 0.5), "wz": (-1.0, 1.0)}


def nudge(axis, sign):
    lo, hi = CMD_RANGE[axis]
    cmd[axis] = max(lo, min(hi, cmd[axis] + sign * CMD_STEP[axis]))
    print(f"[cmd] vx={cmd['vx']:+.2f} vy={cmd['vy']:+.2f} wz={cmd['wz']:+.2f}")


def stop_cmd():
    cmd["vx"] = cmd["vy"] = cmd["wz"] = 0.0
    print("[cmd] stop (all zero)")


# GLFW key codes delivered to the MuJoCo passive-viewer key_callback.
KEY_Q, KEY_X, KEY_J, KEY_L = 81, 88, 74, 76
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262


def key_callback(keycode):
    if keycode == KEY_Q:
        killed["pending"] = True
    elif keycode == KEY_UP:
        nudge("vx", +1)
    elif keycode == KEY_DOWN:
        nudge("vx", -1)
    elif keycode == KEY_LEFT:
        nudge("wz", +1)
    elif keycode == KEY_RIGHT:
        nudge("wz", -1)
    elif keycode == KEY_J:
        nudge("vy", +1)
    elif keycode == KEY_L:
        nudge("vy", -1)
    elif keycode == KEY_X:
        stop_cmd()


_ground = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")


# Action logging: print the policy's raw output each control step. Set LOG_EVERY > 1 to thin it out.
LOG_ACTIONS = True
LOG_EVERY = 1


# --- bring-up + closed-loop inference (identical loop to deploy_genesis) ---------------------------
def run_inference(max_steps=None, spawn_mode="default", rsi_frame=0):
    """Ramp/teleport to start pose, then run closed-loop policy inference like deploy_genesis.
    spawn_mode: 'default' (PD-ramp to default_joint_angles) or 'rsi' (training motion frame).
    Returns (steps_run, fell, final_base_z)."""
    prepare_spawn(spawn_mode, rsi_frame)
    # No warmup burn-in: priming the hidden state with standing frames puts it in a regime the policy
    # never sees at the start of a training rollout, which destabilises the handoff. Genesis just
    # resets and goes — so do we.
    policy.reset()
    last_actions = torch.zeros(len(joint_names), dtype=torch.float32)
    step = 0

    t_next = time.perf_counter()
    while (viewer is None or viewer.is_running()) and (max_steps is None or step < max_steps):
        if killed["pending"] and not killed["on"]:
            killed["on"] = True
            go_limp()
            print("[kill] q pressed → motors off, robot limp")
        if killed["on"]:
            data.ctrl[:] = 0.0
            mj.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            elif max_steps is not None:
                break
            continue

        proj_grav = transform_by_quat(
            g_world, inv_quat(torch.tensor(data.qpos[3:7].copy(), dtype=torch.float32))
        )

        # Fall detection: if the base drops or tilts too far, kill and go limp.
        if data.qpos[2] < FALL_HEIGHT or proj_grav[2].item() > FALL_TILT:
            killed["on"] = True
            go_limp()
            print(f"[kill] fall detected at step {step} → motors off, robot limp")
            mj.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            elif max_steps is not None:
                break
            continue

        obs = build_obs(step, last_actions)
        step += 1

        with torch.no_grad():
            actions = torch.clip(policy(obs), -clip_actions, clip_actions)

        if LOG_ACTIONS and step % LOG_EVERY == 0:
            a = actions.numpy()
            vec = " ".join(f"{v:+.2f}" for v in a)
            print(f"[act] step {step:4d} | z={data.qpos[2]:.3f} | "
                  f"|a| mean={np.abs(a).mean():.3f} max={np.abs(a).max():.3f} | a=[{vec}]")

        # One-step action latency, matching training's simulate_action_latency=True: the motors execute
        # the PREVIOUS action (last_actions, the same value carried in the observation) this step, while
        # the freshly computed action is applied next step.
        motor_targets = default_motor + last_actions * action_scale
        physics_substeps(motor_targets, fixed_target)
        last_actions = actions

        if viewer is not None:
            viewer.sync()
            t_next += control_dt
            sleep_s = t_next - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
    return step, killed["on"], float(data.qpos[2])


def main():
    global viewer
    parser = argparse.ArgumentParser(description="Deploy Genesis-trained policy in MuJoCo")
    parser.add_argument("--spawn", choices=("default", "rsi"), default="default",
                        help="default: PD-ramp to default_joint_angles; rsi: training motion frame")
    parser.add_argument("--rsi-frame", type=int, default=0,
                        help="RSI frame index (slow.npz frames first, then jogging)")
    parser.add_argument("--steps", type=int, default=None, help="headless step limit (no viewer)")
    args = parser.parse_args()

    if _ground >= 0:
        model.geom_group[_ground] = 1
    if args.steps is not None:
        run_inference(max_steps=args.steps, spawn_mode=args.spawn, rsi_frame=args.rsi_frame)
        return
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.opt.geomgroup[0] = 0
        run_inference(spawn_mode=args.spawn, rsi_frame=args.rsi_frame)


if __name__ == "__main__":
    main()
