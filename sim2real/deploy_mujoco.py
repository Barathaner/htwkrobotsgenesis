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
import os
import mujoco as mj
import mujoco.viewer
import math
import torch
import torch.nn as nn
import os
import yaml

with open("/home/luna/Dokumente/git/htwkrobotsgenesis-1/sim2real/deploy.yaml", "r") as f:
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

model = mj.MjModel.from_xml_path(config["policy"]["mujoco_path"])
data = mj.MjData(model)
print(f"model: {model.njnt} joints, {model.nu} actuators, "
      f"nq={model.nq} (qpos size), nv={model.nv} (qvel size)\n")

p = config["policy"]
joint_names = p["joint_names"]          # 16 policy joints, action order
fixed_names = p["fixed_joint_names"]    # PD-held joints (head, shoulders, elbows)
all_names = joint_names + fixed_names

# Per-joint MuJoCo addressing. Unlike Genesis (where control_dofs_position re-sorts the targets), in
# MuJoCo every joint is addressed by its own qpos/dof/actuator id, so no argsort reorder is needed —
# we read and write each joint directly by name.
def joint_info(name):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
    aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
    return {
        "qadr": int(model.jnt_qposadr[jid]),   # index into data.qpos
        "vadr": int(model.jnt_dofadr[jid]),     # index into data.qvel / data.ctrl-joint
        "aid": int(aid),                         # index into data.ctrl (motor actuator)
        "kp": float(p["joint_gains"][name]["kp"]),
        "kd": float(p["joint_gains"][name]["kd"]),
        "effort": float(p["joint_gains"][name]["effort"]),
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

# Control runs at dt=0.02 like training; physics at the XML timestep (0.001) → step it `decimation`
# times per control tick, recomputing the PD torque each substep (what Genesis does internally).
control_dt = p["dt"]
decimation = max(1, round(control_dt / model.opt.timestep))
g_world = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)


def pd_torque(info, target_pos):
    """tau = kp*(target - q) - kd*dq, clipped to ±effort. Mirrors Genesis control_dofs_position PD."""
    q = data.qpos[info["qadr"]]
    dq = data.qvel[info["vadr"]]
    tau = info["kp"] * (target_pos - q) - info["kd"] * dq
    return max(-info["effort"], min(info["effort"], tau))


def apply_pd(motor_targets, fixed_targets):
    """Write PD torques into data.ctrl for one physics substep (called every substep)."""
    for info, tgt in zip(motor_info, motor_targets):
        data.ctrl[info["aid"]] = pd_torque(info, float(tgt))
    for info, tgt in zip(fixed_info, fixed_targets):
        data.ctrl[info["aid"]] = pd_torque(info, float(tgt))


def physics_substeps(motor_targets, fixed_targets, n=None):
    """Advance physics `decimation` (or n) substeps holding the given position targets."""
    for _ in range(decimation if n is None else n):
        apply_pd(motor_targets, fixed_targets)
        mj.mj_step(model, data)


# --- spawn & bring-up -----------------------------------------------------------------------------
# Spawn upright with straight legs (URDF rest = 0) slightly above the ground, then PD-ramp every joint
# to the default pose over n_steps (like the real robot's bring-up and deploy_genesis.prepare_to_default).
mj.mj_resetData(model, data)
data.qpos[0:3] = [0.0, 0.0, 0.80]      # base position (x, y, z)
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # base orientation (wxyz, upright)
mj.mj_forward(model, data)


def prepare_to_default(n_steps=100):
    """Smoothly PD-ramp the policy and fixed joints from their current angles to the default pose."""
    q_motor = torch.tensor([data.qpos[m["qadr"]] for m in motor_info], dtype=torch.float32)
    q_fixed = torch.tensor([data.qpos[f["qadr"]] for f in fixed_info], dtype=torch.float32)
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        motor_targets = (1 - alpha) * q_motor + alpha * default_motor
        fixed_targets = (1 - alpha) * q_fixed + alpha * fixed_target
        physics_substeps(motor_targets, fixed_targets)
        if viewer is not None:
            viewer.sync()


# --- kill switch ----------------------------------------------------------------------------------
# Cut all actuation so the robot collapses limp instead of twitching. Triggered by a detected fall or
# by pressing 'q' in the viewer; once limp the loop just steps physics so it settles on the ground.
FALL_HEIGHT = 0.30   # base height [m] below which it counts as fallen (standing ~0.56)
FALL_TILT = -0.4     # projected-gravity z above this ⇒ tilted >~65° (upright ≈ -1.0)
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


# --- main loop ------------------------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    viewer.opt.geomgroup[0] = 0  # hide collision geoms (group 0), show visual meshes (group 1)

    prepare_to_default()
    policy.reset()  # start the policy with empty LSTM memory

    step = 0
    last_actions = torch.zeros(len(joint_names), dtype=torch.float32)
    while viewer.is_running():
        # Kill switch ('q' pressed): cut actuation once, then just let the robot settle limp.
        if killed["pending"] and not killed["on"]:
            killed["on"] = True
            go_limp()
            print("[kill] q pressed → motors off, robot limp")
        if killed["on"]:
            data.ctrl[:] = 0.0
            mj.mj_step(model, data)
            viewer.sync()
            continue

        # --- read robot state -> observation ------------------------------------------------------
        base_quat = torch.tensor(data.qpos[3:7].copy(), dtype=torch.float32)   # wxyz
        # MuJoCo free-joint qvel[3:6] is already the body-frame angular velocity (= genesis ang_body_vel).
        ang_body_vel = torch.tensor(data.qvel[3:6].copy(), dtype=torch.float32)
        obs_ang_vel = ang_body_vel * obs_scales["ang_vel"]
        proj_grav = transform_by_quat(g_world, inv_quat(base_quat))

        # Fall detection: if the base drops or tilts too far, kill and go limp.
        if data.qpos[2] < FALL_HEIGHT or proj_grav[2].item() > FALL_TILT:
            killed["on"] = True
            go_limp()
            print("[kill] fall detected → motors off, robot limp")
            mj.mj_step(model, data)
            viewer.sync()
            continue

        q = torch.tensor([data.qpos[m["qadr"]] for m in motor_info], dtype=torch.float32)
        dq = torch.tensor([data.qvel[m["vadr"]] for m in motor_info], dtype=torch.float32)
        obs_dof_pos = (q - default_motor) * obs_scales["dof_pos"]
        dof_vel = dq * obs_scales["dof_vel"]

        commands = torch.tensor([cmd["vx"], cmd["vy"], cmd["wz"]], dtype=torch.float32) * commands_scale

        phase = (step % p["gait_period_steps"]) / p["gait_period_steps"]
        clock = torch.tensor(
            [math.sin(phase * 2.0 * math.pi), math.cos(phase * 2.0 * math.pi)], dtype=torch.float32
        )

        # Single proprio frame, layout matching the training proprio group (K1_env._update_observation):
        # ang_vel, projected_gravity, commands, dof_pos, dof_vel, last_actions, gait_clock.
        obs = torch.cat(
            [obs_ang_vel, proj_grav, commands, obs_dof_pos, dof_vel, last_actions, clock], dim=0
        )
        step += 1

        with torch.no_grad():
            actions = torch.clip(policy(obs), -clip_actions, clip_actions)

        # One-step action latency, matching training's simulate_action_latency=True: the motors execute
        # the PREVIOUS action (last_actions, the same value carried in the observation) this step, while
        # the freshly computed action is applied next step.
        motor_targets = default_motor + last_actions * action_scale
        physics_substeps(motor_targets, fixed_target)

        last_actions = actions
        viewer.sync()
