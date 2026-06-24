

import math
import torch
import torch.nn as nn
import genesis as gs
import os
import yaml
import genesis.utils.geom as geom
from genesis.vis.keybindings import Key, KeyAction, Keybind
with open("deploy.yaml", "r") as f:
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


#next steps:
# create genesis environment


# Genesis quirk: control_dofs_position(values, dofs_idx) assigns `values` in ASCENDING dof-index
# order, NOT in the order of `dofs_idx` (the boolean-mask assignment in rigid_solver sorts the
# indices). So the values must be pre-permuted by argsort(dofs_idx) for each target to land on the
# right joint. get_dofs_position, by contrast, returns positionally (in dofs_idx order). Training
# does exactly this (K1_env.py: actions_dof_idx = argsort(motors_dof_idx)). Skipping this scrambles
# every joint target onto the wrong joint → the robot twitches and collapses.
def control_position(robot, values, dofs_idx, order):
    robot.control_dofs_position(values[order], dofs_idx)


def prepare_to_default(robot, motor_dofs, motor_order, motor_default,
                       fixed_dofs, fixed_order, fixed_target, n_steps=100):
    """Smoothly PD-ramp the policy and fixed joints from their current angles (the URDF rest pose at
    spawn) to the default pose over n_steps, like the real robot's bring-up, then hand to the policy.
    Ramping all joints together — crucially the arms to their non-zero defaults (shoulder roll ±1.5),
    not left at the T-pose zero — keeps the robot balanced through the whole ramp."""
    q_motor = robot.get_dofs_position(motor_dofs).clone()
    q_fixed = robot.get_dofs_position(fixed_dofs).clone()
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        target = (1 - alpha) * q_motor + alpha * motor_default
        targetfixed = (1 - alpha) * q_fixed + alpha * fixed_target
        robot.control_dofs_position(target, motor_dofs)
        robot.control_dofs_position(targetfixed, fixed_dofs)
        scene.step()


if __name__ == "__main__":
    gs.init(backend=gs.cpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -2.0, 1.5), camera_lookat=(0.0, 0.0, 0.5), camera_fov=30,
        ),
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),  # match training (K1_env.py)
    )
    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file=os.path.abspath(config["policy"]["urdf_path"]), pos=(0.0, 0.0, 0.56)),
        surface=gs.surfaces.Default(color=(0.7, 0.85, 1.0), opacity=0.55),
    )
    scene.viewer.follow_entity(robot, fixed_axis=(None, None, 1.2), smoothing=0.9)
    scene.build()

    p = config["policy"]
    all_names= p["joint_names"] + p["fixed_joint_names"]
    dofs_idx = [idx for name in all_names for idx in robot.get_joint(name).dofs_idx_local]

    kp = [p["joint_gains"][name]["kp"] for name in all_names]
    kd = [p["joint_gains"][name]["kd"] for name in all_names]
    effort = [p["joint_gains"][name]["effort"] for name in all_names]
    robot.set_dofs_kp(kp, dofs_idx)
    robot.set_dofs_kv(kd, dofs_idx)
    robot.set_dofs_force_range([-e for e in effort], effort, dofs_idx)
    model = load_model(config["policy"]["model"])
    print(model)

    joint_names = p["joint_names"]
    # 16-entry default for the policy joints (joint_names order), used to centre the obs.
    default_motor = torch.tensor(
        [p["default_joint_angles"][name] for name in joint_names],
        dtype=gs.tc_float,
        device=gs.device,
    )
    motor_dofs_robot = [idx for name in joint_names for idx in robot.get_joint(name).dofs_idx_local]
    motor_order = torch.argsort(torch.tensor(motor_dofs_robot))

    # Fixed joints (head etc.) are PD-held at their default every step, exactly like training.
    fixed_names = p["fixed_joint_names"]
    fixed_dofs_robot = [idx for name in fixed_names for idx in robot.get_joint(name).dofs_idx_local]
    fixed_order = torch.argsort(torch.tensor(fixed_dofs_robot))
    fixed_target = torch.tensor(
        [p["default_joint_angles"][name] for name in fixed_names],
        dtype=gs.tc_float,
        device=gs.device,
    )
    clip_actions = p["clip_actions"]


    # Smoothly PD-ramp to the default pose (robot is already standing there), then hand to the policy.
    prepare_to_default(robot, motor_dofs_robot, motor_order, default_motor,
                       fixed_dofs_robot, fixed_order, fixed_target)
    # Start the policy with empty LSTM memory (no stale hidden state from construction).
    model.reset()

    # --- kill switch: cut all actuation so the robot collapses limp instead of twitching ----------
    # Triggered either by a detected fall or by pressing 'q' in the viewer. Once limp, the loop just
    # steps physics so the robot settles on the ground and stays there.
    FALL_HEIGHT = 0.30   # base height [m] below which it counts as fallen (standing ~0.56)
    FALL_TILT = -0.4     # projected-gravity z above this ⇒ tilted >~65° (upright ≈ -1.0)
    killed = {"pending": False, "on": False}

    def go_limp():
        """Zero every joint's stiffness, damping and effort so the robot goes fully passive."""
        zeros = [0.0] * len(dofs_idx)
        robot.set_dofs_kp(zeros, dofs_idx)
        robot.set_dofs_kv(zeros, dofs_idx)
        robot.set_dofs_force_range(zeros, zeros, dofs_idx)

    def request_kill():
        killed["pending"] = True   # set from the viewer key-callback thread; applied in the loop

    # 'q' in the Genesis viewer triggers the kill switch (overwrite=False keeps the camera controls).
    if scene.viewer is not None:
        scene.viewer.register_keybinds(
            Keybind("kill_switch", Key.Q, KeyAction.PRESS, callback=request_kill),
        )

    step = 0
    last_actions = torch.zeros((16,), dtype=gs.tc_float, device=gs.device)
    while True:
        # Kill switch ('q' pressed): cut actuation once, then just let the robot settle limp.
        if killed["pending"] and not killed["on"]:
            killed["on"] = True
            go_limp()
            print("[kill] q pressed → motors off, robot limp")
        if killed["on"]:
            scene.step()
            continue

        q = robot.get_dofs_position(motor_dofs_robot)
        dq = robot.get_dofs_velocity(motor_dofs_robot)
        base_ang_vel = robot.get_ang()
        base_quat = robot.get_quat()
        ang_body_vel = geom.transform_by_quat(base_ang_vel, geom.inv_quat(base_quat))
        obs_ang_vel = ang_body_vel * config["policy"]["obs_scales"]["ang_vel"]
        g_world = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        proj_grav = geom.transform_by_quat(g_world, geom.inv_quat(base_quat))

        # Fall detection: if the base drops or tilts too far, kill the policy and go limp so the
        # robot collapses to the ground instead of twitching under continued inference.
        if robot.get_pos()[2].item() < FALL_HEIGHT or proj_grav[2].item() > FALL_TILT:
            killed["on"] = True
            go_limp()
            print("[kill] fall detected → motors off, robot limp")
            scene.step()
            continue


        commands = torch.tensor([0.5, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
        commands_scale = torch.tensor(
            [
                config["policy"]["commands_scales"]["lin_vel_x"],
                config["policy"]["commands_scales"]["lin_vel_y"],
                config["policy"]["commands_scales"]["ang_vel_yaw"],
            ],
            dtype=gs.tc_float,
            device=gs.device,
        )
        commands = commands * commands_scale

        obs_dof_pos = (q - default_motor) * config["policy"]["obs_scales"]["dof_pos"]
        dof_vel = dq * config["policy"]["obs_scales"]["dof_vel"]

        phase = (step % config["policy"]["gait_period_steps"]) / config["policy"]["gait_period_steps"]
        clock = torch.tensor(
            [math.sin(phase * 2.0 * math.pi), math.cos(phase * 2.0 * math.pi)],
            dtype=gs.tc_float, device=gs.device,
        )

        # Single proprio frame, layout matching the training proprio group (K1_env._update_observation):
        # ang_vel, projected_gravity, commands, dof_pos, dof_vel, last_actions, gait_clock.
        # No frame-stack: the LSTM hidden state (carried inside the model across steps) is the memory.
        obs = torch.cat(
            [obs_ang_vel, proj_grav, commands, obs_dof_pos, dof_vel, last_actions, clock], dim=0
        )
        step += 1


        with torch.no_grad():
            actions = torch.clip(model(obs), -clip_actions, clip_actions)
        # One-step action latency, matching training's simulate_action_latency=True: the motors
        # execute the PREVIOUS action (last_actions) — the same value carried in the observation —
        # while the freshly computed action is applied next step. Applying the current action
        # immediately changes the closed-loop dynamics the policy was trained for and destabilises it.
        targets = default_motor + last_actions * config["policy"]["action_scale"]
        control_position(robot, targets, motor_dofs_robot, motor_order)
        # Hold the non-policy joints at their default, like training does every step.
        control_position(robot, fixed_target, fixed_dofs_robot, fixed_order)

        # carry the clipped action to the next step (executed then, and fed back into the obs)
        last_actions = actions
        scene.step()
