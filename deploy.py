

import math
import torch
import torch.nn as nn
import genesis as gs
import os
import yaml
import genesis.utils.geom as geom
with open("deploy.yaml", "r") as f:
    config = yaml.safe_load(f)

class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 256),     nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, 16), # 16 dofs that are controlled by the policy
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def load_model(checkpoint_path: str) -> tuple[ActorMLP, int, bool]:
    """Load for inference.

    Returns (actor, obs_dim, is_student). obs_dim is inferred from the checkpoint's first
    layer so the network width always matches the weights, regardless of obs layout changes.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["student_state_dict"]
    actor = ActorMLP(config["policy"]["obs_dim"])
    # strict=False: rsl-rl also stores distribution.std_param which inference does not need.
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()
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
        gs.morphs.URDF(file=os.path.abspath("models/K1/K1_22dof.urdf"), pos=(0.0, 0.0, 0.56)),
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

    step = 0
    last_actions = torch.zeros((16,), dtype=gs.tc_float, device=gs.device)
    while True:
        q = robot.get_dofs_position(motor_dofs_robot)
        dq = robot.get_dofs_velocity(motor_dofs_robot)
        base_ang_vel = robot.get_ang()
        base_quat = robot.get_quat()
        ang_body_vel = geom.transform_by_quat(base_ang_vel, geom.inv_quat(base_quat))
        obs_ang_vel = ang_body_vel * config["policy"]["obs_scales"]["ang_vel"]
        g_world = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        proj_grav = geom.transform_by_quat(g_world, geom.inv_quat(base_quat))


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

        # obs layout must match the training proprio group (K1_env.py _update_observation):
        # ang_vel, projected_gravity, commands, dof_pos, dof_vel, last_actions, gait_clock
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
