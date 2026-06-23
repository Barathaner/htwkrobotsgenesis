

import math
from utils import make_dummy_student_obs
import torch
import torch.nn as nn
import genesis as gs
import os
import yaml
import numpy as np
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


def prepare_to_default(robot,dofs_idx,default_targets,n_steps=100):
    "smooth drive joints to default pose"
    q_start = robot.get_dofs_position(dofs_idx).clone()
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        target = (1 - alpha) * q_start + alpha * default_targets
        robot.control_dofs_position(target, dofs_idx)
        scene.step()
    for i in range(n_steps):
        robot.control_dofs_position(default_targets, dofs_idx)
        scene.step()


if __name__ == "__main__":
    gs.init(backend=gs.cpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -2.0, 1.5), camera_lookat=(0.0, 0.0, 0.5), camera_fov=30,
        ),
        sim_options=gs.options.SimOptions(dt=0.02),
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
    dofs_idx = [robot.get_joint(name).dof_idx_local for name in all_names]
    default_pose = torch.tensor(
        [p["default_joint_angles"][name] for name in all_names],
        dtype=gs.tc_float,
        device=gs.device,
    )

    kp = [p["joint_gains"][name]["kp"] for name in all_names]
    kd = [p["joint_gains"][name]["kd"] for name in all_names]
    effort = [p["joint_gains"][name]["effort"] for name in all_names]
    robot.set_dofs_kp(kp, dofs_idx)
    robot.set_dofs_kv(kd, dofs_idx)
    robot.set_dofs_force_range([-e for e in effort], effort, dofs_idx)
    model = load_model(config["policy"]["model"])
    print(model)
    prepare_to_default(robot,dofs_idx,default_pose)
    # qpos layout: 7 base (xyz + quat wxyz) followed by the URDF joints in urdf_joint_names order.
    base = np.array([0.0, 0.0, 0.56, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    urdf_joint_names = config["policy"]["urdf_joint_names"]
    joint_names = config["policy"]["joint_names"]
    default_pose = np.array(
        [config["policy"]["default_joint_angles"][name] for name in urdf_joint_names],
        dtype=np.float32,
    )
    # Map each policy action (joint_names order) to its slot in the URDF qpos (urdf_joint_names order).
    action_to_urdf = [urdf_joint_names.index(name) for name in joint_names]
    motor_dofs_robot = [robot.get_joint(name).dof_idx_local for name in joint_names]
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


        commands = torch.tensor([1.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
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

        obs_dof_pos = q - default_pose
        #TODO: measure default pose on robot in prep mode adn correct order of dofs
        dof_vel = dq * 0.05
        #init with zeros
        
        phase = (step % config["policy"]["gait_period_steps"]) / config["policy"]["gait_period_steps"]
        clock = [math.sin(phase * 2.0 * math.pi), math.cos(phase * 2.0 * math.pi)]
        
        # all together in right order
        obs = torch.cat([obs_ang_vel, proj_grav, commands, obs_dof_pos, dof_vel, last_actions, torch.tensor(clock, dtype=gs.tc_float, device=gs.device)], dim=0)
        step += 1


        with torch.no_grad():
            actions = model(obs)
            actions_scaled = model(obs).squeeze(0).cpu().numpy() * config["policy"]["action_scale"]
        print("motor_commands:", actions_scaled)
        joint_angles = default_pose.copy()
        for i, urdf_idx in enumerate(action_to_urdf):
            joint_angles[urdf_idx] += actions_scaled[i]

        qpos = np.concatenate([base, joint_angles], axis=0)
        robot.set_qpos(torch.as_tensor(qpos, dtype=gs.tc_float, device=gs.device))


        #clean for next iter
        last_actions = actions
        scene.step()