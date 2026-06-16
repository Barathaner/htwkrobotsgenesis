import math
import os

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat, xyz_to_quat
from tensordict import TensorDict


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class _WandbEnvCfg:
    """Thin wrapper: rsl-rl WandbLogWriter.store_config() requires .to_dict()."""

    __slots__ = ("_data",)

    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


class K1Env:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=False,
        record_camera=False,
    ):
        self.num_envs = num_envs
        self.cfg = _WandbEnvCfg(env_cfg)  # rsl-rl Logger / wandb config upload
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = dict(reward_cfg["reward_scales"])
        self.device = gs.device
        self.dt = 0.02
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        scene_kwargs: dict = {
            "sim_options": gs.options.SimOptions(dt=self.dt, substeps=2),
            "rigid_options": gs.options.RigidOptions(
                enable_self_collision=True,
                max_collision_pairs=40,
                tolerance=1e-5,
            ),
            "viewer_options": gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
            ),
            "show_viewer": show_viewer,
        }
        if record_camera:
            scene_kwargs["vis_options"] = gs.options.VisOptions(rendered_envs_idx=[0])
        self.scene = gs.Scene(**scene_kwargs)
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="models/K1/K1_22dof.urdf",
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            )
        )
        if show_viewer:
            from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

            self.scene.viewer.add_plugin(ImGuiOverlayPlugin())

        self.cam = None
        if record_camera:
            vcfg = env_cfg.get("video", {})
            self.cam = self.scene.add_camera(
                res=tuple(vcfg.get("res", [640, 480])),
                pos=tuple(vcfg.get("camera_pos", [3.0, -1.0, 1.5])),
                lookat=tuple(vcfg.get("camera_lookat", [0.0, 0.0, 0.5])),
                fov=vcfg.get("camera_fov", 30),
                GUI=False,
            )

        self.scene.build(n_envs=num_envs)

        # Verfolgungskamera: hält den anfänglichen Versatz Kamera→Rumpf und schwenkt mit dem
        # Roboter mit (lookat folgt dem Rumpf), statt starr zu stehen. update_following() muss
        # je Frame VOR cam.render() aufgerufen werden (siehe k1_train_runner._record_video).
        # smoothing ∈ (0,1): EMA-Glättung (höher = weicher/träger); None = exaktes Tracking.
        if self.cam is not None:
            vcfg = env_cfg.get("video", {})
            if vcfg.get("follow", True):
                self.cam.follow_entity(self.robot, smoothing=vcfg.get("follow_smoothing", 0.9))

        self.num_actions = len(env_cfg["joint_names"])
        self.num_commands = command_cfg["num_commands"]
        assert self.num_actions == env_cfg["num_actions"]

        # Policy actions = die 16 policy-gesteuerten Gelenke (12 Bein + 4 Arm); restliche physische
        # URDF-DOFs (Kopf, Shoulder_Roll, Elbow_Yaw) werden separat fixiert (s. fixed_*).
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        joint_gains = env_cfg["joint_gains"]
        kp = [joint_gains[n]["kp"] for n in env_cfg["joint_names"]]
        kd = [joint_gains[n]["kd"] for n in env_cfg["joint_names"]]
        effort = [joint_gains[n]["effort"] for n in env_cfg["joint_names"]]
        self.robot.set_dofs_kp(kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(kd, self.motors_dof_idx)
        self.robot.set_dofs_force_range([-e for e in effort], effort, self.motors_dof_idx)

        # Fixierte Gelenke (Kopf + fürs Laufen unnötige Arm-/Bein-DOFs): nicht policy-gesteuert, aber
        # per PD auf fester Default-Pose gehalten (sonst schlackern sie lose). Eigene kp/kd setzen
        # (Default wäre 0 → kraftlos) und konstantes Ziel fixed_target jeden Step kommandieren.
        fixed_names = env_cfg["fixed_joint_names"]
        self.fixed_dof_idx = torch.tensor(
            [self.robot.get_joint(n).dof_start for n in fixed_names], dtype=gs.tc_int, device=gs.device
        )
        self.fixed_actions_dof_idx = torch.argsort(self.fixed_dof_idx)
        self.robot.set_dofs_kp([joint_gains[n]["kp"] for n in fixed_names], self.fixed_dof_idx)
        self.robot.set_dofs_kv([joint_gains[n]["kd"] for n in fixed_names], self.fixed_dof_idx)
        fixed_effort = [joint_gains[n]["effort"] for n in fixed_names]
        self.robot.set_dofs_force_range([-e for e in fixed_effort], fixed_effort, self.fixed_dof_idx)
        self.fixed_target = torch.tensor(
            [env_cfg["default_joint_angles"][n] for n in fixed_names], dtype=gs.tc_float, device=gs.device
        ).expand(num_envs, -1)

        # start state
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        # Full-robot initial pose: iterate all physical DOFs (URDF still has 22 joints).
        # Joints not driven by the policy (e.g. head) default to 0.0.
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"].get(joint.name, 0.0) for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, inv_quat(self.init_base_quat))

        # Position robot trunk in world buffer
        self.base_ang_vel = torch.empty((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_pos = torch.empty((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        # Lineare Geschwindigkeit im Heading-Frame (Welt-xy relativ zur Spawn-Yaw je Episode).
        self.base_lin_vel_heading = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        spawn_yaw = quat_to_xyz(self.init_base_quat.unsqueeze(0), rpy=True)[0, 2]
        spawn_rpy = torch.zeros(1, 3, dtype=gs.tc_float, device=gs.device)
        spawn_rpy[0, 2] = spawn_yaw
        self.inv_heading_quat = inv_quat(xyz_to_quat(spawn_rpy)).expand(num_envs, -1).clone()

        self.projected_gravity = torch.empty_like(self.base_ang_vel)
        self.commands = torch.empty((num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [
                obs_cfg["obs_scales"]["lin_vel"],
                obs_cfg["obs_scales"]["lin_vel"],
                obs_cfg["obs_scales"]["ang_vel"],
            ],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.commands_limits = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )

        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][n] for n in env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.dof_pos = torch.empty((num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.dof_vel = torch.empty_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_pos)

        self.actions = torch.zeros_like(self.dof_pos)
        self.last_actions = torch.zeros_like(self.dof_pos)

        # Fuß-Kontakt via net contact force (Genesis links_state.contact_force); start True (Spawn stehend).
        self.feet_idx_local = [self.robot.get_link(n).idx_local for n in ("left_foot_link", "right_foot_link")]
        self.feet_link_idx = torch.tensor(self.feet_idx_local, dtype=gs.tc_int, device=gs.device)
        self.foot_contact_force_threshold = float(self.reward_cfg["foot_contact_force_threshold"])
        self.foot_in_contact = torch.ones((num_envs, 2), dtype=gs.tc_bool, device=gs.device)

        # feet_air_time: Luftphase je Fuß [s]; phasen-gekoppelte Belohnung beim Aufsetzen.
        self.foot_air_time = torch.zeros((num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.feet_air_time_reward = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)

        # leg_symmetry: Strafe für nicht-alternierenden Gang (siehe _reward_leg_symmetry). Beide Füße
        # gleichzeitig in der Luft (Flugphase, im Gehen nie gültig) → sofort bestraft; beide gleichzeitig
        # am Boden (Doppelstütze) → nur die Überdauer über das natürliche Doppelstütz-Fenster hinaus.
        self.both_stance_time = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.leg_symmetry_penalty = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        # leg_symmetry: erlaubte Doppelstütz-Dauer [s]. Der Timer (both_stance_time) zählt eine
        # zusammenhängende Doppelstütze; Gemeinsames Stehen darüber hinaus wird bestraft.
        self.ds_allow_time = float(self.reward_cfg["leg_symmetry_ds_allow_s"])

        # style (feature-matching Style-Reward): belohnt Nähe der Live-Bewegung zur NPZ-Referenz im
        # Feature-Raum (nicht-adversariell, timing-/speed-agnostisch). Setup nur wenn aktiviert.
        self.style_enabled = "style" in self.reward_scales
        if self.style_enabled:
            self._setup_style_reference()

        # Assist-Force-Curriculum: zieht den Trunk in Kommandorichtung; Stärke fällt linear auf 0 bei
        # decay_end_iter. cur_iteration wird vom Runner je Lern-Iteration gesetzt; bleibt es None
        # (Eval/Video), wird keine Kraft angewandt (s. _apply_assist_force).
        self.cur_iteration = None
        self.assist_force_cfg = env_cfg.get("assist_force")
        if self.assist_force_cfg is not None:
            self.assist_force_start = float(self.assist_force_cfg["start_n"])
            self.assist_decay_end_iter = int(self.assist_force_cfg["decay_end_iter"])
            self.trunk_link_idx = self.robot.get_link("Trunk").idx  # globaler Solver-Index
            self._assist_force = torch.zeros((num_envs, 1, 3), dtype=gs.tc_float, device=gs.device)
            self.assist_force_mag = 0.0  # aktuelle Zugkraft [N] (für wandb-Logging, s. step())

        # Obs nur aus deploy-fähigen Größen (IMU + Gelenke + commands + letzte Aktion). base_lin_vel_heading
        # (privilegiert, auf der Hardware nicht messbar) und feet_contact (Deploy liefert nur Fake) sind
        # bewusst NICHT in der Obs — sie bleiben aber als Reward-Eingang erhalten (tracking_lin_vel bzw.
        # feet_slip/leg_symmetry).
        self._obs_slices = {
            "base_ang_vel": self.base_ang_vel.shape[-1],
            "projected_gravity": self.projected_gravity.shape[-1],
            "commands": self.commands.shape[-1],
            "dof_pos": self.dof_pos.shape[-1],
            "dof_vel": self.dof_vel.shape[-1],
            "actions": self.actions.shape[-1],
        }
        self.obs_dim = sum(self._obs_slices.values())

        self.obs_buf = torch.empty((num_envs, self.obs_dim), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((num_envs,), dtype=gs.tc_int, device=gs.device)
        self.extras = {}

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    def _setup_style_reference(self):
        """Lädt die NPZ-Referenz und baut die standardisierte Style-Feature-Matrix (siehe
        _reward_style). Feature je Frame: [root_height(1), projected_gravity(3), dof_pos(16),
        dof_vel(16), foot_clear(2)] = 38 — identische Definition für Referenz und Live, alle
        Größen frame-konsistent (kein Phasen-Clock, timing-/speed-agnostisch).
        """
        path = self.reward_cfg["style_motion_file"]
        if not os.path.isabs(path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(repo_root, path)
        data = np.load(path, allow_pickle=True)

        # 22-DOF-Referenz (NPZ joint_names) → 16 policy-Gelenke in env-Action-Reihenfolge (by name),
        # damit ref dof_pos/dof_vel spaltenweise zu self.dof_pos/self.dof_vel passen.
        ref_jn = [str(n) for n in data["joint_names"]]
        col = [ref_jn.index(n) for n in self.env_cfg["joint_names"]]
        ref = np.concatenate(
            [
                data["root_pos"][:, 2:3],          # root height
                data["projected_gravity"],          # (M,3) Schwerkraft im Body-Frame (Rumpfneigung)
                data["dof_pos"][:, col],            # (M,16)
                data["dof_vel"][:, col],            # (M,16) rad/s
                data["foot_clear"],                 # (M,2) Schwunghöhe je Fuß
            ],
            axis=1,
        ).astype(np.float32)

        mean = ref.mean(0)
        std = ref.std(0) + 1e-6  # per-Feature standardisieren → Distanz nicht von dof_vel dominiert

        # Pro-Gruppe-Gewichte in der Distanz (nach Standardisierung): ohne sie zählen die 16 dof_vel-
        # Dims ~42% der Distanz und die Orientierung nur ~8% — umgekehrt zur IL-Praxis. Gewichte →
        # Gelenkwinkel primär, Orientierung ~20%, Gelenkgeschw. runter, Füße hoch (vgl. DeepMimic-Ratios).
        n = len(self.env_cfg["joint_names"])
        w = np.concatenate([
            np.full(1, 0.5, np.float32),    # root_height
            np.full(3, 2.0, np.float32),    # projected_gravity (Rumpf-Orientierung) — hochgewichtet
            np.full(n, 1.0, np.float32),    # dof_pos (Gelenkwinkel) — Hauptfaktor
            np.full(n, 0.25, np.float32),   # dof_vel — runtergewichtet (sonst geschw.-dominiert)
            np.full(2, 2.0, np.float32),    # foot_clear (Füße) — hochgewichtet
        ])
        sqrt_w = np.sqrt(w)  # in die Features falten → torch.cdist liefert die gewichtete Distanz

        self.style_feat_dim = ref.shape[1]
        self.style_sigma = float(self.reward_cfg.get("style_sigma", 1.0))
        self.style_w_sum = float(w.sum())
        self.style_mean = torch.tensor(mean, dtype=gs.tc_float, device=gs.device)
        self.style_std = torch.tensor(std, dtype=gs.tc_float, device=gs.device)
        self.style_sqrt_w = torch.tensor(sqrt_w, dtype=gs.tc_float, device=gs.device)
        # Referenz vorab standardisiert UND gewichtet ablegen (Live-Feature wird gleich behandelt).
        self.style_ref_feat = torch.tensor((ref - mean) / std, dtype=gs.tc_float, device=gs.device)
        self.style_ref_feat = self.style_ref_feat * self.style_sqrt_w

        # Steh-Fußhöhe (FK auf Default-Pose) als Live-Clearance-Nulllinie — passt zur per-Fuß-Ground
        # der Referenz (foot_clear = z − Steh-z), siehe build_motion_reference.
        q0 = self.init_qpos.unsqueeze(0).expand(self.num_envs, -1).contiguous()
        links_pos, _ = self.robot.forward_kinematics(q0)
        self.style_ground_z = links_pos[0, self.feet_link_idx, 2].clone()  # (2,)
        print(f"[style] ref={self.style_ref_feat.shape[0]} frames  feat_dim={self.style_feat_dim}  "
              f"sigma={self.style_sigma}  ground_z={np.round(self.style_ground_z.cpu().numpy(), 3)}")

    def step(self, actions):
        self.actions.copy_(torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]))
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(
            target_dof_pos[:, self.actions_dof_idx],
            self.motors_dof_idx,
        )
        # Fixierte Gelenke (Kopf + unbenutzte Arm-/Bein-DOFs) konstant auf Default-Pose halten — kein Mitschlackern.
        self.robot.control_dofs_position(
            self.fixed_target[:, self.fixed_actions_dof_idx],
            self.fixed_dof_idx,
        )
        self._apply_assist_force()
        self.scene.step()

        self.episode_length_buf += 1
        # copy into pre-allocated buffers so reset() works outside torch.inference_mode()
        self.base_pos.copy_(self.robot.get_pos())
        self.base_quat.copy_(self.robot.get_quat())
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        world_lin_vel = self.robot.get_vel()
        self.base_lin_vel.copy_(transform_by_quat(world_lin_vel, inv_base_quat))
        self.base_lin_vel_heading.copy_(transform_by_quat(world_lin_vel, self.inv_heading_quat))
        self.base_ang_vel.copy_(transform_by_quat(self.robot.get_ang(), inv_base_quat))
        self.projected_gravity.copy_(transform_by_quat(self.global_gravity, inv_base_quat))
        self.dof_pos.copy_(self.robot.get_dofs_position(self.motors_dof_idx))
        self.dof_vel.copy_(self.robot.get_dofs_velocity(self.motors_dof_idx))
        self._update_foot_contact()

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._resample_commands(
            self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0
        )

        self.reset_buf.copy_(self._termination_mask())

        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf)

        # Assist-Force-Curriculum in wandb loggen (nach _reset_idx, das extras["episode"] neu aufbaut).
        # Nur im Training (cur_iteration gesetzt); ep_extras wird je Step gemittelt → Kurve über Iterationen.
        if self.assist_force_cfg is not None and self.cur_iteration is not None:
            self.extras["episode"]["assist_force_n"] = torch.tensor(self.assist_force_mag, device=self.device)

        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _apply_assist_force(self):
        """Curriculum-Hilfskraft: zieht den Trunk im Welt-Frame in Kommandorichtung. Magnitude fällt
        linear von start_n (Iter 0) auf 0 (decay_end_iter); danach kein Aufruf. Nur im Training aktiv:
        cur_iteration wird vom Runner gesetzt — bleibt es None (Eval/Video), passiert nichts. Die
        Kraft muss je Step neu angewandt werden (Genesis löscht externe Kräfte nach scene.step()).
        """
        if self.assist_force_cfg is None or self.cur_iteration is None:
            return
        frac = max(0.0, 1.0 - self.cur_iteration / self.assist_decay_end_iter)
        self.assist_force_mag = self.assist_force_start * frac  # für wandb-Logging (s. step())
        if self.assist_force_mag <= 0.0:
            return
        # Kommandorichtung im Heading-Frame (horizontal, normiert) → Welt-Frame.
        cmd_dir = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        cmd_dir[:, :2] = self.commands[:, :2]
        norm = torch.linalg.norm(cmd_dir, dim=1, keepdim=True).clamp(min=1e-6)
        unit_world = transform_by_quat(cmd_dir / norm, inv_quat(self.inv_heading_quat))
        self._assist_force[:, 0, :] = unit_world * self.assist_force_mag
        self.scene.rigid_solver.apply_links_external_force(
            force=self._assist_force, links_idx=[self.trunk_link_idx]
        )

    def _set_heading_reference(self, envs_idx=None):
        """Spawn-Yaw je Episode fixieren → Heading-Frame für command_accuracy (Welt-xy ohne Drift).

        Aktuell identisches init_base_quat für alle Envs; bei Random-Yaw-Spawn später base_quat nach Reset lesen.
        """
        quat = self.init_base_quat.unsqueeze(0)
        yaw = quat_to_xyz(quat, rpy=True)[0, 2]
        spawn_rpy = torch.zeros(1, 3, dtype=gs.tc_float, device=gs.device)
        spawn_rpy[0, 2] = yaw
        inv_h = inv_quat(xyz_to_quat(spawn_rpy))
        if envs_idx is None:
            self.inv_heading_quat.copy_(inv_h.expand(self.num_envs, -1))
        else:
            self.inv_heading_quat.index_copy_(0, envs_idx.nonzero(as_tuple=True)[0], inv_h.expand(envs_idx.sum(), -1))

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    def _reset_idx(self, envs_idx=None):
        # reset state
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # reset buffers
        if envs_idx is None:
            self.base_pos[:] = self.init_base_pos
            self.base_quat[:] = self.init_base_quat
            self.projected_gravity[:] = self.init_projected_gravity
            self.dof_pos[:] = self.default_dof_pos
            self.base_lin_vel.zero_()
            self.base_lin_vel_heading.zero_()
            self.base_ang_vel.zero_()
            self._set_heading_reference()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            self.foot_in_contact.fill_(True)
            self.foot_air_time.zero_()
            self.feet_air_time_reward.zero_()
            self.both_stance_time.zero_()
            self.leg_symmetry_penalty.zero_()
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.default_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_lin_vel_heading.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self._set_heading_reference(envs_idx)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)
            self.foot_in_contact.masked_fill_(envs_idx[:, None], True)
            self.foot_air_time.masked_fill_(envs_idx[:, None], 0.0)
            self.feet_air_time_reward.masked_fill_(envs_idx, 0.0)
            self.both_stance_time.masked_fill_(envs_idx, 0.0)
            self.leg_symmetry_penalty.masked_fill_(envs_idx, 0.0)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        # random sample command upon reset
        self._resample_commands(envs_idx)

    def _read_foot_contact_forces(self):
        """Net contact force on foot links (n_envs, 2, 3) [N], world frame.

        Uses RigidEntity.get_links_net_contact_force() → rigid_solver.links_state.contact_force.
        """
        link_forces = self.robot.get_links_net_contact_force()
        if link_forces.ndim == 2:
            link_forces = link_forces.unsqueeze(0)
        return link_forces[:, self.feet_link_idx, :]

    def _foot_in_contact_from_force(self):
        """Bodenkontakt je Fuß: |F_net| > foot_contact_force_threshold."""
        foot_forces = self._read_foot_contact_forces()
        force_norm = torch.linalg.norm(foot_forces, dim=2)
        return force_norm > self.foot_contact_force_threshold

    def _update_foot_contact(self):
        """Fuß-Kontakt via net contact force; verbucht die Lande-Belohnung (feet_air_time) und die
        Symmetrie-Strafe (leg_symmetry). Inference-mode-kompatibel: copy_ / in-place.
        """
        in_contact = self._foot_in_contact_from_force()
        touchdown = in_contact & ~self.foot_in_contact
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        active = (cmd_speed > self.reward_cfg["feet_air_cmd_threshold"]).to(gs.tc_float)

        # feet_air_time (klassisch, legged_gym-Stil): Luftzeit je Fuß akkumulieren und beim Aufsetzen
        # (touchdown) die Luftzeit über feet_air_time_target hinaus gutschreiben → belohnt lange, klare
        # Schritte statt hektischem Trippeln. Danach Timer in Kontakt nullen.
        self.foot_air_time += self.dt
        landing = (self.foot_air_time - self.reward_cfg["feet_air_time_target"]) * touchdown.to(gs.tc_float)
        self.feet_air_time_reward.copy_(landing.sum(dim=1) * active)
        self.foot_air_time *= (~in_contact).to(gs.tc_float)  # Füße am Boden: Timer zurücksetzen

        # leg_symmetry: alternierender Gang = genau ein Fuß schwingt, der andere stützt; dann Wechsel.
        # both_air (beide in der Luft) ist im Gehen nie gültig → sofort als Flag bestrafen. both_stance
        # (Doppelstütze) ist nur kurz natürlich → Dauer akkumulieren und die Überdauer über ds_allow_time
        # hinaus bestrafen. Beides nur bei cmd_speed > feet_air_cmd_threshold (im Stand kein Wechselzwang).
        both_air = (~in_contact[:, 0]) & (~in_contact[:, 1])
        both_stance = in_contact[:, 0] & in_contact[:, 1]
        self.both_stance_time += self.dt
        self.both_stance_time *= both_stance.to(gs.tc_float)  # nur bei Doppelstütze weiterzählen, sonst 0
        ds_excess = torch.clamp(self.both_stance_time - self.ds_allow_time, min=0.0)
        self.leg_symmetry_penalty.copy_((both_air.to(gs.tc_float) + ds_excess) * active)

        self.foot_in_contact.copy_(in_contact)

    def _update_observation(self):
        obs_parts = [
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ]
        for i, part in enumerate(obs_parts):
            assert part.ndim == 2 and part.shape[0] == self.num_envs, f"obs part {i}: bad shape {part.shape}"
        self.obs_buf = torch.cat(obs_parts, dim=-1)
        assert self.obs_buf.shape[-1] == self.obs_dim

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _termination_mask(self):
        """True je Env, wenn Episode endet (Fall, zu tief, Timeout, Sim-Fehler)."""
        terminate = self.episode_length_buf > self.max_episode_length
        terminate = terminate | (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
        terminate = terminate | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
        terminate = terminate | (self.base_pos[:, 2] < 0.50)
        terminate = terminate | self.scene.rigid_solver.get_error_envs_mask()
        return terminate

    # ------------ reward functions ----------------
    # Jede Funktion gibt einen Wert pro Env zurück (0..N).
    # Beitrag zum Step-Reward: funktion() * reward_scales[name]  (scale schon * dt in __init__)

    def _reward_tracking_lin_vel(self):
        """Belohnung: vorwärts/seitwärts wie commands [vx, vy] in Spawn-Heading fahren.

        Misst quadrierten Fehler zwischen Ziel- und Ist-Geschwindigkeit im Heading-Frame
        (base_lin_vel_heading, fixiert bei Episode-Start) — nicht im rotierenden Körper-Frame.
        Verhindert den Spin-Hack: Körper-vx kann hoch bleiben, während die Welt-Bahn driftet.
        exp(-fehler / sigma) → 1.0 bei perfektem Treffer, sinkt bei Abweichung.

        Beispiel (sigma=0.25, scale=1.0, dt=0.02):
          command [0.5, 0.0], Heading-Ist [0.5, 0.0]  → fehler=0      → return 1.0   → +0.02/Step
          command [0.5, 0.0], Heading-Ist [0.3, 0.0]  → fehler=0.04   → return≈0.85  → +0.017/Step
          Spin: Körper-vx=0.5, Heading-vx=0.0         → fehler=0.25   → return≈0.37  → +0.007/Step
        """
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel_heading[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """Belohnung: Drehgeschwindigkeit wie command [yaw] einhalten.

        Wie tracking_lin_vel, aber nur die z-Achse (Drehen um Hochachse).

        Beispiel (sigma=0.25, scale=0.2, dt=0.02):
          command 0.0 rad/s, Ist 0.0  → return 1.0  → +0.002/Step
          command 0.0 rad/s, Ist 0.5  → fehler=0.25 → return≈0.37 → +0.0007/Step
        """
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_survival(self):
        """Belohnung: +1 pro Step solange keine Termination greift (Alive-Bonus).

        Nutzt dieselben Bedingungen wie _termination_mask() — auf dem Fall-/Crash-Step 0.

        Beispiel (scale=1.0, dt=0.02):
          stehend/laufend  → return 1.0 → +0.02/Step
          Kippen/Fall      → return 0.0 → 0/Step
        """
        return (~self._termination_mask()).to(gs.tc_float)

    def _reward_action_rate(self):
        """Strafe: Aktionen sollen sich nicht ruckartig ändern (glatte Bewegung).

        Summiert (letzte_aktion - aktuelle_aktion)² über alle 22 Gelenke.

        Beispiel (scale=-0.005, dt=0.02):
          alle 22 Gelenke ändern sich um 0.1 → sum=22*0.01=0.22 → return 0.22 → -0.000022/Step
          keine Änderung                    → return 0          → 0/Step
        """
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)



    def _reward_feet_air_time(self):
        """Belohnung: lange, klare Schritte (klassisch legged_gym, Event beim Aufsetzen).

        In _update_foot_contact berechnet: je Fuß die Luftzeit akkumulieren und beim Touchdown die
        Luftzeit über feet_air_time_target hinaus gutschreiben → (air_time − target) als einmaliger
        Bonus beim Aufsetzen. Belohnt längere Flugzeiten (weniger Trippeln); zu kurze Schritte
        (air_time < target) geben einen kleinen Malus. Keine Phase-Clock, kein Gauß. Gate:
        cmd_speed > feet_air_cmd_threshold (im Stand kein Reward).

        Beispiel (target=0.3, scale=5 → *dt=0.1):
          Schritt mit air=0.45 → (0.45−0.3)=0.15 → +0.015 beim Touchdown je Fuß
          Schritt mit air=0.20 → (0.20−0.3)=−0.10 → −0.010 beim Touchdown je Fuß
          Stehen (cmd≈0)        → 0
        """
        return self.feet_air_time_reward

    def _reward_feet_slip(self):
        """Strafe: Fuß rutscht/schlurft am Boden — Horizontalgeschwindigkeit des Fußes im Kontakt.

        Echtes Anti-Schlurf-Signal: ein sauber geplanter Stützfuß hat ~0 Horizontalgeschwindigkeit
        → keine Strafe; ein schleifender Fuß wird quadratisch bestraft.

        Beispiel (scale=-0.2, dt=0.02):
          Stützfuß still (v_xy≈0)        → 0           → 0/Step
          Fuß rutscht 0.3 m/s im Kontakt → 0.09        → -0.00036/Step je Fuß
        """
        foot_vel_xy = self.robot.get_links_vel(self.feet_idx_local)[:, :, :2]  # (n,2,2) Welt-xy
        slip = torch.sum(torch.square(foot_vel_xy), dim=2)  # (n,2)
        return (slip * self.foot_in_contact.to(gs.tc_float)).sum(dim=1)


    def _reward_style(self):
        """Belohnung: Live-Bewegung sieht aus wie die NPZ-Referenz (feature-matching, nicht-adversariell).

        Baut je Step dasselbe 38-dim Feature wie die Referenz (root_height, projected_gravity, dof_pos,
        dof_vel, foot_clear), standardisiert mit den Referenz-Statistiken und misst die Distanz zum
        NÄCHSTEN Referenz-Frame im Feature-Raum. Da dof_vel/foot_clear enthalten sind, kodiert das
        BEWEGUNG (eine eingefrorene Pose hat v=0 → weit weg vom Jog) — timing-/speed-agnostisch, kein
        Phasen-Clock. r = exp(−mean_sq_z / (2σ²)) ∈ (0,1]: 1 wenn die Pose+Dynamik im Schnitt < σ Std
        von einem Referenz-Frame entfernt ist.

        Beispiel (σ=1.0, scale positiv → *dt):
          Bewegung trifft Referenz-Stil    → mean_sq_z≈0   → r≈1
          im Schnitt 1 Std daneben         → mean_sq_z≈1   → r≈0.61
          völlig anderer Gang/steif        → mean_sq_z groß → r→0
        """
        foot_z = self.robot.get_links_pos(self.feet_idx_local)[:, :, 2]  # (n,2)
        foot_clear = torch.clamp(foot_z - self.style_ground_z, 0.0, 0.5)
        feat = torch.cat(
            [self.base_pos[:, 2:3], self.projected_gravity, self.dof_pos, self.dof_vel, foot_clear], dim=1
        )
        feat = (feat - self.style_mean) / self.style_std * self.style_sqrt_w  # standardisiert + gewichtet
        # nächster Referenz-Frame je Env: min gewichtete ||feat − ref_k||² (Std-Einheiten), normiert auf Σw
        min_sq = torch.cdist(feat, self.style_ref_feat).pow(2).min(dim=1).values
        mean_sq = min_sq / self.style_w_sum
        return torch.exp(-mean_sq / (2.0 * self.style_sigma**2))

    def _reward_leg_symmetry(self):
        """Strafe: Beine laufen nicht alternierend (anti-phasig: ein Fuß schwingt, der andere stützt).

        Bündelt die Links/Rechts-Symmetrie in EINEM Term. In _update_foot_contact je Step verbucht
        (leg_symmetry_penalty), nur bei cmd_speed > feet_air_cmd_threshold:
          - both_air (beide Füße in der Luft): Flugphase, im Gehen nie gültig → Flag 1.0 sofort bestraft.
          - both_stance (Doppelstütze): nur kurz natürlich → Dauer akkumuliert, bestraft wird die Überdauer
            max(0, both_stance_time − ds_allow_time).

        Beispiel (ds_allow=0.15, scale=-2 → *dt=-0.04):
          sauberer Wechselschritt (Einzelstütze) → 0          → 0/Step
          beide Füße in der Luft (Hüpfen)         → 1.0        → -0.04/Step
          Doppelstütze 0.30 s (zu lang stehend)   → excess 0.15 → -0.006/Step
          Stehen (cmd≈0)                          → 0
        """
        return self.leg_symmetry_penalty

    def _reward_dof_vel(self):
        """Strafe: hohe Gelenkgeschwindigkeiten dämpfen (Energie/Vibration), ergänzt action_rate.

        action_rate bestraft nur Aktions-Differenzen; dieser Term greift die tatsächlichen
        Gelenkgeschwindigkeiten ab → unterdrückt hochfrequentes Zittern bei konstanter Aktion.

        Beispiel (scale=-2e-4, dt=0.02):
          alle 20 Gelenke ~1 rad/s → sum≈20 → -0.00008/Step
          ruhiger Stand            → sum≈0  → 0/Step
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)



    def _reward_orientation(self):
        """Strafe: Rumpf aufrecht halten — Roll UND Pitch dämpfen (gegen Kippen/Nicken).

        Regelt Roll UND Pitch: roll² + pitch² zieht den Rumpf gegen senkrecht (0°/0°). base_euler
        liegt in Grad vor → in Radian umrechnen, damit die Skala physikalisch sinnvoll ist.

        Beispiel (scale=-5.0, dt=0.02):
          roll=0°,  pitch=0°  → 0                  → 0/Step
          roll=10°, pitch=0°  → 0.0305 rad²        → -0.0030/Step
          roll=0°,  pitch=10° → 0.0305 rad²        → -0.0030/Step
          roll=10°, pitch=10° → 0.0610 rad²        → -0.0061/Step
        """
        roll_pitch = torch.deg2rad(self.base_euler[:, :2])
        return torch.sum(torch.square(roll_pitch), dim=1)

    def _reward_ang_vel_xy(self):
        """Strafe: Roll-/Pitch-Raten des Rumpfes dämpfen (kein Kippeln/Schwanken).

        Summiert (ω_x² + ω_y²) der Körper-Winkelgeschwindigkeit.

        Beispiel (scale=-0.05, dt=0.02):
          ω_xy=0           → 0      → 0/Step
          ω_x=0.5 rad/s    → 0.25   → -0.00025/Step
        """
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_ang_vel_z(self):
        """Strafe: Yaw-Rate vom Kommando abweichen (direkte ω_z-Dämpfung, ergänzt tracking_ang_vel).

        tracking_ang_vel belohnt nur exp(-err/sigma) und bleibt bei cmd=0 bei ω_z>0 noch leicht positiv.
        Dieser Term bestraft (ω_z − cmd_yaw)² quadratisch → Spin-Hacks werden teuer.

        Beispiel (scale=-1.0, dt=0.02, cmd_yaw=0):
          ω_z=0     → 0    → 0/Step
          ω_z=0.5   → 0.25 → -0.005/Step
          ω_z=1.0   → 1.0  → -0.02/Step
        """
        return torch.square(self.base_ang_vel[:, 2] - self.commands[:, 2])

