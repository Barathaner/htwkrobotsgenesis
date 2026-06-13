import math

import genesis as gs
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

        self.total_steps = 0                                                   # monotoner globaler Step-Zähler

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
        # foot_step_quality: zerfallende On-Beat-Landequalität je Fuß (Latch bei Touchdown, sonst
        # *feet_air_decay/Step). Koppelt beide Füße: eine Landung zählt nur, soweit der ANDERE Fuß
        # zuletzt auch on-beat gelandet ist → ein schleifender Fuß kann feet_air_time nicht allein tragen.
        self.foot_step_quality = torch.zeros((num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.feet_air_decay = float(self.reward_cfg["feet_air_decay"])

        # foot_roll / flat_foot: Ankle_Pitch- und Knee_Pitch-Spaltenindizes in dof_pos (Reihenfolge = joint_names).
        jn = env_cfg["joint_names"]
        self.ankle_pitch_idx = torch.tensor(
            [jn.index("Left_Ankle_Pitch"), jn.index("Right_Ankle_Pitch")], dtype=gs.tc_int, device=gs.device
        )
        self.knee_pitch_idx = torch.tensor(
            [jn.index("Left_Knee_Pitch"), jn.index("Right_Knee_Pitch")], dtype=gs.tc_int, device=gs.device
        )
        self.foot_roll_window = int(self.reward_cfg["foot_roll_window_steps"])
        # Rolling-History der Ankle_Pitch-Winkel (je Fuß, letzte N Steps) für die Zeh-Pose vor Liftoff.
        self.ankle_pitch_hist = torch.zeros((num_envs, 2, self.foot_roll_window), dtype=gs.tc_float, device=gs.device)
        self.foot_flat_time = torch.zeros((num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.foot_roll_reward = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.foot_flat_penalty = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)

        # push_robustness (Stage 3): je Env zufällig getaktet (2–5 s), Stoß als Soll-Δv [m/s].
        self.push_enabled = bool(self.reward_cfg.get("push_enabled", False))
        self.push_interval_min_steps = max(1, int(self.reward_cfg["push_interval_min_s"] / self.dt))
        self.push_interval_max_steps = max(
            self.push_interval_min_steps + 1, int(self.reward_cfg["push_interval_max_s"] / self.dt)
        )
        # je Env Episoden-Step des nächsten Pushes (zufällig 2–5 s, dekorreliert)
        self.next_push_step = self._sample_push_interval((num_envs,))
        # Welt-Frame Linear-DOFs der Floating-Base (Index 0,1,2 = x,y,z); xy für Horizontal-Push.
        self.base_xy_dof_idx = torch.tensor([0, 1], dtype=gs.tc_int, device=gs.device)

        # Assist-Force-Curriculum: HILFSKRAFT am Rumpf-COM, die IN Kommandorichtung schiebt — hilft dem
        # Roboter anzulaufen (Anschubhilfe). Magnitude startet voll und fällt LINEAR über decay_steps auf
        # scale_min ab (kein Glocken-Anstieg mehr). Begründung/Mechanik: _apply_assist_force.
        ac = env_cfg.get("assist", {})
        self.assist_enabled = bool(ac.get("enabled", False))
        self.assist_force = float(ac.get("force", 30.0))             # [N] Schub in Kommandorichtung bei Step 0
        self.assist_decay_steps = max(1, int(ac.get("decay_steps", 100_000)))
        self.assist_scale_min = float(ac.get("scale_min", 0.0))
        self.assist_scale = 1.0 if self.assist_enabled else 0.0      # Logging vor dem 1. Step (Start: voll)
        self.assist_link_idx = [self.robot.base_link_idx]         # globaler Index des Rumpf-Links
        self.assist_force_buf = torch.zeros((num_envs, 1, 3), dtype=gs.tc_float, device=gs.device)

        # Phase-Clock (Siekmann/Margolis): periodischer Gangtakt φ∈[0,1). φ läuft jeden Step weiter;
        # sin/cos(2πφ) gehen in die Obs, damit die Policy den Takt timen kann.
        self.gait_period_steps = max(1, int(self.reward_cfg["gait_period_s"] / self.dt))
        self.gait_stance_ratio = float(self.reward_cfg["gait_stance_ratio"])
        self.gait_phase_offset = float(self.reward_cfg["gait_phase_offset"])
        self.gait_phase = torch.rand((num_envs,), dtype=gs.tc_float, device=gs.device)
        # Soll-Schwungdauer = (1 − stance_ratio) · Periode: Ziel-Luftzeit je Schritt fürs Kontaktmodell.
        self.gait_swing_time = (1.0 - self.gait_stance_ratio) * float(self.reward_cfg["gait_period_s"])
        self.feet_air_sigma = float(self.reward_cfg["feet_air_sigma"])

        self._obs_slices = {
            "base_ang_vel": self.base_ang_vel.shape[-1],
            "projected_gravity": self.projected_gravity.shape[-1],
            "commands": self.commands.shape[-1],
            # Heading-Frame xy-Geschwindigkeit (Welt-vx,vy relativ zur Spawn-Yaw): macht den
            # Drift FÜR DIE POLICY BEOBACHTBAR. projected_gravity kodiert nur roll/pitch und ist
            # yaw-invariant → ohne diesen Term kann die Policy ihre Heading-Abweichung gar nicht
            # sehen, command_accuracy wäre ein nicht lernbarer (nicht-Markovscher) Reward.
            "base_lin_vel_heading": 2,
            "dof_pos": self.dof_pos.shape[-1],
            "dof_vel": self.dof_vel.shape[-1],
            "actions": self.actions.shape[-1],
            "gait_phase": 2,  # sin/cos(2πφ)
            # L/R-Fußkontakt (0/1): macht die Stützsituation FÜR DIE POLICY beobachtbar. Bisher nur
            # von den Rewards (feet_air_time, gait_phase, feet_slip) genutzt — die Policy musste
            # Alternation/Doppelstütze aus Gelenkwinkeln raten. Mit Kontakt-State kann sie aktiv
            # Wechselschritt timen (welches Bein trägt, wann abheben) → unterstützt zweibeinigen Gang.
            "feet_contact": 2,
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
        # Assist-Force VOR scene.step() aufprägen, damit sie in die Integration dieses Steps eingeht.
        if self.assist_enabled:
            self._apply_assist_force()
        self.scene.step()

        self.episode_length_buf += 1
        self.total_steps += 1  # globaler Takt fürs Curriculum-Cooldown
        # Phase-Clock weiterdrehen: ein voller Zyklus je gait_period_steps Steps (in-place, mod 1).
        self.gait_phase.add_(1.0 / self.gait_period_steps).remainder_(1.0)
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

        if self.push_enabled:
            self._apply_push()

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
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

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
            self.foot_step_quality.zero_()
            self.ankle_pitch_hist.zero_()
            self.foot_flat_time.zero_()
            self.foot_roll_reward.zero_()
            self.foot_flat_penalty.zero_()
            self.next_push_step.copy_(self._sample_push_interval((self.num_envs,)))
            self.gait_phase.uniform_(0.0, 1.0)
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
            self.foot_step_quality.masked_fill_(envs_idx[:, None], 0.0)
            self.ankle_pitch_hist.masked_fill_(envs_idx[:, None, None], 0.0)
            self.foot_flat_time.masked_fill_(envs_idx[:, None], 0.0)
            self.foot_roll_reward.masked_fill_(envs_idx, 0.0)
            self.foot_flat_penalty.masked_fill_(envs_idx, 0.0)
            torch.where(envs_idx, self._sample_push_interval((self.num_envs,)), self.next_push_step,
                        out=self.next_push_step)
            torch.where(envs_idx, torch.rand_like(self.gait_phase), self.gait_phase, out=self.gait_phase)

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
        # Assist-Force-Curriculum: aktuelle Stützstärke mitloggen
        self.extras["episode"]["assist_scale"] = torch.tensor(
            self.assist_scale, dtype=gs.tc_float, device=gs.device
        )

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
        """Fuß-Kontakt via net contact force; zählt Touchdowns (Flanke hoch→Kontakt).

        Akkumuliert außerdem foot_air_time (dt pro Step ohne Bodenkontakt) und verbucht beim
        Aufsetzen die Lande-Belohnung (feet_air_time). Inference-mode-kompatibel: copy_ / in-place.
        """
        in_contact = self._foot_in_contact_from_force()
        touchdown = in_contact & ~self.foot_in_contact

        # feet_air_time (phasen-gekoppelt): dt für die Luftphase addieren, dann beim Touchdown nur
        # belohnen, wenn die Landung INS STANCE-FENSTER der Phase-Clock fällt (richtiger Takt) UND die
        # vorausgegangene Luftzeit nahe der Soll-Schwungdauer (1−stance_ratio)·Periode liegt (Gauß).
        # → bewertet OB der Schritt zum Takt passt, nicht bloß die Dauer. Danach Timer in Kontakt nullen.
        self.foot_air_time += self.dt
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        active = (cmd_speed > self.reward_cfg["feet_air_cmd_threshold"]).to(gs.tc_float)
        on_beat_land = touchdown & self._desired_stance()  # Aufsetzen im richtigen Phase-Fenster
        air_quality = torch.exp(-torch.square(self.foot_air_time - self.gait_swing_time) / self.feet_air_sigma)
        # Symmetrie-Kopplung: Landung je Fuß × On-Beat-Güte des ANDEREN Fußes (foot_step_quality, vor
        # Update gelesen) → schleift ein Fuß (Güte→0), zählt auch die gute Landung des anderen kaum.
        other_quality = self.foot_step_quality[:, [1, 0]]
        landing = air_quality * on_beat_land.to(gs.tc_float) * other_quality
        self.feet_air_time_reward.copy_(landing.sum(dim=1) * active)
        # foot_step_quality: pro Step zerfallen, bei On-Beat-Landung auf die frische air_quality latchen
        self.foot_step_quality.mul_(self.feet_air_decay)
        self.foot_step_quality.copy_(torch.where(on_beat_land, air_quality, self.foot_step_quality))
        self.foot_air_time *= (~in_contact).to(gs.tc_float)  # Füße am Boden: Timer zurücksetzen

        # --- foot_roll (Heel-to-Toe) & flat_foot über Ankle_Pitch ---
        liftoff = (~in_contact) & self.foot_in_contact
        ankle_pitch = self.dof_pos[:, self.ankle_pitch_idx]  # (n,2)
        roll_active = (cmd_speed > self.reward_cfg["foot_roll_cmd_threshold"]).to(gs.tc_float)
        sigma = self.reward_cfg["foot_roll_sigma"]

        # Rolling-History fortschreiben (älteste raus, aktuellen Step rein).
        self.ankle_pitch_hist[:, :, :-1] = self.ankle_pitch_hist[:, :, 1:].clone()
        self.ankle_pitch_hist[:, :, -1] = ankle_pitch

        # Touchdown → Ferse zuerst (heel_target); Liftoff → Zeh-Abdruck über die letzten N Stance-Steps.
        heel_rew = torch.exp(
            -torch.square(ankle_pitch - self.reward_cfg["ankle_heel_target"]) / sigma
        ) * touchdown.to(gs.tc_float)
        toe_rew = torch.exp(
            -torch.square(self.ankle_pitch_hist - self.reward_cfg["ankle_toe_target"]) / sigma
        ).mean(dim=2) * liftoff.to(gs.tc_float)
        self.foot_roll_reward.copy_((heel_rew + toe_rew).sum(dim=1) * roll_active)

        # flat_foot: |Ankle_Pitch| zu klein während Bodenkontakt → Plattfuß-Dauer akkumulieren.
        flat = (ankle_pitch.abs() < self.reward_cfg["foot_flat_threshold"]) & in_contact
        self.foot_flat_time += self.dt
        self.foot_flat_time *= flat.to(gs.tc_float)  # nur bei flach-in-Kontakt weiterzählen, sonst 0
        flat_excess = torch.clamp(self.foot_flat_time - self.reward_cfg["foot_flat_time"], min=0.0)
        self.foot_flat_penalty.copy_(flat_excess.sum(dim=1) * roll_active)

        self.foot_in_contact.copy_(in_contact)

    def _desired_stance(self):
        """Soll-Bodenkontakt je Fuß (n,2) aus der Phase-Clock: True = soll Stance, False = soll Swing.

        Gemeinsames Kontaktmodell für gait_phase (Off-Beat-Strafe) UND die phasen-gekoppelte
        Lande-Belohnung (feet_air_time): linkes Bein folgt gait_phase, rechtes um gait_phase_offset
        versetzt; Stance solange Phase < gait_stance_ratio.
        """
        phase = torch.stack(
            [self.gait_phase, (self.gait_phase + self.gait_phase_offset).remainder(1.0)], dim=1
        )
        return phase < self.gait_stance_ratio

    def _sample_push_interval(self, batch_shape):
        """Zufälliges Push-Intervall [steps] im konfigurierten Band (je Env dekorreliert)."""
        return torch.randint(
            self.push_interval_min_steps,
            self.push_interval_max_steps + 1,
            batch_shape,
            dtype=gs.tc_int,
            device=gs.device,
        )

    def _apply_push(self):
        """Stage-3 Robustheit: horizontaler Stoß auf den Rumpf als Soll-Δv [m/s] (NICHT roher Impuls).

        Je Env eigener Zeitplan (next_push_step, 2–5 s zufällig → dekorrelierte Störungen statt
        batch-synchroner Schläge). Beim Auslösen wird ein Δv in zufälliger Horizontalrichtung
        (push_vel_min..max) auf die Welt-xy-Geschwindigkeit der Base addiert; das Δv ist so bemessen,
        dass es aufholbar bleibt.
        """
        due = self.episode_length_buf >= self.next_push_step
        envs_idx = due.nonzero(as_tuple=False).reshape(-1)
        if envs_idx.numel() == 0:
            return
        n = envs_idx.numel()
        angle = torch.rand(n, dtype=gs.tc_float, device=gs.device) * (2.0 * math.pi)
        lo = self.reward_cfg["push_vel_min"]
        hi = self.reward_cfg["push_vel_max"]
        dv = torch.rand(n, dtype=gs.tc_float, device=gs.device) * (hi - lo) + lo
        cur = self.robot.get_vel()[envs_idx]  # Welt-Frame Linear-Geschwindigkeit der Base
        new_xy = torch.stack([cur[:, 0] + dv * torch.cos(angle), cur[:, 1] + dv * torch.sin(angle)], dim=1)
        self.robot.set_dofs_velocity(new_xy, dofs_idx_local=self.base_xy_dof_idx, envs_idx=envs_idx)
        # nächsten Push für genau diese Envs neu auslosen
        self.next_push_step[envs_idx] = self.episode_length_buf[envs_idx] + self._sample_push_interval((n,))

    def _apply_assist_force(self):
        """Hilfskraft-Curriculum: konstanter Schub am Rumpf-COM IN Kommandorichtung — eine Anschubhilfe,
        die dem Roboter das Anlaufen erleichtert (Gegenteil eines Widerstands). Gegenstück zum Push
        (_apply_push): Push STÖRT impulsiv, diese Hilfe SCHIEBT gerichtet und blendet über die Zeit aus.

          force = scale · force · cmd_dir        Schub entlang der Kommandorichtung (Einheitsvektor);
                                                 inaktiv bei ~null Kommando.

        Am Rumpf-COM (ref='root_com' → reine Linearkraft, KEIN Stör-Drehmoment). NICHT in der
        Observation → die Policy erlebt nur die veränderte Dynamik, die über das Curriculum verschwindet.

        Heading == Welt gilt bei Identity-Spawn (base_init_quat = Einheit); cmd_dir (Heading-Frame) ist
        dann Welt-xy. Bei späterem Random-Yaw-Spawn müsste cmd_dir vor dem Anwenden nach Welt rotiert werden.

        Skala = LINEARER Abbau über die Zeit (start stark, dann ausblenden):
          scale = max(scale_min, 1 − total_steps/decay_steps)

        Beispiel (force=30, decay_steps=1e5):
          total_steps=0     → scale=1.00 → F≈+30 N (volle Anschubhilfe vorwärts)
          total_steps=5e4   → scale=0.50 → F≈+15 N
          total_steps≥1e5   → scale=0    → frei laufen
        """
        self.assist_scale = max(self.assist_scale_min, 1.0 - self.total_steps / self.assist_decay_steps)
        if self.assist_scale <= 0.0:
            return
        cmd_speed = torch.norm(self.commands[:, :2], dim=1, keepdim=True)
        cmd_dir = self.commands[:, :2] / (cmd_speed + 1e-6)
        force_xy = self.assist_scale * self.assist_force * cmd_dir * (cmd_speed > 1e-3).to(gs.tc_float)
        self.assist_force_buf[:, 0, 0] = force_xy[:, 0]
        self.assist_force_buf[:, 0, 1] = force_xy[:, 1]
        self.scene.rigid_solver.apply_links_external_force(
            self.assist_force_buf, links_idx=self.assist_link_idx, ref="root_com"
        )

    def _update_observation(self):
        phase_2pi = self.gait_phase * (2.0 * math.pi)
        obs_parts = [
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            # Heading-Frame xy-Geschwindigkeit (gleiche Skala wie commands) → Policy sieht ihre
            # Ist-Geschwindigkeit in Spawn-Richtung und kann Drift aktiv ausregeln (schließt den
            # Regelkreis für command_accuracy, der sonst keinen beobachtbaren Fehler hätte).
            self.base_lin_vel_heading[:, :2] * self.obs_scales["lin_vel"],
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
            torch.stack([torch.sin(phase_2pi), torch.cos(phase_2pi)], dim=1),  # Phase-Clock
            # L/R-Fußkontakt, zentriert auf {-0.5, +0.5} (nullnah wie die übrigen Obs-Terme).
            self.foot_in_contact.to(gs.tc_float) - 0.5,
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
        terminate = terminate | (self.base_pos[:, 2] < 0.35)
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

    def _reward_lin_vel_z(self):
        """Strafe: nicht in der Höhe hopsen (vz soll ≈ 0).

        Quadriert die vertikale Geschwindigkeit des Rumpfes.

        Beispiel (scale=-1.0, dt=0.02):
          vz=0.0   → return 0      → 0/Step
          vz=0.1   → return 0.01   → -0.01/Step
          vz=0.5   → return 0.25   → -0.25/Step
        """
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        """Strafe: Aktionen sollen sich nicht ruckartig ändern (glatte Bewegung).

        Summiert (letzte_aktion - aktuelle_aktion)² über alle 22 Gelenke.

        Beispiel (scale=-0.005, dt=0.02):
          alle 22 Gelenke ändern sich um 0.1 → sum=22*0.01=0.22 → return 0.22 → -0.000022/Step
          keine Änderung                    → return 0          → 0/Step
        """
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        """Strafe: Gelenke sollen nahe der Default-Standpose bleiben.

        Summiert |ist_winkel - default_winkel| über alle 22 Gelenke [rad].

        Beispiel (scale=-0.1, dt=0.02):
          alle Gelenke 0.05 rad daneben → sum=22*0.05=1.1 → return 1.1 → -0.0022/Step
          exakt Default-Pose              → return 0         → 0/Step
        """
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        """Strafe: Rumpfhöhe (Trunk z) soll base_height_target halten.

        Quadriert Abweichung in Metern.

        Beispiel (target=0.53 m, scale=-50.0, dt=0.02 → effektiv -1.0):
          z=0.53 → return 0       → 0/Step
          z=0.54 → return 0.0001  → -0.0001/Step
          z=0.50 → return 0.0009  → -0.0009/Step
          z=0.45 → return 0.0064  → -0.0064/Step
        """
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_feet_air_time(self):
        """Belohnung: phasen-gekoppelter, kadenz-treuer Schritt (Event beim Aufsetzen).

        In _update_foot_contact berechnet (gemeinsames Kontaktmodell mit gait_phase): beim Touchdown
        nur belohnt, wenn die Landung INS STANCE-FENSTER der Phase-Clock fällt (richtiger Takt), und
        zwar exp(−(air_time − gait_swing_time)² / feet_air_sigma) — voll nur, wenn die Luftzeit nahe der
        Soll-Schwungdauer (1−stance_ratio)·gait_period_s liegt. Misst also OB der Schritt zum Takt passt,
        nicht bloß die Dauer. Off-Beat oder zu kurze/lange Luftzeit → kaum Reward. Gate: cmd_speed >
        feet_air_cmd_threshold (im Stand kein Reward); Alternation folgt aus dem Phasenversatz.

        Symmetrie-Kopplung: jede Landung wird mit der zerfallenden On-Beat-Güte des ANDEREN Fußes
        (foot_step_quality) multipliziert → ein schleifender/dragender Fuß (Güte→0) zieht den Term
        gegen 0, auch wenn das andere Bein sauber schreitet. Ein Fuß kann ihn so nicht allein tragen.

        Beispiel (swing_time=0.44, sigma=0.02, scale=7 → *dt=0.14):
          beide on-beat (other_q≈0.8), air=0.44 → 1.0·0.8 ≈ 0.8 → +0.11/Step
          ein Fuß dragt (other_q≈0)              → ≈ 0        → 0
          Off-Beat-Landung / cmd≈0               → 0
        """
        return self.feet_air_time_reward

    def _reward_swing_clearance(self):
        """Strafe: Fuß bleibt im Swing-Fenster der Phase-Clock am Boden (Schleif-/Drag-Fuß).

        Schärfere, kontinuierliche Variante der gait_phase-Off-Beat-Strafe: bestraft je Fuß den
        Höhen-FEHLBETRAG max(0, feet_air_height_target − z), wenn die Phase-Clock SWING vorschreibt
        (~_desired_stance). Ein Fuß, der schreiten SOLL, aber tief am Boden schleift, zahlt proportional
        zur Tiefe; ein sauber abhebender Fuß (z ≥ target) zahlt 0. Gate: cmd_speed > gait_cmd_threshold.

        Beispiel (target=0.12, scale=-10 → *dt=-0.2):
          Schwungfuß z=0.12 (sauber)  → 0       → 0/Step
          Drag-Fuß  z=0.05 im Swing   → 0.07    → -0.014/Step je Fuß
          Stehen (cmd≈0)              → 0
        """
        foot_z = self.robot.get_links_pos(self.feet_idx_local)[:, :, 2]  # (n,2)
        desired_swing = (~self._desired_stance()).to(gs.tc_float)
        shortfall = torch.clamp(self.reward_cfg["feet_air_height_target"] - foot_z, min=0.0)
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        active = (cmd_speed > self.reward_cfg["gait_cmd_threshold"]).to(gs.tc_float)
        return (shortfall * desired_swing).sum(dim=1) * active

    def _reward_foot_roll(self):
        """Belohnung: sauberes Heel-to-Toe-Abrollen (kein separater Zeh/Hacken-Link → Ankle_Pitch als Proxy).

        In _update_foot_contact je Fuß als Event-Reward verbucht (analog feet_air_time):
          Touchdown → Ankle_Pitch nahe ankle_heel_target (-0.15 rad, Ferse zuerst),
          Liftoff   → Ankle_Pitch der letzten foot_roll_window Stance-Steps nahe ankle_toe_target (+0.10 rad).
        exp(-(Δ)²/foot_roll_sigma) je Event, über beide Füße summiert; nur wenn cmd_speed > foot_roll_cmd_threshold.

        Beispiel (heel=-0.15, sigma=0.05, scale=0.5 → *dt=0.01):
          Touchdown mit Ankle=-0.15 → exp(0)=1.0      → contrib≈+0.01 (einmalig)
          Touchdown mit Ankle=-0.30 → exp(-0.0225/0.05)≈0.64 → +0.0064
          Stehen (cmd≈0)            → 0
        """
        return self.foot_roll_reward

    def _reward_flat_foot(self):
        """Strafe: Fuß bleibt während Bodenkontakt zu lange flach (|Ankle_Pitch| < foot_flat_threshold).

        foot_flat_time akkumuliert dt solange ein Fuß flach UND am Boden ist (Reset sonst), siehe
        _update_foot_contact. Bestraft wird die Überdauer über foot_flat_time hinaus, je Fuß
          excess = max(0, flat_time - foot_flat_time),  über beide summiert; nur cmd_speed > foot_roll_cmd_threshold.

        Beispiel (thresh=0.03 rad, foot_flat_time=0.4 s, scale=-0.2 → *dt=0.004):
          Fuß 0.30 s flach am Boden (normaler Stützfuß) → excess=0    → 0/Step
          Fuß 0.60 s flach hängend (Schlurfen)         → excess=0.20 → -0.0008/Step (wächst weiter)
          Stehen (cmd≈0)                                → 0
        """
        return self.foot_flat_penalty

    def _reward_feet_slip(self):
        """Strafe: Fuß rutscht/schlurft am Boden — Horizontalgeschwindigkeit des Fußes im Kontakt.

        Echtes Anti-Schlurf-Signal (der Höhen-Proxy allein sieht Rutschen nicht): ein sauber
        geplanter Stützfuß hat ~0 Horizontalgeschwindigkeit → keine Strafe; ein schleifender Fuß
        wird quadratisch bestraft. Ergänzt flat_foot (das jetzt nur grob dauer-flache Füße erfasst).

        Beispiel (scale=-0.2, dt=0.02):
          Stützfuß still (v_xy≈0)        → 0           → 0/Step
          Fuß rutscht 0.3 m/s im Kontakt → 0.09        → -0.00036/Step je Fuß
        """
        foot_vel_xy = self.robot.get_links_vel(self.feet_idx_local)[:, :, :2]  # (n,2,2) Welt-xy
        slip = torch.sum(torch.square(foot_vel_xy), dim=2)  # (n,2)
        return (slip * self.foot_in_contact.to(gs.tc_float)).sum(dim=1)

    def _reward_close_feet(self):
        """Strafe: Füße kommen sich seitlich zu nah (Kollisions-/Verhakungsschutz, SPRINT-Stil).

        Seitlicher Abstand = |y_links − y_rechts| im BASIS-Frame (rotationsinvariant, anders als
        Welt-y bei gedrehtem Rumpf). max(0, d_min − Abstand) → greift erst unter dem Mindestabstand
        d_min und wird hart gewichtet (scale −100) gegen kreuzende/kollidierende Füße beim Gehen/Drehen.

        Beispiel (d_min=0.15, scale=-100 → *dt=-2.0):
          Abstand 0.18 m → 0     → 0/Step
          Abstand 0.10 m → 0.05  → -0.10/Step
        """
        foot_pos = self.robot.get_links_pos(self.feet_idx_local)  # (n,2,3) Welt
        inv_bq = inv_quat(self.base_quat)
        left_base = transform_by_quat(foot_pos[:, 0] - self.base_pos, inv_bq)   # (n,3) im Basis-Frame
        right_base = transform_by_quat(foot_pos[:, 1] - self.base_pos, inv_bq)
        lateral_dist = torch.abs(left_base[:, 1] - right_base[:, 1])
        return torch.clamp(self.reward_cfg["close_feet_d_min"] - lateral_dist, min=0.0)

    def _reward_feet_air_height(self):
        """Belohnung: Schwungfuß hält eine saubere Soll-Flughöhe (Gauß um feet_air_height_target).

        Nur für Füße in der Luft (~foot_in_contact): exp(−|z_link − h_target|/sigma) → 1.0 bei exakter
        Höhe, fällt zu beiden Seiten ab → definierte Bodenfreiheit (kein Schleifen, kein Überheben).
        z ist die Höhe des Fuß-LINK-Ursprungs (sitzt ~5-6 cm über der Sohle) → h_target entsprechend
        höher als die reine Sohlen-Clearance wählen.

        Beispiel (h_target=0.12, sigma=0.03, scale=5 → *dt=0.1):
          Schwungfuß z=0.12 → exp(0)=1.0           → +0.1/Step je Fuß
          Schwungfuß z=0.09 → exp(-0.03/0.03)≈0.37 → +0.037/Step
          Fuß am Boden      → is_air=0             → 0
        """
        foot_z = self.robot.get_links_pos(self.feet_idx_local)[:, :, 2]  # (n,2)
        is_air = (~self.foot_in_contact).to(gs.tc_float)
        h_target = self.reward_cfg["feet_air_height_target"]
        sigma = self.reward_cfg["feet_air_height_sigma"]
        return (torch.exp(-torch.abs(foot_z - h_target) / sigma) * is_air).sum(dim=1)

    def _reward_feet_parallel(self):
        """Belohnung: beide Füße zeigen in dieselbe Richtung (parallel) → kein Spreizen/Einwärtsdrehen.

        Vergleicht die in die Horizontale projizierten Vorwärtsachsen (lokale +x) beider Füße: bei
        parallelen Füßen sind die normierten xy-Vektoren gleich → Differenz 0. exp(−||Δ||²/sigma).
        Yaw/Heading-basiert → robust gegen das Anstellen (Pitch) des Schwungfußes im Schritt.

        Beispiel (sigma=0.1, scale=1.0 → *dt=0.02):
          Füße parallel (Δ≈0) → exp(0)=1.0    → +0.02/Step
          Füße ~25° verdreht  → ||Δ||²≈0.19   → exp(-1.9)≈0.15
        """
        foot_quat = self.robot.get_links_quat(self.feet_idx_local)  # (n,2,4)
        world_fwd = torch.tensor([1.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
        fwd = transform_by_quat(world_fwd, foot_quat.reshape(-1, 4)).reshape(-1, 2, 3)[:, :, :2]  # (n,2,2) xy
        fwd = fwd / (torch.norm(fwd, dim=2, keepdim=True) + 1e-6)
        diff = torch.sum(torch.square(fwd[:, 0] - fwd[:, 1]), dim=1)  # (n,)
        return torch.exp(-diff / self.reward_cfg["feet_parallel_sigma"])

    def _reward_feet_guidance(self):
        """Strafe: Schwungfuß unterschreitet die Soll-Flugzeit t_target (Mindest-Airtime, SPRINT-Stil).

        Solange ein Fuß in der Luft ist UND seine bisherige foot_air_time noch < t_target liegt, wird
        die Lücke max(0, t_target − air_time) bestraft → drängt zu längeren, klaren Schritten statt
        hektischem Trippeln. foot_air_time wird in _update_foot_contact geführt (Reset bei Kontakt).
        t_target folgt dem Paper aus Schrittfrequenz/Swing-Anteil (D_swing/f_cmd).

        Beispiel (t_target=0.16, scale=-15 → *dt=-0.3):
          Schwungfuß air_time=0.05 → Lücke 0.11 → -0.033/Step je Fuß
          Schwungfuß air_time≥0.16 → Lücke 0    → 0
          Fuß am Boden             → is_air=0   → 0
        """
        is_air = (~self.foot_in_contact).to(gs.tc_float)
        gap = torch.clamp(self.reward_cfg["feet_guidance_t_target"] - self.foot_air_time, min=0.0)
        return (gap * is_air).sum(dim=1)

    def _reward_gait_phase(self):
        """Strafe: Fuß-Kontakt passt nicht zum Phase-Clock-Takt (Siekmann/Margolis-Stil).

        Nutzt dasselbe Kontaktmodell wie die phasen-gekoppelte feet_air_time-Belohnung (_desired_stance):
        bestraft je Fuß die Abweichung Soll-Kontakt vs. Ist-Kontakt (Höhen-Proxy), nur bei
        cmd_speed > gait_cmd_threshold (im Stand kein Takt-Zwang). gait_phase liefert die kontinuierliche
        Off-Beat-Strafe, feet_air_time die positive Belohnung fürs korrekte Stepping → zusammen ein Modell.

        Beispiel (offset=0.5, stance_ratio=0.6, scale=-0.8 → *dt=-0.016):
          mismatch=1 (ein Fuß off-beat) → -0.016/Step
          cmd≈0 (Stehen)                → 0
        """
        mismatch = (self._desired_stance() != self.foot_in_contact).to(gs.tc_float).sum(dim=1)
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        active = (cmd_speed > self.reward_cfg["gait_cmd_threshold"]).to(gs.tc_float)
        return mismatch * active

    def _reward_dof_vel(self):
        """Strafe: hohe Gelenkgeschwindigkeiten dämpfen (Energie/Vibration), ergänzt action_rate.

        action_rate bestraft nur Aktions-Differenzen; dieser Term greift die tatsächlichen
        Gelenkgeschwindigkeiten ab → unterdrückt hochfrequentes Zittern bei konstanter Aktion.

        Beispiel (scale=-2e-4, dt=0.02):
          alle 20 Gelenke ~1 rad/s → sum≈20 → -0.00008/Step
          ruhiger Stand            → sum≈0  → 0/Step
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_torso_pitch(self):
        """Strafe: Rumpf-Neigung via projected_gravity gx — zieht aktiv Richtung gf (SPRINT r_reg).

        gx = Schwerkraft-Komponente im Körper-x (positiv = leicht nach vorne geneigt).
        gf = torso_pitch_target (fix), gb = torso_pitch_backward (0).
        rt = (max(0, gx−gf) + max(0, gf−gx) + max(0, gb−gx))²
          → Minimum bei gx=gf; unter gf wird aktiv gezogen; über gf und Rück-Neigung extra bestraft.

        Beispiel (gf=0.1, gb=0, scale=-10, dt=0.02):
          gx=0.10 (Soll)          → term=0     → 0/Step
          gx=0.05 (zu wenig)      → term=0.05  → sq=0.0025 → -0.0005/Step
          gx=0.0  (aufrecht)      → term=0.1   → sq=0.01   → -0.002/Step
          gx=-0.05 (Rück-Neigung) → term=0.25  → sq=0.0625 → -0.0125/Step
          gx=0.15 (zu weit vorne) → term=0.05  → sq=0.0025 → -0.0005/Step
        """
        gx = self.projected_gravity[:, 0]
        gf = self.reward_cfg["torso_pitch_target"]
        gb = self.reward_cfg["torso_pitch_backward"]
        term = (
            torch.clamp(gx - gf, min=0.0)
            + torch.clamp(gf - gx, min=0.0)
            + torch.clamp(gb - gx, min=0.0)
        )
        return torch.square(term)

    def _reward_orientation(self):
        """Strafe: seitliches Kippen dämpfen (nur Roll — Pitch über torso_pitch asymmetrisch).

        base_euler liegt in Grad vor → in Radian umrechnen, damit die Skala physikalisch sinnvoll ist.

        Beispiel (scale=-2.0, dt=0.02):
          roll=0°   → 0           → 0/Step
          roll=10°  → 0.0305 rad² → -0.0012/Step
          roll=20°  → 0.1220 rad² → -0.0049/Step
        """
        roll = torch.deg2rad(self.base_euler[:, 0])
        return torch.square(roll)

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

    def _reward_support_pose(self):
        """Belohnung: leichte Kniebeugung [support_knee_min, support_knee_max] beim Stehen/langsamen Gehen.

        Federnde Stützpose (Knie nicht durchgestreckt) → push-resistenter Stand. Nur wenn
        cmd_speed < support_pose_cmd_speed; je Knie 1.0 wenn im Band, sonst exp-Abfall zum Bandrand.

        Beispiel (band=[0.4,0.8], scale=0.2 → *dt=0.004):
          beide Knie 0.6 rad (im Band) → 2.0 → +0.008/Step
          beide Knie 0.2 rad (zu steif)→ exp(-(0.2)²/0.05)*2≈0.91 → +0.0018/Step
          schnelles Gehen (cmd hoch)   → 0
        """
        knee = self.dof_pos[:, self.knee_pitch_idx]  # (n,2)
        lo = self.reward_cfg["support_knee_min"]
        hi = self.reward_cfg["support_knee_max"]
        below = torch.clamp(lo - knee, min=0.0)
        above = torch.clamp(knee - hi, min=0.0)
        dist2 = torch.square(below + above)  # 0 im Band, sonst quadratischer Abstand zum nächsten Rand
        in_band_score = torch.exp(-dist2 / self.reward_cfg["support_pose_sigma"]).sum(dim=1)
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        active = (cmd_speed < self.reward_cfg["support_pose_cmd_speed"]).to(gs.tc_float)
        return in_band_score * active


# if __name__ == "__main__":
#     gs.init(backend=gs.gpu, logging_level="warning")
#     config_path = os.path.join(os.path.dirname(__file__), "config", "k1_env.yaml")
#     with open(config_path, "r") as f:
#         all_cfg = yaml.safe_load(f)
#     env = K1Env(
#         1,
#         env_cfg=all_cfg["env_cfg"],
#         obs_cfg=all_cfg["obs_cfg"],
#         reward_cfg=all_cfg["reward_cfg"],
#         command_cfg=all_cfg["command_cfg"],
#         show_viewer=False,
#     )
#     print("OK", env.robot.n_dofs, env.robot.n_links)
#     print("num_actions", env.num_actions, "obs_dim", env.obs_dim)
#     print("Run tests: .venv/bin/python walk/test_k1_env.py --test a")
