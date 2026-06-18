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
        self.rand_cfg = env_cfg.get("randomization", {})

        scene_kwargs: dict = {
            "sim_options": gs.options.SimOptions(dt=self.dt, substeps=2),
            "rigid_options": gs.options.RigidOptions(
                enable_self_collision=True,
                max_collision_pairs=40,
                tolerance=1e-5,
                batch_dofs_info=bool(self.rand_cfg),
                batch_links_info=bool(self.rand_cfg),
            ),
            "viewer_options": gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
            ),
            "show_viewer": show_viewer,
        }
        if record_camera:
            scene_kwargs["vis_options"] = gs.options.VisOptions(rendered_envs_idx=list(range(num_envs)))
        self.scene = gs.Scene(**scene_kwargs)
        vcfg_vis = env_cfg.get("video", {})
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="models/K1/K1_22dof.urdf",
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            ),
            surface=gs.surfaces.Default(
                color=tuple(float(c) for c in vcfg_vis.get("robot_color", [0.82, 0.10, 0.10])),
                roughness=float(vcfg_vis.get("robot_roughness", 0.6)),
            ),
        )
        self.hero_ghost = None
        if record_camera and env_cfg.get("video", {}).get("hero_ghost", True):
            from k1_hero_ghost import HeroGhost

            self.hero_ghost = HeroGhost(self.scene, env_cfg, reward_cfg, num_envs)
        if show_viewer:
            from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

            self.scene.viewer.add_plugin(ImGuiOverlayPlugin())

        self.cam = None
        if record_camera:
            vcfg = env_cfg.get("video", {})
            if num_envs > 1:
                cam_pos = tuple(vcfg.get("multi_camera_pos", [0.0, -9.0, 5.0]))
                cam_lookat = tuple(vcfg.get("multi_camera_lookat", [0.0, 0.0, 0.5]))
                cam_fov = vcfg.get("multi_camera_fov", 55)
            else:
                cam_pos = tuple(vcfg.get("camera_pos", [3.0, -1.0, 1.5]))
                cam_lookat = tuple(vcfg.get("camera_lookat", [0.0, 0.0, 0.5]))
                cam_fov = vcfg.get("camera_fov", 30)
            self.cam = self.scene.add_camera(
                res=tuple(vcfg.get("res", [640, 480])),
                pos=cam_pos,
                lookat=cam_lookat,
                fov=cam_fov,
                GUI=False,
            )

        build_kwargs: dict = {"n_envs": num_envs}
        if record_camera and num_envs > 1:
            vcfg = env_cfg.get("video", {})
            build_kwargs["env_spacing"] = tuple(vcfg.get("env_spacing", [3.0, 3.0]))
            build_kwargs["n_envs_per_row"] = 2
        self.scene.build(**build_kwargs)
        self.base_link_idx_local = self.robot.links[0].idx_local

        # Verfolgungskamera: hält den anfänglichen Versatz Kamera→Rumpf und schwenkt mit dem
        # Roboter mit (lookat folgt dem Rumpf), statt starr zu stehen. update_following() muss
        # je Frame VOR cam.render() aufgerufen werden (siehe k1_train_runner._record_video).
        # smoothing ∈ (0,1): EMA-Glättung (höher = weicher/träger); None = exaktes Tracking.
        if self.cam is not None and num_envs == 1:
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
        self.nom_kp = torch.tensor(kp, dtype=gs.tc_float, device=gs.device)
        self.nom_kd = torch.tensor(kd, dtype=gs.tc_float, device=gs.device)

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

        # Command-Tracking auf der ZEITGEMITTELTEN (EMA-)Geschwindigkeit statt der momentanen: eine
        # natürliche Gangart pendelt innerhalb jedes Schritts → gegen ein konstantes Command kann der
        # Momentanwert nie ≈1 erreichen. Über ein Schrittzyklus-Fenster gemittelt fällt die Oszillation
        # raus → die perfekte Referenz (korrektes MITTEL) wird zum Reward-Optimum. tracking misst das
        # "Was" (mittlere Sollgeschw.), style das "Wie" (Dynamik je Frame) — kein Konflikt mehr.
        self.tracking_ema_window_s = float(reward_cfg.get("tracking_ema_window_s", 0.5))
        self.vel_ema_alpha = math.exp(-self.dt / self.tracking_ema_window_s)
        self.lin_vel_ema = torch.zeros((num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.ang_vel_z_ema = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)

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
        # foot_step_quality: zerfallende On-Beat-Landequalität je Fuß; koppelt Alternation (s. _update_foot_contact).
        self.foot_step_quality = torch.zeros((num_envs, 2), dtype=gs.tc_float, device=gs.device)

        # leg_symmetry: Strafe für nicht-alternierenden Gang (siehe _reward_leg_symmetry). Sowohl die
        # Doppelstütze (beide am Boden) als auch die Flugphase (beide in der Luft) sind als kurze
        # Übergänge natürlich (Doppelstütze beim Gehen, Flugphase beim Laufen/Jog) → es wird jeweils
        # nur die ÜBERDAUER über ein erlaubtes Fenster bestraft, nicht das Auftreten an sich.
        self.both_stance_time = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.both_air_time = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.leg_symmetry_penalty = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        # leg_symmetry: erlaubte Doppelstütz-/Flugdauer [s]. Die Timer zählen je eine
        # zusammenhängende Doppelstütze bzw. Flugphase; nur die Überdauer darüber hinaus wird bestraft.
        self.ds_allow_time = float(self.reward_cfg["leg_symmetry_ds_allow_s"])
        self.flight_allow_time = float(self.reward_cfg.get("leg_symmetry_flight_allow_s", 0.20))

        # Phase-Clock (Siekmann/Margolis): φ∈[0,1), feste Periode; sin/cos(2πφ) in Obs.
        self.gait_period_steps = max(1, int(reward_cfg["gait_period_s"] / self.dt))
        self.gait_stance_ratio = float(reward_cfg["gait_stance_ratio"])
        self.gait_phase_offset = float(reward_cfg["gait_phase_offset"])
        self.gait_swing_time = (1.0 - self.gait_stance_ratio) * float(reward_cfg["gait_period_s"])
        self.feet_air_sigma = float(reward_cfg["feet_air_sigma"])
        self.feet_air_decay = float(reward_cfg["feet_air_decay"])
        self.gait_phase = torch.rand((num_envs,), dtype=gs.tc_float, device=gs.device)

        # Random push state (always allocated; only used when push_enabled=True)
        self.push_force_buf = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.push_steps_remaining = torch.zeros((num_envs,), dtype=gs.tc_int, device=gs.device)

        # Domain randomization state buffers (per-env; updated each episode reset)
        if self.rand_cfg:
            n_links = self.robot.n_links
            self.curr_friction_ratios = torch.ones((num_envs, n_links), dtype=gs.tc_float, device=gs.device)
            self.curr_mass_shifts = torch.zeros((num_envs, n_links), dtype=gs.tc_float, device=gs.device)
            self.curr_com_shifts = torch.zeros((num_envs, n_links, 3), dtype=gs.tc_float, device=gs.device)
            self.curr_motor_strength = torch.ones((num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
            # Original link masses for ratio-based mass randomization (shape: n_links)
            self.orig_link_masses = torch.tensor(
                [link.inertial_mass for link in self.robot.links], dtype=gs.tc_float, device=gs.device
            )

        # style (feature-matching Style-Reward): belohnt Nähe der Live-Bewegung zur NPZ-Referenz im
        # Feature-Raum (nicht-adversariell, timing-/speed-agnostisch). Setup nur wenn aktiviert.
        self.amp_disc = None  # set inside _setup_style_reference if amp_disc_enabled
        self.style_enabled = "style" in self.reward_scales
        if self.style_enabled:
            self._setup_style_reference()

        # Obs nur aus deploy-fähigen Größen (IMU + Gelenke + commands + letzte Aktion). base_lin_vel_heading
        # (privilegiert, auf der Hardware nicht messbar) und feet_contact (Deploy liefert nur Fake) sind
        # bewusst NICHT in der Obs — sie bleiben aber als Reward-Eingang erhalten (tracking_lin_vel bzw.
        # feet_slip/leg_symmetry).
        self._obs_slices = {
            "base_ang_vel": self.base_ang_vel.shape[-1],
            "projected_gravity": self.projected_gravity.shape[-1],
            "commands": self.commands.shape[-1],
            # ── privileged (sim-only) ──────────────────────────────────────────
            "base_lin_vel_heading": 2,   # vx/vy im Heading-Frame
            "lin_vel_ema": 2,            # geglättete Ist-Geschw. (= was tracking_lin_vel misst)
            "ang_vel_z_ema": 1,          # geglättete Yaw-Rate   (= was tracking_ang_vel misst)
            "base_height": 1,            # Rumpfhöhe [m]
            "base_lin_vel_z": 1,         # vertikale Geschw. [m/s] (Fallen/Hüpfen)
            # ── deploy-fähig ──────────────────────────────────────────────────
            "dof_pos": self.dof_pos.shape[-1],
            "dof_vel": self.dof_vel.shape[-1],
            "actions": self.actions.shape[-1],
            "gait_phase": 2,             # sin/cos(2πφ) — Phase-Clock für Schritt-Timing
            "feet_contact": 2,             # L/R-Kontakt, zentriert auf ±0.5
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

        if self.hero_ghost is not None:
            self.hero_ghost.attach_after_build(self)
            print(f"[video] hero ghost ON — {os.path.basename(self.hero_ghost._motion_path)}")

        self._init_curriculum()
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

        # AMP: build normalized reference transitions (s_t, s_{t+1}) for discriminator training
        if self.reward_cfg.get("amp_disc_enabled", False):
            self._setup_amp_discriminator(ref, mean, std)

    def _setup_amp_discriminator(self, ref_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        from amp_discriminator import AMPDiscriminator

        amp_obs_dim = ref_raw.shape[1]  # 38
        hidden = tuple(int(h) for h in self.reward_cfg.get("amp_hidden_dims", [256, 128]))
        lr = float(self.reward_cfg.get("amp_lr", 1e-4))
        gp_w = float(self.reward_cfg.get("amp_grad_penalty", 10.0))
        replay_size = int(self.reward_cfg.get("amp_replay_size", 100_000))

        self.amp_disc = AMPDiscriminator(amp_obs_dim, hidden, lr, gp_w, device=str(gs.device))
        self._amp_obs_dim = amp_obs_dim

        # Reference transitions: all consecutive (s_t, s_{t+1}) pairs from NPZ, pre-normalized
        ref_norm = (ref_raw - mean) / std
        s  = torch.tensor(ref_norm[:-1], dtype=gs.tc_float, device=gs.device)
        sn = torch.tensor(ref_norm[1:],  dtype=gs.tc_float, device=gs.device)
        self._amp_ref_buf = torch.cat([s, sn], dim=-1)  # (T-1, 76)

        # Circular replay buffer for policy transitions
        self._amp_replay = torch.zeros((replay_size, 2 * amp_obs_dim), dtype=gs.tc_float, device=gs.device)
        self._amp_replay_ptr = 0
        self._amp_replay_n = 0

        # Current/previous AMP obs buffers (updated every step)
        self._amp_obs_prev = torch.zeros((self.num_envs, amp_obs_dim), dtype=gs.tc_float, device=gs.device)
        self._amp_obs_curr = torch.zeros((self.num_envs, amp_obs_dim), dtype=gs.tc_float, device=gs.device)

        print(f"[AMP] disc ON  obs_dim={amp_obs_dim}  ref_transitions={len(self._amp_ref_buf)}  replay={replay_size}")

    def _build_amp_obs(self) -> torch.Tensor:
        """38-dim AMP state: same features as style reference, normalized with ref statistics."""
        foot_z = self.robot.get_links_pos(self.feet_idx_local)[:, :, 2]
        foot_clear = torch.clamp(foot_z - self.style_ground_z, 0.0, 0.5)
        feat = torch.cat([
            self.base_pos[:, 2:3],
            self.projected_gravity,
            self.dof_pos,
            self.dof_vel,
            foot_clear,
        ], dim=1)
        return (feat - self.style_mean) / self.style_std

    def _amp_push_replay(self) -> None:
        """Append current (prev, curr) transitions to the replay buffer."""
        trans = torch.cat([self._amp_obs_prev, self._amp_obs_curr], dim=-1)  # (n, 76)
        n = self.num_envs
        size = self._amp_replay.shape[0]
        end = (self._amp_replay_ptr + n) % size
        if end > self._amp_replay_ptr:
            self._amp_replay[self._amp_replay_ptr:end] = trans
        else:
            cut = size - self._amp_replay_ptr
            self._amp_replay[self._amp_replay_ptr:] = trans[:cut]
            self._amp_replay[:end] = trans[cut:]
        self._amp_replay_ptr = end
        self._amp_replay_n = min(self._amp_replay_n + n, size)

    def update_amp_disc(self, batch_size: int = 512) -> dict[str, float]:
        """Sample real/fake transitions, update discriminator. Called by runner after PPO update."""
        if self.amp_disc is None or self._amp_replay_n < batch_size:
            return {}
        ri = torch.randint(len(self._amp_ref_buf), (batch_size,), device=gs.device)
        real = self._amp_ref_buf[ri]
        fi = torch.randint(self._amp_replay_n, (batch_size,), device=gs.device)
        fake = self._amp_replay[fi]
        return self.amp_disc.update(*real.chunk(2, -1), *fake.chunk(2, -1))

    # ── Curriculum ────────────────────────────────────────────────────────────

    def _init_curriculum(self) -> None:
        curr_cfg = self.env_cfg.get("curriculum", {})
        if not curr_cfg.get("enabled", False):
            self._curriculum_phases: list | None = None
            return
        phases = curr_cfg.get("phases", [])
        if not phases:
            self._curriculum_phases = None
            return
        self._curriculum_phases = phases
        self._curriculum_phase_idx = 0
        # Phase 0 overrides command ranges and style file (style already loaded
        # with the base yaml value; re-apply so phase 0 takes precedence).
        self._apply_curriculum_phase(0)
        print(f"[curriculum] {len(phases)} phases | starting → '{phases[0]['name']}'")

    def _apply_curriculum_phase(self, phase_idx: int) -> None:
        phase = self._curriculum_phases[phase_idx]

        # ── command ranges ────────────────────────────────────────────────────
        cmd_ranges = phase.get("command_ranges", {})
        for key, val in cmd_ranges.items():
            self.command_cfg[key] = val
        # Rebuild the pre-computed tensors used by _resample_commands.
        self.commands_limits = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )

        # ── style reference + AMP discriminator ──────────────────────────────
        style_file = phase.get("style_motion_file")
        if style_file and self.style_enabled:
            self.reward_cfg["style_motion_file"] = style_file
            # _setup_style_reference re-builds style_ref_feat and (if amp_disc_enabled)
            # calls _setup_amp_discriminator which resets disc + replay + AMP obs.
            self._setup_style_reference()

        print(f"[curriculum] phase {phase_idx}: '{phase['name']}' | "
              f"vx={self.command_cfg['lin_vel_x_range']}  "
              f"style={os.path.basename(phase.get('style_motion_file', '?'))}")

    def advance_curriculum_phase(self) -> str:
        """Advance to the next curriculum phase. Returns the new phase name."""
        if self._curriculum_phases is None:
            return "none"
        next_idx = self._curriculum_phase_idx + 1
        if next_idx >= len(self._curriculum_phases):
            return self._curriculum_phases[-1]["name"]   # already at final phase
        self._curriculum_phase_idx = next_idx
        self._apply_curriculum_phase(next_idx)
        return self._curriculum_phases[next_idx]["name"]

    @property
    def curriculum_active(self) -> bool:
        return self._curriculum_phases is not None

    @property
    def curriculum_phase_name(self) -> str:
        if not self.curriculum_active:
            return "none"
        return self._curriculum_phases[self._curriculum_phase_idx]["name"]

    @property
    def curriculum_phase_cfg(self) -> dict | None:
        if not self.curriculum_active:
            return None
        return self._curriculum_phases[self._curriculum_phase_idx]

    @property
    def curriculum_at_final_phase(self) -> bool:
        if not self.curriculum_active:
            return True
        return self._curriculum_phase_idx >= len(self._curriculum_phases) - 1

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
        if self.rand_cfg.get("push_enabled", False):
            self._maybe_apply_push()
        self.scene.step()

        self.episode_length_buf += 1
        # Phase-Clock weiterdrehen (in-place, mod 1) — vor Kontakt-Update für On-Beat-Gating.
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
        self._update_command_tracking_ema()

        if self.amp_disc is not None:
            self._amp_obs_prev = self._amp_obs_curr
            self._amp_obs_curr = self._build_amp_obs()
            self._amp_push_replay()

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        self.rew_buf.nan_to_num_(0.0)  # physics NaN (e.g. sim explosion) must not crash training

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

    def _set_heading_reference(self, envs_idx=None, quats=None):
        """Spawn-Yaw je Episode fixieren → Heading-Frame für Command-Tracking (Welt-xy ohne Drift).

        quats: (num_envs, 4) per-env spawn quats from _build_noisy_init_qpos; if None falls back to
        init_base_quat (legacy, used when randomization is off).
        """
        if quats is None:
            quat = self.init_base_quat.unsqueeze(0)
            yaw = quat_to_xyz(quat, rpy=True)[0, 2]
            spawn_rpy = torch.zeros(1, 3, dtype=gs.tc_float, device=gs.device)
            spawn_rpy[0, 2] = yaw
            inv_h = inv_quat(xyz_to_quat(spawn_rpy))
            if envs_idx is None:
                self.inv_heading_quat.copy_(inv_h.expand(self.num_envs, -1))
            else:
                self.inv_heading_quat.index_copy_(0, envs_idx.nonzero(as_tuple=True)[0], inv_h.expand(envs_idx.sum(), -1))
        else:
            # quats is (num_envs, 4) — extract yaw per env and compute per-env inv_heading_quat
            yaw = quat_to_xyz(quats, rpy=True)[:, 2]  # (num_envs,)
            spawn_rpy = torch.zeros(self.num_envs, 3, dtype=gs.tc_float, device=gs.device)
            spawn_rpy[:, 2] = yaw
            inv_h = inv_quat(xyz_to_quat(spawn_rpy))  # (num_envs, 4)
            if envs_idx is None:
                self.inv_heading_quat.copy_(inv_h)
            else:
                reset_indices = envs_idx.nonzero(as_tuple=True)[0]
                self.inv_heading_quat[reset_indices] = inv_h[reset_indices]

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
        # reset state with init-state noise + random yaw
        noisy_qpos, init_quats_batch = self._build_noisy_init_qpos(envs_idx)
        self.robot.set_qpos(noisy_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # reset buffers
        if envs_idx is None:
            self.base_pos[:] = self.init_base_pos
            self.base_quat.copy_(init_quats_batch)
            self.projected_gravity[:] = self.init_projected_gravity
            self.dof_pos[:] = self.default_dof_pos
            self.base_lin_vel.zero_()
            self.base_lin_vel_heading.zero_()
            self.base_ang_vel.zero_()
            self.lin_vel_ema.zero_()
            self.ang_vel_z_ema.zero_()
            self._set_heading_reference(quats=init_quats_batch)
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
            self.both_stance_time.zero_()
            self.both_air_time.zero_()
            self.leg_symmetry_penalty.zero_()
            self.gait_phase.uniform_(0.0, 1.0)
            self.push_force_buf.zero_()
            self.push_steps_remaining.zero_()
            if self.amp_disc is not None:
                self._amp_obs_prev.zero_()
                self._amp_obs_curr.zero_()
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], init_quats_batch, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.default_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_lin_vel_heading.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.lin_vel_ema.masked_fill_(envs_idx[:, None], 0.0)
            self.ang_vel_z_ema.masked_fill_(envs_idx, 0.0)
            self._set_heading_reference(envs_idx, quats=init_quats_batch)
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
            self.both_stance_time.masked_fill_(envs_idx, 0.0)
            self.both_air_time.masked_fill_(envs_idx, 0.0)
            self.leg_symmetry_penalty.masked_fill_(envs_idx, 0.0)
            torch.where(envs_idx, torch.rand_like(self.gait_phase), self.gait_phase, out=self.gait_phase)
            self.push_force_buf.masked_fill_(envs_idx[:, None], 0.0)
            self.push_steps_remaining.masked_fill_(envs_idx, 0)
            if self.amp_disc is not None:
                self._amp_obs_prev.masked_fill_(envs_idx[:, None], 0.0)
                self._amp_obs_curr.masked_fill_(envs_idx[:, None], 0.0)

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
        # domain randomization: re-sample physics per episode
        self._randomize_domain(envs_idx)

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

        # feet_air_time (phasen-gekoppelt): Touchdown nur on-beat + Gauß um Soll-Schwungdauer;
        # Symmetrie via foot_step_quality des ANDEREN Fußes (s. _desired_stance).
        self.foot_air_time += self.dt
        on_beat_land = touchdown & self._desired_stance()
        air_quality = torch.exp(-torch.square(self.foot_air_time - self.gait_swing_time) / self.feet_air_sigma)
        other_quality = self.foot_step_quality[:, [1, 0]]
        landing = air_quality * on_beat_land.to(gs.tc_float) * other_quality
        self.feet_air_time_reward.copy_(landing.sum(dim=1) * active)
        self.foot_step_quality.mul_(self.feet_air_decay)
        self.foot_step_quality.copy_(torch.where(on_beat_land, air_quality, self.foot_step_quality))
        self.foot_air_time *= (~in_contact).to(gs.tc_float)

        # leg_symmetry: alternierender Gang = genau ein Fuß schwingt, der andere stützt; dann Wechsel.
        # both_stance (Doppelstütze) und both_air (Flugphase) sind als KURZE Übergänge natürlich
        # (Doppelstütze beim Gehen, Flugphase beim Laufen/Jog). Daher je einen Timer akkumulieren und
        # nur die Überdauer über das erlaubte Fenster (ds_allow_time / flight_allow_time) bestrafen —
        # sonst würde jeder saubere Laufschritt für seine Flugphase bestraft. Nur bei cmd_speed >
        # feet_air_cmd_threshold (im Stand kein Wechselzwang).
        both_air = (~in_contact[:, 0]) & (~in_contact[:, 1])
        both_stance = in_contact[:, 0] & in_contact[:, 1]
        self.both_stance_time += self.dt
        self.both_stance_time *= both_stance.to(gs.tc_float)  # nur bei Doppelstütze weiterzählen, sonst 0
        ds_excess = torch.clamp(self.both_stance_time - self.ds_allow_time, min=0.0)
        self.both_air_time += self.dt
        self.both_air_time *= both_air.to(gs.tc_float)  # nur bei Flugphase weiterzählen, sonst 0
        air_excess = torch.clamp(self.both_air_time - self.flight_allow_time, min=0.0)
        self.leg_symmetry_penalty.copy_((air_excess + ds_excess) * active)

        self.foot_in_contact.copy_(in_contact)

    def _desired_stance(self):
        """Soll-Bodenkontakt je Fuß (n,2): True = Stance-Fenster laut Phase-Clock.

        Linkes Bein folgt gait_phase, rechtes um gait_phase_offset versetzt (typisch 0.5 = anti-phasig).
        Stance solange Phase < gait_stance_ratio. Gemeinsames Modell für gait_phase-Strafe und feet_air_time.
        """
        phase = torch.stack(
            [self.gait_phase, (self.gait_phase + self.gait_phase_offset).remainder(1.0)], dim=1
        )
        return phase < self.gait_stance_ratio

    def _update_command_tracking_ema(self):
        """EMA-Tiefpass der Heading-Lineargeschw. (xy) und Yaw-Rate für das Command-Tracking.

        v_ema = α·v_ema + (1−α)·v, mit α = exp(−dt / tracking_ema_window_s). Glättet die
        natürliche Innerhalb-des-Schritts-Oszillation heraus, sodass tracking_* die mittlere
        Sollgeschwindigkeit misst (siehe __init__). Inference-mode-kompatibel (in-place).
        """
        a = self.vel_ema_alpha
        self.lin_vel_ema.mul_(a).add_(self.base_lin_vel_heading[:, :2], alpha=1.0 - a)
        self.ang_vel_z_ema.mul_(a).add_(self.base_ang_vel[:, 2], alpha=1.0 - a)

    def _update_observation(self):
        phase_2pi = self.gait_phase * (2.0 * math.pi)
        obs_parts = [
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            # privileged
            self.base_lin_vel_heading[:, :2] * self.obs_scales["lin_vel"],
            self.lin_vel_ema * self.obs_scales["lin_vel"],
            self.ang_vel_z_ema.unsqueeze(1) * self.obs_scales["ang_vel"],
            self.base_pos[:, 2:3],
            self.base_lin_vel[:, 2:3] * self.obs_scales["lin_vel"],
            # deploy-fähig
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
            torch.stack([torch.sin(phase_2pi), torch.cos(phase_2pi)], dim=1),
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
        terminate = terminate | (self.base_pos[:, 2] < 0.50)
        terminate = terminate | self.scene.rigid_solver.get_error_envs_mask()
        return terminate

    # ------------ reward functions ----------------
    # Jede Funktion gibt einen Wert pro Env zurück (0..N).
    # Beitrag zum Step-Reward: funktion() * reward_scales[name]  (scale schon * dt in __init__)

    def _reward_tracking_lin_vel(self):
        """Belohnung: vorwärts/seitwärts wie commands [vx, vy] in Spawn-Heading fahren.

        Misst den quadrierten Fehler zwischen Ziel und der ZEITGEMITTELTEN (EMA, ~tracking_ema_window_s)
        Ist-Geschwindigkeit im Heading-Frame (lin_vel_ema) — nicht der momentanen und nicht im
        rotierenden Körper-Frame. Die EMA glättet die natürliche Innerhalb-des-Schritts-Oszillation,
        sodass eine perfekte Gangart mit korrektem Geschwindigkeits-MITTEL ≈1.0 erreicht (sonst zieht
        die Stride-Oszillation den Momentanwert dauerhaft unter 1). Verhindert weiter den Spin-Hack
        (Heading-Frame statt Körper-Frame). exp(-fehler / sigma) → 1.0 bei perfektem Treffer.

        Beispiel (sigma=0.2, scale=2.0, dt=0.02):
          command [0.5, 0.0], EMA-Ist [0.5, 0.0]  → fehler=0      → return 1.0
          command [0.5, 0.0], EMA-Ist [0.3, 0.0]  → fehler=0.04   → return≈0.82
        """
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.lin_vel_ema), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """Belohnung: Drehgeschwindigkeit wie command [yaw] einhalten.

        Wie tracking_lin_vel, aber nur die z-Achse (Drehen um Hochachse) und auf der ZEITGEMITTELTEN
        Yaw-Rate (ang_vel_z_ema). Die EMA mittelt die starke Yaw-Oszillation je Schritt (Hüft-/Schulter-
        rotation beim Gehen) heraus → perfekte Gangart mit Soll-Drehrate erreicht ≈1.0.

        Beispiel (sigma=0.2):
          command 0.0 rad/s, EMA-Ist 0.0  → return 1.0
          command 0.0 rad/s, EMA-Ist 0.2  → fehler=0.04 → return≈0.82
        """
        ang_vel_error = torch.square(self.commands[:, 2] - self.ang_vel_z_ema)
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
        """Strafe (gebunden ∈[0,1)): ruckartige Aktionsänderungen (glatte Bewegung).

        tanh(Σ(Δaction)² / action_rate_sigma): 0 bei glatter Bewegung, →1 bei großen Sprüngen.
        Vorzeichen via negative Scale. Kleineres σ = strenger.
        """
        sq = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return torch.tanh(sq / self.reward_cfg["action_rate_sigma"])



    def _reward_feet_air_time(self):
        """Belohnung: phasen-gekoppelter, kadenz-treuer Schritt (Event beim Aufsetzen).

        In _update_foot_contact: Touchdown nur on-beat (_desired_stance) und
        exp(−(air_time − gait_swing_time)² / feet_air_sigma), gekoppelt über foot_step_quality
        des anderen Fußes. Gate: cmd_speed > feet_air_cmd_threshold.
        """
        return self.feet_air_time_reward

    def _reward_feet_slip(self):
        """Strafe (gebunden ∈[0,1)): Fuß rutscht/schlurft am Boden (Horizontalgeschw. im Kontakt).

        tanh(Σ v_xy² der Füße im Kontakt / feet_slip_sigma): 0 bei stillem Stützfuß, →1 bei Schlurfen.
        """
        foot_vel_xy = self.robot.get_links_vel(self.feet_idx_local)[:, :, :2]  # (n,2,2) Welt-xy
        slip = torch.sum(torch.square(foot_vel_xy), dim=2)  # (n,2)
        s = (slip * self.foot_in_contact.to(gs.tc_float)).sum(dim=1)
        return torch.tanh(s / self.reward_cfg["feet_slip_sigma"])


    def _reward_style(self):
        if self.amp_disc is not None:
            # AMP adversarial reward: discriminator judges whether (s_{t-1}, s_t) looks like reference
            return self.amp_disc.reward(self._amp_obs_prev, self._amp_obs_curr)

        # Fallback: nearest-neighbour feature matching (non-adversarial, timing-agnostic)
        foot_z = self.robot.get_links_pos(self.feet_idx_local)[:, :, 2]
        foot_clear = torch.clamp(foot_z - self.style_ground_z, 0.0, 0.5)
        feat = torch.cat(
            [self.base_pos[:, 2:3], self.projected_gravity, self.dof_pos, self.dof_vel, foot_clear], dim=1
        )
        feat = (feat - self.style_mean) / self.style_std * self.style_sqrt_w
        min_sq = torch.cdist(feat, self.style_ref_feat).pow(2).min(dim=1).values
        return torch.exp(-min_sq / self.style_w_sum / (2.0 * self.style_sigma**2))

    def _reward_leg_symmetry(self):
        """Strafe: Beine laufen nicht alternierend (anti-phasig: ein Fuß schwingt, der andere stützt).

        Bündelt die Links/Rechts-Symmetrie in EINEM Term. In _update_foot_contact je Step verbucht
        (leg_symmetry_penalty), nur bei cmd_speed > feet_air_cmd_threshold. Sowohl Doppelstütze als
        auch Flugphase sind als KURZE Übergänge natürlich (Doppelstütze beim Gehen, Flugphase beim
        Laufen/Jog); bestraft wird je nur die Überdauer über das erlaubte Fenster:
          - both_air (Flugphase):   max(0, both_air_time   − flight_allow_time)
          - both_stance (Doppelstütze): max(0, both_stance_time − ds_allow_time)

        Ausgabe gebunden ∈[0,1): tanh(leg_symmetry_penalty / leg_symmetry_sigma). 0 bei sauberem
        Wechselschritt (inkl. normaler Lauf-Flugphase), →1 bei zu langer Flug-/Doppelstützphase
        (Hüpfen, Stillstand auf beiden Beinen). Stehen (cmd≈0) → 0.
        """
        return torch.tanh(self.leg_symmetry_penalty / self.reward_cfg["leg_symmetry_sigma"])

    def _reward_gait_phase(self):
        """Strafe: Fuß-Kontakt passt nicht zum Phase-Clock-Takt (Off-Beat / falsches Swing-Fenster).

        Je Fuß 1 bei mismatch (Soll-Stance ≠ Ist-Kontakt), summiert über beide Füße → 0–2.
        Nutzt dasselbe Kontaktmodell wie feet_air_time (_desired_stance). Gate: cmd_speed >
        feet_air_cmd_threshold.
        """
        mismatch = (self._desired_stance() != self.foot_in_contact).to(gs.tc_float).sum(dim=1)
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        active = (cmd_speed > self.reward_cfg["feet_air_cmd_threshold"]).to(gs.tc_float)
        return mismatch * active

    def _reward_dof_vel(self):
        """Strafe (gebunden ∈[0,1)): hohe Gelenkgeschwindigkeiten dämpfen (Energie/Vibration).

        tanh(Σ dof_vel² / dof_vel_sigma): 0 bei ruhigem Stand, →1 bei schnellem/zitterndem
        Gelenkverlauf. Ergänzt action_rate (das nur Aktions-Differenzen abgreift).
        """
        sq = torch.sum(torch.square(self.dof_vel), dim=1)
        return torch.tanh(sq / self.reward_cfg["dof_vel_sigma"])



    def _reward_orientation(self):
        """Strafe (gebunden ∈[0,1)): Rumpf aufrecht halten — Roll/Pitch dämpfen (gegen Kippen/Nicken).

        tanh((roll²+pitch²)[rad²] / orientation_sigma): 0 aufrecht, →1 bei starker Neigung.
        base_euler liegt in Grad vor → in Radian umrechnen.
        """
        roll_pitch = torch.deg2rad(self.base_euler[:, :2])
        sq = torch.sum(torch.square(roll_pitch), dim=1)
        return torch.tanh(sq / self.reward_cfg["orientation_sigma"])

    def _reward_ang_vel_xy(self):
        """Strafe (gebunden ∈[0,1)): Roll-/Pitch-Raten des Rumpfes dämpfen (kein Kippeln/Schwanken).

        tanh((ω_x²+ω_y²) / ang_vel_xy_sigma): 0 bei ruhigem Rumpf, →1 bei starkem Schwanken.
        """
        sq = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return torch.tanh(sq / self.reward_cfg["ang_vel_xy_sigma"])

    def _reward_ang_vel_z(self):
        """Strafe (gebunden ∈[0,1)): Yaw-Rate vom Kommando abweichen (ergänzt tracking_ang_vel).

        Wie tracking_ang_vel auf der ZEITGEMITTELTEN Yaw-Rate (ang_vel_z_ema), nicht der momentanen —
        die oszilliert je Schritt und würde sonst eine perfekte Referenz fälschlich bestrafen.
        tanh((ω_z_ema − cmd_yaw)² / ang_vel_z_sigma): 0 bei Soll-Drehrate, →1 bei Spin.
        """
        sq = torch.square(self.ang_vel_z_ema - self.commands[:, 2])
        return torch.tanh(sq / self.reward_cfg["ang_vel_z_sigma"])

    def _reward_base_height(self):
        """Strafe (gebunden ∈[0,1)): Rumpfhöhe vom Zielwert abweichen (Hocken/Aufbäumen).

        tanh((z − z_target)² / base_height_sigma): 0 bei Zielshöhe, →1 bei starker Abweichung.
        Typischer Walking-Regularizer: hält den Roboter in aufrechter Gehhöhe statt zu hocken.
        Beispiel (target=0.56, sigma=0.01):
          z=0.56 → 0         z=0.46 → tanh(0.01/0.01)≈0.76     z=0.36 → tanh(0.04/0.01)≈1.0
        """
        sq = torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
        return torch.tanh(sq / self.reward_cfg["base_height_sigma"])

    def _reward_lin_vel_z(self):
        """Strafe (gebunden ∈[0,1)): vertikale Rumpfgeschwindigkeit dämpfen (kein Hüpfen/Bouncing).

        tanh(vz² / lin_vel_z_sigma): 0 bei ruhigem Gehen, →1 bei starkem Auf/Ab-Schwingen.
        Ergänzt base_height (die nur die Position sieht): reagiert früher auf Bounce-Dynamik.
        Beispiel (sigma=0.5):
          vz=0.0 → 0     vz=0.5 m/s → tanh(0.5)≈0.46     vz=1.0 m/s → tanh(2.0)≈0.96
        """
        sq = torch.square(self.base_lin_vel[:, 2])
        return torch.tanh(sq / self.reward_cfg["lin_vel_z_sigma"])

    # ------------ randomization helpers ----------------

    def _build_noisy_init_qpos(self, envs_idx):
        """Build (num_envs, qpos_dim) init qpos with random yaw + optional dof/height noise.

        Returns (qpos_batch, quats_batch), both (num_envs, ...).
        Non-reset envs' rows also get noise — Genesis ignores them via the envs_idx mask.
        When rand_cfg is empty all noise values default to 0 → identical to fixed init_qpos.
        """
        # Base position with optional height noise
        pos = self.init_base_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        h_noise = float(self.rand_cfg.get("init_base_height_noise", 0.0))
        if h_noise > 0:
            pos[:, 2] = pos[:, 2] + (2 * torch.rand(self.num_envs, device=gs.device) - 1) * h_noise

        # Random yaw
        yaw_range = self.rand_cfg.get("init_yaw_range", [0.0, 0.0])
        yaw_lo, yaw_hi = float(yaw_range[0]), float(yaw_range[1])
        if yaw_lo != yaw_hi:
            yaw = yaw_lo + (yaw_hi - yaw_lo) * torch.rand(self.num_envs, device=gs.device)
        else:
            yaw = torch.full((self.num_envs,), yaw_lo, dtype=gs.tc_float, device=gs.device)
        spawn_rpy = torch.zeros(self.num_envs, 3, dtype=gs.tc_float, device=gs.device)
        spawn_rpy[:, 2] = yaw
        quats = xyz_to_quat(spawn_rpy)  # (num_envs, 4)

        # DOF position noise applied to all URDF joints
        n_urdf_dofs = self.init_dof_pos.shape[0]
        dof_noise = float(self.rand_cfg.get("init_dof_pos_noise", 0.0))
        if dof_noise > 0:
            noise = (2 * torch.rand(self.num_envs, n_urdf_dofs, device=gs.device) - 1) * dof_noise
            dof_pos = self.init_dof_pos.unsqueeze(0) + noise
        else:
            dof_pos = self.init_dof_pos.unsqueeze(0).expand(self.num_envs, -1)

        qpos_batch = torch.cat([pos, quats, dof_pos], dim=1)  # (num_envs, 3+4+n_urdf_dofs)
        return qpos_batch, quats

    def _randomize_domain(self, envs_idx):
        """Re-sample physics parameters for reset envs. Called at the end of every _reset_idx."""
        if not self.rand_cfg:
            return
        if envs_idx is None:
            reset_idx = torch.arange(self.num_envs, device=gs.device)
        else:
            reset_idx = envs_idx.nonzero(as_tuple=True)[0]
        n = len(reset_idx)
        if n == 0:
            return

        if self.rand_cfg.get("randomize_friction", False):
            lo, hi = self.rand_cfg["friction_range"]
            self.curr_friction_ratios[reset_idx] = (
                lo + (hi - lo) * torch.rand(n, self.robot.n_links, device=gs.device)
            )
            self.robot.set_friction_ratio(
                self.curr_friction_ratios, links_idx_local=list(range(self.robot.n_links))
            )

        if self.rand_cfg.get("randomize_mass", False):
            lo, hi = self.rand_cfg["mass_ratio_range"]
            ratios = lo + (hi - lo) * torch.rand(n, self.robot.n_links, device=gs.device)
            # Convert ratio to additive shift: shift = (ratio - 1) * original_mass.
            # We store (ratio - 1) scaled by orig_link_masses so set_mass_shift receives kg.
            self.curr_mass_shifts[reset_idx] = (ratios - 1.0) * self.orig_link_masses
            self.robot.set_mass_shift(
                self.curr_mass_shifts, links_idx_local=list(range(self.robot.n_links))
            )

        if self.rand_cfg.get("randomize_com", False):
            lo, hi = self.rand_cfg["com_shift_range"]
            self.curr_com_shifts[reset_idx] = (
                lo + (hi - lo) * torch.rand(n, self.robot.n_links, 3, device=gs.device)
            )
            self.robot.set_COM_shift(
                self.curr_com_shifts, links_idx_local=list(range(self.robot.n_links))
            )

        if self.rand_cfg.get("randomize_motor_strength", False):
            lo, hi = self.rand_cfg["motor_strength_range"]
            self.curr_motor_strength[reset_idx] = (
                lo + (hi - lo) * torch.rand(n, self.num_actions, device=gs.device)
            )
            kp = self.nom_kp.unsqueeze(0) * self.curr_motor_strength  # (num_envs, num_actions)
            kd = self.nom_kd.unsqueeze(0) * self.curr_motor_strength
            self.robot.set_dofs_kp(kp, self.motors_dof_idx)
            self.robot.set_dofs_kv(kd, self.motors_dof_idx)

    def _maybe_apply_push(self):
        """Stochastically apply random horizontal impulse forces to the robot base link."""
        interval_s = float(self.rand_cfg.get("push_interval_s", 5.0))
        fmax = float(self.rand_cfg.get("push_force_xy_max", 150.0))
        tmax = float(self.rand_cfg.get("push_torque_z_max", 30.0))
        dur = int(self.rand_cfg.get("push_duration_steps", 5))

        # Poisson trigger: probability p = dt / interval per step
        trigger = torch.rand(self.num_envs, device=gs.device) < (self.dt / interval_s)

        # New push force (random horizontal direction + magnitude)
        new_force = torch.zeros(self.num_envs, 3, dtype=gs.tc_float, device=gs.device)
        new_force[:, :2] = (2 * torch.rand(self.num_envs, 2, device=gs.device) - 1) * fmax

        # Start new push (overwrite) on triggered envs; decrement counter on ongoing ones
        self.push_force_buf = torch.where(trigger[:, None], new_force, self.push_force_buf)
        new_remaining = torch.full((self.num_envs,), dur, dtype=gs.tc_int, device=gs.device)
        decremented = torch.clamp(self.push_steps_remaining - 1, min=0)
        self.push_steps_remaining = torch.where(trigger, new_remaining, decremented)

        # Apply force and torque to base link only where steps_remaining > 0
        active = (self.push_steps_remaining > 0).to(gs.tc_float)
        force_3d = (self.push_force_buf * active[:, None]).unsqueeze(1)  # (n_envs, 1, 3)
        self.scene.sim.rigid_solver.apply_links_external_force(
            force=force_3d, links_idx=[self.base_link_idx_local]
        )
        torque_3d = torch.zeros_like(force_3d)
        torque_3d[:, 0, 2] = (2 * torch.rand(self.num_envs, device=gs.device) - 1) * tmax * active
        self.scene.sim.rigid_solver.apply_links_external_torque(
            torque=torque_3d, links_idx=[self.base_link_idx_local]
        )

