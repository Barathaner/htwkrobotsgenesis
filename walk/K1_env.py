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

        # All 22 joints: policy actions = legs + arms + head
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

        # Kopfgelenke: nicht policy-gesteuert, aber per PD auf fester Pose gehalten (sonst schlackern sie lose).
        # Eigene kp/kd setzen (Default wäre 0 → kraftlos) und konstantes Ziel head_target jeden Step kommandieren.
        head_names = env_cfg["head_joint_names"]
        self.head_dof_idx = torch.tensor(
            [self.robot.get_joint(n).dof_start for n in head_names], dtype=gs.tc_int, device=gs.device
        )
        self.head_actions_dof_idx = torch.argsort(self.head_dof_idx)
        self.robot.set_dofs_kp([joint_gains[n]["kp"] for n in head_names], self.head_dof_idx)
        self.robot.set_dofs_kv([joint_gains[n]["kd"] for n in head_names], self.head_dof_idx)
        head_effort = [joint_gains[n]["effort"] for n in head_names]
        self.robot.set_dofs_force_range([-e for e in head_effort], head_effort, self.head_dof_idx)
        self.head_target = torch.tensor(
            [env_cfg["default_joint_angles"][n] for n in head_names], dtype=gs.tc_float, device=gs.device
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

        # Geglättete Tracking-Qualität (mittlere RAW command_accuracy/Step der endenden Episoden, 0..1)
        # UND geglättete Gangqualität (mittleres air_gate/Step = „schreitet wirklich"). Beide treiben
        # das Style-Curriculum-Gate (s. _update_style_curriculum); der vx-Sollbereich ist fix.
        self.cmd_curr_perf = 0.0
        self.gait_curr_perf = 0.0
        self.total_steps = 0                                                   # monotoner globaler Step-Zähler

        # Reward-Curriculum (Style-Ramp): erst LAUFEN, dann STIL. Polish-Terme (Knie beugen,
        # Heel-to-Toe-Abrollen, Stützpose …) werden erst hochgefahren, NACHDEM Laufen steht. Gate
        # öffnet (gelatcht), sobald Warmup abgelaufen UND cmd_curr_perf > threshold UND gait_curr_perf
        # > gait_threshold — also erst wenn der Roboter das Kommando trackt UND dabei wirklich schreitet
        # (nicht nur vorwärts gleitet). Dann läuft style_weight über style_ramp_s Sim-Sekunden linear
        # 0→1 und skaliert in step() die Scales der style_terms; alle übrigen Terme bleiben stets voll.
        sc = self.reward_cfg.get("style_curriculum", {})
        self.style_enabled = bool(sc.get("enabled", False))
        self.style_threshold = float(sc.get("threshold", 0.5))
        self.style_gait_threshold = float(sc.get("gait_threshold", 0.5))  # min. air_gate-Mittel fürs Gate
        self.perf_ema = float(sc.get("ema", 0.1))  # Glättung der cmd_curr_perf/gait_curr_perf-Schätzung
        # Harte Zeit-Sperre: das Gate kann frühestens nach warmup_s Sim-Sekunden öffnen, egal wie
        # hoch cmd_curr_perf ist → verhindert sofortiges Aktivieren, solange der Gang noch instabil ist.
        self.style_warmup_steps = max(0, int(float(sc.get("warmup_s", 0.0)) / self.dt))
        self.style_ramp_steps = max(1, int(float(sc.get("ramp_s", 60.0)) / self.dt))
        self.style_terms = set(sc.get("terms", []))
        self.style_start_step = -1                 # total_steps bei Gate-Öffnung (-1 = noch zu)
        self.style_weight = 0.0 if self.style_enabled else 1.0  # aus → konstant 1 (Verhalten wie ohne Ramp)
        if self.style_enabled:
            unknown = self.style_terms - set(self.reward_scales)
            assert not unknown, f"style_curriculum.terms nicht in reward_scales aktiv: {sorted(unknown)}"

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

        # Fuß-Kontakt-Proxy (Höhe): foot_in_contact startet True (Spawn stehend → kein Fake-Touchdown).
        self.feet_idx_local = [self.robot.get_link(n).idx_local for n in ("left_foot_link", "right_foot_link")]
        self.foot_in_contact = torch.ones((num_envs, 2), dtype=gs.tc_bool, device=gs.device)

        # feet_air_time: Luftphase je Fuß [s]; Belohnung beim Aufsetzen (legged-gym-Stil).
        self.foot_air_time = torch.zeros((num_envs, 2), dtype=gs.tc_float, device=gs.device)
        self.feet_air_time_reward = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)

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

        # Assist-Force-Curriculum: WIDERSTAND am Rumpf-COM, der ENTGEGEN der kommandierten Bewegung
        # zieht — der Roboter muss sich dagegen lehnen, um Fortschritt zu machen (Gegenstück zum Push,
        # aber gerichtet: Push stört impulsiv, dieser Widerstand bremst dauerhaft). Magnitude folgt
        # einer Glockenkurve: klein zu Beginn → Maximum bei peak_steps → wieder kleiner → ausgeblendet.
        # Begründung/Mechanik: _apply_assist_force.
        ac = env_cfg.get("assist", {})
        self.assist_enabled = bool(ac.get("enabled", False))
        self.assist_force_kp = float(ac.get("force_kp", 40.0))
        self.assist_force_dir = float(ac.get("force_dir", 30.0))
        self.assist_force_max = float(ac.get("force_max", 80.0))
        self.assist_start_scale = float(ac.get("start_scale", 0.2))  # kleine Anfangskraft (Glocken-Start)
        self.assist_peak_steps = max(1, int(ac.get("peak_steps", 50_000)))  # Step des Kraft-Maximums
        self.assist_decay_steps = max(1, int(ac.get("decay_steps", 200_000)))
        self.assist_scale_min = float(ac.get("scale_min", 0.0))
        # Performance-Gate: Assist zusätzlich zum Zeit-Abbau zurückfahren, sobald command_accuracy
        # steigt — aber nur bis perf_floor (Rest bleibt, der Zeit-Abbau blendet final aus).
        self.assist_accuracy_target = max(1e-6, float(ac.get("accuracy_target", 0.7)))
        self.assist_perf_floor = float(ac.get("perf_floor", 0.5))
        self.assist_perf_alpha = float(ac.get("perf_ema", 0.01))
        # Gate-Signal: command_accuracy (Heading-Frame, drift-bewusst), sonst tracking_lin_vel.
        self.assist_perf_key = "command_accuracy" if "command_accuracy" in self.reward_scales else "tracking_lin_vel"
        self.assist_perf_ema = 0.0                                 # EMA der mittleren Gate-Reward-Qualität
        self.assist_scale = self.assist_start_scale if self.assist_enabled else 0.0  # Logging vor dem 1. Step
        self.assist_link_idx = [self.robot.base_link_idx]         # globaler Index des Rumpf-Links
        self.assist_force_buf = torch.zeros((num_envs, 1, 3), dtype=gs.tc_float, device=gs.device)

        # Phase-Clock (Siekmann/Margolis): periodischer Gangtakt φ∈[0,1). φ läuft jeden Step weiter;
        # sin/cos(2πφ) gehen in die Obs, damit die Policy den Takt timen kann.
        self.gait_period_steps = max(1, int(self.reward_cfg["gait_period_s"] / self.dt))
        self.gait_stance_ratio = float(self.reward_cfg["gait_stance_ratio"])
        self.gait_phase_offset = float(self.reward_cfg["gait_phase_offset"])
        self.gait_phase = torch.rand((num_envs,), dtype=gs.tc_float, device=gs.device)

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
        # command_accuracy ∈ [0,1] je Env, EINMAL pro Step in step() berechnet (RAW Tracking-Güte,
        # exp(-Heading-Vel-Fehler/sigma)). Single Source für: den command_accuracy-Reward, die
        # gait_phase-Kopplung, die feet_air_time-Kopplung und cmd_curr_perf (Style-Gate).
        self.command_accuracy = torch.ones((num_envs,), dtype=gs.tc_float, device=gs.device)
        # air_gate ∈ [floor,1] je Env: misst, ob der Roboter WIRKLICH schreitet — Schwungfuß-Luftzeit
        # max(foot_air_time)/ref, gesättigt auf 1. Bei Schlurfen/Gleiten (Füße bleiben am Boden,
        # max_air≈0) → air_gate≈floor. Koppelt Tracking AN Gang (command_accuracy × air_gate, s.
        # _reward_command_accuracy) → kein Tracking-Reward für „Strecke ohne Schritt".
        self.air_gate = torch.ones((num_envs,), dtype=gs.tc_float, device=gs.device)
        # Air-Gate-Kopplung: feet_air_time-Belohnung mit command_accuracy stufen → Abheben wird nur
        # belohnt, soweit das Kommando auch getrackt wird (kein Reward fürs Treten auf der Stelle ohne
        # Vortrieb). Aus → reines cmd_speed-Gate (feet_air_cmd_threshold).
        self.feet_air_couple_cmd_acc = bool(self.reward_cfg.get("feet_air_couple_command_accuracy", False))
        # Tracking AN Gang koppeln: command_accuracy-Reward zusätzlich mit air_gate stufen (s. o.).
        self.cmd_acc_couple_air = bool(self.reward_cfg.get("command_accuracy_couple_air", False))
        self.air_gate_ref = max(1e-6, float(self.reward_cfg.get("air_gate_ref_s", 0.1)))  # Luftzeit für Gate=1
        self.air_gate_floor = float(self.reward_cfg.get("air_gate_floor", 0.0))           # Rest-Gate bei 0 Luft
        # gait_phase-Taktstrafe zusätzlich mit command_accuracy stufen (Takt-Zwang nur bei gutem Tracking).
        self.gait_couple_cmd_acc = bool(self.reward_cfg.get("gait_couple_command_accuracy", False))
        # Episoden-Akkumulatoren (RAW, ungated) fürs Style-Gate: mittlere Tracking-Güte und mittlere
        # Gangqualität (air_gate) je Episode → cmd_curr_perf / gait_curr_perf in _update_style_curriculum.
        self.episode_cmd_acc_sum = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.episode_air_gate_sum = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
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
        # Kopf konstant auf fester Pose halten (geradeaus, leicht nach unten) — kein Mitschlackern.
        self.robot.control_dofs_position(
            self.head_target[:, self.head_actions_dof_idx],
            self.head_dof_idx,
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

        # command_accuracy je Env [0,1] EINMAL berechnen (RAW Tracking-Güte, Single Source) — der
        # command_accuracy-Reward, die gait_phase-/feet_air_time-Kopplungen und cmd_curr_perf lesen
        # denselben Wert, unabhängig von der Auswertungsreihenfolge der Reward-Funktionen. commands
        # sind in diesem Step noch stabil (Resampling passiert erst nach der Reward-Schleife).
        ca_err = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel_heading[:, :2]), dim=1)
        self.command_accuracy.copy_(torch.exp(-ca_err / self.reward_cfg["tracking_sigma"]))
        # air_gate ∈ [floor,1]: Schwungfuß-Luftzeit max(foot_air_time)/ref, gesättigt → misst echtes
        # Schreiten. foot_air_time ist hier frisch (in _update_foot_contact bereits aktualisiert/genullt).
        max_air = self.foot_air_time.max(dim=1).values
        self.air_gate.copy_(torch.clamp(max_air / self.air_gate_ref, min=self.air_gate_floor, max=1.0))
        # RAW-Akkumulatoren fürs Style-Gate (ungated, längenunabhängig in _update_style_curriculum normiert).
        self.episode_cmd_acc_sum += self.command_accuracy
        self.episode_air_gate_sum += self.air_gate

        self.rew_buf.zero_()
        style_w = self._current_style_weight()  # 0→1 Ramp, skaliert nur die style_terms
        self.style_weight = style_w
        for name, reward_func in self.reward_functions.items():
            raw = reward_func()
            scale = self.reward_scales[name]
            if name in self.style_terms:
                scale = scale * style_w
            rew = raw * scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
            # command_accuracy-EMA fürs Assist-Performance-Gate (raw ∈ [0,1], vor *scale).
            if self.assist_enabled and name == self.assist_perf_key:
                self.assist_perf_ema += self.assist_perf_alpha * (raw.mean().item() - self.assist_perf_ema)

        self._resample_commands(
            self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0
        )

        self.reset_buf.copy_(self.episode_length_buf > self.max_episode_length)
        self.reset_buf.logical_or_(torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
        self.reset_buf.logical_or_(torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
        self.reset_buf.logical_or_(self.base_pos[:, 2] < 0.35)
        self.reset_buf.logical_or_(self.scene.rigid_solver.get_error_envs_mask())

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
        # Style-Curriculum-Gate VOR dem Buffer-Reset auswerten (braucht episode_sums / episode_length).
        if envs_idx is not None:
            self._update_style_curriculum(envs_idx)

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
            self.ankle_pitch_hist.zero_()
            self.foot_flat_time.zero_()
            self.foot_roll_reward.zero_()
            self.foot_flat_penalty.zero_()
            self.episode_cmd_acc_sum.zero_()
            self.episode_air_gate_sum.zero_()
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
            self.ankle_pitch_hist.masked_fill_(envs_idx[:, None, None], 0.0)
            self.foot_flat_time.masked_fill_(envs_idx[:, None], 0.0)
            self.foot_roll_reward.masked_fill_(envs_idx, 0.0)
            self.foot_flat_penalty.masked_fill_(envs_idx, 0.0)
            self.episode_cmd_acc_sum.masked_fill_(envs_idx, 0.0)
            self.episode_air_gate_sum.masked_fill_(envs_idx, 0.0)
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
        # Curriculum-Fortschritt mitloggen (Style-Ramp-Gewicht + Gate-Signale)
        self.extras["episode"]["curriculum_style_weight"] = torch.tensor(
            self.style_weight, dtype=gs.tc_float, device=gs.device)
        self.extras["episode"]["curriculum_cmd_perf"] = torch.tensor(
            self.cmd_curr_perf, dtype=gs.tc_float, device=gs.device)
        self.extras["episode"]["curriculum_gait_perf"] = torch.tensor(
            self.gait_curr_perf, dtype=gs.tc_float, device=gs.device)
        # Assist-Force-Curriculum: aktuelle Stützstärke + Gate-Signal (command_accuracy-EMA) mitloggen
        self.extras["episode"]["assist_scale"] = torch.tensor(
            self.assist_scale, dtype=gs.tc_float, device=gs.device
        )
        self.extras["episode"]["assist_perf_ema"] = torch.tensor(
            self.assist_perf_ema, dtype=gs.tc_float, device=gs.device
        )

        # random sample command upon reset
        self._resample_commands(envs_idx)

    def _update_style_curriculum(self, envs_idx):
        """Pflegt cmd_curr_perf (Tracking) + gait_curr_perf (Schreiten) und öffnet das Style-Ramp-Gate.

        Misst je endender Episode die mittleren RAW-Signale PRO STEP — normiert auf die ECHTE
        Episodenlänge (episode_length_buf), nicht auf max_episode_length. Sonst skaliert die Qualität
        mit der Episodendauer: kurze Episoden (z. B. ~75 statt 1000 Steps) lägen selbst bei perfektem
        Tracking nur bei ~length/max_len → die Gates öffneten NIE. Mit Echtlängen-Normierung sind
        beide Größen ∈ [0,1] längenunabhängig:
          cmd_curr_perf  = mittlere RAW command_accuracy/Step  (trackt es das Kommando?)
          gait_curr_perf = mittleres air_gate/Step             (schreitet es wirklich, statt zu gleiten?)
        Beide werden EMA-geglättet. Das Gate öffnet (einmalig gelatcht), sobald Warmup abgelaufen UND
        cmd_curr_perf > style_threshold UND gait_curr_perf > style_gait_threshold — verhindert, dass
        die Style-Ramp auf einem vorwärts GLEITENDEN (aber nicht schreitenden) Roboter startet.
        """
        if not self.style_enabled:
            return
        n = int(envs_idx.sum())
        if n == 0:
            return
        ep_len = self.episode_length_buf[envs_idx].clamp(min=1).to(gs.tc_float)
        batch_cmd = (self.episode_cmd_acc_sum[envs_idx] / ep_len).mean().item()
        batch_gait = (self.episode_air_gate_sum[envs_idx] / ep_len).mean().item()
        self.cmd_curr_perf = (1.0 - self.perf_ema) * self.cmd_curr_perf + self.perf_ema * batch_cmd
        self.gait_curr_perf = (1.0 - self.perf_ema) * self.gait_curr_perf + self.perf_ema * batch_gait

        # Style-Gate öffnen (einmalig gelatcht): Warmup abgelaufen UND Tracking gut UND echtes Schreiten.
        if (
            self.style_start_step < 0
            and self.total_steps >= self.style_warmup_steps
            and self.cmd_curr_perf > self.style_threshold
            and self.gait_curr_perf > self.style_gait_threshold
        ):
            self.style_start_step = self.total_steps

    def _current_style_weight(self):
        """Aktuelles Style-Gewicht ∈ [0,1] (monoton, gelatcht) — Faktor für die style_terms in step().

        Gate zu (cmd_curr_perf nie über style_threshold) → 0: keine Style-Terme, reines Laufenlernen.
        Ab Gate-Öffnung lineare Rampe über style_ramp_steps Sim-Steps auf 1 (sanfter Übergang, kein
        harter Sprung, der einen funktionierenden Gang zurück Richtung Stehen zerrt). Style aus → 1.
        """
        if not self.style_enabled:
            return 1.0
        if self.style_start_step < 0:
            return 0.0
        return min(1.0, max(0.0, (self.total_steps - self.style_start_step) / self.style_ramp_steps))

    def _update_foot_contact(self):
        """Fußhöhe-Proxy: Fuß am Boden wenn z < foot_contact_height; zählt Touchdowns (Flanke hoch→Kontakt).

        Akkumuliert außerdem foot_air_time (dt pro Step ohne Bodenkontakt) und verbucht beim
        Aufsetzen die Lande-Belohnung (feet_air_time) — beide nutzen denselben foot_z-Read.
        Inference-mode-kompatibel: copy_ / in-place +=, *=, masked-fill.
        """
        foot_z = self.robot.get_links_pos(self.feet_idx_local)[:, :, 2]
        in_contact = foot_z < self.reward_cfg["foot_contact_height"]
        touchdown = in_contact & ~self.foot_in_contact

        # feet_air_time: erst dt für alle Füße addieren (zählt diesen Step zur Luftphase),
        # dann beim Touchdown die Luftphase belohnen, danach Füße in Kontakt auf 0 setzen.
        self.foot_air_time += self.dt
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        active = (cmd_speed > self.reward_cfg["feet_air_cmd_threshold"]).to(gs.tc_float)
        # Alternations-Gate: Landung nur belohnen, wenn der ANDERE Fuß am Boden ist
        # (echtes Wechselschreiten / Double-Support, kein Hüpfen auf einem Bein).
        other_in_contact = in_contact[:, [1, 0]].to(gs.tc_float)
        landing = torch.clamp(
            self.foot_air_time - self.reward_cfg["feet_air_time_min"],
            min=0.0,
            max=self.reward_cfg["feet_air_time_max"],
        ) * touchdown.to(gs.tc_float) * other_in_contact
        self.feet_air_time_reward.copy_(landing.sum(dim=1) * active)
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
        """Widerstands-Curriculum: virtueller Zug am Rumpf-COM, der ENTGEGEN der kommandierten
        Bewegung wirkt — der Roboter muss sich dagegen lehnen und aktiv dagegen anlaufen, um
        Fortschritt zu erzielen (Resistance-Training). Gegenstück zum Push (_apply_push): Push STÖRT
        impulsiv, dieser Widerstand BREMST dauerhaft und gerichtet. Zwei Anteile (Heading-/Welt-xy):

          force = scale · ( −force_kp·v_ist        geschwindigkeitsproportionaler Drag (entgegen der
                                                    Ist-Bewegung): bremst Vorwärtsfahrt, dämpft Drift
                          − force_dir·cmd_dir )     konstanter Gegen-Zug ENTGEGEN der Kommandorichtung
                                                    (Einheitsvektor): „Gegenwind", gegen den der COM
                                                    nach vorn gelehnt werden muss → erzwingt Anlehnen.

        |Gesamtkraft| auf force_max [N] gedeckelt; am Rumpf-COM (ref='root_com' → reine Linearkraft,
        KEIN Stör-Drehmoment). NICHT in der Observation → die Policy erlebt den Widerstand nur als
        veränderte Dynamik, die über das Curriculum erst zunimmt und dann verschwindet.

        Heading == Welt gilt bei Identity-Spawn (base_init_quat = Einheit); base_lin_vel_heading ist
        dann Welt-xy. Bei späterem Random-Yaw-Spawn müssten v_ist und cmd_dir von Heading nach Welt
        rotiert werden, bevor die Kraft im Weltframe angewandt wird.

        Skala = time_scale · perf_factor. time_scale folgt einer GLOCKENKURVE (Dreieck) über die Zeit:
          total_steps ≤ peak_steps:  ramp start_scale → 1.0   (kleine Anfangskraft, steigend)
          total_steps > peak_steps:  ramp 1.0 → scale_min     (fallend bis ausgeblendet)
          perf_factor = 1 − (1−perf_floor) · clamp(acc_ema/accuracy_target, 0, 1)
        command_accuracy (EMA) hoch → perf_factor fällt auf perf_floor (nicht 0): läuft der Roboter
        gut, wird der Widerstand etwas entlastet, ganz weg ist er erst nach dem Zeit-Abbau.

        Beispiel (force_dir=30, start_scale=0.2, peak_steps=5e4, decay_steps=2e5, perf_floor=0.5):
          total_steps=0,     acc=0.0 → time=0.20, perf=1.0  → scale=0.20 → F≈−6 N (leichter Gegenwind)
          total_steps=5e4,   acc=0.5 → time=1.00, perf≈0.64 → scale≈0.64 → F≈−19 N (max. Widerstand)
          total_steps≥2e5                                   → time=0      → scale=0 (frei laufen)
        """
        if self.total_steps <= self.assist_peak_steps:
            # Anstieg: kleine Anfangskraft (start_scale) linear hoch zum Maximum (1.0) bei peak_steps
            time_scale = self.assist_start_scale + (1.0 - self.assist_start_scale) * (
                self.total_steps / self.assist_peak_steps
            )
        else:
            # Abfall: Maximum (1.0) linear hinunter auf scale_min am Ende des Decay-Fensters
            span = max(1, self.assist_decay_steps - self.assist_peak_steps)
            time_scale = max(self.assist_scale_min, 1.0 - (self.total_steps - self.assist_peak_steps) / span)
        acc = min(1.0, max(0.0, self.assist_perf_ema / self.assist_accuracy_target))
        perf_factor = 1.0 - (1.0 - self.assist_perf_floor) * acc
        self.assist_scale = time_scale * perf_factor
        if self.assist_scale <= 0.0:
            return
        # (1) geschwindigkeitsproportionaler Drag ENTGEGEN der Ist-Bewegung
        force_xy = -self.assist_force_kp * self.base_lin_vel_heading[:, :2]
        # (2) konstanter Gegen-Zug ENTGEGEN der Kommandorichtung (Einheitsvektor); inaktiv bei ~null Kommando
        cmd_speed = torch.norm(self.commands[:, :2], dim=1, keepdim=True)
        cmd_dir = self.commands[:, :2] / (cmd_speed + 1e-6)
        force_xy = force_xy - self.assist_force_dir * cmd_dir * (cmd_speed > 1e-3).to(gs.tc_float)
        # gemeinsame Curriculum-Skalierung, dann Betrag der Gesamtkraft deckeln
        force_xy = self.assist_scale * force_xy
        mag = torch.norm(force_xy, dim=1, keepdim=True)
        force_xy = force_xy * torch.clamp(self.assist_force_max / (mag + 1e-6), max=1.0)
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

    # ------------ reward functions ----------------
    # Jede Funktion gibt einen Wert pro Env zurück (0..N).
    # Beitrag zum Step-Reward: funktion() * reward_scales[name]  (scale schon * dt in __init__)

    def _reward_tracking_lin_vel(self):
        """Belohnung: vorwärts/seitwärts wie commands [vx, vy] fahren.

        Misst quadrierten Fehler zwischen Ziel- und Ist-Geschwindigkeit (Körper-Frame).
        exp(-fehler / sigma) → 1.0 bei perfektem Treffer, sinkt bei Abweichung.

        Beispiel (sigma=0.25, scale=1.0, dt=0.02):
          command [0.5, 0.0], Ist [0.5, 0.0]  → fehler=0      → return 1.0   → +0.02/Step
          command [0.5, 0.0], Ist [0.3, 0.0]  → fehler=0.04   → return≈0.85  → +0.017/Step
          command [0.5, 0.0], Ist [0.0, 0.0]  → fehler=0.25   → return≈0.37  → +0.007/Step
        """
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_command_accuracy(self):
        """Belohnung: Geschwindigkeit im Heading-/Weltframe wie command [vx, vy] (ohne Yaw-Drift).

        Misst Ist-Geschwindigkeit im Spawn-Heading-Frame (Welt-xy relativ zur Start-Yaw), nicht im
        Körperframe. Bei cmd [0.5, 0] soll der Roboter 0.5 m/s in Spawn-Vorwärtsrichtung laufen —
        auch wenn er sich unbeabsichtigt dreht (dann wäre Körper-vx ok, Heading-vx aber falsch).

        Beispiel (sigma=0.25, scale=1.0, dt=0.02):
          cmd [0.5, 0], Heading [0.5, 0]  → return 1.0
          cmd [0.5, 0], Roboter 90° gedreht, Körper-vx=0.5 → Heading≈[0, 0.5] → return≈0.37

        RAW-Wert wird in step() vorab berechnet (self.command_accuracy) und auch von den Kopplungen
        (gait_phase, feet_air_time) sowie cmd_curr_perf genutzt → eine gemeinsame Quelle.

        Tracking AN Gang koppeln (command_accuracy_couple_air): der REWARD (nicht das RAW-Signal) wird
        mit air_gate ∈ [floor,1] gestuft → Tracking-Reward gibt es nur, soweit der Roboter wirklich
        SCHREITET. Bekämpft „keine Strecke": vorwärts gleiten/schlurfen (Füße bleiben am Boden,
        air_gate≈floor) bringt dann kaum command_accuracy-Reward. Symmetrisch zur Gang→Tracking-
        Kopplung (feet_air_time × command_accuracy). tracking_lin_vel (Körperframe) bleibt ungated →
        der Vortriebs-Lernanreiz fürs Anlaufen bleibt erhalten (kein Bootstrap-Deadlock).
        """
        if self.cmd_acc_couple_air:
            return self.command_accuracy * self.air_gate
        return self.command_accuracy

    def _reward_tracking_ang_vel(self):
        """Belohnung: Drehgeschwindigkeit wie command [yaw] einhalten.

        Wie tracking_lin_vel, aber nur die z-Achse (Drehen um Hochachse).

        Beispiel (sigma=0.25, scale=0.2, dt=0.02):
          command 0.0 rad/s, Ist 0.0  → return 1.0  → +0.002/Step
          command 0.0 rad/s, Ist 0.5  → fehler=0.25 → return≈0.37 → +0.0007/Step
        """
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

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
        """Belohnung: Füße sollen genug abheben (gegen Schlurfen/Trippeln ohne Bodenfreiheit).

        Legged-Gym-Stil: Belohnung NUR beim Aufsetzen (Touchdown), proportional zur
        vorausgegangenen Luftphase. foot_air_time wird in _update_foot_contact akkumuliert
        (dt pro Step ohne Bodenkontakt) und bei Landung in feet_air_time_reward verbucht:
          je Fuß  clamp(air_time - feet_air_time_min, 0, feet_air_time_max),  über beide summiert.
        Nur wenn cmd_speed > feet_air_cmd_threshold (beim Stehen kein Reward) UND der andere Fuß
        beim Aufsetzen am Boden ist (Alternations-Gate → kein Einbein-Hüpfen, siehe _update_foot_contact).

        Air-Gate-Kopplung (feet_air_couple_command_accuracy): die Lande-Belohnung wird zusätzlich mit
        command_accuracy ∈ [0,1] gestuft → Abheben zählt nur, soweit der Roboter das Kommando auch
        trackt. Damit verschwindet der Anreiz, auf der Stelle zu treten/zu trippeln (Füße heben, aber
        kein Vortrieb): command_accuracy ist dann niedrig → die Air-Belohnung schrumpft. Der binäre
        cmd_speed-Gate bleibt als Floor erhalten. Der dominante command_accuracy-Reward (scale 1.0)
        verhindert Sandbagging (schlecht tracken lohnt nie).

        Beispiel (min=0.05, max=0.25, scale=3.0 → effektiv *dt=0.06, dt=0.02):
          Landung raw=0.18, command_accuracy=1.0 (trackt) → contrib≈+0.0108 (voll)
          Landung raw=0.18, command_accuracy=0.3 (Stelle) → contrib≈+0.0032 (gestuft)
          kein Touchdown in diesem Step                   → raw=0 → 0
          cmd≈0 (Stehen)                                  → raw=0 → 0
        """
        if self.feet_air_couple_cmd_acc:
            return self.feet_air_time_reward * self.command_accuracy
        return self.feet_air_time_reward

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

    def _reward_gait_phase(self):
        """Strafe: Fuß-Kontakt passt nicht zum Phase-Clock-Takt (Siekmann/Margolis-Stil).

        Der Takt schreibt je Bein vor, wann es am Boden sein soll (Stance) und wann in der Luft
        (Swing): linkes Bein folgt gait_phase, rechtes Bein um gait_phase_offset (0.5) versetzt
        → erzwingt Alternation und feste Kadenz. Stance, solange Phase < gait_stance_ratio.
        Bestraft wird je Fuß die Abweichung von Soll-Kontakt vs. Ist-Kontakt (Höhen-Proxy),
        nur bei cmd_speed > gait_cmd_threshold (im Stand kein Takt-Zwang). sin/cos(2πφ) liegt
        in der Observation → die Policy kann den Takt aktiv timen.

        Gang-Kopplung (gait_couple_command_accuracy): die Strafe wird zusätzlich mit command_accuracy
        ∈ [0,1] skaliert → der Takt-Zwang greift nur, wenn der Roboter das Kommando auch trackt.
        Trackt er schlecht (Stolpern/Erholung), entfällt die Off-Beat-Strafe weitgehend, statt das
        ohnehin kämpfende Tracking weiter zu untergraben ("erst tracken, dann Kadenz"). Der dominante
        command_accuracy-Reward verhindert dabei Sandbagging.

        Beispiel (offset=0.5, stance_ratio=0.6, scale=-0.3 → *dt=-0.006):
          mismatch=1, command_accuracy=1.0 (perfektes Tracking) → -0.006/Step (voller Takt-Zwang)
          mismatch=1, command_accuracy=0.3 (schlechtes Tracking) → -0.0018/Step (entkoppelt)
          cmd≈0 (Stehen)                                         → 0
        """
        left_stance = self.gait_phase < self.gait_stance_ratio
        right_stance = ((self.gait_phase + self.gait_phase_offset) % 1.0) < self.gait_stance_ratio
        desired_contact = torch.stack([left_stance, right_stance], dim=1)  # (n,2) True = soll am Boden
        mismatch = (desired_contact != self.foot_in_contact).to(gs.tc_float).sum(dim=1)
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        active = (cmd_speed > self.reward_cfg["gait_cmd_threshold"]).to(gs.tc_float)
        penalty = mismatch * active
        if self.gait_couple_cmd_acc:
            penalty = penalty * self.command_accuracy
        return penalty

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
        """Strafe: Rumpf soll aufrecht bleiben — roll² + pitch² (dynamische Stabilität, push-resistent).

        base_euler liegt in Grad vor → in Radian umrechnen, damit die Skala physikalisch sinnvoll ist.

        Beispiel (scale=-2.0, dt=0.02):
          roll=0°, pitch=0°  → 0           → 0/Step
          roll=10°, pitch=0° → 0.0305 rad² → -0.0012/Step
          roll=20°, pitch=10°→ 0.1523 rad² → -0.0061/Step
        """
        rp = torch.deg2rad(self.base_euler[:, :2])
        return torch.sum(torch.square(rp), dim=1)

    def _reward_ang_vel_xy(self):
        """Strafe: Roll-/Pitch-Raten des Rumpfes dämpfen (kein Kippeln/Schwanken).

        Summiert (ω_x² + ω_y²) der Körper-Winkelgeschwindigkeit.

        Beispiel (scale=-0.05, dt=0.02):
          ω_xy=0           → 0      → 0/Step
          ω_x=0.5 rad/s    → 0.25   → -0.00025/Step
        """
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

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
