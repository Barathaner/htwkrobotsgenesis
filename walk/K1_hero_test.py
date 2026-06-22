"""Hero-Agent-Test: spielt die NPZ-Referenzmotion 'perfekt' ab (kinematischer Replay), schickt jeden
Frame durch die Reward-Funktionen von K1Env und loggt Rewards + ein annotiertes Video nach wandb.

Sinn: Obergrenze/Sanity-Check der Reward-Gestaltung. Ein perfekter Imitator der Referenz sollte
hohe style/tracking-Rewards und niedrige Strafterme bekommen. Weicht das ab, stimmt etwas im
Reward (Feature-Definition, Skalen, Vorzeichen) oder im Retarget nicht.

Replay-Prinzip: pro Frame wird der Roboter via set_qpos auf die Referenz-Pose gesetzt (FK an →
Fußpositionen + Rendering korrekt) und die internen K1Env-Buffer werden direkt aus der NPZ befüllt
(zuverlässiger als get_vel nach Teleport). Es läuft KEINE Physik — die Bewegung ist exakt die NPZ.
Policy-Actions werden kinematisch approximiert (action = (dof_pos − default) / action_scale), damit
action_rate und andere action-abhängige Terme im Replay sinnvoll geloggt werden.

Das Command wird je Frame auf die Referenz-Geschwindigkeit (Heading-Frame) gesetzt → tracking ≈ 1.
Im Video werden Command-Richtung (3D-Pfeil + 2D-HUD) und -Geschwindigkeit eingeblendet.

Usage (from repo root, mit dem genesis-venv):
  /pfad/zum/python walk/K1_hero_test.py
  ... walk/K1_hero_test.py --no-wandb --out logs/hero/hero.mp4
  ... walk/K1_hero_test.py --motion walk/data/motions/k1_jogging_motion.npz --max-frames 200
"""

import argparse
import os
import sys

import genesis as gs
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

from k1_video_overlay import render_annotated_frame

WALK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(WALK_DIR)
CONFIG_PATH = os.path.join(WALK_DIR, "config", "k1_env.yaml")


def load_cfgs():
    with open(CONFIG_PATH, "r") as f:
        c = yaml.safe_load(f)
    return c["env_cfg"], c["obs_cfg"], c["reward_cfg"], c["command_cfg"]


def main():
    parser = argparse.ArgumentParser(description="K1 Hero-Agent-Test (perfekter NPZ-Replay → Rewards → wandb)")
    parser.add_argument("--motion", default=None, help="NPZ-Pfad; Default = reward_cfg.style_motion_file")
    parser.add_argument("--out", default=os.path.join("logs", "hero", "hero_test.mp4"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--project", default="k1-locomotion")
    parser.add_argument("--name", default="hero-agent-test")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Gesamt-Replay-Schritte (0 = episode_length_s, Referenz wird dabei geloopt)")
    parser.add_argument("--cmd-mode", choices=["instant", "mean"], default="instant",
                        help="instant: Command folgt je Frame der Referenz (tracking≈1, Formel-Validierung); "
                             "mean: ein konstantes Command wie im Training (tracking oszilliert)")
    parser.add_argument("--cmd-vx", type=float, default=None, help="konst. vx [m/s] (cmd-mode=mean); Def=Ref-Mittel")
    parser.add_argument("--cmd-vy", type=float, default=None, help="konst. vy [m/s]; Default = Ref-Mittel")
    parser.add_argument("--cmd-yaw", type=float, default=None, help="konst. yaw [rad/s]; Default = Ref-Mittel")
    # Kamera (seitlicher, etwas weiter weg als die Trainings-Defaults für bessere Sicht auf den Roboter).
    parser.add_argument("--cam-pos", type=float, nargs=3, default=[1.2, -4.2, 1.3], help="Kamera-Offset zum Roboter")
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=[0.0, 0.0, 0.5], help="Kamera-Blickpunkt")
    parser.add_argument("--cam-fov", type=float, default=32.0, help="Kamera-FOV (kleiner = stärker gezoomt)")
    parser.add_argument("--follow-smoothing", type=float, default=0.9, help="Verfolgungskamera-Glättung (0..1)")
    parser.add_argument("--contact-height", type=float, default=0.06,
                        help="Fuß gilt als Bodenkontakt, wenn foot_clear < diesem Wert [m]. KINEMATISCHE "
                             "Kontakterkennung für den Replay: ohne scene.step() liefert get_links_net_contact_force() "
                             "immer 0 → feet_air_time/feet_slip/leg_symmetry sonst dauerhaft 0.")
    args = parser.parse_args()

    if WALK_DIR not in sys.path:
        sys.path.insert(0, WALK_DIR)
    from K1_env import K1Env

    env_cfg, obs_cfg, reward_cfg, command_cfg = load_cfgs()

    # Kamera für den Hero-Test überschreiben: seitlicher + etwas weiter weg als Trainings-Defaults.
    # follow_entity hält diesen Offset und schwenkt mit dem Roboter mit (siehe K1Env.__init__).
    vid = dict(env_cfg.get("video") or {})
    vid.update(
        {
            "camera_pos": list(args.cam_pos),
            "camera_lookat": list(args.cam_lookat),
            "camera_fov": args.cam_fov,
            "follow": True,
            "follow_smoothing": args.follow_smoothing,
        }
    )
    env_cfg["video"] = vid

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    env = K1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        record_camera=True,
    )
    assert env.cam is not None, "record_camera=True hat keine Kamera erzeugt"
    dev = gs.device

    # Replay läuft OHNE scene.step() → get_links_net_contact_force() ist konstant 0, daher wären
    # feet_air_time/feet_slip/leg_symmetry sonst dauerhaft 0. Kontakt stattdessen kinematisch aus der
    # Fußhöhe ableiten (foot_clear < contact_height) und in dieselbe Env-Buchhaltung einspeisen, die
    # auch das Training nutzt (env._update_foot_contact ruft _foot_in_contact_from_force).
    if getattr(env, "style_ground_z", None) is not None:
        _foot_ground_z = env.style_ground_z  # Steh-Fußhöhe (FK auf Default-Pose), wie im style-Reward
    else:
        _q0 = env.init_qpos.unsqueeze(0).expand(env.num_envs, -1).contiguous()
        _links_pos, _ = env.robot.forward_kinematics(_q0)
        _foot_ground_z = _links_pos[0, env.feet_link_idx, 2].clone()

    def _kinematic_foot_contact():
        foot_z = env.robot.get_links_pos(env.feet_idx_local)[:, :, 2]  # (1, 2) Welt-z
        return (foot_z - _foot_ground_z) < args.contact_height          # (1, 2) bool

    env._foot_in_contact_from_force = _kinematic_foot_contact

    # ---- Motion laden + auf Roboter-/Policy-Gelenkreihenfolge mappen ----
    motion_path = args.motion or reward_cfg["style_motion_file"]
    if not os.path.isabs(motion_path):
        motion_path = os.path.join(REPO_ROOT, motion_path)
    m = np.load(motion_path, allow_pickle=True)
    npz_jn = [str(x) for x in m["joint_names"]]
    T = int(m["root_pos"].shape[0])  # volle Referenzlänge = Loop-Quelle
    fps = float(m["fps"])
    # Wie im Training: die Referenz endlos loopen, die "Episode" aber bei episode_length_s kappen.
    # n_steps = Gesamt-Replay-Länge; --max-frames überschreibt (für schnelle Tests).
    episode_length_s = float(env_cfg["episode_length_s"])
    n_steps = args.max_frames if args.max_frames else round(episode_length_s * fps)

    robot_joint_names = [j.name for j in env.robot.joints[1:]]  # 22 DOF in Roboter-Reihenfolge (wie init_dof_pos)
    col_full = [npz_jn.index(n) for n in robot_joint_names]      # → NPZ-Spalten (22)
    col_motor = [npz_jn.index(n) for n in env_cfg["joint_names"]]  # 16 Policy-Gelenke → NPZ-Spalten

    root_pos = torch.tensor(m["root_pos"], dtype=gs.tc_float, device=dev)
    root_quat = torch.tensor(m["root_quat"], dtype=gs.tc_float, device=dev)  # bereits wxyz (Genesis-Konvention)
    root_lin_vel = torch.tensor(m["root_lin_vel"], dtype=gs.tc_float, device=dev)  # Welt-Frame
    root_ang_vel_body = torch.tensor(m["root_ang_vel_body"], dtype=gs.tc_float, device=dev)  # Body-Frame
    dof_pos_npz = torch.tensor(m["dof_pos"], dtype=gs.tc_float, device=dev)
    dof_vel_npz = torch.tensor(m["dof_vel"], dtype=gs.tc_float, device=dev)

    # Netto-Vorwärtsversatz je komplettem Motion-Zyklus (nur x/y), damit der geloopte Replay
    # vorwärts weiterläuft statt zum Startpunkt zurückzuspringen. z bleibt periodisch → kein Höhendrift.
    loop_disp = (root_pos[T - 1] - root_pos[0]).clone()
    loop_disp[2] = 0.0

    reward_names = sorted(n[len("_reward_"):] for n in dir(env) if n.startswith("_reward_"))
    print(f"[hero] motion={os.path.basename(motion_path)} frames={T} steps={n_steps} "
          f"(~{n_steps / fps:.1f}s) fps={fps}  rewards={reward_names}")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project=args.project,
            name=args.name,
            config={
                "motion": os.path.basename(motion_path),
                "frames": n_steps,
                "motion_frames": T,
                "fps": fps,
                "reward_scales": dict(reward_cfg["reward_scales"]),
            },
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames = []
    episode_sums = {n: 0.0 for n in reward_names}  # skalierte Beiträge (raw*scale*dt)
    raw_sums = {n: 0.0 for n in reward_names}       # rohe Outputs ∈[0,1] → mittlerer erreichbarer Wert

    global_gravity = env.global_gravity.unsqueeze(0)  # (1,3)
    inv_base_init_quat = env.inv_base_init_quat
    inv_heading_quat = env.inv_heading_quat  # (1,4), Heading = Spawn-Yaw (hier Identität)

    # Command-Modus:
    #  instant (Default): Command = MOMENTANE Referenz-Geschwindigkeit je Frame → tracking_lin/ang_vel
    #    werden exakt exp(0)=1.0. Validiert, dass die Reward-FORMEL ihr Maximum für perfektes Folgen
    #    erreicht (echter Mimic-Test). Der Pfeil/HUD wackelt mit dem Gang — das ist hier korrekt.
    #  mean: EIN konstantes Command (Mittel der Referenz) für die ganze Episode, wie im Training. Dann
    #    misst tracking_* die Abweichung der natürlich oszillierenden Jog-Geschwindigkeit vom festen
    #    Ziel → schwankt (kein Bug, das ist das reale Trainingssignal).
    hv = transform_by_quat(root_lin_vel[:T], inv_heading_quat.expand(T, -1))  # (T,3) Heading-Frame
    mean_vx, mean_vy = float(hv[:, 0].mean()), float(hv[:, 1].mean())
    mean_yaw = float(root_ang_vel_body[:T, 2].mean())
    cmd_vx = args.cmd_vx if args.cmd_vx is not None else mean_vx
    cmd_vy = args.cmd_vy if args.cmd_vy is not None else mean_vy
    cmd_yaw = args.cmd_yaw if args.cmd_yaw is not None else mean_yaw
    speed = float((cmd_vx**2 + cmd_vy**2) ** 0.5)
    if args.cmd_mode == "mean":
        env.commands[:, 0], env.commands[:, 1], env.commands[:, 2] = cmd_vx, cmd_vy, cmd_yaw
        print(f"[hero] cmd-mode=mean (konstant): vx={cmd_vx:+.3f} vy={cmd_vy:+.3f} yaw={cmd_yaw:+.3f} |v|={speed:.3f}")
    else:
        print(f"[hero] cmd-mode=instant: Command folgt je Frame der Referenz → tracking_* sollte ≈1.0 sein "
              f"(Ref-Mittel |v|≈{(mean_vx**2 + mean_vy**2) ** 0.5:.3f})")

    if use_wandb:
        wandb.config.update({"cmd_mode": args.cmd_mode, "cmd_vx": cmd_vx, "cmd_vy": cmd_vy,
                             "cmd_yaw": cmd_yaw, "cmd_speed": speed})

    action_clip = env.env_cfg["clip_actions"]
    action_scale = env.env_cfg["action_scale"]

    def kinematic_action(dof_row: torch.Tensor) -> torch.Tensor:
        """Policy-Action aus Gelenkwinkel (Inverse von K1Env.step: target = action*scale + default)."""
        return torch.clip((dof_row - env.default_dof_pos) / action_scale, -action_clip, action_clip)

    # Warm-start: actions/last_actions auf Frame 0 → kein künstlicher action_rate-Spike (last=0).
    a0 = kinematic_action(dof_pos_npz[0, col_motor].unsqueeze(0))
    env.actions.copy_(a0)
    env.last_actions.copy_(a0)

    with torch.no_grad():
        for step in range(n_steps):
            t = step % T                          # Referenz endlos loopen
            pos_offset = (step // T) * loop_disp  # je Zyklus um den Netto-Vorlauf versetzen
            bq = root_quat[t].unsqueeze(0)  # (1,4) wxyz
            inv_bq = inv_quat(bq)

            # 1) Roboter-Pose setzen (FK an → Fußpositionen + Render). KEINE Physik.
            dof_full = dof_pos_npz[t, col_full]
            qpos = torch.cat([root_pos[t] + pos_offset, root_quat[t], dof_full]).unsqueeze(0)  # (1,29)
            env.robot.set_qpos(qpos, zero_velocity=False, skip_forward=False)
            # Geschwindigkeiten für feet_slip/get_links_vel: [lin_world(3), ang_world(3), joints(22)]
            ang_world = transform_by_quat(root_ang_vel_body[t].unsqueeze(0), bq)[0]
            vel_full = torch.cat([root_lin_vel[t], ang_world, dof_vel_npz[t, col_full]]).unsqueeze(0)
            try:
                env.robot.set_dofs_velocity(vel_full, skip_forward=False)
            except Exception:
                pass  # feet_slip dann approximativ; restliche Rewards unberührt

            # 2) K1Env-Buffer direkt aus NPZ befüllen (wie env.step() nach scene.step(), aber aus Referenz)
            wlv = root_lin_vel[t].unsqueeze(0)
            env.base_pos.copy_((root_pos[t] + pos_offset).unsqueeze(0))
            env.base_quat.copy_(bq)
            env.base_euler = quat_to_xyz(transform_quat_by_quat(inv_base_init_quat, bq), rpy=True, degrees=True)
            env.base_lin_vel.copy_(transform_by_quat(wlv, inv_bq))
            env.base_lin_vel_heading.copy_(transform_by_quat(wlv, inv_heading_quat))
            env.base_ang_vel.copy_(root_ang_vel_body[t].unsqueeze(0))
            env.projected_gravity.copy_(transform_by_quat(global_gravity, inv_bq))
            env.dof_pos.copy_(env.robot.get_dofs_position(env.motors_dof_idx))
            env.dof_vel.copy_(dof_vel_npz[t, col_motor].unsqueeze(0))

            # Kinematische Policy-Actions approximieren (Inverse von step(): target = action*scale + default).
            kin_action = kinematic_action(env.dof_pos)
            env.last_actions.copy_(env.actions)
            env.actions.copy_(kin_action)

            env._update_command_tracking_ema()  # EMA-Geschw. für das (zeitgemittelte) Command-Tracking

            # Command setzen: instant → folgt der EMA-Geschwindigkeit (tracking_* = exp(0) = 1.0,
            # validiert das Formel-Maximum). mean → bereits vor der Schleife konstant gesetzt.
            if args.cmd_mode == "instant":
                env.commands[:, 0] = env.lin_vel_ema[:, 0]
                env.commands[:, 1] = env.lin_vel_ema[:, 1]
                env.commands[:, 2] = env.ang_vel_z_ema
                cmd_vx = float(env.commands[0, 0])
                cmd_vy = float(env.commands[0, 1])
                cmd_yaw = float(env.commands[0, 2])
                speed = float(torch.norm(env.commands[0, :2]).item())

            env._update_foot_contact()

            # 4) Rewards: rohen Output (gebunden ∈[0,1]) und skalierten Beitrag je Term
            row = {}
            total = 0.0
            for n in reward_names:
                raw = float(getattr(env, "_reward_" + n)().mean().item())
                row[f"reward_raw/{n}"] = raw
                raw_sums[n] += raw
                scale = env.reward_scales.get(n)  # bereits * dt; None für deaktivierte Terme
                if scale is not None:
                    contrib = raw * scale
                    row[f"reward_step/{n}"] = contrib
                    total += contrib
                    episode_sums[n] += contrib
            row["reward_step/total"] = total

            # Zur Kontrolle: tatsächliche (momentane) Referenz-Geschwindigkeit im Heading-Frame loggen.
            row["cmd/speed"] = speed
            row["cmd/vx"] = cmd_vx
            row["cmd/vy"] = cmd_vy
            row["cmd/yaw"] = cmd_yaw
            row["actual/vx"] = float(env.base_lin_vel_heading[0, 0])
            row["actual/vy"] = float(env.base_lin_vel_heading[0, 1])
            row["actual/speed"] = float(torch.norm(env.base_lin_vel_heading[0, :2]).item())

            frames.append(render_annotated_frame(env, force_render=True))

            if use_wandb:
                wandb.log(row, step=step)

    # ---- Video schreiben + zu wandb ----
    imageio.mimsave(args.out, frames, fps=int(round(fps)))
    print(f"[hero] Video geschrieben: {args.out}  ({len(frames)} Frames)")

    duration_s = n_steps / fps
    summary = {f"episode/rew_{n}": s / duration_s for n, s in episode_sums.items()}
    summary["episode/return_total"] = sum(episode_sums.values())
    # ROHE Mittel ∈[0,1]: das ist der "Hero"-Wert je Term (1.0 = perfekt). Nicht mit reward_step
    # (raw*scale*dt, daher winzig) oder reward_raw-Kurven verwechseln.
    raw_mean = {f"episode/raw_mean_{n}": s / n_steps for n, s in raw_sums.items()}
    summary.update(raw_mean)
    print("\n[hero] ROHE Reward-Mittel (∈[0,1], 1.0=perfekt) — der eigentliche Hero-Maßstab:")
    for n in reward_names:
        print(f"  raw_mean/{n:18s} {raw_sums[n] / n_steps:6.3f}")
    print("[hero] skalierte (raw*scale*dt) Episodenmittel:")
    for k, v in summary.items():
        if k.startswith("episode/rew_") or k == "episode/return_total":
            print(f"  {k:34s} {v:+.4f}")

    if use_wandb:
        import wandb

        wandb.log({"hero_video": wandb.Video(args.out, fps=int(round(fps)), format="mp4")}, step=n_steps)
        wandb.log(summary, step=n_steps)
        wandb.finish()


if __name__ == "__main__":
    main()
