"""Hero-Agent-Test: spielt die NPZ-Referenzmotion 'perfekt' ab (kinematischer Replay), schickt jeden
Frame durch die Reward-Funktionen von K1Env und loggt Rewards + ein annotiertes Video nach wandb.

Sinn: Obergrenze/Sanity-Check der Reward-Gestaltung. Ein perfekter Imitator der Referenz sollte
hohe style/tracking-Rewards und niedrige Strafterme bekommen. Weicht das ab, stimmt etwas im
Reward (Feature-Definition, Skalen, Vorzeichen) oder im Retarget nicht.

Replay-Prinzip: pro Frame wird der Roboter via set_qpos auf die Referenz-Pose gesetzt (FK an →
Fußpositionen + Rendering korrekt) und die internen K1Env-Buffer werden direkt aus der NPZ befüllt
(zuverlässiger als get_vel nach Teleport). Es läuft KEINE Physik — die Bewegung ist exakt die NPZ.

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

import cv2
import genesis as gs
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

WALK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(WALK_DIR)
CONFIG_PATH = os.path.join(WALK_DIR, "config", "k1_env.yaml")


def load_cfgs():
    with open(CONFIG_PATH, "r") as f:
        c = yaml.safe_load(f)
    return c["env_cfg"], c["obs_cfg"], c["reward_cfg"], c["command_cfg"]


def draw_hud(rgb: np.ndarray, cmd_vx: float, cmd_vy: float, speed: float) -> np.ndarray:
    """Brennt die Command-Geschwindigkeit als Text in den Frame (Richtung zeigt der 3D-Pfeil über dem Kopf)."""
    lines = [
        f"cmd speed: {speed:0.2f} m/s",
        f"cmd vx/vy: {cmd_vx:+0.2f} / {cmd_vy:+0.2f} m/s",
    ]
    for i, s in enumerate(lines):
        org = (14, 30 + 28 * i)
        cv2.putText(rgb, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(rgb, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 60), 1, cv2.LINE_AA)
    return rgb


def project_to_pixels(world_pts: np.ndarray, cam) -> tuple[np.ndarray, np.ndarray]:
    """Projiziert Welt-Punkte (N,3) in Pixelkoordinaten der Kamera. Gibt (uv (N,2), z_cam (N,)) zurück;
    z_cam > 0 = vor der Kamera. extrinsics ist ein cached_property (folgt der bewegten Kamera NICHT) →
    Welt→Kamera hier je Frame frisch aus cam.transform (OpenCV-Konvention: y/z-Spalten gespiegelt)."""
    T = np.asarray(cam.transform, dtype=np.float64)
    if T.ndim == 3:
        T = T[0]
    Tf = T.copy()
    Tf[:3, 1:3] *= -1  # Genesis-Kameraachsen → OpenCV (x rechts, y runter, z vorwärts)
    E = np.linalg.inv(Tf)  # Welt → Kamera
    hom = np.concatenate([np.asarray(world_pts, np.float64), np.ones((len(world_pts), 1))], axis=1)
    cam_pts = (E @ hom.T).T[:, :3]
    z = cam_pts[:, 2]
    zc = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = np.stack([cam.f * cam_pts[:, 0] / zc + cam.cx, cam.f * cam_pts[:, 1] / zc + cam.cy], axis=1)
    return uv, z


def draw_command_arrow(rgb: np.ndarray, env, cmd_unit_world: torch.Tensor, speed: float) -> None:
    """Zeichnet den Command-Pfeil als 2D-Projektion eines 3D-Pfeils über dem Kopf (Genesis-Debug-Objekte
    erscheinen nicht im Offscreen-Kamerarender, daher selbst projizieren). Horizontal in Kommandorichtung,
    Länge ~ Geschwindigkeit; verankert in der Welt, folgt dem Roboter."""
    horiz = cmd_unit_world.clone()
    horiz[2] = 0.0
    n = torch.norm(horiz).clamp(min=1e-6)
    horiz = horiz / n
    length = 0.35 + 0.55 * speed  # Mindestlänge + Skalierung mit Tempo
    base = env.base_pos[0].detach().cpu().numpy()
    p0 = base + np.array([0.0, 0.0, 0.65])  # über dem Kopf
    p1 = p0 + horiz.detach().cpu().numpy() * length

    uv, z = project_to_pixels(np.stack([p0, p1]), env.cam)
    if z[0] <= 0 or z[1] <= 0:  # hinter der Kamera → nicht zeichnen
        return
    a = (int(round(uv[0, 0])), int(round(uv[0, 1])))
    b = (int(round(uv[1, 0])), int(round(uv[1, 1])))
    cv2.arrowedLine(rgb, a, b, (0, 0, 0), 7, cv2.LINE_AA, tipLength=0.3)  # Kontur
    cv2.arrowedLine(rgb, a, b, (60, 255, 60), 4, cv2.LINE_AA, tipLength=0.3)


def render_frame(env, cmd_unit_world: torch.Tensor, cmd_vx: float, cmd_vy: float, speed: float) -> np.ndarray:
    """Rendern (force_render=True, da kein scene.step() → sonst _t-Guard → eingefrorener Render), dann
    den projizierten 3D-Command-Pfeil über dem Kopf und den Geschwindigkeits-Text aufbrennen."""
    # force_render=True übernimmt die per set_qpos gesetzte Pose in den Renderer und führt die Kamera nach.
    out = env.cam.render(force_render=True)
    rgb = out[0] if isinstance(out, (tuple, list)) else out
    rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3]).astype(np.uint8)
    draw_command_arrow(rgb, env, cmd_unit_world, speed)  # nach render → cam.transform ist aktuell
    return draw_hud(rgb, cmd_vx, cmd_vy, speed)


def main():
    parser = argparse.ArgumentParser(description="K1 Hero-Agent-Test (perfekter NPZ-Replay → Rewards → wandb)")
    parser.add_argument("--motion", default=None, help="NPZ-Pfad; Default = reward_cfg.style_motion_file")
    parser.add_argument("--out", default=os.path.join("logs", "hero", "hero_test.mp4"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--project", default="k1-locomotion")
    parser.add_argument("--name", default="hero-agent-test")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = ganze Motion")
    parser.add_argument("--cmd-vx", type=float, default=None, help="konst. vx [m/s]; Default = Ref-Mittel")
    parser.add_argument("--cmd-vy", type=float, default=None, help="konst. vy [m/s]; Default = Ref-Mittel")
    parser.add_argument("--cmd-yaw", type=float, default=None, help="konst. yaw [rad/s]; Default = Ref-Mittel")
    # Kamera (seitlicher, etwas weiter weg als die Trainings-Defaults für bessere Sicht auf den Roboter).
    parser.add_argument("--cam-pos", type=float, nargs=3, default=[1.2, -4.2, 1.3], help="Kamera-Offset zum Roboter")
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=[0.0, 0.0, 0.5], help="Kamera-Blickpunkt")
    parser.add_argument("--cam-fov", type=float, default=32.0, help="Kamera-FOV (kleiner = stärker gezoomt)")
    parser.add_argument("--follow-smoothing", type=float, default=0.9, help="Verfolgungskamera-Glättung (0..1)")
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

    # ---- Motion laden + auf Roboter-/Policy-Gelenkreihenfolge mappen ----
    motion_path = args.motion or reward_cfg["style_motion_file"]
    if not os.path.isabs(motion_path):
        motion_path = os.path.join(REPO_ROOT, motion_path)
    m = np.load(motion_path, allow_pickle=True)
    npz_jn = [str(x) for x in m["joint_names"]]
    T = int(m["root_pos"].shape[0])
    if args.max_frames:
        T = min(T, args.max_frames)
    fps = float(m["fps"])

    robot_joint_names = [j.name for j in env.robot.joints[1:]]  # 22 DOF in Roboter-Reihenfolge (wie init_dof_pos)
    col_full = [npz_jn.index(n) for n in robot_joint_names]      # → NPZ-Spalten (22)
    col_motor = [npz_jn.index(n) for n in env_cfg["joint_names"]]  # 16 Policy-Gelenke → NPZ-Spalten

    root_pos = torch.tensor(m["root_pos"], dtype=gs.tc_float, device=dev)
    root_quat = torch.tensor(m["root_quat"], dtype=gs.tc_float, device=dev)  # bereits wxyz (Genesis-Konvention)
    root_lin_vel = torch.tensor(m["root_lin_vel"], dtype=gs.tc_float, device=dev)  # Welt-Frame
    root_ang_vel_body = torch.tensor(m["root_ang_vel_body"], dtype=gs.tc_float, device=dev)  # Body-Frame
    dof_pos_npz = torch.tensor(m["dof_pos"], dtype=gs.tc_float, device=dev)
    dof_vel_npz = torch.tensor(m["dof_vel"], dtype=gs.tc_float, device=dev)

    reward_names = sorted(n[len("_reward_"):] for n in dir(env) if n.startswith("_reward_"))
    print(f"[hero] motion={os.path.basename(motion_path)} frames={T} fps={fps}  rewards={reward_names}")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project=args.project,
            name=args.name,
            config={
                "motion": os.path.basename(motion_path),
                "frames": T,
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
    heading_quat = inv_quat(inv_heading_quat)

    # KONSTANTES Command für die ganze Episode (wie im Training, das nur alle resampling_time_s neu zieht).
    # Default = Mittel der Referenz-Geschwindigkeit im Heading-Frame → fair erreichbares Ziel für den
    # perfekten Imitator. tracking_* misst dann die Abweichung der momentanen Jog-Geschwindigkeit von
    # diesem festen Ziel (oszilliert pro Schritt) — NICHT mehr per-Frame nachgezogen.
    hv = transform_by_quat(root_lin_vel[:T], inv_heading_quat.expand(T, -1))  # (T,3) Heading-Frame
    mean_vx, mean_vy = float(hv[:, 0].mean()), float(hv[:, 1].mean())
    mean_yaw = float(root_ang_vel_body[:T, 2].mean())
    cmd_vx = args.cmd_vx if args.cmd_vx is not None else mean_vx
    cmd_vy = args.cmd_vy if args.cmd_vy is not None else mean_vy
    cmd_yaw = args.cmd_yaw if args.cmd_yaw is not None else mean_yaw
    env.commands[:, 0] = cmd_vx
    env.commands[:, 1] = cmd_vy
    env.commands[:, 2] = cmd_yaw
    speed = float((cmd_vx**2 + cmd_vy**2) ** 0.5)
    # Command-Richtung (Heading → Welt) für den 3D-Pfeil — konstant über die Episode.
    cmd_dir = torch.tensor([cmd_vx, cmd_vy, 0.0], dtype=gs.tc_float, device=dev)
    cmd_unit_world = transform_by_quat((cmd_dir / max(speed, 1e-6)).unsqueeze(0), heading_quat)[0]
    print(f"[hero] konstantes Command: vx={cmd_vx:+.3f} vy={cmd_vy:+.3f} yaw={cmd_yaw:+.3f}  |v|={speed:.3f} m/s")

    if use_wandb:
        wandb.config.update({"cmd_vx": cmd_vx, "cmd_vy": cmd_vy, "cmd_yaw": cmd_yaw, "cmd_speed": speed})

    with torch.no_grad():
        for t in range(T):
            bq = root_quat[t].unsqueeze(0)  # (1,4) wxyz
            inv_bq = inv_quat(bq)

            # 1) Roboter-Pose setzen (FK an → Fußpositionen + Render). KEINE Physik.
            dof_full = dof_pos_npz[t, col_full]
            qpos = torch.cat([root_pos[t], root_quat[t], dof_full]).unsqueeze(0)  # (1,29)
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
            env.base_pos.copy_(root_pos[t].unsqueeze(0))
            env.base_quat.copy_(bq)
            env.base_euler = quat_to_xyz(transform_quat_by_quat(inv_base_init_quat, bq), rpy=True, degrees=True)
            env.base_lin_vel.copy_(transform_by_quat(wlv, inv_bq))
            env.base_lin_vel_heading.copy_(transform_by_quat(wlv, inv_heading_quat))
            env.base_ang_vel.copy_(root_ang_vel_body[t].unsqueeze(0))
            env.projected_gravity.copy_(transform_by_quat(global_gravity, inv_bq))
            env.dof_pos.copy_(env.robot.get_dofs_position(env.motors_dof_idx))
            env.dof_vel.copy_(dof_vel_npz[t, col_motor].unsqueeze(0))

            # Command ist konstant (vor der Schleife gesetzt); tracking_* misst die Abweichung der
            # momentanen Referenz-Geschwindigkeit (base_lin_vel_heading/base_ang_vel) vom festen Ziel.
            env._update_foot_contact()  # füllt foot_in_contact, feet_air_time_reward, leg_symmetry_penalty

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

            frames.append(render_frame(env, cmd_unit_world, cmd_vx, cmd_vy, speed))

            if use_wandb:
                wandb.log(row, step=t)

    # ---- Video schreiben + zu wandb ----
    imageio.mimsave(args.out, frames, fps=int(round(fps)))
    print(f"[hero] Video geschrieben: {args.out}  ({len(frames)} Frames)")

    duration_s = T / fps
    summary = {f"episode/rew_{n}": s / duration_s for n, s in episode_sums.items()}
    summary["episode/return_total"] = sum(episode_sums.values())
    # ROHE Mittel ∈[0,1]: das ist der "Hero"-Wert je Term (1.0 = perfekt). Nicht mit reward_step
    # (raw*scale*dt, daher winzig) oder reward_raw-Kurven verwechseln.
    raw_mean = {f"episode/raw_mean_{n}": s / T for n, s in raw_sums.items()}
    summary.update(raw_mean)
    print("\n[hero] ROHE Reward-Mittel (∈[0,1], 1.0=perfekt) — der eigentliche Hero-Maßstab:")
    for n in reward_names:
        print(f"  raw_mean/{n:18s} {raw_sums[n] / T:6.3f}")
    print("[hero] skalierte (raw*scale*dt) Episodenmittel:")
    for k, v in summary.items():
        if k.startswith("episode/rew_") or k == "episode/return_total":
            print(f"  {k:34s} {v:+.4f}")

    if use_wandb:
        import wandb

        wandb.log({"hero_video": wandb.Video(args.out, fps=int(round(fps)), format="mp4")}, step=T)
        wandb.log(summary, step=T)
        wandb.finish()


if __name__ == "__main__":
    main()
