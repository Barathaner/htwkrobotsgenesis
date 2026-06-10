"""K1 Schritt-0 Debug-Skript (kein RL).

Ziele:
  1. URDF laedt ohne Fehler.
  2. Gelenke / DOF-Indizes / Limits auflisten -> models/K1/output.txt
  3. Roboter im Viewer spawnen, faellt er sinnvoll?
  4. Stehende Default-Pose (Radiant) definieren.
  5. PD-Gains (kp/kd) finden, die die Pose gegen die Schwerkraft halten.

Wichtig (per Genesis-Introspektion ermittelt):
  - K1_22dof.urdf hat eine FLOATING BASE: root_joint = DOF 0..5 (nicht ansteuern).
  - Es gibt 22 aktuierte Gelenke (lokale DOF-Indizes 6..27).
  - Genesis ordnet die DOFs NICHT in URDF-Reihenfolge -> immer per Gelenkname mappen!

Aufruf:
  .venv/bin/python models/K1/K1_debug.py            # mit Viewer
  HEADLESS=1 .venv/bin/python models/K1/K1_debug.py # ohne Viewer (CI/Test)
"""

import os

import numpy as np
import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
from genesis.vis.keybindings import Key, KeyAction, Keybind
from genesis.vis.viewer_plugins.base import ViewerPlugin

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "K1_22dof.urdf")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.txt")
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

# Spawn-Hoehe: bei Null-Pose liegt der Fuss ~0.514 m unter dem Trunk.
# Etwas hoeher spawnen, damit die Fuesse knapp ueber dem Boden starten und sich setzen.
BASE_INIT_POS = (0.0, 0.0, 0.56)

# Stehende Default-Pose [rad] je Gelenkname. Null = gestreckte Beine (stabil auf flachem Boden).
# Beine bewusst leicht gebeugt fuer robusteres Stehen; bei Bedarf hier tunen.
DEFAULT_POSE = {
    # Kopf
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    # Arme (haengen seitlich)
    "ALeft_Shoulder_Pitch": 0.0,
    "ARight_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": 0.0,
    "Right_Shoulder_Roll": 0.0,
    "Left_Elbow_Pitch": 0.0,
    "Right_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": 0.0,
    "Right_Elbow_Yaw": 0.0,
    # Beine (leichte Hocke: Hip/Knee/Ankle gleichen sich aus -> Fuss flach)
    "Left_Hip_Pitch": -0.30,
    "Right_Hip_Pitch": -0.30,
    "Left_Hip_Roll": 0.0,
    "Right_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.60,
    "Right_Knee_Pitch": 0.60,
    "Left_Ankle_Pitch": -0.30,
    "Right_Ankle_Pitch": -0.30,
    "Left_Ankle_Roll": 0.0,
    "Right_Ankle_Roll": 0.0,
}

# PD-Gains [kp, kv] je Gelenkname. Grob an den URDF-effort-Limits orientiert.
GAINS = {
    "AAHead_yaw": (20.0, 0.5),
    "Head_pitch": (20.0, 0.5),
    "ALeft_Shoulder_Pitch": (40.0, 1.0),
    "ARight_Shoulder_Pitch": (40.0, 1.0),
    "Left_Shoulder_Roll": (40.0, 1.0),
    "Right_Shoulder_Roll": (40.0, 1.0),
    "Left_Elbow_Pitch": (40.0, 1.0),
    "Right_Elbow_Pitch": (40.0, 1.0),
    "Left_Elbow_Yaw": (40.0, 1.0),
    "Right_Elbow_Yaw": (40.0, 1.0),
    "Left_Hip_Pitch": (200.0, 5.0),
    "Right_Hip_Pitch": (200.0, 5.0),
    "Left_Hip_Roll": (200.0, 5.0),
    "Right_Hip_Roll": (200.0, 5.0),
    "Left_Hip_Yaw": (150.0, 4.0),
    "Right_Hip_Yaw": (150.0, 4.0),
    "Left_Knee_Pitch": (250.0, 6.0),
    "Right_Knee_Pitch": (250.0, 6.0),
    "Left_Ankle_Pitch": (120.0, 3.0),
    "Right_Ankle_Pitch": (120.0, 3.0),
    "Left_Ankle_Roll": (120.0, 3.0),
    "Right_Ankle_Roll": (120.0, 3.0),
}

# Drehmoment-Grenzen [Nm] aus der URDF (effort), je Gelenkname.
EFFORT = {
    "AAHead_yaw": 6.0, "Head_pitch": 6.0,
    "ALeft_Shoulder_Pitch": 14.0, "ARight_Shoulder_Pitch": 14.0,
    "Left_Shoulder_Roll": 14.0, "Right_Shoulder_Roll": 14.0,
    "Left_Elbow_Pitch": 14.0, "Right_Elbow_Pitch": 14.0,
    "Left_Elbow_Yaw": 14.0, "Right_Elbow_Yaw": 14.0,
    "Left_Hip_Pitch": 68.0, "Right_Hip_Pitch": 68.0,
    "Left_Hip_Roll": 76.0, "Right_Hip_Roll": 76.0,
    "Left_Hip_Yaw": 38.3, "Right_Hip_Yaw": 38.3,
    "Left_Knee_Pitch": 112.0, "Right_Knee_Pitch": 112.0,
    "Left_Ankle_Pitch": 38.3, "Right_Ankle_Pitch": 38.3,
    "Left_Ankle_Roll": 38.3, "Right_Ankle_Roll": 38.3,
}


def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
        ),
        show_viewer=not HEADLESS,
    )

    # Boden NICHT vergessen - sonst faellt der Roboter ins Unendliche.
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file=URDF, pos=BASE_INIT_POS))

    plugin = ImGuiOverlayPlugin()
    scene.viewer.add_plugin(plugin)
    scene.build()

    # --- Nur ansteuerbare Gelenke (1 DOF), Basis (root_joint, 6 DOF) ausgeschlossen ---
    motor_joints = [j for j in robot.joints if j.n_dofs == 1]
    motor_names = [j.name for j in motor_joints]
    motor_dofs_local = [j._dof_idx_local() for j in motor_joints]

    # --- Gelenkinfo in Datei schreiben ---
    with open(OUTPUT, "w") as f:
        f.write(f"# K1_22dof.urdf  n_dofs={robot.n_dofs}  n_links={robot.n_links}\n")
        f.write(f"# Basis: root_joint = DOF 0..5 (floating, nicht ansteuern)\n")
        f.write(f"# {len(motor_joints)} ansteuerbare Gelenke (lokale DOF-Indizes):\n\n")
        f.write(f"{'joint_name':28s} {'dof_local':>9s} {'lower':>8s} {'upper':>8s}\n")
        for j in motor_joints:
            lim = np.array(j.dofs_limit).reshape(-1)
            f.write(f"{j.name:28s} {j._dof_idx_local():>9d} {lim[0]:>8.3f} {lim[1]:>8.3f}\n")
    print(f"Gelenkinfo -> {OUTPUT}")

    # --- PD-Gains, Force-Range und Default-Pose in lokaler DOF-Reihenfolge aufbauen ---
    kp = np.array([GAINS[n][0] for n in motor_names], dtype=np.float32)
    kv = np.array([GAINS[n][1] for n in motor_names], dtype=np.float32)
    eff = np.array([EFFORT[n] for n in motor_names], dtype=np.float32)
    target = np.array([DEFAULT_POSE[n] for n in motor_names], dtype=np.float32)

    robot.set_dofs_kp(kp, motor_dofs_local)
    robot.set_dofs_kv(kv, motor_dofs_local)
    robot.set_dofs_force_range(-eff, eff, motor_dofs_local)

    # Startpose einmal hart setzen (Roboter beginnt in der Zielpose).
    robot.set_dofs_position(target, motor_dofs_local, zero_velocity=True)

    # --- Halte-Test: PD jeden Schritt auf die Zielpose ---
    print("Halte-Test laeuft. Beobachte Beckenhoehe (sollte ~konstant bleiben).")
    steps = 100 if HEADLESS else 100000
    start_h = float(robot.get_pos()[2])
    for i in range(steps):
        robot.control_dofs_position(target, motor_dofs_local)
        scene.step()
        if i % 20 == 0:
            h = float(robot.get_pos()[2])
            print(f"step {i:5d}  Beckenhoehe = {h:.3f} m  (delta {h - start_h:+.3f})")


if __name__ == "__main__":
    main()
