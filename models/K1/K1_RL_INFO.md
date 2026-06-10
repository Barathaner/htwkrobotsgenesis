# Booster K1 – Referenz für Reinforcement Learning (PPO, Gym-artige Struktur)

Alle Werte wurden per Genesis-Introspektion (Version 1.1.0) aus
`models/K1/K1_22dof.urdf` ermittelt und mit einem Steh-Test verifiziert
(`models/K1/K1_debug.py`).

---

## 1. Asset

| | |
|---|---|
| URDF | `models/K1/K1_22dof.urdf` |
| MJCF | `models/K1/K1_22dof.xml` |
| Locomotion-URDF (Arme/Kopf fixed) | `models/K1/K1_locomotion.urdf` |
| Meshes | `models/K1/meshes/*.STL` |
| Trunk-Masse | 6.5 kg |
| Lädt fehlerfrei? | Ja (keine korrupten `.dae` wie bei Go2) |

`K1_22dof.urdf` lädt sauber. Wenn du für reines Laufen eine **kleinere
Action-Space** willst, nimm `K1_locomotion.urdf` (Kopf/Arme sind dort `fixed`).

---

## 2. DOF-Struktur (WICHTIG)

```
n_dofs  = 28   ->  6 (floating base) + 22 (aktuiert)
n_links = 23
```

- **`root_joint`** = lokale DOF **0..5** = floating base (Position + Orientierung).
  **Niemals** ansteuern. Im `qpos` sind das die ersten 7 Einträge (3 pos + 4 quat).
- **22 aktuierte Gelenke** = lokale DOF **6..27**.

> ⚠️ Genesis ordnet die DOFs **NICHT** in URDF-Reihenfolge. Immer per
> **Gelenkname** mappen, nie per fester Index-Annahme:
> ```python
> motor_joints = [j for j in robot.joints if j.n_dofs == 1]
> motor_dofs_local = [j._dof_idx_local() for j in motor_joints]
> ```

---

## 3. Gelenke, lokale DOF-Indizes, Limits, Drehmoment

| Gelenk | dof_local | lower [rad] | upper [rad] | effort [Nm] | vel [rad/s] | Gruppe |
|---|---:|---:|---:|---:|---:|---|
| AAHead_yaw | 6 | -1.000 | 1.000 | 6.0 | 18.0 | Kopf |
| ALeft_Shoulder_Pitch | 7 | -3.316 | 1.220 | 14.0 | 18.0 | Arm |
| ARight_Shoulder_Pitch | 8 | -3.316 | 1.220 | 14.0 | 18.0 | Arm |
| Left_Hip_Pitch | 9 | -3.000 | 2.210 | 68.0 | 7.1 | **Bein** |
| Right_Hip_Pitch | 10 | -3.000 | 2.210 | 68.0 | 7.1 | **Bein** |
| Head_pitch | 11 | -0.349 | 0.855 | 6.0 | 18.0 | Kopf |
| Left_Shoulder_Roll | 12 | -1.740 | 1.570 | 14.0 | 18.0 | Arm |
| Right_Shoulder_Roll | 13 | -1.570 | 1.740 | 14.0 | 18.0 | Arm |
| Left_Hip_Roll | 14 | -0.400 | 1.570 | 76.0 | 12.9 | **Bein** |
| Right_Hip_Roll | 15 | -1.570 | 0.400 | 76.0 | 12.9 | **Bein** |
| Left_Elbow_Pitch | 16 | -2.270 | 2.270 | 14.0 | 18.0 | Arm |
| Right_Elbow_Pitch | 17 | -2.270 | 2.270 | 14.0 | 18.0 | Arm |
| Left_Hip_Yaw | 18 | -1.000 | 1.000 | 38.3 | 18.1 | **Bein** |
| Right_Hip_Yaw | 19 | -1.000 | 1.000 | 38.3 | 18.1 | **Bein** |
| Left_Elbow_Yaw | 20 | -2.440 | 0.000 | 14.0 | 18.0 | Arm |
| Right_Elbow_Yaw | 21 | 0.000 | 2.440 | 14.0 | 18.0 | Arm |
| Left_Knee_Pitch | 22 | 0.000 | 2.230 | 112.0 | 12.5 | **Bein** |
| Right_Knee_Pitch | 23 | 0.000 | 2.230 | 112.0 | 12.5 | **Bein** |
| Left_Ankle_Pitch | 24 | -0.870 | 0.345 | 38.3 | 18.1 | **Bein** |
| Right_Ankle_Pitch | 25 | -0.870 | 0.345 | 38.3 | 18.1 | **Bein** |
| Left_Ankle_Roll | 26 | -0.345 | 0.345 | 38.3 | 18.1 | **Bein** |
| Right_Ankle_Roll | 27 | -0.345 | 0.345 | 38.3 | 18.1 | **Bein** |

Die 12 **Bein-Gelenke** sind die typische Action-Space für reines Laufen.
Arme (8) + Kopf (2) kannst du anfangs fixieren oder mit hohem `kp` auf Default halten.

---

## 4. Verifizierte Steh-Pose + PD-Gains

Getestet mit `K1_debug.py`: Roboter steht **stabil über 600 Schritte**
(12 s bei dt=0.02), Beckenhöhe konstant **0.531 m**, kein Umfallen/Drift.

### Spawn

```python
base_init_pos  = (0.0, 0.0, 0.56)   # Fuesse starten knapp ueber Boden
base_init_quat = (1.0, 0.0, 0.0, 0.0)
# eingeschwungene Trunk-Hoehe nach dem Setzen ~ 0.531 m
```

### Default-Pose (stehende Hocke) [rad]

```python
DEFAULT_POSE = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0, "ARight_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": 0.2, "Right_Shoulder_Roll": -0.2,
    "Left_Elbow_Pitch": 0.0, "Right_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": 0.0, "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": -0.30, "Right_Hip_Pitch": -0.30,
    "Left_Hip_Roll": 0.0, "Right_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0, "Right_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.60, "Right_Knee_Pitch": 0.60,
    "Left_Ankle_Pitch": -0.30, "Right_Ankle_Pitch": -0.30,
    "Left_Ankle_Roll": 0.0, "Right_Ankle_Roll": 0.0,
}
```

### PD-Gains [kp, kv]

```python
GAINS = {
    # Kopf
    "AAHead_yaw": (20, 0.5), "Head_pitch": (20, 0.5),
    # Arme
    "ALeft_Shoulder_Pitch": (40, 1), "ARight_Shoulder_Pitch": (40, 1),
    "Left_Shoulder_Roll": (40, 1),  "Right_Shoulder_Roll": (40, 1),
    "Left_Elbow_Pitch": (40, 1),    "Right_Elbow_Pitch": (40, 1),
    "Left_Elbow_Yaw": (40, 1),      "Right_Elbow_Yaw": (40, 1),
    # Beine
    "Left_Hip_Pitch": (200, 5),  "Right_Hip_Pitch": (200, 5),
    "Left_Hip_Roll": (200, 5),   "Right_Hip_Roll": (200, 5),
    "Left_Hip_Yaw": (150, 4),    "Right_Hip_Yaw": (150, 4),
    "Left_Knee_Pitch": (250, 6), "Right_Knee_Pitch": (250, 6),
    "Left_Ankle_Pitch": (120, 3),"Right_Ankle_Pitch": (120, 3),
    "Left_Ankle_Roll": (120, 3), "Right_Ankle_Roll": (120, 3),
}
```

PD setzen (lokale DOF-Reihenfolge, per Name aufgebaut):

```python
robot.set_dofs_kp(kp_array, motor_dofs_local)
robot.set_dofs_kv(kv_array, motor_dofs_local)
robot.set_dofs_force_range(-effort_array, effort_array, motor_dofs_local)
```

---

## 5. Empfohlene RL-Konfiguration (PPO)

### Steuerung
- **Regelfrequenz**: 50 Hz → `dt = 0.02`, `substeps = 2`
- **Aktion**: 12 Bein-Gelenke (oder 22 inkl. Arme/Kopf)
- **Action-Mapping**: `target = action * action_scale + default_pose`,
  `action_scale ≈ 0.25`
- **Latenz**: 1 Step (`simulate_action_latency = True`)

### Beobachtung (Vorschlag, ~3+3+3+3N analog Go2)
```
base_ang_vel           (3)   * 0.25
projected_gravity      (3)
commands               (3)   [vx, vy, wyaw] * scale
(dof_pos - default)    (N)   * 1.0
dof_vel                (N)   * 0.05
last_actions           (N)
```
N = 12 (nur Beine) → Obs-Dim = 45. N = 22 → Obs-Dim = 75.

### Termination
```python
reset |= episode_length > max_len
reset |= abs(roll)  > 30°            # base_euler[:,0]
reset |= abs(pitch) > 30°            # base_euler[:,1]
reset |= base_pos[:,2] < 0.35        # Trunk zu tief
reset |= scene.rigid_solver.get_error_envs_mask()   # NaN / Physik-Blowup
```

### Scene / Solver
```python
rigid_options = RigidOptions(
    enable_self_collision = True,   # Humanoid: Arme/Beine koennen kollidieren
    max_collision_pairs   = 40,     # mehr als Go2 (20)
)
```

### Reward-Bausteine (einzeln zuschalten, in dieser Reihenfolge)
1. `base_height`  → Trunk-Ziel ~0.53 m halten
2. `similar_to_default` → nahe Default-Pose bleiben
3. `lin_vel_z` (penalty) → kein Hüpfen
4. `action_rate` (penalty) → glatte Bewegungen
5. `tracking_lin_vel` → Zielgeschwindigkeit folgen
6. `tracking_ang_vel` → Drehung folgen

### Go2 → K1 Unterschiede
| Go2 | K1 |
|---|---|
| `slice(6, 18)` bei control | **per `motor_dofs_local` (Name-Mapping)** |
| base_init_pos z=0.42 | **z=0.56** (eingeschwungen ~0.531) |
| base_height_target=0.3 | **~0.53** (Trunk) |
| max_collision_pairs=20 | **40** |
| self_collision False | **True** (Humanoid) |
| kp=20 / kd=0.5 | **Beine 150–250 / 4–6**, Arme 40/1, Kopf 20/0.5 |
| 4 Beine, quadruped | 2 Beine, **Balance schwerer** |

---

## 6. Steh-Test ausführen

```bash
# mit Viewer
.venv/bin/python models/K1/K1_debug.py

# headless (CI / schneller Check)
HEADLESS=1 .venv/bin/python models/K1/K1_debug.py
```

Erwartung: Beckenhöhe pendelt sich bei ~0.531 m ein und bleibt konstant.
Gelenk-Dump landet in `models/K1/output.txt`.
