Sachen nicht vergessen:
o.35 ist magic number abbruch umkippen
Commands intervalle sagen was erlaubt ist



What should the robot do?
### Verhalten (Task)
- **Laufen lernen** — stabile Gangbewegung auf ebenem Untergrund
- **Geschwindigkeit und Richtung anpassbar** — über Commands `[vx, vy, yaw_rate]` (m/s, rad/s)
- **Gewählte Richtung halten** — Heading/Yaw-Rate zum Command (`tracking_ang_vel`)
- **Ruhig stehen bleiben** — bei Command ≈ 0 nicht zappeln, Höhe und Pose halten
- **Gerade bleiben / Balance** — Roll/Pitch klein, Rumpf aufrecht
- **Smooth, nicht zackig** — glatte Gelenk-/Action-Änderungen (`action_rate`)
- **Nicht hinfallen** — Episode endet bei Umfallen / zu starker Neigung / zu niedriger Höhe

- muss drehen können auf befehl
- auch rückwärts laufen 
- richtung halten

### Robustheit (später / noch nicht im Training)
- Stabil gegen **Stöße / Schubsen** (andere Roboter, Impulse)
- Stabil auf **verschiedenen Kunstrasen-Untergründen** (Reibung, Unebenheit, Domain Randomization)


- Laufen lernen
- Geschwindigkeit und Richtung anpassbar
- gewählte Richtung halten können
- nicht zappeln
- kopf und arme bewegen sich random
- ruhig stehen bleiben.
- gerade bleiben
- balance halten
- bewegen smooth machen und nicht zackig
- stabil gegen stöße, schubsen von anderen roboter
- stabil auf verschiedenen kunstrasenuntergrund
- nicht hinfallen.
- muss drehen können auf befehl
- auch rückwärts laufen 

## Curriculum — Rewards & Commands

Vier unabhängige Curricula in `walk/config/k1_env.yaml` + `K1_env`:

| Curriculum | Schalter | Gate (wann es startet) | Was skaliert | wandb-Metrik |
|---|---|---|---|---|
| **Command** | `command_cfg.curriculum.enabled` | `cmd_curr_perf > 0.8` (EMA von `command_accuracy`) | `lin_vel_x_range` Obergrenze: 0.8 → 1.2 m/s (+0.1, Cooldown 10 s) | `curriculum_lin_vel_x_max` |
| **Style** | `reward_cfg.style_curriculum.enabled` | `cmd_curr_perf > 0.5` (gelatcht) | Polish-Terme × `style_weight` 0→1 über 60 s Sim-Zeit | `curriculum_style_weight` |
| **Assist** | `env_cfg.assist.enabled` | `command_accuracy`-EMA + Zeit-Abbau | Stützkraft am Rumpf-COM → 0 über ~200k Steps | `assist_scale`, `assist_perf_ema` |
| **Push** | `reward_cfg.push_enabled` | manuell (Stage 4) | zufälliger Horizontal-Stoß (Δv); Recovery über Task-Rewards | — |

**Prinzip:** Erst **Task** (laufen, nicht fallen), dann **Gait** (Takt, Abheben), dann **Style** (Feintuning), zuletzt **Robustheit** (Stöße, DR).

---

### Reward-Tabelle nach Curriculum-Stufe

| Reward | Scale | Stufe | Gewicht im Training | Aktiv wenn | Ziel |
|---|---:|---|---|---|---|
| `command_accuracy` | +1.0 | **Task** | immer 100 % | immer | Weltframe: Soll-Geschw. in Spawn-Richtung, kein Drift |
| `tracking_lin_vel` | +1.0 | **Task** | immer 100 % | immer | Körperframe-Geschwindigkeit zum Command |
| `tracking_ang_vel` | +0.25 | **Task** | immer 100 % | immer | Yaw-Rate zum Command |
| `base_height` | −50.0 | **Survival** | immer 100 % | immer | Trunk-Höhe ~0.53 m |
| `lin_vel_z` | −1.0 | **Survival** | immer 100 % | immer | Nicht hopsen |
| `orientation` | −2.0 | **Survival** | immer 100 % | immer | Roll/Pitch klein (aufrecht) |
| `ang_vel_xy` | −0.05 | **Survival** | immer 100 % | immer | Kipp-Raten dämpfen |
| `action_rate` | −0.005 | **Survival** | immer 100 % | immer | Glatte Aktionen |
| `dof_vel` | −2e−4 | **Survival** | immer 100 % | immer | Leichte Energiebremse |
| Termination | — | **Survival** | hard reset | roll/pitch > 30° oder z < 0.35 m | Nicht umfallen |
| `feet_air_time` | +3.0 | **Gait** | immer 100 % | cmd_speed > 0.2 m/s | Fuß hebt ab, Wechselschritt |
| `feet_slip` | −0.6 | **Gait** | immer 100 % | Fuß in Kontakt | Kein Schlurfen/Rutschen |
| `gait_phase` | −1.0 | **Gait** | immer 100 % | cmd_speed > 0.2 m/s | Fuß-Kontakt passt zum Phasen-Takt |
| `foot_roll` | +0.5 | **Style** | × `style_weight` 0→1 | Gate + cmd > 0.2 | Heel-Strike / Toe-Off (Ankle-Pitch) |
| `flat_foot` | −0.2 | **Style** | × `style_weight` 0→1 | Gate + Kontakt > 0.4 s flach | Kein dauerhaft flacher Schlurf-Fuß |
| `support_pose` | +0.2 | **Style** | × `style_weight` 0→1 | Gate + cmd < 0.6 m/s | Knie leicht gebeugt (Federung) |
| `similar_to_default` | −0.03 | **Style** | × `style_weight` 0→1 | Gate | Nahe Default-Pose, ruhige Haltung |

**Style-Terme** (`style_curriculum.terms`): `foot_roll`, `flat_foot`, `support_pose`, `similar_to_default`

---

### Command-Curriculum (Geschwindigkeit)

| Phase | `lin_vel_x` | `lin_vel_y` | `ang_vel` | Voraussetzung |
|---|---|---|---|---|
| Start | 0.3 – 0.8 m/s | 0 | 0 | — |
| +Stufe | +0.1 m/s Obergrenze | 0 | 0 | `cmd_curr_perf > 0.8`, Cooldown 10 s |
| Ziel | 0.3 – **1.2** m/s | 0 | 0 | `lin_vel_x_max_limit` |
| Stage B (später) | −0.4 – 1.2 | 0 | 0 | Rückwärts manuell in YAML |
| Stage C (später) | wie B | ±0.2 | ±0.5 | Seitwärts + Drehen manuell |

---

### Empfohlene Trainings-Stages (manuell)

| Stage | Fokus | YAML-Änderungen | Erfolgskriterium |
|---|---|---|---|
| **0** | Stehen / Balance | nur Survival-Terme, cmd ≈ 0 | z stabil, kein Fall |
| **1** | Vorwärts laufen | Task + Gait, Command-Curriculum an | `command_accuracy` ↑, Video ok |
| **2** | Gang-Stil | `style_curriculum.enabled: true` | `style_weight → 1`, Heel/Toe sichtbar |
| **3** | Mehr Commands | `lin_vel_y`, `ang_vel`, ggf. negatives vx | alle Richtungen, Fall rate ≈ 0 |
| **4** | Robustheit | `push_enabled: true`, Reibung DR | Stoß überlebt, Tracking bleibt stabil |

---

### Reward-Design — Kurzreferenz (Ziel → Mechanismus)

| Ziel | Reward | Status |
|---|---|---|
| Command-Geschwindigkeit + Richtung | `command_accuracy` + `tracking_lin_vel` | ✅ |
| Drehen | `tracking_ang_vel` | ✅ |
| Nicht hopsen / Höhe / aufrecht | `lin_vel_z`, `base_height`, `orientation` | ✅ |
| Smooth | `action_rate`, `dof_vel` | ✅ |
| Echter Gang | `feet_air_time`, `gait_phase`, `feet_slip` | ✅ |
| Heel-to-Toe | `foot_roll`, `flat_foot` (Style-Ramp) | ✅ gerampt |
| Stöße | `push_enabled` (Env-Stoß, Task-Rewards für Recovery) | ⏸ vorbereitet, aus |
| Kunstrasen / Terrain | — | ❌ noch offen |

e.g. „Walk forward at commanded speed, stay upright, don’t hop, don’t fall.“

What is success?
eingestellte commands  halten über zeitraum
fall rate 0
wwenig höhenvarianz
nur balance actions ansonsten 0 action
hoher reward, video plausibel


Measurable: mean forward velocity, distance per episode, fall rate, height variance.

What is forbidden?
Fall, joint limits, excessive torque, bouncing.

What is optional / nice-to-have?
Smooth motion, energy efficiency, natural arm swing.