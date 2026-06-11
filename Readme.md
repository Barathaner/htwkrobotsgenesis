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

## Reward-Design — Übersetzung Ziel → Mechanismus
| Ziel | Wie messen? | Aktuell in `k1_env.yaml` | Status |
|---|---|---|---|
| Command-Geschwindigkeit fahren | ‖cmd_xy − vel_xy‖ | `tracking_lin_vel` (+1.0) | ✅ aktiv |
| Richtung / Drehen halten | ‖cmd_yaw − ω_z‖ | `tracking_ang_vel` (+0.2) | ✅ aktiv |
| Nicht hopsen | vz² | `lin_vel_z` (−1.0) | ✅ aktiv |
| Höhe halten | (z − z_target)² | `base_height` (−50.0), target 0.53 m | ✅ aktiv |
| Smooth / nicht zackig | ‖a_t − a_{t−1}‖² | `action_rate` (−0.005) | ✅ aktiv |
| Nahe Default-Pose | ‖q − q_default‖ | `similar_to_default` | ⚠️ auskommentiert |
| Nicht umfallen | Roll/Pitch, z < Schwelle | Termination (30°, z < 0.35 m) | ✅ aktiv |
| Größere Schritte / Gang | Fuß-Luftphase, Clearance | — | ❌ fehlt |
| Stöße / Schubsen | Impuls-Randomization | — | ❌ fehlt |
| Kunstrasen / Terrain | Reibung, Boden, Unebenheit | — | ❌ fehlt |
| Gelenk-/Drehmoment-Limits | \|τ\|, Limit-Annäherung | — | ❌ fehlt |

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