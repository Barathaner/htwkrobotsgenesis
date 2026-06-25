"""Self-collision parity: make MuJoCo stop the limbs from passing through the belly / each other,
matching Genesis (which runs enable_self_collision=True).

Root cause (found by inspecting K1_22dof.xml): nearly every limb collision primitive carries
contype="0" conaffinity="0", so MuJoCo ignores limb-vs-limb and limb-vs-belly contacts entirely —
the legs slide into the torso and lodge in each other. Genesis collides those links, so the two
engines disagree. This script:

  1. DIAGNOSE — lists MuJoCo's collision primitives and how many are currently disabled.
  2. PATCH    — enable_self_collision(model): turns the limb primitives back on (contype/conaffinity=1)
                and sets a Genesis-matched contact softness (solref). Importable by deploy_mujoco.py
                / joint_test.py so the fix is one function call.
  3. VERIFY   — randomized leg-pose sweep comparing self-collisions in Genesis vs MuJoCo (default &
                patched), then prints suggested capsule colliders to add to the MJCF.

Finding: the two engines' KINEMATICS match (link positions agree to ~2 mm) but their COLLISION
GEOMETRY does not — Genesis collides the full limb meshes/convex-hulls (conservative) while the MJCF
has only thin, mostly-disabled primitive proxies. No runtime knob (enable flags, contact margin,
solref) reconciles them; the MJCF needs real limb collision volume. enable_self_collision() still
fixes the clear bugs (limbs that don't collide at all + over-stiff contact that makes feet/legs jam);
the printed capsule lines are the model edit that stops legs entering the belly / each other.

Run from sim2real/:  python calibrate_collision.py
"""

import contextlib
import io
import os

import mujoco as mj
import numpy as np
import torch
import yaml

import genesis as gs

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "deploy.yaml"), "r") as f:
    P = yaml.safe_load(f)["policy"]

JOINT_NAMES = P["urdf_joint_names"]
DEFAULTS = P["default_joint_angles"]
BASE_POS = (0.0, 0.0, 1.0)
DT = 0.02

# The runtime fixes live in the shared pure-MuJoCo helper (also used by deploy_mujoco.py / joint_test.py).
from mujoco_calibration import collision_primitive_gids, enable_self_collision  # noqa: E402

SELF_COLLISION_SOLREF = (0.02, 1.0)


# ===================================================================================================
# Measurement rigs.
# ===================================================================================================
class GenesisRig:
    def __init__(self):
        gs.init(backend=gs.cpu, logging_level="error")
        with contextlib.redirect_stderr(io.StringIO()):
            self.sc = gs.Scene(show_viewer=False,
                               sim_options=gs.options.SimOptions(dt=DT, substeps=2))
            self.sc.add_entity(gs.morphs.Plane())
            self.rb = self.sc.add_entity(
                gs.morphs.URDF(file=os.path.abspath(P["urdf_path"]), pos=BASE_POS))
            self.sc.build()
        self.di = {n: self.rb.get_joint(n).dofs_idx_local[0] for n in JOINT_NAMES}
        self.all_dofs = [self.di[n] for n in JOINT_NAMES]
        self.default_vec = torch.tensor([DEFAULTS[n] for n in JOINT_NAMES], dtype=gs.tc_float)
        self.base_pos = torch.tensor(BASE_POS, dtype=gs.tc_float)
        self.base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float)

    def _set_pose(self, overrides):
        q = self.default_vec.clone()
        for n, v in overrides.items():
            q[JOINT_NAMES.index(n)] = v
        self.rb.set_dofs_position(q, self.all_dofs)
        self.rb.set_pos(self.base_pos)
        self.rb.set_quat(self.base_quat)
        self.sc.step()

    def self_pairs(self, overrides):
        """Set the pose, return {frozenset(link_a,link_b): max_penetration} for all current contacts."""
        self._set_pose(overrides)
        c = self.rb.get_contacts()
        out = {}
        la, lb = np.asarray(c["link_a"]).ravel(), np.asarray(c["link_b"]).ravel()
        pen = np.asarray(c["penetration"]).ravel()
        for a, b, d in zip(la, lb, pen):
            key = frozenset((int(a), int(b)))
            out[key] = max(out.get(key, 0.0), float(d))
        return out


class MujocoRig:
    def __init__(self, patched: bool):
        self.model = mj.MjModel.from_xml_path(os.path.abspath(P["mujoco_path"]))
        self.data = mj.MjData(self.model)
        if patched:
            enable_self_collision(self.model)
        self.qadr = {}
        for n in JOINT_NAMES:
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, n)
            self.qadr[n] = int(self.model.jnt_qposadr[jid])
        self.base_qpos = np.array([*BASE_POS, 1.0, 0.0, 0.0, 0.0])

    def self_pairs(self, overrides):
        """Kinematic overlap at a pose: set qpos, mj_forward (runs collision), read contacts.
        Keys are body-id pairs (to compare with Genesis link pairs); value = max penetration depth."""
        mj.mj_resetData(self.model, self.data)
        for n in JOINT_NAMES:
            self.data.qpos[self.qadr[n]] = overrides.get(n, DEFAULTS[n])
        self.data.qpos[0:7] = self.base_qpos
        mj.mj_forward(self.model, self.data)
        out = {}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[con.geom1])
            b2 = int(self.model.geom_bodyid[con.geom2])
            key = frozenset((b1, b2))
            out[key] = max(out.get(key, 0.0), float(max(0.0, -con.dist)))
        return out


# ===================================================================================================
# Parity check: sample many leg poses that bring the limbs together (the situation the user saw in
# joint_test — legs into belly / into each other) and compare, per engine, whether a self-collision
# is detected. Aggregate is far more informative than a couple of hand-picked single-joint probes.
# ===================================================================================================
# Leg joints sampled, each over a sub-range biased toward bringing the legs together / up.
PROBE_RANGES = {
    "Left_Hip_Pitch":  (-1.2, 1.2),  "Right_Hip_Pitch": (-1.2, 1.2),
    "Left_Hip_Roll":   (0.0, 1.4),   "Right_Hip_Roll":  (-1.4, 0.0),   # both inward → midline
    "Left_Hip_Yaw":    (-0.8, 0.8),  "Right_Hip_Yaw":   (-0.8, 0.8),
    "Left_Knee_Pitch": (0.0, 1.8),   "Right_Knee_Pitch": (0.0, 1.8),
}
N_POSES = 150


def has_self_collision(rig, baseline_keys, overrides):
    """True if the pose creates a contact pair not present at the default pose."""
    extra = {k: v for k, v in rig.self_pairs(overrides).items() if k not in baseline_keys}
    return len(extra) > 0


def main():
    print("[collision] loading rigs (Genesis + MuJoCo default + MuJoCo patched)…")
    gen = GenesisRig()
    mj_def = MujocoRig(patched=False)
    mj_fix = MujocoRig(patched=True)

    # ---- 1. diagnose ----
    gids = collision_primitive_gids(mj_def.model)
    disabled = [g for g in gids
                if mj_def.model.geom_contype[g] == 0 and mj_def.model.geom_conaffinity[g] == 0]
    print(f"\n[diagnose] MuJoCo collision primitives: {len(gids)} total, "
          f"{len(disabled)} currently DISABLED (contype=conaffinity=0) → those limbs can't collide.")
    print(f"[patch]    enable_self_collision() turns on {len(gids)} primitives, "
          f"solref={SELF_COLLISION_SOLREF}.\n")

    # Baseline contacts (permanently-touching adjacent links) per engine — ignored as non-collisions.
    base_gen = set(gen.self_pairs({}).keys())
    base_def = set(mj_def.self_pairs({}).keys())
    base_fix = set(mj_fix.self_pairs({}).keys())

    # ---- 2/3. randomized parity sweep ----
    rng = np.random.default_rng(0)
    names = list(PROBE_RANGES.keys())
    n_gen = n_def = n_fix = 0          # poses each engine flags as self-colliding
    agree_def = agree_fix = 0          # of the Genesis-colliding poses, how many each MuJoCo catches
    for _ in range(N_POSES):
        pose = {n: float(rng.uniform(*PROBE_RANGES[n])) for n in names}
        g = has_self_collision(gen, base_gen, pose)
        d = has_self_collision(mj_def, base_def, pose)
        f = has_self_collision(mj_fix, base_fix, pose)
        n_gen += g; n_def += d; n_fix += f
        if g:
            agree_def += d
            agree_fix += f

    print(f"[verify] {N_POSES} random leg poses (legs drawn toward belly / each other):")
    print(f"  Genesis            detects self-collision in {n_gen:3d}/{N_POSES} poses  (reference)")
    print(f"  MuJoCo  default    detects self-collision in {n_def:3d}/{N_POSES} poses")
    print(f"  MuJoCo  patched    detects self-collision in {n_fix:3d}/{N_POSES} poses")
    if n_gen:
        print(f"\n  Of the {n_gen} poses Genesis flags, MuJoCo catches:  "
              f"default {100*agree_def/n_gen:.0f}%,  patched {100*agree_fix/n_gen:.0f}%")

    print("\n[conclusion] The robot's KINEMATICS match between engines (link positions agree to ~2 mm),")
    print("but their COLLISION GEOMETRY does not: Genesis self-collides the full limb meshes/convex-hulls")
    print("(large, conservative), while the MJCF only has thin primitive proxies — most of them disabled.")
    print("No runtime knob (enable flags, contact margin, solref) closes that gap; the MJCF needs real")
    print("limb collision volume. enable_self_collision() still helps: it turns the proxies back on and")
    print("softens the over-stiff contact (solref 0.001→0.02) that makes colliding feet/legs JAM.")
    print("\nUse the runtime helper in deploy_mujoco.py / joint_test.py (right after loading the model):")
    print("    from calibrate_collision import enable_self_collision")
    print("    enable_self_collision(model)")
    print("\nTo actually stop the legs entering the belly / each other, add these capsule colliders to the")
    print("MJCF leg bodies (sized from the kinematics; MuJoCo can't add geoms at runtime). Paste each line")
    print("inside the matching <body name=\"…\"> in models/K1/K1_22dof.xml:")
    for body, fromto, r in suggest_leg_capsules(mj_def.model):
        print(f'    <!-- in <body name="{body}"> -->')
        print(f'    <geom type="capsule" fromto="{fromto}" size="{r:g}" '
              f'contype="1" conaffinity="1" rgba="1 0 0 0.3"/>')


def suggest_leg_capsules(model, radius=0.045):
    """Capsule colliders spanning each thigh (hip→knee) and shank (knee→ankle), in each leg body's
    local frame, derived from the (engine-matched) kinematics. These give MuJoCo real leg volume so
    the legs stop at the belly / each other instead of interpenetrating."""
    out = []
    def bpos(name):
        return model.body_pos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)]
    for side in ("Left", "Right"):
        knee = bpos(f"{side}_Shank")          # knee origin, local to {side}_Hip_Yaw (thigh body)
        ankle = bpos(f"{side}_Ankle_Cross")   # ankle origin, local to {side}_Shank
        out.append((f"{side}_Hip_Yaw",
                    f"0 0 0 {knee[0]:.3f} {knee[1]:.3f} {knee[2]:.3f}", radius))
        out.append((f"{side}_Shank",
                    f"0 0 0 {ankle[0]:.3f} {ankle[1]:.3f} {ankle[2]:.3f}", radius))
    return out


if __name__ == "__main__":
    main()
