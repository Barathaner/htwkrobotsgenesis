"""Pure-MuJoCo calibration helpers (no Genesis import) shared by deploy_mujoco.py, joint_test.py and
calibrate_collision.py. Two runtime patches that bring MuJoCo closer to the Genesis training plant:

  apply_mujoco_calibration(model) — load deploy_mujoco_calib.yaml: PD gains + joint limits (deploy_mujoco.py).
  apply_calibrated_gains(model)   — kp/kv only (joint_test.py).
  enable_self_collision(model)   — ensure limb collision primitives are on and contact solref matches
                                   Genesis (~0.02 s). MJCF K1_22dof.xml now bakes this in; the call is
                                   idempotent and still refreshes solref if the XML is reverted.
"""

import os

import mujoco as mj
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIB = os.path.join(HERE, "deploy_mujoco_calib.yaml")
DEFAULT_DEPLOY = os.path.join(HERE, "deploy.yaml")

_PRIMITIVE_TYPES = {int(mj.mjtGeom.mjGEOM_SPHERE), int(mj.mjtGeom.mjGEOM_CAPSULE),
                    int(mj.mjtGeom.mjGEOM_CYLINDER), int(mj.mjtGeom.mjGEOM_BOX)}


def apply_calibrated_gains(model, calib_path=DEFAULT_CALIB):
    """Set each position actuator's kp/kv from the calibration file (gainprm[0]=kp, biasprm=[0,-kp,-kv]).
    Returns the number of actuators updated."""
    if not os.path.exists(calib_path):
        print(f"[calib] {calib_path} not found — run calibrate_pd.py first; using MJCF gains.")
        return 0
    with open(calib_path, "r") as f:
        gains = yaml.safe_load(f).get("calibrated_joint_gains", {})
    n = 0
    for name, g in gains.items():
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            print(f"[calib] warning: no actuator '{name}' in model — skipped")
            continue
        kp, kv = float(g["kp"]), float(g["kv"])
        model.actuator_gainprm[aid, 0] = kp
        model.actuator_biasprm[aid, 1] = -kp
        model.actuator_biasprm[aid, 2] = -kv
        n += 1
    print(f"[calib] applied calibrated kp/kv to {n}/{len(gains)} actuators "
          f"from {os.path.basename(calib_path)}")
    return n


def apply_joint_limits(model, calib_path=DEFAULT_CALIB):
    """Align MJCF joint ranges with Genesis/URDF limits from calibrate_pd.py."""
    if not os.path.exists(calib_path):
        return 0
    with open(calib_path, "r") as f:
        limits = yaml.safe_load(f).get("joint_limits", {})
    n = 0
    for name, lim in limits.items():
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        model.jnt_range[jid, 0] = float(lim["lower"])
        model.jnt_range[jid, 1] = float(lim["upper"])
        n += 1
    if n:
        print(f"[calib] applied joint limits for {n} joints from {os.path.basename(calib_path)}")
    return n


def apply_mujoco_calibration(model, calib_path=DEFAULT_CALIB):
    """Apply PD gains + joint limits from deploy_mujoco_calib.yaml (output of calibrate_pd.py)."""
    n_g = apply_calibrated_gains(model, calib_path)
    n_l = apply_joint_limits(model, calib_path)
    if n_g == 0:
        return 0
    # Spot-check: confirm runtime gains differ from MJCF training values.
    aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "Left_Hip_Pitch")
    if aid >= 0:
        kp = float(model.actuator_gainprm[aid, 0])
        kv = float(-model.actuator_biasprm[aid, 2])
        print(f"[calib] example Left_Hip_Pitch: kp={kp:.1f} kv={kv:.1f} (MJCF default was 200/5)")
    return n_g + n_l


def collision_primitive_gids(model):
    """Geom ids of the limb/body collision PROXIES (primitive shapes), excluding the ground plane."""
    ground = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")
    return [g for g in range(model.ngeom)
            if g != ground and int(model.geom_type[g]) in _PRIMITIVE_TYPES]


def enable_self_collision(model, solref=(0.02, 1.0)):
    """Enable contype/conaffinity on the limb collision primitives and soften their contact to a
    Genesis-like time-constant. MuJoCo's filterparent already excludes parent↔child links, so only
    non-adjacent pairs start colliding. Returns the number of geoms enabled."""
    gids = collision_primitive_gids(model)
    for g in gids:
        model.geom_contype[g] = 1
        model.geom_conaffinity[g] = 1
        model.geom_solref[g, 0] = solref[0]
        model.geom_solref[g, 1] = solref[1]
    print(f"[calib] enabled self-collision on {len(gids)} limb primitives (solref={solref})")
    return len(gids)
