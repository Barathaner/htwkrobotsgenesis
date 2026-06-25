"""Deploy-parity calibration: find, per joint, the MuJoCo position-actuator kp/kv that REPRODUCE
the Genesis closed-loop step response, and align the MuJoCo joint limits to Genesis (URDF).

Why: the same kp/kv realise different dynamics in the two engines — Genesis is softer & underdamped,
MuJoCo stiffer & well-damped (see joint_test.py findings). A Genesis-trained policy therefore expects
the Genesis response. This script fits MuJoCo gains so deploy_mujoco.py reproduces what the policy was
trained against. NOTE: this is a band-aid for sim2sim parity — it makes MuJoCo behave like Genesis,
which is NOT the same as matching the real robot (hardware ≈ stiff MuJoCo, not Genesis).

Method, per joint (base pinned in both sims, all other joints PD-held at default):
  1. Genesis reference: step the joint default → default+Δ at the TRAINING gains; record q(t) for T ticks.
  2. MuJoCo: grid-search (then local-refine) kp,kv to minimise SSE against the Genesis trajectory.
Outputs sim2real/deploy_mujoco_calib.yaml (per-joint kp/kv + Genesis joint limits) and prints a table
plus copy-pasteable MJCF <position .../> lines.

Run from sim2real/:  python calibrate_pd.py            # all joints
                     python calibrate_pd.py --joints Head_pitch Left_Knee_Pitch   # subset, fast
"""

import argparse
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
GAINS = P["joint_gains"]
BASE_POS = (0.0, 0.0, 1.0)
DT = 0.02
T_TICKS = 40          # control ticks recorded per step test
STEP_DELTA = 0.30     # [rad] commanded step magnitude (clamped into the joint range)


# ---------------------------------------------------------------------------------------------------
# Genesis: build once, pinned base, record a per-joint step response at the training gains.
# ---------------------------------------------------------------------------------------------------
class GenesisRig:
    def __init__(self):
        gs.init(backend=gs.cpu, logging_level="error")
        with contextlib.redirect_stderr(io.StringIO()):
            self.sc = gs.Scene(show_viewer=False,
                               sim_options=gs.options.SimOptions(dt=DT, substeps=2))
            self.sc.add_entity(gs.morphs.Plane())
            self.rb = gs.morphs.URDF(file=os.path.abspath(P["urdf_path"]), pos=BASE_POS)
            self.rb = self.sc.add_entity(self.rb)
            self.sc.build()
        self.di = {n: self.rb.get_joint(n).dofs_idx_local[0] for n in JOINT_NAMES}
        self.all_dofs = [self.di[n] for n in JOINT_NAMES]
        self.base_dofs = sorted(set(range(self.rb.n_dofs)) - set(self.all_dofs))
        self.rb.set_dofs_kp([GAINS[n]["kp"] for n in JOINT_NAMES], self.all_dofs)
        self.rb.set_dofs_kv([GAINS[n]["kd"] for n in JOINT_NAMES], self.all_dofs)
        self.default_vec = torch.tensor([DEFAULTS[n] for n in JOINT_NAMES], dtype=gs.tc_float)
        self.base_pos = torch.tensor(BASE_POS, dtype=gs.tc_float)
        self.base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float)
        self.base_zero = torch.zeros(len(self.base_dofs), dtype=gs.tc_float)

    def limit(self, name):
        lo, hi = self.rb.get_dofs_limit([self.di[name]])
        return float(lo[0]), float(hi[0])

    def reset_to_default(self):
        self.rb.set_dofs_position(self.default_vec, self.all_dofs)
        self.rb.set_dofs_velocity(torch.zeros(len(self.all_dofs), dtype=gs.tc_float), self.all_dofs)
        self.rb.set_pos(self.base_pos)
        self.rb.set_quat(self.base_quat)

    def step_response(self, name, target):
        """Hold all joints at default except `name`, which steps to `target`. Return q(t), len T."""
        self.reset_to_default()
        tgt = self.default_vec.clone()
        tgt[JOINT_NAMES.index(name)] = target
        di = self.di[name]
        traj = np.zeros(T_TICKS)
        for k in range(T_TICKS):
            self.rb.control_dofs_position(tgt, self.all_dofs)
            self.rb.set_dofs_velocity(self.base_zero, self.base_dofs)
            self.rb.set_pos(self.base_pos)
            self.rb.set_quat(self.base_quat)
            self.sc.step()
            traj[k] = float(self.rb.get_dofs_position([di])[0])
        return traj


# ---------------------------------------------------------------------------------------------------
# MuJoCo: same step test, but with tunable kp/kv on the tested joint's <position> actuator.
# ---------------------------------------------------------------------------------------------------
class MujocoRig:
    def __init__(self):
        self.model = mj.MjModel.from_xml_path(os.path.abspath(P["mujoco_path"]))
        self.data = mj.MjData(self.model)
        self.aid = {n: mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES}
        self.qadr = {}
        self.jrange = {}
        for n in JOINT_NAMES:
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, n)
            self.qadr[n] = int(self.model.jnt_qposadr[jid])
            self.jrange[n] = (float(self.model.jnt_range[jid, 0]), float(self.model.jnt_range[jid, 1]))
        self.base_qpos = np.array([*BASE_POS, 1.0, 0.0, 0.0, 0.0])
        self.dec = max(1, round(DT / self.model.opt.timestep))

    def set_gains(self, name, kp, kv):
        """Rewrite a position actuator's kp/kv in place (gainprm[0]=kp, biasprm=[0,-kp,-kv])."""
        aid = self.aid[name]
        self.model.actuator_gainprm[aid, 0] = kp
        self.model.actuator_biasprm[aid, 1] = -kp
        self.model.actuator_biasprm[aid, 2] = -kv

    def step_response(self, name, target):
        mj.mj_resetData(self.model, self.data)
        for n in JOINT_NAMES:
            self.data.qpos[self.qadr[n]] = DEFAULTS[n]
        self.data.qpos[0:7] = self.base_qpos
        mj.mj_forward(self.model, self.data)
        ctrl = {n: DEFAULTS[n] for n in JOINT_NAMES}
        ctrl[name] = target
        traj = np.zeros(T_TICKS)
        for k in range(T_TICKS):
            for n in JOINT_NAMES:
                self.data.ctrl[self.aid[n]] = ctrl[n]
            for _ in range(self.dec):
                self.data.qpos[0:7] = self.base_qpos
                self.data.qvel[0:6] = 0.0
                mj.mj_step(self.model, self.data)
            traj[k] = float(self.data.qpos[self.qadr[name]])
        return traj


def pick_target(name, lo, hi):
    """A step target = default ± STEP_DELTA, kept inside the range with a small margin."""
    d = DEFAULTS[name]
    margin = 0.02
    up, dn = min(hi - margin, d + STEP_DELTA), max(lo + margin, d - STEP_DELTA)
    return up if (up - d) >= (d - dn) else dn   # step toward whichever side has more room


def fit_joint(name, gen_traj, target, mjr):
    """Grid-search then local-refine MuJoCo (kp,kv) to minimise SSE vs the Genesis trajectory."""
    train_kp, train_kv = GAINS[name]["kp"], GAINS[name]["kd"]

    def sse(kp, kv):
        mjr.set_gains(name, kp, kv)
        return float(np.sum((mjr.step_response(name, target) - gen_traj) ** 2))

    kp_grid = train_kp * np.linspace(0.05, 2.50, 16)
    kv_grid = np.linspace(0.0, max(6.0, train_kv * 5.0), 14)
    best = (train_kp, train_kv, float("inf"))
    for kp in kp_grid:
        for kv in kv_grid:
            e = sse(kp, kv)
            if e < best[2]:
                best = (kp, kv, e)
    # local refine around the best grid point
    bkp, bkv, _ = best
    for kp in np.linspace(bkp * 0.8, bkp * 1.2, 7):
        for kv in np.linspace(max(0.0, bkv - 1.0), bkv + 1.0, 7):
            e = sse(kp, kv)
            if e < best[2]:
                best = (kp, kv, e)
    return best  # (kp, kv, sse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", nargs="*", default=JOINT_NAMES, help="subset of joints to calibrate")
    args = ap.parse_args()
    todo = [j for j in args.joints if j in JOINT_NAMES]

    print(f"[calib] building Genesis + MuJoCo rigs … ({len(todo)} joints)")
    gen = GenesisRig()
    mjr = MujocoRig()

    out_gains, out_limits, rows, limit_rows = {}, {}, [], []
    for name in todo:
        g_lo, g_hi = gen.limit(name)                 # Genesis/URDF limits (authoritative)
        m_lo, m_hi = mjr.jrange[name]                # MuJoCo MJCF limits
        out_limits[name] = {"lower": round(g_lo, 6), "upper": round(g_hi, 6)}
        if abs(g_lo - m_lo) > 1e-4 or abs(g_hi - m_hi) > 1e-4:
            limit_rows.append((name, (g_lo, g_hi), (m_lo, m_hi)))

        target = pick_target(name, g_lo, g_hi)
        gen_traj = gen.step_response(name, target)
        kp, kv, err = fit_joint(name, gen_traj, target, mjr)
        # RMS tracking error after fitting, in degrees, as a quality read-out.
        mjr.set_gains(name, kp, kv)
        rms_deg = float(np.sqrt(np.mean((mjr.step_response(name, target) - gen_traj) ** 2))) * 180 / np.pi
        out_gains[name] = {"kp": round(float(kp), 3), "kv": round(float(kv), 3)}
        rows.append((name, GAINS[name]["kp"], GAINS[name]["kd"], kp, kv, rms_deg))
        print(f"  {name:<22} train(kp={GAINS[name]['kp']:.0f},kv={GAINS[name]['kd']:.1f}) "
              f"→ mujoco(kp={kp:7.2f},kv={kv:5.2f})  rms={rms_deg:.2f}°")

    # ---- write calibration file ----
    out_path = os.path.join(HERE, "deploy_mujoco_calib.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(
            {"calibrated_joint_gains": out_gains, "joint_limits": out_limits},
            f, sort_keys=False, default_flow_style=False,
        )
    print(f"\n[calib] wrote {out_path}")

    # ---- joint-limit mismatch report ----
    if limit_rows:
        print("\nJoint-limit mismatches (Genesis/URDF vs MuJoCo/MJCF) — align MJCF <joint range> to Genesis:")
        for name, (glo, ghi), (mlo, mhi) in limit_rows:
            print(f"  {name:<22} genesis=[{glo:+.3f},{ghi:+.3f}]  mujoco=[{mlo:+.3f},{mhi:+.3f}]")
    else:
        print("\nJoint limits already match between Genesis and MuJoCo. ✓")

    # ---- copy-pasteable MJCF actuator lines ----
    print("\nMJCF <position> lines for K1_22dof.xml (matched to Genesis response):")
    for name, *_ , in rows:
        kp, kv = out_gains[name]["kp"], out_gains[name]["kv"]
        eff = GAINS[name]["effort"]
        print(f'    <position name="{name}" joint="{name}" kp="{kp:g}" kv="{kv:g}" '
              f'forcelimited="true" forcerange="-{eff:g} {eff:g}"/>')


if __name__ == "__main__":
    main()
