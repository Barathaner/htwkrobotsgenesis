"""Smoke test: run a trained K1 walking policy on the real Booster K1 robot.

Loads a distillation STUDENT checkpoint (walk/k1_distill.py, key "student_state_dict",
59-dim onboard proprio obs) by default, and also accepts a TEACHER checkpoint
(walk/K1_train.py, key "actor_state_dict") — the kind is auto-detected and the observation
is built to match. The student needs no privileged/faked inputs, so it is the deployable one.

Requires:
  - booster_robotics_sdk_python  (build from ~/git/booster_robotics_sdk)
  - PyTorch CPU

Usage (from repo root, running directly ON the robot):
  python deploy.py \\
      --model logs/k1-distill/model_2000.pt \\
      --interface 127.0.0.1 \\
      --duration 10 \\
      --vx 0.0 --vy 0.0 --wz 0.0

The robot must already be standing / held safely.
Press Ctrl+C to stop; the script then zeros all commands (damp mode).
Note: use --interface eth0 only when running from an EXTERNAL PC connected via Ethernet.

Joint-space note:
  Policy uses 16 URDF joints. The K1 SDK uses 23 physical motor indices
  (B1JointIndex). Ankle joints pass through a parallel linkage:
    sdk_crank_up = ankle_pitch + ankle_roll
    sdk_crank_dn = ankle_pitch - ankle_roll
  Verify this on your robot before trusting ankle commands.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import signal
import sys
import time

import torch
import torch.nn as nn

from booster_robotics_sdk_python import (
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    ChannelFactory,
    LowCmd,
    LowCmdType,
    MotorCmd,
    RobotMode,
)

# ---------------------------------------------------------------------------
# Robot constants (must match k1_env.yaml)
# ---------------------------------------------------------------------------

DT = 0.02  # policy step [s] = 50 Hz

# Policy joint order (matches env_cfg.joint_names in k1_env.yaml)
JOINT_NAMES = [
    "ALeft_Shoulder_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",   # virtual – mapped to CrankUpLeft/CrankDownLeft
    "Left_Ankle_Roll",    # virtual – mapped to CrankUpLeft/CrankDownLeft
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",  # virtual – mapped to CrankUpRight/CrankDownRight
    "Right_Ankle_Roll",   # virtual – mapped to CrankUpRight/CrankDownRight
]
NUM_ACTIONS = len(JOINT_NAMES)  # 16

DEFAULT_JOINT_ANGLES = {
    "ALeft_Shoulder_Pitch":  0.0,
    "Left_Elbow_Yaw":       -1.0,
    "ARight_Shoulder_Pitch": 0.0,
    "Right_Elbow_Yaw":       1.0,
    "Left_Hip_Pitch":       -0.30,
    "Left_Hip_Roll":         0.0,
    "Left_Hip_Yaw":          0.0,
    "Left_Knee_Pitch":       0.60,
    "Left_Ankle_Pitch":     -0.30,
    "Left_Ankle_Roll":       0.0,
    "Right_Hip_Pitch":      -0.30,
    "Right_Hip_Roll":        0.0,
    "Right_Hip_Yaw":         0.0,
    "Right_Knee_Pitch":      0.60,
    "Right_Ankle_Pitch":    -0.30,
    "Right_Ankle_Roll":      0.0,
}
DEFAULT_DOF_POS = torch.tensor(
    [DEFAULT_JOINT_ANGLES[n] for n in JOINT_NAMES], dtype=torch.float32
)

ACTION_SCALE = 0.25
CLIP_ACTIONS = 100.0

# Fixed joints (held at default via PD, not policy-controlled)
FIXED_JOINT_DEFAULTS = {
    "AAHead_yaw":          0.0,
    "Head_pitch":          0.2,
    "Left_Shoulder_Roll": -1.5,
    "Right_Shoulder_Roll": 1.5,
    "Left_Elbow_Pitch":    0.1,
    "Right_Elbow_Pitch":   0.1,
}

# ---------------------------------------------------------------------------
# K1 joint indices (JointIndexK1: 22 joints, no waist)
# The Python binding only exposes the generic B1JointIndex (23 joints with
# waist=10).  K1 omits that slot, so all leg indices are one less than the
# generic enum.  The SDK expects motor_cmd arrays with exactly 22 entries.
# ---------------------------------------------------------------------------
K1_JOINT_CNT = 22

class K1Ji:  # mirrors JointIndexK1 from b1_api_const.hpp
    kHeadYaw             =  0;  kHeadPitch           =  1
    kLeftShoulderPitch   =  2;  kLeftShoulderRoll    =  3
    kLeftElbowPitch      =  4;  kLeftElbowYaw        =  5
    kRightShoulderPitch  =  6;  kRightShoulderRoll   =  7
    kRightElbowPitch     =  8;  kRightElbowYaw       =  9
    kLeftHipPitch        = 10;  kLeftHipRoll         = 11
    kLeftHipYaw          = 12;  kLeftKneePitch       = 13
    kCrankUpLeft         = 14;  kCrankDownLeft       = 15
    kRightHipPitch       = 16;  kRightHipRoll        = 17
    kRightHipYaw         = 18;  kRightKneePitch      = 19
    kCrankUpRight        = 20;  kCrankDownRight      = 21

# Offset from K1Ji leg index → motor_state_parallel array index
K1_LEG_OFFSET = K1Ji.kLeftHipPitch  # = 10

# Parallel-linkage crank indices (JointIndexK1); matches deploy configs [14,15,20,21]
K1_CRANK_INDICES = [
    K1Ji.kCrankUpLeft, K1Ji.kCrankDownLeft,
    K1Ji.kCrankUpRight, K1Ji.kCrankDownRight,
]

# SDK joint index → (kp, kd, effort_limit) — keyed by K1Ji (22-joint ordering)
SDK_JOINT_GAINS: dict[int, tuple[float, float, float]] = {
    K1Ji.kHeadYaw:            (40.0,  1.0,   6.0),
    K1Ji.kHeadPitch:          (40.0,  1.0,   6.0),
    K1Ji.kLeftShoulderPitch:  (40.0,  1.0,  14.0),
    K1Ji.kLeftShoulderRoll:   (40.0,  1.0,  14.0),
    K1Ji.kLeftElbowPitch:     (40.0,  1.0,  14.0),
    K1Ji.kLeftElbowYaw:       (40.0,  1.0,  14.0),
    K1Ji.kRightShoulderPitch: (40.0,  1.0,  14.0),
    K1Ji.kRightShoulderRoll:  (40.0,  1.0,  14.0),
    K1Ji.kRightElbowPitch:    (40.0,  1.0,  14.0),
    K1Ji.kRightElbowYaw:      (40.0,  1.0,  14.0),
    K1Ji.kLeftHipPitch:       (200.0, 5.0,  68.0),
    K1Ji.kLeftHipRoll:        (200.0, 5.0,  76.0),
    K1Ji.kLeftHipYaw:         (150.0, 4.0,  38.3),
    K1Ji.kLeftKneePitch:      (250.0, 6.0, 112.0),
    K1Ji.kCrankUpLeft:        (  0.0, 1.0,  20.0),  # torque-controlled; kp=0
    K1Ji.kCrankDownLeft:      (  0.0, 1.0,  15.0),  # torque-controlled; kp=0
    K1Ji.kRightHipPitch:      (200.0, 5.0,  68.0),
    K1Ji.kRightHipRoll:       (200.0, 5.0,  76.0),
    K1Ji.kRightHipYaw:        (150.0, 4.0,  38.3),
    K1Ji.kRightKneePitch:     (250.0, 6.0, 112.0),
    K1Ji.kCrankUpRight:       (  0.0, 1.0,  20.0),  # torque-controlled; kp=0
    K1Ji.kCrankDownRight:     (  0.0, 1.0,  15.0),  # torque-controlled; kp=0
}

# Stiffness used in the torque formula for crank joints:
#   tau = clip((target - current) × stiffness, ±torque_limit)
# Values from Parameter_Walk_k1.yaml.
CRANK_STIFFNESS: dict[int, float] = {
    K1Ji.kCrankUpLeft:   25.0,
    K1Ji.kCrankDownLeft: 25.0,
    K1Ji.kCrankUpRight:  25.0,
    K1Ji.kCrankDownRight: 25.0,
}

# Obs scales (from k1_env.yaml)
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05

# Gait clock (from k1_env.yaml: reward_cfg.gait_period_s). MUST match the training config
# the checkpoint was produced with, or the phase clock desyncs from the learned gait.
GAIT_PERIOD_S  = 0.80
GAIT_PERIOD_STEPS = max(1, round(GAIT_PERIOD_S / DT))  # 40

# ---------------------------------------------------------------------------
# Policy network (mirrors rsl-rl MLPModel with ELU + GaussianDistribution)
# ---------------------------------------------------------------------------
#
# Two checkpoint kinds are supported (auto-detected in load_actor):
#   - STUDENT (distillation, walk/k1_distill.py): key "student_state_dict",
#     onboard-only obs (proprio group, 59-dim) — no privileged/faked inputs.
#   - TEACHER (PPO/AMP, walk/K1_train.py):        key "actor_state_dict",
#     legacy privileged obs (63-dim here) with placeholder lin_vel_heading/foot_contact.
# Both share the same MLP (512→256→128, ELU); only the input width differs.

STUDENT_OBS_DIM = 59  # proprio group: ang_vel(3)+grav(3)+cmd(3)+dof_pos(16)+dof_vel(16)+act(16)+clock(2)
TEACHER_OBS_DIM = 63  # legacy: STUDENT layout + lin_vel_heading(2) + foot_contact(2)


class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 256),     nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


DEFAULT_MODEL_DIR = "logs/k1-distill"  # distillation student checkpoints live here


def resolve_model(path: str) -> str:
    """Accept a .pt file or a log dir (auto-picks the highest-numbered model_*.pt).

    Mirrors _resolve_teacher in walk/k1_distill.py so deploy keeps working as the
    student trains and new checkpoints appear (the default points at the dir, not a
    fixed model_N.pt that may not exist yet).
    """
    path = os.path.abspath(path)
    if os.path.isdir(path):
        cands = sorted(
            glob.glob(os.path.join(path, "model_*.pt")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
        )
        if not cands:
            raise FileNotFoundError(f"No model_*.pt found in {path}")
        print(f"Auto-selected latest student checkpoint: {cands[-1]}")
        return cands[-1]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_actor(checkpoint_path: str) -> tuple[ActorMLP, int, bool]:
    """Load a student (preferred) or teacher actor for inference.

    Returns (actor, obs_dim, is_student). obs_dim is inferred from the checkpoint's first
    layer so the network width always matches the weights, regardless of obs layout changes.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "student_state_dict" in ckpt:
        state_dict, is_student = ckpt["student_state_dict"], True
    elif "actor_state_dict" in ckpt:
        state_dict, is_student = ckpt["actor_state_dict"], False
    else:
        raise KeyError(
            f"{checkpoint_path}: no 'student_state_dict' or 'actor_state_dict' found "
            f"(keys: {list(ckpt.keys())})"
        )

    obs_dim = int(state_dict["mlp.0.weight"].shape[1])  # first Linear: (512, obs_dim)
    actor = ActorMLP(obs_dim)
    # strict=False: rsl-rl also stores distribution.std_param which inference does not need.
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()
    return actor, obs_dim, is_student


# ---------------------------------------------------------------------------
# Ankle ↔ Crank conversion (K1 parallel linkage, 1:1 ratio)
# ---------------------------------------------------------------------------

def ankle_to_crank(ankle_pitch: float, ankle_roll: float) -> tuple[float, float]:
    """Convert URDF ankle (pitch, roll) → SDK crank (up, down)."""
    return ankle_pitch + ankle_roll, ankle_pitch - ankle_roll


def crank_to_ankle(crank_up: float, crank_down: float) -> tuple[float, float]:
    """Convert SDK crank (up, down) → URDF ankle (pitch, roll)."""
    return (crank_up + crank_down) / 2.0, (crank_up - crank_down) / 2.0


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def rpy_to_projected_gravity(roll: float, pitch: float) -> list[float]:
    """Gravity vector [0,0,-1] projected into body frame from roll/pitch (yaw-independent)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return [-sp, sr * cp, -cr * cp]


# ---------------------------------------------------------------------------
# SDK state → policy DOF arrays
# ---------------------------------------------------------------------------

class RobotStateBuffer:
    """Copies SDK state values inside the callback; never holds a C++ reference.

    The SDK frees / recycles its LowState object as soon as the callback
    returns.  Storing self._state = state and reading it later from the main
    thread is a use-after-free → segfault.  Instead we copy every value we
    need into plain Python floats inside update(), which is safe.

    Joint layout: motor_state_serial has ALL 22 K1 joints in K1Ji order
    (indices 0-21).  motor_state_parallel mirrors the same 22 slots but
    provides different sensor data (not used for state here).  This matches
    exactly what every working K1 deploy script does.
    """

    def __init__(self) -> None:
        self._ready = False
        self._q   = [0.0] * K1_JOINT_CNT   # q  for all K1Ji 0-21, from serial
        self._dq  = [0.0] * K1_JOINT_CNT   # dq for all K1Ji 0-21, from serial
        self._gyro = [0.0, 0.0, 0.0]
        self._rpy  = [0.0, 0.0, 0.0]
        self._raw_serial_len   = 0
        self._raw_parallel_len = 0

    def update(self, state) -> None:
        """Called on the SDK background thread; copy everything out immediately."""
        ms = state.motor_state_serial
        self._raw_serial_len   = len(ms)
        self._raw_parallel_len = len(state.motor_state_parallel)
        for i in range(min(len(ms), K1_JOINT_CNT)):
            self._q[i]  = ms[i].q
            self._dq[i] = ms[i].dq
        imu = state.imu_state
        self._gyro[0] = imu.gyro[0]; self._gyro[1] = imu.gyro[1]; self._gyro[2] = imu.gyro[2]
        self._rpy[0]  = imu.rpy[0];  self._rpy[1]  = imu.rpy[1];  self._rpy[2]  = imu.rpy[2]
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def get_dof_pos_vel(self) -> tuple[list[float], list[float]]:
        """Return (dof_pos, dof_vel) in URDF policy order (16 joints).

        Physical L/R swap: SDK kLeft* indices are wired to the robot's physical
        RIGHT side and kRight* to the physical LEFT side.  So policy "left" slots
        are fed from K1Ji.kRight* encoders and vice versa.
        """
        q  = self._q
        dq = self._dq

        # SDK "left" cranks = physical RIGHT ankle
        sdk_l_ap, sdk_l_ar     = crank_to_ankle(q[K1Ji.kCrankUpLeft],   q[K1Ji.kCrankDownLeft])
        sdk_l_ap_v, sdk_l_ar_v = crank_to_ankle(dq[K1Ji.kCrankUpLeft],  dq[K1Ji.kCrankDownLeft])
        # SDK "right" cranks = physical LEFT ankle
        sdk_r_ap, sdk_r_ar     = crank_to_ankle(q[K1Ji.kCrankUpRight],  q[K1Ji.kCrankDownRight])
        sdk_r_ap_v, sdk_r_ar_v = crank_to_ankle(dq[K1Ji.kCrankUpRight], dq[K1Ji.kCrankDownRight])

        pos = [
            q[K1Ji.kRightShoulderPitch],  # policy: ALeft_Shoulder_Pitch
            q[K1Ji.kRightElbowYaw],       # policy: Left_Elbow_Yaw
            q[K1Ji.kLeftShoulderPitch],   # policy: ARight_Shoulder_Pitch
            q[K1Ji.kLeftElbowYaw],        # policy: Right_Elbow_Yaw
            q[K1Ji.kRightHipPitch],       # policy: Left_Hip_Pitch
            q[K1Ji.kRightHipRoll],        # policy: Left_Hip_Roll
            q[K1Ji.kRightHipYaw],         # policy: Left_Hip_Yaw
            q[K1Ji.kRightKneePitch],      # policy: Left_Knee_Pitch
            sdk_r_ap, sdk_r_ar,           # policy: Left_Ankle_Pitch/Roll ← physical left (SDK right)
            q[K1Ji.kLeftHipPitch],        # policy: Right_Hip_Pitch
            q[K1Ji.kLeftHipRoll],         # policy: Right_Hip_Roll
            q[K1Ji.kLeftHipYaw],          # policy: Right_Hip_Yaw
            q[K1Ji.kLeftKneePitch],       # policy: Right_Knee_Pitch
            sdk_l_ap, sdk_l_ar,           # policy: Right_Ankle_Pitch/Roll ← physical right (SDK left)
        ]
        vel = [
            dq[K1Ji.kRightShoulderPitch],
            dq[K1Ji.kRightElbowYaw],
            dq[K1Ji.kLeftShoulderPitch],
            dq[K1Ji.kLeftElbowYaw],
            dq[K1Ji.kRightHipPitch],
            dq[K1Ji.kRightHipRoll],
            dq[K1Ji.kRightHipYaw],
            dq[K1Ji.kRightKneePitch],
            sdk_r_ap_v, sdk_r_ar_v,
            dq[K1Ji.kLeftHipPitch],
            dq[K1Ji.kLeftHipRoll],
            dq[K1Ji.kLeftHipYaw],
            dq[K1Ji.kLeftKneePitch],
            sdk_l_ap_v, sdk_l_ar_v,
        ]
        return pos, vel

    def get_imu(self) -> tuple[list[float], float, float]:
        return list(self._gyro), self._rpy[0], self._rpy[1]

    def is_fallen(self, roll_limit: float = 1.0, pitch_limit: float = 1.0) -> bool:
        return abs(self._rpy[0]) > roll_limit or abs(self._rpy[1]) > pitch_limit

    def get_crank_pos(self) -> tuple[float, float, float, float]:
        return (
            self._q[K1Ji.kCrankUpLeft],
            self._q[K1Ji.kCrankDownLeft],
            self._q[K1Ji.kCrankUpRight],
            self._q[K1Ji.kCrankDownRight],
        )


# ---------------------------------------------------------------------------
# SDK command builder
# ---------------------------------------------------------------------------

_K1JI_NAMES: dict[int, str] = {
    K1Ji.kHeadYaw:            "HeadYaw",
    K1Ji.kHeadPitch:          "HeadPitch",
    K1Ji.kLeftShoulderPitch:  "L_ShoulderPitch",
    K1Ji.kLeftShoulderRoll:   "L_ShoulderRoll",
    K1Ji.kLeftElbowPitch:     "L_ElbowPitch",
    K1Ji.kLeftElbowYaw:       "L_ElbowYaw",
    K1Ji.kRightShoulderPitch: "R_ShoulderPitch",
    K1Ji.kRightShoulderRoll:  "R_ShoulderRoll",
    K1Ji.kRightElbowPitch:    "R_ElbowPitch",
    K1Ji.kRightElbowYaw:      "R_ElbowYaw",
    K1Ji.kLeftHipPitch:       "L_HipPitch",
    K1Ji.kLeftHipRoll:        "L_HipRoll",
    K1Ji.kLeftHipYaw:         "L_HipYaw",
    K1Ji.kLeftKneePitch:      "L_KneePitch",
    K1Ji.kCrankUpLeft:        "L_CrankUp",
    K1Ji.kCrankDownLeft:      "L_CrankDown",
    K1Ji.kRightHipPitch:      "R_HipPitch",
    K1Ji.kRightHipRoll:       "R_HipRoll",
    K1Ji.kRightHipYaw:        "R_HipYaw",
    K1Ji.kRightKneePitch:     "R_KneePitch",
    K1Ji.kCrankUpRight:       "R_CrankUp",
    K1Ji.kCrankDownRight:     "R_CrankDown",
}


def dump_debug(
    state_buf: "RobotStateBuffer",
    dof_pos: list[float],
    dof_vel: list[float],
    filtered_dof_pos: list[float],
    low_cmd: "LowCmd",
    path: str = "joint_debug.txt",
) -> None:
    """Write a one-shot snapshot of joint state and commands to *path*.

    Uses the already-copied Python floats in state_buf — never touches the
    SDK C++ object (which may have been freed by the time we run).

    Section 1 — Raw copied motor values (serial + parallel):
        K1Ji index, name, q, dq for every slot.
    Section 2 — URDF policy mapping (what the policy sees / commands):
        JOINT_NAMES order with observed pos, default, filtered target.
    Section 3 — LowCmd sent to robot:
        Every K1Ji slot: name, q, tau, kp, kd.
    """
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("SECTION 1 — motor_state_serial (all 22 K1 joints)")
    lines.append(f"  raw len at last callback: serial={state_buf._raw_serial_len}"
                 f"  parallel={state_buf._raw_parallel_len}")
    lines.append(f"  {'K1Ji_idx':>8}  {'K1Ji_name':<18}  {'q':>10}  {'dq':>10}")
    for i in range(K1_JOINT_CNT):
        lines.append(
            f"  K1Ji[{i:2d}]   {_K1JI_NAMES.get(i,'???'):<18}"
            f"  {state_buf._q[i]:+10.4f}  {state_buf._dq[i]:+10.4f}"
        )

    lines.append("")
    lines.append("SECTION 2 — URDF policy joint order (16 joints)")
    lines.append(f"  {'#':>2}  {'URDF_name':<24}  {'obs_pos':>10}  {'default':>10}  {'filtered_tgt':>12}")
    for i, (name, obs, default, tgt) in enumerate(
        zip(JOINT_NAMES, dof_pos, DEFAULT_DOF_POS.tolist(), filtered_dof_pos)
    ):
        lines.append(f"  {i:2d}  {name:<24}  {obs:+10.4f}  {default:+10.4f}  {tgt:+12.4f}")

    lines.append("")
    lines.append("SECTION 3 — LowCmd sent to robot (K1Ji order, 22 slots)")
    lines.append(f"  {'K1Ji_idx':>8}  {'K1Ji_name':<18}  {'q':>10}  {'tau':>10}  {'kp':>6}  {'kd':>6}  ctrl")
    for i in range(K1_JOINT_CNT):
        mc = low_cmd.motor_cmd[i]
        ctrl = "TORQUE" if i in K1_CRANK_INDICES else "pos"
        lines.append(
            f"  K1Ji[{i:2d}]   {_K1JI_NAMES.get(i,'???'):<18}  {mc.q:+10.4f}  {mc.tau:+10.4f}"
            f"  {mc.kp:6.1f}  {mc.kd:6.1f}  {ctrl}"
        )

    lines.append("=" * 70)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[debug] joint snapshot written to {path}")


def alloc_low_cmd() -> LowCmd:
    """Allocate the LowCmd once; reuse it every step via update_low_cmd()."""
    motor_cmds = [MotorCmd() for _ in range(K1_JOINT_CNT)]
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    low_cmd.motor_cmd = motor_cmds
    # Write constant fields once (kp/kd for non-crank joints never change)
    for idx in range(K1_JOINT_CNT):
        kp, kd, _ = SDK_JOINT_GAINS.get(idx, (0.0, 0.0, 0.0))
        mc = low_cmd.motor_cmd[idx]
        mc.dq     = 0.0
        mc.weight = 0.0
        mc.kp     = kp   # 0.0 for crank joints (torque mode)
        mc.kd     = kd
        mc.tau    = 0.0
        mc.q      = 0.0
    return low_cmd


def update_low_cmd(
    low_cmd: LowCmd,
    target_dof_pos: list[float],
    current_crank_pos: tuple[float, float, float, float],
) -> None:
    """Update the pre-allocated LowCmd in-place (no heap allocation per step).

    Crank joints: kp=0, tau = clip((target−current)×stiffness, ±limit).
    All other joints: position control, kp/kd already set in alloc_low_cmd.
    """
    (
        l_sh_pitch, l_el_yaw,
        r_sh_pitch, r_el_yaw,
        l_hip_p, l_hip_r, l_hip_y, l_knee,
        l_ank_p, l_ank_r,
        r_hip_p, r_hip_r, r_hip_y, r_knee,
        r_ank_p, r_ank_r,
    ) = target_dof_pos

    l_crank_up_tgt, l_crank_dn_tgt = ankle_to_crank(l_ank_p, l_ank_r)
    r_crank_up_tgt, r_crank_dn_tgt = ankle_to_crank(r_ank_p, r_ank_r)
    crank_cur_ul, crank_cur_dl, crank_cur_ur, crank_cur_dr = current_crank_pos

    mc = low_cmd.motor_cmd

    # Position-controlled joints.
    # Physical L/R swap: send policy "left" outputs to SDK right motors and vice versa.
    mc[K1Ji.kHeadYaw].q           = FIXED_JOINT_DEFAULTS["AAHead_yaw"]
    mc[K1Ji.kHeadPitch].q         = FIXED_JOINT_DEFAULTS["Head_pitch"]
    mc[K1Ji.kRightShoulderPitch].q = l_sh_pitch                              # policy left → SDK right (physical left)
    mc[K1Ji.kRightShoulderRoll].q  = FIXED_JOINT_DEFAULTS["Left_Shoulder_Roll"]
    mc[K1Ji.kRightElbowPitch].q    = FIXED_JOINT_DEFAULTS["Left_Elbow_Pitch"]
    mc[K1Ji.kRightElbowYaw].q      = l_el_yaw
    mc[K1Ji.kLeftShoulderPitch].q  = r_sh_pitch                              # policy right → SDK left (physical right)
    mc[K1Ji.kLeftShoulderRoll].q   = FIXED_JOINT_DEFAULTS["Right_Shoulder_Roll"]
    mc[K1Ji.kLeftElbowPitch].q     = FIXED_JOINT_DEFAULTS["Right_Elbow_Pitch"]
    mc[K1Ji.kLeftElbowYaw].q       = r_el_yaw
    mc[K1Ji.kRightHipPitch].q      = l_hip_p                                 # policy left leg → SDK right (physical left)
    mc[K1Ji.kRightHipRoll].q       = l_hip_r
    mc[K1Ji.kRightHipYaw].q        = l_hip_y
    mc[K1Ji.kRightKneePitch].q     = l_knee
    mc[K1Ji.kLeftHipPitch].q       = r_hip_p                                 # policy right leg → SDK left (physical right)
    mc[K1Ji.kLeftHipRoll].q        = r_hip_r
    mc[K1Ji.kLeftHipYaw].q         = r_hip_y
    mc[K1Ji.kLeftKneePitch].q      = r_knee

    # Torque-controlled crank joints (same L/R swap).
    def _crank(idx: int, tgt: float, cur: float) -> None:
        _, _, limit = SDK_JOINT_GAINS[idx]
        mc[idx].q   = cur   # hold current position; no position-mode jump
        mc[idx].tau = max(-limit, min(limit, (tgt - cur) * CRANK_STIFFNESS[idx]))

    # policy left ankle → SDK right cranks (physical left ankle)
    _crank(K1Ji.kCrankUpRight,   l_crank_up_tgt, crank_cur_ur)
    _crank(K1Ji.kCrankDownRight, l_crank_dn_tgt, crank_cur_dr)
    # policy right ankle → SDK left cranks (physical right ankle)
    _crank(K1Ji.kCrankUpLeft,    r_crank_up_tgt, crank_cur_ul)
    _crank(K1Ji.kCrankDownLeft,  r_crank_dn_tgt, crank_cur_dl)


def damp_cmd() -> LowCmd:
    """All motors set to zero torque (emergency stop)."""
    motor_cmds = [MotorCmd() for _ in range(K1_JOINT_CNT)]
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    low_cmd.motor_cmd = motor_cmds
    for i in range(K1_JOINT_CNT):
        low_cmd.motor_cmd[i].q   = 0.0
        low_cmd.motor_cmd[i].dq  = 0.0
        low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.motor_cmd[i].kp  = 0.0
        low_cmd.motor_cmd[i].kd  = 2.0  # gentle damping
        low_cmd.motor_cmd[i].weight = 0.0
    return low_cmd


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def build_obs(
    gyro: list[float],
    roll: float,
    pitch: float,
    commands: list[float],          # [vx, vy, wz]
    dof_pos: list[float],           # URDF order, 16 joints
    dof_vel: list[float],
    last_actions: list[float],
    gait_phase: float,              # ∈ [0, 1)
    *,
    privileged: bool,               # True = teacher (adds faked heading + contact); False = student
    foot_contact: list[float] | None = None,   # [L, R] ∈ {0, 1}; only used when privileged
    expected_dim: int | None = None,
) -> torch.Tensor:
    proj_g = rpy_to_projected_gravity(roll, pitch)

    ang_vel_obs  = [g * OBS_SCALE_ANG_VEL for g in gyro]
    cmd_scaled   = [
        commands[0] * OBS_SCALE_LIN_VEL,
        commands[1] * OBS_SCALE_LIN_VEL,
        commands[2] * OBS_SCALE_ANG_VEL,
    ]

    pos_rel = [(p - d) * OBS_SCALE_DOF_POS for p, d in zip(dof_pos, DEFAULT_DOF_POS.tolist())]
    vel_obs  = [v * OBS_SCALE_DOF_VEL for v in dof_vel]

    phase_2pi = gait_phase * 2.0 * math.pi
    clock = [math.sin(phase_2pi), math.cos(phase_2pi)]

    # Student (proprio) layout: ang_vel, grav, cmd, dof_pos, dof_vel, last_action, clock.
    # The student is trained WITHOUT base_lin_vel_heading and foot_contact (privileged /
    # un-deployable), so they are simply omitted here — no fake values fed to the network.
    flat = ang_vel_obs + proj_g + cmd_scaled
    if privileged:
        flat += [0.0, 0.0]  # base_lin_vel_heading: not observable without odometry → 0
    flat += pos_rel + vel_obs + last_actions + clock
    if privileged:
        fc = foot_contact if foot_contact is not None else [1.0, 1.0]
        flat += [c - 0.5 for c in fc]

    if expected_dim is not None:
        assert len(flat) == expected_dim, f"obs dim mismatch: {len(flat)} != {expected_dim}"
    return torch.tensor(flat, dtype=torch.float32).unsqueeze(0)


def run(args: argparse.Namespace) -> None:
    model_path = resolve_model(args.model)
    print(f"Loading model: {model_path}")
    actor, obs_dim, is_student = load_actor(model_path)
    kind = "STUDENT (onboard proprio obs)" if is_student else "TEACHER (privileged obs)"
    print(f"Model loaded: {kind}. OBS_DIM={obs_dim}, NUM_ACTIONS={NUM_ACTIONS}")

    state_buf = RobotStateBuffer()
    ChannelFactory.Instance().Init(0, args.interface)

    subscriber = B1LowStateSubscriber(state_buf.update)
    subscriber.InitChannel()

    publisher = B1LowCmdPublisher()
    publisher.InitChannel()

    client = B1LocoClient()
    client.Init()

    # Graceful shutdown on Ctrl+C
    running = True
    def _sigint_handler(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _sigint_handler)

    print("Waiting for first robot state message…")
    t0 = time.monotonic()
    while not state_buf.ready:
        if time.monotonic() - t0 > 5.0:
            print("ERROR: no state received after 5 s. Check network interface.", file=sys.stderr)
            sys.exit(1)
        time.sleep(0.01)
    print("State received.")

    # --- Roboter in Custom-Modus schalten (wie altes Script) ---
    print("Switching to Custom mode…")
    client.ChangeMode(RobotMode.kCustom)
    time.sleep(0.5)
    print("Custom mode active. Starting policy loop.")

    commands   = [args.vx, args.vy, args.wz]
    last_actions: list[float] = [0.0] * NUM_ACTIONS
    # Glättungsfilter-Startwert: Standardposition des Roboters
    filtered_dof_pos: list[float] = DEFAULT_DOF_POS.tolist()
    gait_phase = 0.0
    step = 0
    fall_detected = False
    t_start = time.monotonic()
    t_next  = t_start
    low_cmd = alloc_low_cmd()   # allocate once; updated in-place each step

    while running:
        now = time.monotonic()
        if args.duration > 0 and (now - t_start) >= args.duration:
            print(f"\nSmoke test done ({args.duration} s).")
            break

        # --- read state ---
        gyro, roll, pitch = state_buf.get_imu()
        dof_pos, dof_vel  = state_buf.get_dof_pos_vel()

        # --- Sturzdetektor (wie altes Script: |rpy| > 1.0 rad) ---
        if state_buf.is_fallen():
            print(
                f"\nFALL DETECTED: roll={math.degrees(roll):+.1f}°  "
                f"pitch={math.degrees(pitch):+.1f}° — Notabbruch!",
                file=sys.stderr,
            )
            fall_detected = True
            break

        # --- build observation ---
        # Student: foot_contact is omitted entirely. Teacher: faked as both-in-contact.
        obs = build_obs(
            gyro=gyro,
            roll=roll,
            pitch=pitch,
            commands=commands,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            last_actions=last_actions,
            gait_phase=gait_phase,
            privileged=not is_student,
            foot_contact=[1.0, 1.0],
            expected_dim=obs_dim,
        )

        # --- policy inference ---
        with torch.no_grad():
            actions = actor(obs).squeeze(0)  # (16,)
        actions = torch.clamp(actions, -CLIP_ACTIONS, CLIP_ACTIONS)

        # --- action → target joint positions ---
        raw_dof_pos = (actions * ACTION_SCALE + DEFAULT_DOF_POS).tolist()
        last_actions = actions.tolist()

        # --- Glättungsfilter: 80 % alt + 20 % neu (wie altes Script) ---
        filtered_dof_pos = [
            0.8 * f + 0.2 * r for f, r in zip(filtered_dof_pos, raw_dof_pos)
        ]

        # --- send command ---
        update_low_cmd(low_cmd, filtered_dof_pos, state_buf.get_crank_pos())
        publisher.Write(low_cmd)

        # --- one-shot debug dump on first step ---
        if step == 0:
            dump_debug(state_buf, dof_pos, dof_vel, filtered_dof_pos, low_cmd)

        # --- advance phase clock ---
        gait_phase = (gait_phase + 1.0 / GAIT_PERIOD_STEPS) % 1.0
        step += 1

        if step % 50 == 0:
            elapsed = now - t_start
            act_max  = float(actions.abs().max())
            act_mean = float(actions.abs().mean())
            # delta between current filtered target and default pose (key leg joints)
            d = [filtered_dof_pos[i] - DEFAULT_DOF_POS[i].item() for i in range(NUM_ACTIONS)]
            print(
                f"[{elapsed:6.1f}s] step={step:5d} "
                f"roll={math.degrees(roll):+.1f}° pitch={math.degrees(pitch):+.1f}° "
                f"phase={gait_phase:.2f} | "
                f"act max={act_max:.3f} mean={act_mean:.3f} | "
                f"Δhip_L={d[4]:+.3f} Δknee_L={d[7]:+.3f} "
                f"Δhip_R={d[10]:+.3f} Δknee_R={d[13]:+.3f}"
            )

        # --- pace to 50 Hz ---
        t_next += DT
        sleep_remaining = t_next - time.monotonic()
        if sleep_remaining > 0:
            time.sleep(sleep_remaining)

    # --- safe stop ---
    print("Sending damp commands…")
    for _ in range(10):
        publisher.Write(damp_cmd())
        time.sleep(0.02)

    print("Switching to Damping mode…")
    client.ChangeMode(RobotMode.kDamping)
    print("Done." if not fall_detected else "Stopped due to fall detection.")


def main() -> None:
    parser = argparse.ArgumentParser(description="K1 policy smoke test")
    parser.add_argument("--model",     default=DEFAULT_MODEL_DIR,
                        help="Checkpoint .pt, or a log dir (auto-picks the latest model_*.pt). "
                             "Defaults to the distillation student dir 'logs/k1-distill'. A student "
                             "(student_state_dict) or teacher (actor_state_dict) is auto-detected.")
    parser.add_argument("--interface", default="127.0.0.1",
                        help="Network interface or IP for SDK (use 127.0.0.1 when running on the robot)")
    parser.add_argument("--duration",  type=float, default=10.0,
                        help="Test duration in seconds (0 = run until Ctrl+C)")
    parser.add_argument("--vx",        type=float, default=0.0,  help="Commanded forward velocity [m/s]")
    parser.add_argument("--vy",        type=float, default=0.0,  help="Commanded lateral velocity [m/s]")
    parser.add_argument("--wz",        type=float, default=0.0,  help="Commanded yaw rate [rad/s]")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
