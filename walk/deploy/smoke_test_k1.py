"""Smoke test: run a trained K1 walking policy on the real Booster K1 robot.

Requires:
  - booster_robotics_sdk_python  (build from ~/git/booster_robotics_sdk)
  - PyTorch CPU

Usage (from repo root, running directly ON the robot):
  python walk/deploy/smoke_test_k1.py \\
      --model model_1250.pt \\
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
import math
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

# Gait clock (from k1_env.yaml)
GAIT_PERIOD_S  = 1.1
GAIT_PERIOD_STEPS = max(1, round(GAIT_PERIOD_S / DT))  # 55

# ---------------------------------------------------------------------------
# Policy network (mirrors rsl-rl MLPModel with ELU + GaussianDistribution)
# ---------------------------------------------------------------------------

OBS_DIM = 63  # see k1_env.yaml header comment


class ActorMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(OBS_DIM, 512), nn.ELU(),
            nn.Linear(512, 256),     nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def load_actor(checkpoint_path: str) -> ActorMLP:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = ActorMLP()
    # strict=False: rsl-rl also stores distribution.std_param which we don't need for inference
    actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
    actor.eval()
    return actor


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
    """Thread-safe latest-state buffer filled by the subscriber callback."""

    def __init__(self) -> None:
        self._state = None

    def update(self, state) -> None:
        self._state = state

    @property
    def ready(self) -> bool:
        return self._state is not None

    def get_dof_pos_vel(self) -> tuple[list[float], list[float]]:
        """Return (dof_pos, dof_vel) in URDF policy order (16 joints).

        K1 layout: motor_state_serial = head+arms (K1Ji 0-9, 10 entries),
        motor_state_parallel = legs (K1Ji 10-21, 12 entries,
        parallel[i] = K1Ji[i + K1_LEG_OFFSET]).
        """
        s = self._state
        ms = s.motor_state_serial
        mp = s.motor_state_parallel

        def q(k1_idx: int) -> float:
            if k1_idx < len(ms):
                return ms[k1_idx].q
            return mp[k1_idx - K1_LEG_OFFSET].q

        def dq(k1_idx: int) -> float:
            if k1_idx < len(ms):
                return ms[k1_idx].dq
            return mp[k1_idx - K1_LEG_OFFSET].dq

        l_ap, l_ar     = crank_to_ankle(q(K1Ji.kCrankUpLeft),   q(K1Ji.kCrankDownLeft))
        r_ap, r_ar     = crank_to_ankle(q(K1Ji.kCrankUpRight),  q(K1Ji.kCrankDownRight))
        l_ap_v, l_ar_v = crank_to_ankle(dq(K1Ji.kCrankUpLeft),  dq(K1Ji.kCrankDownLeft))
        r_ap_v, r_ar_v = crank_to_ankle(dq(K1Ji.kCrankUpRight), dq(K1Ji.kCrankDownRight))

        pos = [
            q(K1Ji.kLeftShoulderPitch),
            q(K1Ji.kLeftElbowYaw),
            q(K1Ji.kRightShoulderPitch),
            q(K1Ji.kRightElbowYaw),
            q(K1Ji.kLeftHipPitch),
            q(K1Ji.kLeftHipRoll),
            q(K1Ji.kLeftHipYaw),
            q(K1Ji.kLeftKneePitch),
            l_ap, l_ar,
            q(K1Ji.kRightHipPitch),
            q(K1Ji.kRightHipRoll),
            q(K1Ji.kRightHipYaw),
            q(K1Ji.kRightKneePitch),
            r_ap, r_ar,
        ]
        vel = [
            dq(K1Ji.kLeftShoulderPitch),
            dq(K1Ji.kLeftElbowYaw),
            dq(K1Ji.kRightShoulderPitch),
            dq(K1Ji.kRightElbowYaw),
            dq(K1Ji.kLeftHipPitch),
            dq(K1Ji.kLeftHipRoll),
            dq(K1Ji.kLeftHipYaw),
            dq(K1Ji.kLeftKneePitch),
            l_ap_v, l_ar_v,
            dq(K1Ji.kRightHipPitch),
            dq(K1Ji.kRightHipRoll),
            dq(K1Ji.kRightHipYaw),
            dq(K1Ji.kRightKneePitch),
            r_ap_v, r_ar_v,
        ]
        return pos, vel

    def get_imu(self) -> tuple[list[float], float, float]:
        """Return (gyro[3], roll, pitch)."""
        imu = self._state.imu_state
        return list(imu.gyro), imu.rpy[0], imu.rpy[1]

    def is_fallen(self, roll_limit: float = 1.0, pitch_limit: float = 1.0) -> bool:
        """True wenn Roll oder Pitch den Sicherheitsgrenzwert überschreiten (Sturzerkennung)."""
        imu = self._state.imu_state
        return abs(imu.rpy[0]) > roll_limit or abs(imu.rpy[1]) > pitch_limit

    def get_crank_pos(self) -> tuple[float, float, float, float]:
        """Return raw crank motor positions (up_left, dn_left, up_right, dn_right).

        Used by the torque controller for the parallel-linkage ankle joints.
        """
        mp = self._state.motor_state_parallel
        return (
            mp[K1Ji.kCrankUpLeft   - K1_LEG_OFFSET].q,
            mp[K1Ji.kCrankDownLeft - K1_LEG_OFFSET].q,
            mp[K1Ji.kCrankUpRight  - K1_LEG_OFFSET].q,
            mp[K1Ji.kCrankDownRight - K1_LEG_OFFSET].q,
        )


# ---------------------------------------------------------------------------
# SDK command builder
# ---------------------------------------------------------------------------

def build_low_cmd(
    target_dof_pos: list[float],              # URDF policy order (16 joints)
    current_crank_pos: tuple[float, float, float, float],  # (up_l, dn_l, up_r, dn_r)
) -> LowCmd:
    """Map 16-DOF URDF targets → 22-motor K1 LowCmd (K1Ji ordering, SERIAL mode).

    Crank joints use the torque approach (kp=0) to avoid issues with the
    parallel-linkage non-linearity: tau = clip((target-current)×stiffness, ±limit).
    All other joints use position control (kp/kd from SDK_JOINT_GAINS).
    """
    motor_cmds = [MotorCmd() for _ in range(K1_JOINT_CNT)]

    # Policy targets in URDF order
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

    # Position targets for all joints (K1Ji indices)
    pos_targets: dict[int, float] = {
        K1Ji.kLeftShoulderPitch:  l_sh_pitch,
        K1Ji.kLeftElbowYaw:       l_el_yaw,
        K1Ji.kRightShoulderPitch: r_sh_pitch,
        K1Ji.kRightElbowYaw:      r_el_yaw,
        K1Ji.kLeftHipPitch:       l_hip_p,
        K1Ji.kLeftHipRoll:        l_hip_r,
        K1Ji.kLeftHipYaw:         l_hip_y,
        K1Ji.kLeftKneePitch:      l_knee,
        K1Ji.kRightHipPitch:      r_hip_p,
        K1Ji.kRightHipRoll:       r_hip_r,
        K1Ji.kRightHipYaw:        r_hip_y,
        K1Ji.kRightKneePitch:     r_knee,
        # Fixed joints
        K1Ji.kHeadYaw:            FIXED_JOINT_DEFAULTS["AAHead_yaw"],
        K1Ji.kHeadPitch:          FIXED_JOINT_DEFAULTS["Head_pitch"],
        K1Ji.kLeftShoulderRoll:   FIXED_JOINT_DEFAULTS["Left_Shoulder_Roll"],
        K1Ji.kRightShoulderRoll:  FIXED_JOINT_DEFAULTS["Right_Shoulder_Roll"],
        K1Ji.kLeftElbowPitch:     FIXED_JOINT_DEFAULTS["Left_Elbow_Pitch"],
        K1Ji.kRightElbowPitch:    FIXED_JOINT_DEFAULTS["Right_Elbow_Pitch"],
    }

    # Torque targets for crank joints (target crank position, current crank position)
    crank_torque: dict[int, tuple[float, float, float]] = {
        K1Ji.kCrankUpLeft:   (l_crank_up_tgt, crank_cur_ul, CRANK_STIFFNESS[K1Ji.kCrankUpLeft]),
        K1Ji.kCrankDownLeft: (l_crank_dn_tgt, crank_cur_dl, CRANK_STIFFNESS[K1Ji.kCrankDownLeft]),
        K1Ji.kCrankUpRight:  (r_crank_up_tgt, crank_cur_ur, CRANK_STIFFNESS[K1Ji.kCrankUpRight]),
        K1Ji.kCrankDownRight:(r_crank_dn_tgt, crank_cur_dr, CRANK_STIFFNESS[K1Ji.kCrankDownRight]),
    }

    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    low_cmd.motor_cmd = motor_cmds

    for idx in range(K1_JOINT_CNT):
        _, kd, effort_limit = SDK_JOINT_GAINS.get(idx, (0.0, 0.0, 0.0))
        mc = low_cmd.motor_cmd[idx]
        mc.dq     = 0.0
        mc.weight = 0.0

        if idx in crank_torque:
            tgt, cur, stiff = crank_torque[idx]
            mc.q   = cur                                          # hold current to avoid position jump
            mc.kp  = 0.0
            mc.kd  = kd
            mc.tau = max(-effort_limit, min(effort_limit, (tgt - cur) * stiff))
        else:
            kp, _, _ = SDK_JOINT_GAINS.get(idx, (0.0, 0.0, 0.0))
            mc.q   = pos_targets.get(idx, 0.0)
            mc.kp  = kp
            mc.kd  = kd
            mc.tau = 0.0

    return low_cmd


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
    foot_contact: list[float],      # [L, R] ∈ {0, 1}
) -> torch.Tensor:
    proj_g = rpy_to_projected_gravity(roll, pitch)

    ang_vel_obs  = [g * OBS_SCALE_ANG_VEL for g in gyro]
    cmd_scaled   = [
        commands[0] * OBS_SCALE_LIN_VEL,
        commands[1] * OBS_SCALE_LIN_VEL,
        commands[2] * OBS_SCALE_ANG_VEL,
    ]
    # base_lin_vel_heading: not directly observable without odometry → set to 0
    lin_vel_heading = [0.0, 0.0]

    pos_rel = [(p - d) * OBS_SCALE_DOF_POS for p, d in zip(dof_pos, DEFAULT_DOF_POS.tolist())]
    vel_obs  = [v * OBS_SCALE_DOF_VEL for v in dof_vel]

    phase_2pi = gait_phase * 2.0 * math.pi
    clock = [math.sin(phase_2pi), math.cos(phase_2pi)]

    contact_obs = [c - 0.5 for c in foot_contact]

    flat = (
        ang_vel_obs         # 3
        + proj_g            # 3
        + cmd_scaled        # 3
        + lin_vel_heading   # 2
        + pos_rel           # 16
        + vel_obs           # 16
        + last_actions      # 16
        + clock             # 2
        + contact_obs       # 2
    )
    assert len(flat) == OBS_DIM, f"obs dim mismatch: {len(flat)} != {OBS_DIM}"
    return torch.tensor(flat, dtype=torch.float32).unsqueeze(0)


def run(args: argparse.Namespace) -> None:
    print(f"Loading model: {args.model}")
    actor = load_actor(args.model)
    print("Model loaded. OBS_DIM=63, NUM_ACTIONS=16")

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

        # simple foot contact estimate: both feet always in contact during stand test
        foot_contact = [1.0, 1.0]

        # --- build observation ---
        obs = build_obs(
            gyro=gyro,
            roll=roll,
            pitch=pitch,
            commands=commands,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            last_actions=last_actions,
            gait_phase=gait_phase,
            foot_contact=foot_contact,
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
        low_cmd = build_low_cmd(filtered_dof_pos, state_buf.get_crank_pos())
        publisher.Write(low_cmd)

        # --- advance phase clock ---
        gait_phase = (gait_phase + 1.0 / GAIT_PERIOD_STEPS) % 1.0
        step += 1

        if step % 50 == 0:
            elapsed = now - t_start
            print(
                f"[{elapsed:6.1f}s] step={step:5d} "
                f"roll={math.degrees(roll):+.1f}° pitch={math.degrees(pitch):+.1f}° "
                f"phase={gait_phase:.2f}"
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
    parser.add_argument("--model",     default="logs/foot-rool-penaly/model_1250.pt",
                        help="Path to model checkpoint (.pt)")
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
