OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
import math
import torch

OBS_DIM = 59

STUDENT_OBS_LABELS = [
    # ── base_ang_vel (body frame, ×0.25) ─────────────────────────────────────
    "0: base_ang_vel_x",       # float32; Roh: ωx [rad/s], kein Hard-Clip; Obs = ωx×0.25, typ. ~[-2.5, 2.5]
    "1: base_ang_vel_y",       # float32; Roh: ωy [rad/s], kein Hard-Clip; Obs = ωy×0.25, typ. ~[-2.5, 2.5]
    "2: base_ang_vel_z",       # float32; Roh: ωz [rad/s], kein Hard-Clip; Obs = ωz×0.25, typ. ~[-2.5, 2.5]
    # ── projected_gravity (body frame, unskaliert) ─────────────────────────────
    "3: projected_gravity_x",  # float32; Einheitsvektor-Komponente ∈ [-1, 1]; aufrecht ≈ 0
    "4: projected_gravity_y",  # float32; Einheitsvektor-Komponente ∈ [-1, 1]; aufrecht ≈ 0
    "5: projected_gravity_z",  # float32; Einheitsvektor-Komponente ∈ [-1, 1]; aufrecht ≈ -1
    # ── commands (×2.0 für vx/vy, ×0.25 für wz) ────────────────────────────────
    "6: command_lin_vel_x",    # float32; Roh vx [m/s] typ. [-0.2, 1.3] → Obs [-0.4, 2.6]
    "7: command_lin_vel_y",    # float32; Roh vy [m/s] typ. [-0.4, 0.5] → Obs [-0.8, 1.0]
    "8: command_ang_vel_yaw",  # float32; Roh wz [rad/s] typ. [-0.5, 0.5] → Obs [-0.125, 0.125]
    # ── dof_pos = (q - default_q) × 1.0 [rad] ────────────────────────────────
    "9: dof_pos_ALeft_Shoulder_Pitch",   # float32; Obs ∈ [-3.316,  1.220]
    "10: dof_pos_Left_Elbow_Yaw",        # float32; Obs ∈ [-1.440,  1.000]
    "11: dof_pos_ARight_Shoulder_Pitch", # float32; Obs ∈ [-3.316,  1.220]
    "12: dof_pos_Right_Elbow_Yaw",       # float32; Obs ∈ [-1.000,  1.440]
    "13: dof_pos_Left_Hip_Pitch",        # float32; Obs ∈ [-2.700,  2.510]
    "14: dof_pos_Left_Hip_Roll",         # float32; Obs ∈ [-0.400,  1.570]
    "15: dof_pos_Left_Hip_Yaw",          # float32; Obs ∈ [-1.000,  1.000]
    "16: dof_pos_Left_Knee_Pitch",       # float32; Obs ∈ [-0.600,  1.630]
    "17: dof_pos_Left_Ankle_Pitch",      # float32; Obs ∈ [-0.570,  0.645]
    "18: dof_pos_Left_Ankle_Roll",       # float32; Obs ∈ [-0.345,  0.345]
    "19: dof_pos_Right_Hip_Pitch",       # float32; Obs ∈ [-2.700,  2.510]
    "20: dof_pos_Right_Hip_Roll",        # float32; Obs ∈ [-1.570,  0.400]
    "21: dof_pos_Right_Hip_Yaw",         # float32; Obs ∈ [-1.000,  1.000]
    "22: dof_pos_Right_Knee_Pitch",      # float32; Obs ∈ [-0.600,  1.630]
    "23: dof_pos_Right_Ankle_Pitch",     # float32; Obs ∈ [-0.570,  0.645]
    "24: dof_pos_Right_Ankle_Roll",      # float32; Obs ∈ [-0.345,  0.345]
    # ── dof_vel = dq × 0.05 [rad/s] ──────────────────────────────────────────
    "25: dof_vel_ALeft_Shoulder_Pitch",   # float32; Roh dq [rad/s], kein Hard-Clip; Obs = dq×0.05, typ. ~[-0.75, 0.75]
    "26: dof_vel_Left_Elbow_Yaw",         # float32; dito
    "27: dof_vel_ARight_Shoulder_Pitch",  # float32; dito
    "28: dof_vel_Right_Elbow_Yaw",        # float32; dito
    "29: dof_vel_Left_Hip_Pitch",         # float32; dito
    "30: dof_vel_Left_Hip_Roll",          # float32; dito
    "31: dof_vel_Left_Hip_Yaw",           # float32; dito
    "32: dof_vel_Left_Knee_Pitch",        # float32; dito
    "33: dof_vel_Left_Ankle_Pitch",       # float32; dito
    "34: dof_vel_Left_Ankle_Roll",        # float32; dito
    "35: dof_vel_Right_Hip_Pitch",        # float32; dito
    "36: dof_vel_Right_Hip_Roll",         # float32; dito
    "37: dof_vel_Right_Hip_Yaw",          # float32; dito
    "38: dof_vel_Right_Knee_Pitch",       # float32; dito
    "39: dof_vel_Right_Ankle_Pitch",      # float32; dito
    "40: dof_vel_Right_Ankle_Roll",       # float32; dito
    # ── last_action (Policy-Output, unskaliert) ──────────────────────────────
    "41: last_action_ALeft_Shoulder_Pitch",   # float32; ∈ [-100.0, 100.0] (clip_actions)
    "42: last_action_Left_Elbow_Yaw",         # float32; ∈ [-100.0, 100.0]
    "43: last_action_ARight_Shoulder_Pitch",  # float32; ∈ [-100.0, 100.0]
    "44: last_action_Right_Elbow_Yaw",        # float32; ∈ [-100.0, 100.0]
    "45: last_action_Left_Hip_Pitch",         # float32; ∈ [-100.0, 100.0]
    "46: last_action_Left_Hip_Roll",          # float32; ∈ [-100.0, 100.0]
    "47: last_action_Left_Hip_Yaw",           # float32; ∈ [-100.0, 100.0]
    "48: last_action_Left_Knee_Pitch",        # float32; ∈ [-100.0, 100.0]
    "49: last_action_Left_Ankle_Pitch",     # float32; ∈ [-100.0, 100.0]
    "50: last_action_Left_Ankle_Roll",      # float32; ∈ [-100.0, 100.0]
    "51: last_action_Right_Hip_Pitch",        # float32; ∈ [-100.0, 100.0]
    "52: last_action_Right_Hip_Roll",         # float32; ∈ [-100.0, 100.0]
    "53: last_action_Right_Hip_Yaw",          # float32; ∈ [-100.0, 100.0]
    "54: last_action_Right_Knee_Pitch",       # float32; ∈ [-100.0, 100.0]
    "55: last_action_Right_Ankle_Pitch",      # float32; ∈ [-100.0, 100.0]
    "56: last_action_Right_Ankle_Roll",       # float32; ∈ [-100.0, 100.0]
    # ── gait phase clock (φ ∈ [0,1), Periode 0.80 s = 40 Policy-Schritte) ────
    "57: gait_phase_sin",  # float32; sin(2πφ) ∈ [-1.0, 1.0]
    "58: gait_phase_cos",  # float32; cos(2πφ) ∈ [-1.0, 1.0]
]
def make_dummy_student_obs(
    *,
    batch_size: int = 1,
    mode: str = "nominal",          # "nominal" | "random"
    vx: float = 0.3,                # Roh [m/s]
    vy: float = 0.0,
    wz: float = 0.0,                # Roh [rad/s]
    gait_phase: float = 0.0,        # φ ∈ [0, 1)
    roll: float = 0.0,              # [rad], nur für nominal/random gravity
    pitch: float = 0.0,
    seed: int | None = None,
    device: str = "cpu",
) -> torch.Tensor:

    if len(STUDENT_OBS_LABELS) != OBS_DIM:
        raise ValueError(f"STUDENT_OBS_LABELS length {len(STUDENT_OBS_LABELS)} != OBS_DIM {OBS_DIM}")

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    def _grav_from_rpy(roll_: float, pitch_: float) -> list[float]:
        cr, sr = math.cos(roll_), math.sin(roll_)
        cp, sp = math.cos(pitch_), math.sin(pitch_)
        return [sp, -sr * cp, -cr * cp]

    if mode == "nominal":
        ang_vel = [0.0, 0.0, 0.0]
        grav = _grav_from_rpy(roll, pitch)
        cmd = [
            vx * OBS_SCALE_LIN_VEL,
            vy * OBS_SCALE_LIN_VEL,
            wz * OBS_SCALE_ANG_VEL,
        ]
        dof_pos = [0.0] * 16          # exakt Default-Pose
        dof_vel = [0.0] * 16
        last_action = [0.0] * 16
    elif mode == "random":
        ang_vel = (torch.rand(3, generator=gen) * 2.0 - 1.0).mul(2.5).tolist()   # Obs ~[-2.5, 2.5]
        grav = _grav_from_rpy(
            (torch.rand((), generator=gen).item() * 2.0 - 1.0) * 0.3,
            (torch.rand((), generator=gen).item() * 2.0 - 1.0) * 0.3,
        )
        cmd = [
            (torch.rand((), generator=gen).item() * 1.5) * OBS_SCALE_LIN_VEL,   # vx ~[0, 1.5] m/s
            (torch.rand((), generator=gen).item() * 0.8 - 0.4) * OBS_SCALE_LIN_VEL,
            (torch.rand((), generator=gen).item() * 1.0 - 0.5) * OBS_SCALE_ANG_VEL,
        ]
        dof_pos = (torch.rand(16, generator=gen) * 0.4 - 0.2).tolist()          # kleine Abweichung
        dof_vel = (torch.rand(16, generator=gen) * 2.0 - 1.0).mul(0.5).tolist() # Obs ~[-0.5, 0.5]
        last_action = (torch.rand(16, generator=gen) * 2.0 - 1.0).mul(5.0).tolist()
    else:
        raise ValueError(f"unknown mode: {mode!r} (use 'nominal' or 'random')")

    phase_2pi = gait_phase * 2.0 * math.pi
    clock = [math.sin(phase_2pi), math.cos(phase_2pi)]

    flat = ang_vel + grav + cmd + dof_pos + dof_vel + last_action + clock
    assert len(flat) == OBS_DIM, f"dummy obs dim {len(flat)} != {OBS_DIM}"

    obs = torch.tensor(flat, dtype=torch.float32, device=device).unsqueeze(0)
    return obs.expand(batch_size, -1).contiguous()