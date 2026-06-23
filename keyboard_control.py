"""
Keyboard-driven joint-level debug controller for Booster K1.

Every physical joint (all 22 K1Ji slots) is listed in a curses TUI.
Navigate with UP/DOWN, nudge the selected joint with + / -.

Usage (on the robot or external PC):
  python walk/deploy/joint_debug_keyboard.py [--interface 127.0.0.1]

Keys:
  ↑ / ↓        navigate joint list
  PgUp / PgDn  jump 5 joints
  + or =       increase selected joint target by step
  - or _       decrease selected joint target by step
  [ / ]        halve / double step size
  r            reset ALL joints to defaults
  q or ESC     quit — sends damp + kDamping mode
"""

from __future__ import annotations

import argparse
import curses
import math
import sys
import time

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

# ── K1 constants (mirrors smoke_test_k1.py) ──────────────────────────────────

DT = 0.02  # 50 Hz

K1_JOINT_CNT = 22


class K1Ji:
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


K1_CRANK_INDICES = {
    K1Ji.kCrankUpLeft, K1Ji.kCrankDownLeft,
    K1Ji.kCrankUpRight, K1Ji.kCrankDownRight,
}

JOINT_NAMES = [
    "HeadYaw",          # 0
    "HeadPitch",        # 1
    "L_ShoulderPitch",  # 2
    "L_ShoulderRoll",   # 3
    "L_ElbowPitch",     # 4
    "L_ElbowYaw",       # 5
    "R_ShoulderPitch",  # 6
    "R_ShoulderRoll",   # 7
    "R_ElbowPitch",     # 8
    "R_ElbowYaw",       # 9
    "L_HipPitch",       # 10
    "L_HipRoll",        # 11
    "L_HipYaw",         # 12
    "L_KneePitch",      # 13
    "L_CrankUp",        # 14  torque-controlled
    "L_CrankDown",      # 15  torque-controlled
    "R_HipPitch",       # 16
    "R_HipRoll",        # 17
    "R_HipYaw",         # 18
    "R_KneePitch",      # 19
    "R_CrankUp",        # 20  torque-controlled
    "R_CrankDown",      # 21  torque-controlled
]

# Default targets for all 22 joints.
# Crank defaults = ankle_to_crank(ankle_pitch=-0.30, ankle_roll=0.0):
#   crank_up = -0.30 + 0.0 = -0.30,  crank_down = -0.30 - 0.0 = -0.30
DEFAULT_POS: list[float] = [
     0.0,   # 0  HeadYaw
     0.2,   # 1  HeadPitch
     0.0,   # 2  L_ShoulderPitch
    -1.5,   # 3  L_ShoulderRoll
     0.1,   # 4  L_ElbowPitch
    -1.0,   # 5  L_ElbowYaw
     0.0,   # 6  R_ShoulderPitch
     1.5,   # 7  R_ShoulderRoll
     0.1,   # 8  R_ElbowPitch
     1.0,   # 9  R_ElbowYaw
    -0.30,  # 10 L_HipPitch
     0.0,   # 11 L_HipRoll
     0.0,   # 12 L_HipYaw
     0.60,  # 13 L_KneePitch
    -0.30,  # 14 L_CrankUp
    -0.30,  # 15 L_CrankDown
    -0.30,  # 16 R_HipPitch
     0.0,   # 17 R_HipRoll
     0.0,   # 18 R_HipYaw
     0.60,  # 19 R_KneePitch
    -0.30,  # 20 R_CrankUp
    -0.30,  # 21 R_CrankDown
]

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
    K1Ji.kCrankUpLeft:        (  0.0, 1.0,  20.0),
    K1Ji.kCrankDownLeft:      (  0.0, 1.0,  15.0),
    K1Ji.kRightHipPitch:      (200.0, 5.0,  68.0),
    K1Ji.kRightHipRoll:       (200.0, 5.0,  76.0),
    K1Ji.kRightHipYaw:        (150.0, 4.0,  38.3),
    K1Ji.kRightKneePitch:     (250.0, 6.0, 112.0),
    K1Ji.kCrankUpRight:       (  0.0, 1.0,  20.0),
    K1Ji.kCrankDownRight:     (  0.0, 1.0,  15.0),
}

CRANK_STIFFNESS: dict[int, float] = {
    K1Ji.kCrankUpLeft:    25.0,
    K1Ji.kCrankDownLeft:  25.0,
    K1Ji.kCrankUpRight:   25.0,
    K1Ji.kCrankDownRight: 25.0,
}


# ── Robot state ───────────────────────────────────────────────────────────────

class RobotStateBuffer:
    def __init__(self) -> None:
        self._ready = False
        self._q   = [0.0] * K1_JOINT_CNT
        self._dq  = [0.0] * K1_JOINT_CNT
        self._rpy = [0.0, 0.0, 0.0]

    def update(self, state) -> None:
        ms = state.motor_state_serial
        for i in range(min(len(ms), K1_JOINT_CNT)):
            self._q[i]  = ms[i].q
            self._dq[i] = ms[i].dq
        imu = state.imu_state
        self._rpy[0] = imu.rpy[0]; self._rpy[1] = imu.rpy[1]; self._rpy[2] = imu.rpy[2]
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def q(self, idx: int) -> float:
        return self._q[idx]

    @property
    def rpy(self) -> list[float]:
        return list(self._rpy)

    def is_fallen(self, roll_limit: float = 1.0, pitch_limit: float = 1.0) -> bool:
        return abs(self._rpy[0]) > roll_limit or abs(self._rpy[1]) > pitch_limit


# ── SDK command helpers ───────────────────────────────────────────────────────

def alloc_low_cmd() -> LowCmd:
    motor_cmds = [MotorCmd() for _ in range(K1_JOINT_CNT)]
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    low_cmd.motor_cmd = motor_cmds
    for idx in range(K1_JOINT_CNT):
        kp, kd, _ = SDK_JOINT_GAINS.get(idx, (0.0, 0.0, 0.0))
        mc = low_cmd.motor_cmd[idx]
        mc.dq = 0.0; mc.weight = 0.0; mc.kp = kp; mc.kd = kd; mc.tau = 0.0; mc.q = 0.0
    return low_cmd


def update_low_cmd(low_cmd: LowCmd, targets: list[float], state_buf: RobotStateBuffer) -> None:
    for idx in range(K1_JOINT_CNT):
        mc = low_cmd.motor_cmd[idx]
        if idx in K1_CRANK_INDICES:
            cur = state_buf.q(idx)
            tgt = targets[idx]
            _, _, limit = SDK_JOINT_GAINS[idx]
            mc.q   = cur
            mc.tau = max(-limit, min(limit, (tgt - cur) * CRANK_STIFFNESS[idx]))
        else:
            mc.q = targets[idx]


def damp_cmd() -> LowCmd:
    motor_cmds = [MotorCmd() for _ in range(K1_JOINT_CNT)]
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    low_cmd.motor_cmd = motor_cmds
    for i in range(K1_JOINT_CNT):
        low_cmd.motor_cmd[i].q = 0.0; low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].tau = 0.0; low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 2.0; low_cmd.motor_cmd[i].weight = 0.0
    return low_cmd


# ── Curses TUI ────────────────────────────────────────────────────────────────

HEADER_LINES = 6   # title + IMU + step/status + separator + column header + separator
FOOTER_LINES = 2   # separator + help line


def _draw(
    stdscr,
    state_buf: RobotStateBuffer,
    targets: list[float],
    selected: int,
    scroll: int,
    step: float,
    status: str,
) -> None:
    h, w = stdscr.getmaxyx()
    stdscr.erase()
    rpy = state_buf.rpy

    # ── header ────────────────────────────────────────────────────────────────
    stdscr.addstr(0, 0, "K1 Joint Debug Controller", curses.A_BOLD)
    stdscr.addstr(
        1, 0,
        f"IMU  roll={math.degrees(rpy[0]):+6.1f}°  "
        f"pitch={math.degrees(rpy[1]):+6.1f}°  "
        f"yaw={math.degrees(rpy[2]):+6.1f}°",
    )
    stdscr.addstr(2, 0, f"Step: {step:.4f} rad    {status}")
    stdscr.addstr(3, 0, "─" * min(w - 1, 72))
    stdscr.addstr(4, 0, f"  {'#':>2}  {'Joint':<18}  {'Target':>10}  {'Current':>10}  {'Δ':>8}  Type")
    stdscr.addstr(5, 0, "─" * min(w - 1, 72))

    # ── joint rows (scrollable viewport) ─────────────────────────────────────
    viewport = h - HEADER_LINES - FOOTER_LINES
    for row_idx in range(viewport):
        joint_idx = scroll + row_idx
        if joint_idx >= K1_JOINT_CNT:
            break
        screen_row = HEADER_LINES + row_idx
        cur_q = state_buf.q(joint_idx)
        delta = targets[joint_idx] - cur_q
        is_crank = joint_idx in K1_CRANK_INDICES
        ctrl_type = "CRANK(τ)" if is_crank else "pos"
        line = (
            f"  {joint_idx:>2}  {JOINT_NAMES[joint_idx]:<18}"
            f"  {targets[joint_idx]:>+10.4f}"
            f"  {cur_q:>+10.4f}"
            f"  {delta:>+8.4f}"
            f"  {ctrl_type}"
        )
        attr = curses.color_pair(0)
        if joint_idx == selected:
            attr = curses.color_pair(1) | curses.A_BOLD
        elif is_crank:
            attr = curses.color_pair(2)
        try:
            stdscr.addstr(screen_row, 0, line[: w - 1], attr)
        except curses.error:
            pass

    # ── footer ────────────────────────────────────────────────────────────────
    footer_row = h - FOOTER_LINES
    try:
        stdscr.addstr(footer_row, 0, "─" * min(w - 1, 72))
        stdscr.addstr(
            footer_row + 1, 0,
            "↑↓:select  PgUp/PgDn:jump5  +/-:nudge  [/]:step  r:reset  q/ESC:quit",
        )
    except curses.error:
        pass

    stdscr.refresh()


def _tui(stdscr, args: argparse.Namespace, state_buf: RobotStateBuffer,
         publisher: B1LowCmdPublisher, client: B1LocoClient) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)    # selected
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # crank

    targets  = list(DEFAULT_POS)
    selected = 0
    scroll   = 0
    step     = 0.05
    status   = "Ready — robot in Custom mode"
    low_cmd  = alloc_low_cmd()
    t_next   = time.monotonic()

    while True:
        h, _ = stdscr.getmaxyx()
        viewport = max(1, h - HEADER_LINES - FOOTER_LINES)

        # ── keyboard input ────────────────────────────────────────────────────
        try:
            key = stdscr.getch()
        except Exception:
            key = -1

        if key == curses.KEY_UP:
            selected = (selected - 1) % K1_JOINT_CNT
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % K1_JOINT_CNT
        elif key == curses.KEY_PPAGE:
            selected = max(0, selected - 5)
        elif key == curses.KEY_NPAGE:
            selected = min(K1_JOINT_CNT - 1, selected + 5)
        elif key in (ord('+'), ord('=')):
            targets[selected] += step
            status = f"[{selected}] {JOINT_NAMES[selected]} → {targets[selected]:+.4f} rad"
        elif key in (ord('-'), ord('_')):
            targets[selected] -= step
            status = f"[{selected}] {JOINT_NAMES[selected]} → {targets[selected]:+.4f} rad"
        elif key == ord('['):
            step = max(0.001, step / 2.0)
            status = f"Step size: {step:.4f} rad"
        elif key == ord(']'):
            step = min(1.0, step * 2.0)
            status = f"Step size: {step:.4f} rad"
        elif key == ord('r'):
            targets[:] = list(DEFAULT_POS)
            status = "All joints reset to defaults"
        elif key in (ord('q'), 27):  # q or ESC
            break

        # ── keep selection visible (scroll to follow) ─────────────────────────
        if selected < scroll:
            scroll = selected
        elif selected >= scroll + viewport:
            scroll = selected - viewport + 1

        # ── fall detection ────────────────────────────────────────────────────
        if state_buf.is_fallen():
            rpy = state_buf.rpy
            status = (
                f"FALL DETECTED  roll={math.degrees(rpy[0]):+.1f}°  "
                f"pitch={math.degrees(rpy[1]):+.1f}°"
            )
            _draw(stdscr, state_buf, targets, selected, scroll, step, status)
            break

        # ── send command ──────────────────────────────────────────────────────
        update_low_cmd(low_cmd, targets, state_buf)
        publisher.Write(low_cmd)

        # ── draw ──────────────────────────────────────────────────────────────
        _draw(stdscr, state_buf, targets, selected, scroll, step, status)

        # ── pace to 50 Hz ─────────────────────────────────────────────────────
        t_next += DT
        wait = t_next - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        else:
            t_next = time.monotonic()


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    state_buf = RobotStateBuffer()
    ChannelFactory.Instance().Init(0, args.interface)

    subscriber = B1LowStateSubscriber(state_buf.update)
    subscriber.InitChannel()

    publisher = B1LowCmdPublisher()
    publisher.InitChannel()

    client = B1LocoClient()
    client.Init()

    print("Waiting for first robot state message…")
    t0 = time.monotonic()
    while not state_buf.ready:
        if time.monotonic() - t0 > 5.0:
            print("ERROR: no state received after 5 s. Check --interface.", file=sys.stderr)
            sys.exit(1)
        time.sleep(0.01)

    print("State received. Switching to Custom mode…")
    client.ChangeMode(RobotMode.kCustom)
    time.sleep(0.5)

    try:
        curses.wrapper(_tui, args, state_buf, publisher, client)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nSending damp commands…")
        dc = damp_cmd()
        for _ in range(10):
            publisher.Write(dc)
            time.sleep(0.02)
        client.ChangeMode(RobotMode.kDamping)
        print("Switched to Damping mode. Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="K1 joint-level keyboard debug controller")
    parser.add_argument(
        "--interface", default="127.0.0.1",
        help="Network interface or IP for Booster SDK (127.0.0.1 when on the robot)",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
