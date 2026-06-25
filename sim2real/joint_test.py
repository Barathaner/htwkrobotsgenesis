"""Manual single-joint test rig: drive every K1 joint by hand, mirrored LIVE in Genesis + MuJoCo.

The robot is held FIXED in the air (base welded to the world) in both simulators, so it can't fall —
you just watch how each joint moves. The exact same position target is written to Genesis and to
MuJoCo every control tick, so the two viewers move in lock-step.

Controls (ARROW KEYS ONLY). Keys are picked up from ANY of three places — whichever has focus:
    • the MuJoCo viewer window   (most reliable — just click it)
    • the Genesis viewer window
    • this terminal
    Left  / Right : select previous / next joint
    Up    / Down  : increase / decrease the selected joint's target angle
    r             : reset all joints to the default pose
    0             : zero the selected joint   (terminal / MuJoCo window only)
    q  (or Ctrl-C): quit

Run it from the sim2real/ directory:  python joint_test.py
"""

import collections
import os
import select
import sys
import termios
import time
import tty

import mujoco as mj
import mujoco.viewer
import numpy as np
import torch
import yaml

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

from mujoco_calibration import apply_mujoco_calibration, enable_self_collision

# Shared event queue. The viewer key-callbacks (which fire on the viewer's own thread) push events
# here; the main loop drains it alongside the terminal reader, so every input surface feeds one path.
EVENT_QUEUE: "collections.deque[str]" = collections.deque()

# GLFW key codes delivered to the MuJoCo passive-viewer key_callback (same codes as deploy_mujoco.py).
_GLFW_EVENT = {
    265: "up", 264: "down", 263: "right", 262: "left",
    82: "reset", 48: "zero", 81: "quit", 256: "quit",  # R, '0', Q, Esc
}


def mj_key_callback(keycode):
    ev = _GLFW_EVENT.get(keycode)
    if ev:
        EVENT_QUEUE.append(ev)

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "deploy.yaml"), "r") as f:
    config = yaml.safe_load(f)
p = config["policy"]

# Canonical joint order (all 22 DOFs), shared by both sims and by the keyboard selector.
JOINT_NAMES = p["urdf_joint_names"]
DEFAULTS = p["default_joint_angles"]

STEP_RAD = 0.05          # angle change per Up/Down press [rad]
CONTROL_DT = 0.02        # control/loop period [s]
BASE_POS = (0.0, 0.0, 1.0)   # where the fixed base hangs (matches K1_22dof.xml Trunk pos)


# ---------------------------------------------------------------------------------------------------
# Non-blocking terminal arrow-key reader (one of three input surfaces; the viewer windows are the
# others). Uses raw os.read on the fd — NOT buffered sys.stdin.read — because mixing select() with
# Python's buffered text stream is the classic reason "the terminal reads nothing". If stdin isn't a
# real TTY (e.g. launched from an IDE "Run" button), it disables itself with a clear message rather
# than silently doing nothing — use a viewer window in that case.
# ---------------------------------------------------------------------------------------------------
class KeyReader:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        if not self.enabled:
            print("[keys] stdin is not a TTY → terminal keys OFF. Click the MuJoCo (or Genesis) "
                  "window and press the arrow keys there, or run from a real terminal.")
            return
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def poll(self):
        """Return a list of high-level events: 'up','down','left','right','reset','zero','quit'."""
        if not self.enabled:
            return []
        raw = b""
        while select.select([self.fd], [], [], 0)[0]:
            raw += os.read(self.fd, 32)
        s = raw.decode("utf-8", "ignore")
        events, i = [], 0
        while i < len(s):
            c = s[i]
            if c == "\x1b" and s[i + 1:i + 2] == "[":
                mapped = {"A": "up", "B": "down", "C": "right", "D": "left"}.get(s[i + 2:i + 3])
                if mapped:
                    events.append(mapped)
                i += 3
            elif c in ("q", "\x03"):     # q or Ctrl-C
                events.append("quit"); i += 1
            elif c == "r":
                events.append("reset"); i += 1
            elif c == "0":
                events.append("zero"); i += 1
            else:
                i += 1
        return events

    def restore(self):
        if self.enabled:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def build_genesis():
    """Genesis scene with PD position control. The URDF morph's `fixed=True` is ignored for this
    humanoid (Genesis still builds a 6-dof floating base), so we instead pin the base every step in
    the main loop — same idea as the MuJoCo side."""
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -1.8, 1.6), camera_lookat=(0.0, 0.0, 1.0), camera_fov=30,
        ),
        sim_options=gs.options.SimOptions(dt=CONTROL_DT, substeps=2),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file=os.path.abspath(p["urdf_path"]), pos=BASE_POS),
        surface=gs.surfaces.Default(color=(0.7, 0.85, 1.0), opacity=0.9),
    )
    scene.build()

    dofs_idx = [robot.get_joint(n).dofs_idx_local[0] for n in JOINT_NAMES]
    # The base free joint owns whatever dofs the joints don't (the leading 6: pos + rot).
    base_dofs = sorted(set(range(robot.n_dofs)) - set(dofs_idx))
    kp = [p["joint_gains"][n]["kp"] for n in JOINT_NAMES]
    kd = [p["joint_gains"][n]["kd"] for n in JOINT_NAMES]
    effort = [p["joint_gains"][n]["effort"] for n in JOINT_NAMES]
    robot.set_dofs_kp(kp, dofs_idx)
    robot.set_dofs_kv(kd, dofs_idx)
    robot.set_dofs_force_range([-e for e in effort], effort, dofs_idx)
    return scene, robot, dofs_idx, base_dofs


def build_mujoco():
    """MuJoCo model with the free base joint pinned each step so the robot stays fixed in the air.
    Load/calibration matches deploy_mujoco.py (apply_mujoco_calibration + enable_self_collision)."""
    model = mj.MjModel.from_xml_path(os.path.abspath(p["mujoco_path"]))
    calib = p.get("mujoco_calib", "deploy_mujoco_calib.yaml")
    if not os.path.isabs(calib):
        calib = os.path.join(HERE, calib)
    apply_mujoco_calibration(model, calib)
    enable_self_collision(model)
    data = mj.MjData(model)
    aids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES]
    # Free "world_joint" occupies qpos[0:7] (pos+quat) and qvel[0:6]; we reset it every step.
    base_qpos = np.array([*BASE_POS, 1.0, 0.0, 0.0, 0.0])
    # Per-joint range straight from the model, used to clamp targets for both sims.
    ranges = {}
    for n in JOINT_NAMES:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n)
        lo, hi = model.jnt_range[jid]
        ranges[n] = (float(lo), float(hi))
    decimation = max(1, round(CONTROL_DT / model.opt.timestep))
    return model, data, aids, base_qpos, ranges, decimation


def main():
    scene, robot, gen_dofs, gen_base_dofs = build_genesis()
    model, data, mj_aids, base_qpos, ranges, decimation = build_mujoco()

    gen_base_pos = torch.tensor(BASE_POS, dtype=gs.tc_float, device=gs.device)
    gen_base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
    gen_base_zerovel = torch.zeros(len(gen_base_dofs), dtype=gs.tc_float, device=gs.device)

    # Genesis viewer keybinds → same shared queue (so keys also work when the Genesis window is focused).
    # overwrite=False keeps the viewer's built-in camera controls; arrows aren't used by the camera.
    if scene.viewer is not None:
        try:
            scene.viewer.register_keybinds(
                Keybind("jt_up",    Key.UP,    KeyAction.PRESS, callback=EVENT_QUEUE.append, args=("up",)),
                Keybind("jt_down",  Key.DOWN,  KeyAction.PRESS, callback=EVENT_QUEUE.append, args=("down",)),
                Keybind("jt_left",  Key.LEFT,  KeyAction.PRESS, callback=EVENT_QUEUE.append, args=("left",)),
                Keybind("jt_right", Key.RIGHT, KeyAction.PRESS, callback=EVENT_QUEUE.append, args=("right",)),
                Keybind("jt_reset", Key.R,     KeyAction.PRESS, callback=EVENT_QUEUE.append, args=("reset",)),
                Keybind("jt_quit",  Key.Q,     KeyAction.PRESS, callback=EVENT_QUEUE.append, args=("quit",)),
            )
        except Exception as e:  # don't let a keybind API mismatch kill the whole tool
            print(f"[keys] Genesis keybinds unavailable ({e}); use the MuJoCo window or the terminal.")

    # Live target angle per joint, both sims share it. Start at the default pose.
    targets = {n: float(DEFAULTS[n]) for n in JOINT_NAMES}
    sel = 0  # index into JOINT_NAMES of the currently selected joint

    def clamp(name, val):
        lo, hi = ranges[name]
        return max(lo, min(hi, val))

    def status():
        n = JOINT_NAMES[sel]
        lo, hi = ranges[n]
        print(f"\r[{sel + 1:2d}/{len(JOINT_NAMES)}] {n:<22}  target={targets[n]:+.3f} rad  "
              f"range=[{lo:+.2f},{hi:+.2f}]        ", end="", flush=True)

    keys = KeyReader()
    print(__doc__)
    print("Joints:", ", ".join(f"{i+1}:{n}" for i, n in enumerate(JOINT_NAMES)))
    status()

    # Genesis target tensor, rebuilt each step from `targets` in gen_dofs order.
    gen_target = torch.zeros(len(JOINT_NAMES), dtype=gs.tc_float, device=gs.device)

    def drain_events():
        """All input surfaces in one list: terminal reader + viewer key-callback queue."""
        evs = keys.poll()
        while EVENT_QUEUE:
            evs.append(EVENT_QUEUE.popleft())
        return evs

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=mj_key_callback) as viewer:
            t_next = time.perf_counter()
            running = True
            while running and viewer.is_running():
                for ev in drain_events():
                    n = JOINT_NAMES[sel]
                    if ev == "quit":
                        running = False
                    elif ev == "left":
                        sel = (sel - 1) % len(JOINT_NAMES); status()
                    elif ev == "right":
                        sel = (sel + 1) % len(JOINT_NAMES); status()
                    elif ev == "up":
                        targets[n] = clamp(n, targets[n] + STEP_RAD); status()
                    elif ev == "down":
                        targets[n] = clamp(n, targets[n] - STEP_RAD); status()
                    elif ev == "zero":
                        targets[n] = clamp(n, 0.0); status()
                    elif ev == "reset":
                        for m in JOINT_NAMES:
                            targets[m] = float(DEFAULTS[m])
                        status()

                # --- write the SAME targets to both sims ---
                # Genesis: one control_dofs_position per joint (single dof → no ordering ambiguity).
                for i, name in enumerate(JOINT_NAMES):
                    gen_target[i] = targets[name]
                    robot.control_dofs_position(gen_target[i:i + 1], [gen_dofs[i]])
                # Pin the floating base so the robot hangs fixed instead of falling.
                robot.set_dofs_velocity(gen_base_zerovel, gen_base_dofs)
                robot.set_pos(gen_base_pos)
                robot.set_quat(gen_base_quat)
                scene.step()

                # MuJoCo: position actuator target = joint target; pin the free base; step physics.
                for aid, name in zip(mj_aids, JOINT_NAMES):
                    data.ctrl[aid] = targets[name]
                for _ in range(decimation):
                    data.qpos[0:7] = base_qpos
                    data.qvel[0:6] = 0.0
                    mj.mj_step(model, data)
                viewer.sync()

                t_next += CONTROL_DT
                sleep_s = t_next - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        keys.restore()
        print("\nbye.")


if __name__ == "__main__":
    main()
