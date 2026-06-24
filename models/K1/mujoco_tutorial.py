"""
MuJoCo tutorial for the Booster K1 (22 DoF).

What this shows, step by step:
  1. Load the robot model + create simulation data
  2. Inspect the model: joints, actuators, the qpos/qvel layout
  3. Open the interactive viewer
  4. MOVE the robot: drive the joints with a simple PD position controller
  5. READ the joints: print position / velocity / applied torque each step

Run it (from the repo root):
    ./rltrainingvenv/bin/python models/K1/mujoco_tutorial.py
    ./rltrainingvenv/bin/python models/K1/mujoco_tutorial.py --gravity   # let it fall

Controls in the viewer: drag to orbit, scroll to zoom, double-click a body to select,
Ctrl+drag a selected body to push it. Press Esc / close the window to quit.
"""

import argparse
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

# This file lives next to K1_22dof.xml, and the XML references meshes via meshdir="meshes/",
# so loading by this path makes the mesh paths resolve correctly.
import os
HERE = os.path.dirname(__file__)
ROBOT_PATH = os.path.join(HERE, "K1_22dof.xml")  # robot only
SCENE_PATH = os.path.join(HERE, "scene.xml")     # robot + camera + visuals (includes the robot)


def main(use_gravity: bool, use_scene: bool) -> None:
    XML_PATH = SCENE_PATH if use_scene else ROBOT_PATH
    # ----------------------------------------------------------------------------------
    # 1. LOAD THE MODEL
    # ----------------------------------------------------------------------------------
    # MjModel  = the *static* description (bodies, joints, actuators, meshes...). Read-only.
    # MjData   = the *dynamic* state (qpos, qvel, ctrl, sensor readings...). Changes every step.
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)

    if not use_gravity:
        # Float the robot so you clearly see the commanded joint motion instead of it
        # collapsing under its own weight. Remove this to study real dynamics.
        model.opt.gravity[:] = 0.0

    # ----------------------------------------------------------------------------------
    # 2. INSPECT THE MODEL
    # ----------------------------------------------------------------------------------
    # qpos layout: [free joint: x y z qw qx qy qz] then one scalar per hinge joint.
    # qvel layout: [free joint: vx vy vz wx wy wz] then one scalar per hinge joint.
    # So a hinge joint's value lives at qpos[jnt_qposadr] and qvel[jnt_dofadr].
    print(f"model: {model.njnt} joints, {model.nu} actuators, "
          f"nq={model.nq} (qpos size), nv={model.nv} (qvel size)\n")

    # Collect the actuated (hinge) joints, in actuator order. Each <motor> drives one joint.
    actuated = []
    for a in range(model.nu):
        act_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, a)
        jnt_id = model.actuator_trnid[a, 0]          # joint this actuator drives
        jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        qadr = model.jnt_qposadr[jnt_id]             # index into data.qpos
        vadr = model.jnt_dofadr[jnt_id]              # index into data.qvel
        lo, hi = model.jnt_range[jnt_id]             # joint limits (radians)
        actuated.append((act_name, jnt_id, qadr, vadr, lo, hi))
        print(f"  act[{a:2d}] {act_name:24s} joint={jnt_name:22s} "
              f"qpos[{qadr:2d}] qvel[{vadr:2d}] range=[{lo:+.2f},{hi:+.2f}]")
    print()

    qadr = np.array([j[2] for j in actuated])  # qpos indices of the 22 motors
    vadr = np.array([j[3] for j in actuated])  # qvel indices of the 22 motors

    # ----------------------------------------------------------------------------------
    # 3. SET AN INITIAL POSE
    # ----------------------------------------------------------------------------------
    # mj_resetData zeroes everything, then key the floating base above the ground.
    mj.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 1.0]      # base position x, y, z
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] # base orientation quaternion (w, x, y, z)
    mj.mj_forward(model, data)            # recompute derived quantities for the new state

    # Hold every joint at 0 rad; we'll wiggle the elbows on top of that.
    target = np.zeros(model.nu)
    kp, kd = 40.0, 2.0                    # PD gains: stiffness, damping

    # Find the two elbow actuators by name so we can drive just those.
    elbow_ids = [i for i, j in enumerate(actuated) if "Elbow_Pitch" in j[0]]

    # ----------------------------------------------------------------------------------
    # 4 + 5. RUN THE LOOP: move (PD control) and read joints
    # ----------------------------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[0] = 0  # hide collision geoms (group 0), show visual meshes (group 1)
        start = time.time()
        last_print = 0.0

        while viewer.is_running():
            t = data.time

            # --- MOVE: build a per-step joint target, then a PD torque toward it ---
            cmd = target.copy()
            cmd[elbow_ids] = 1.0 * np.sin(2.0 * np.pi * 0.5 * t)  # 0.5 Hz elbow swing

            q = data.qpos[qadr]   # current actuated joint positions
            dq = data.qvel[vadr]  # current actuated joint velocities
            data.ctrl[:] = kp * (cmd - q) - kd * dq  # torque command -> the <motor> actuators

            # --- STEP the physics one timestep (model.opt.timestep = 0.001 s here) ---
            mj.mj_step(model, data)

            # --- READ: print joint state a few times a second ---
            if t - last_print > 0.5:
                last_print = t
                # data.sensordata holds the IMU readings declared in the XML <sensor> block.
                print(f"t={t:5.2f}s  L_elbow: pos={q[elbow_ids[0]]:+.3f} rad "
                      f"vel={dq[elbow_ids[0]]:+.3f} rad/s  "
                      f"torque={data.ctrl[elbow_ids[0]]:+.2f} Nm")

            viewer.sync()

            # Keep wall-clock roughly in sync with sim time so it plays at real speed.
            dt = model.opt.timestep
            ahead = (data.time) - (time.time() - start)
            if ahead > 0:
                time.sleep(ahead)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gravity", action="store_true",
                   help="enable gravity (robot will fall over without a balance controller)")
    p.add_argument("--scene", action="store_true",
                   help="load models/K1/scene.xml (adds tracking cameras + visuals)")
    args = p.parse_args()
    main(use_gravity=args.gravity, use_scene=args.scene)
