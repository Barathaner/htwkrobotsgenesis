# load model inference

#create mujoco scene
# load robot urdf in mujoco
# get robot data observations
# transform everything into model observation
# send to model
# get actions
# send to robot
# kill switches
# add joint limit termination
import os

import mujoco as mj
import mujoco.viewer
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(REPO_ROOT, "models", "K1", "K1_22dof.urdf")

model = mj.MjModel.from_xml_path(URDF_PATH)
data = mj.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.geomgroup[0] = 0  # hide collision geoms (group 0), show visual meshes (group 1)
    while viewer.is_running():
        mj.mj_step(model, data)
        viewer.sync()
