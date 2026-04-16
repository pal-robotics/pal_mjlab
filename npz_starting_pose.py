import numpy as np

# Load the file
data = np.load('/home/lorenzobarbieri/Downloads/motion(1).npz')
# 1. Internal Joint Angles
initial_joints = data['joint_pos'][0]

# 2. Base Position (usually the first entry in body_pos_w if it's the root)
# Note: Check the shape. If it's [frames, num_bodies, 3], 
# the root is usually index 0.
base_pos = data['body_pos_w'][0, 0] 

# 3. Base Orientation (Quaternion)
base_quat = data['body_quat_w'][0, 0]

print(f"--- Starting Configuration ---")
print(f"Base Position: {base_pos}")
print(f"Base Quat:     {base_quat}")
print(f"Joint Angles:  {initial_joints}")

import mujoco
import numpy as np

# Change this path to the actual MJCF/XML file used by your task
# It is usually located in your mjlab assets folder
model_path = "/home/lorenzobarbieri/pal_mjlab/src/pal_mjlab/robots/pal_kangaroo/xmls/kangaroo.xml" 

try:
    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"{'Index':<8} | {'Joint Name':<25}")
    print("-" * 35)
    
    # joint_names starts from the floating base usually, 
    # but your npz 'joint_pos' only has the 26 controllable joints.
    # We skip the "freejoint" (the first 7 qpos slots)
    
    # In MuJoCo, model.jnt_qposadr gives the starting address of each joint
    # We want the names of the joints that correspond to the 26 positions.
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    
    # Most mjlab robots have a 'root' or 'freejoint' at index 0.
    # If joint_pos has 26, it's likely everything AFTER the freejoint.
    controllable_joints = [name for name in joint_names if name != 'root' and name != 'freejoint']

    for i, name in enumerate(controllable_joints):
        print(f"{i:<8} | {name:<25}")

except Exception as e:
    print(f"Error loading model: {e}")