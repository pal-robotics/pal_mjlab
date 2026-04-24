import numpy as np
import sys
import torch
import yaml
import os

# Script to extract the first pose from an MJLab NPZ motion file and generate a PAL trajectory YAML
# Usage: python npz_starting_pose.py <path_to_npz>

if len(sys.argv) < 2:
    print("Usage: python npz_starting_pose.py <path_to_motion_file>")
    sys.exit(1)

path = sys.argv[1]

# Load data based on file extension
if path.endswith(".npz"):
    data = np.load(path)
    # Extract first frame
    joint_pos_all = data['joint_pos'][0]
elif path.endswith(".csv"):
    data = np.loadtxt(path, delimiter=",")
    # CSV format: root_pos(3), root_quat(4), joint_pos(N)
    # So joint_pos starts from index 7
    if data.ndim == 1:
        joint_pos_all = data[7:]
    else:
        joint_pos_all = data[0, 7:]
else:
    print(f"ERROR: Unsupported file format '{path}'. Use .npz or .csv")
    sys.exit(1)
JOINT_NAMES_ALL = [
    "pelvis_1_joint", "pelvis_2_joint",
    "arm_left_1_joint", "arm_left_2_joint", "arm_left_3_joint", "arm_left_4_joint",
    "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint", "arm_right_4_joint",
    "leg_left_1_joint", "leg_left_2_joint", "leg_left_3_joint", "leg_left_length_joint", 
    "leg_left_4_joint", "leg_left_5_joint", "leg_left_femur_joint", "leg_left_knee_joint",
    "leg_right_1_joint", "leg_right_2_joint", "leg_right_3_joint", "leg_right_length_joint", 
    "leg_right_4_joint", "leg_right_5_joint", "leg_right_femur_joint", "leg_right_knee_joint"
]

# Identify actuated joints (exclude closed-chain passive joints)
ACTUATED_NAMES = [
    name for name in JOINT_NAMES_ALL 
    if "femur" not in name and "knee" not in name
]

print(f"Motion: {path}")
print("-" * 40)


# Map values to actuated joints
actuated_values = []
print("INITIAL ACTUATED POSE:")
for name in ACTUATED_NAMES:
    idx = JOINT_NAMES_ALL.index(name)
    val = float(joint_pos_all[idx])
    actuated_values.append(val)
    print(f"  {name:25}: {val:8.4f}")

# Generate YAML in PAL play_motion format
yaml_data = {
    "play_motion": {
        "motions": {
            "goto_initial_pose": {
                "joints": ACTUATED_NAMES,
                "points": [
                    {
                        "positions": actuated_values,
                        "time_from_start": 5.0
                    }
                ]
            }
        }
    }
}

output_yaml = "goto_initial_pose.yaml"
with open(output_yaml, "w") as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

print("-" * 40)
print(f"SUCCESS: Trajectory YAML saved to '{output_yaml}'")
print("This trajectory will take 5 seconds to align the robot to the MJLab start pose.")