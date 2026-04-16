import onnx

# Point this to your actual ONNX model path
model_path = "/home/lorenzobarbieri/policies/louis_baseline.onnx"
model = onnx.load(model_path)

joint_names = []
default_pos = []

# Parse the metadata
for prop in model.metadata_props:
    if prop.key == "joint_names":
        joint_names = prop.value.split(',')
    elif prop.key == "default_joint_pos":
        default_pos = [float(x) for x in prop.value.split(',')]

# Print them out in a nice YAML format for ROS
print("--- INITIAL JOINT POSITIONS ---")
for name, pos in zip(joint_names, default_pos):
    print(f"{name}: {pos}")