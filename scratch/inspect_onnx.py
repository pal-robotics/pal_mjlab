import onnx
import sys

model_path = "/home/lorenzobarbieri/docker_mounts/ferrum/catkin_ws/src/pal_policy_deployer/pal_policy_deployer/models/imitation/imitation_v12_mjlab/new_tutto_nohistory.onnx"
model = onnx.load(model_path)
meta = model.metadata_props

print("=== Metadata ===")
for prop in meta:
    print(f"{prop.key}: {prop.value}")

print("\n=== Inputs ===")
for input in model.graph.input:
    print(f"Input: {input.name}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
