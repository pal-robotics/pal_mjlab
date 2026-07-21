#!/usr/bin/env python3
"""
Script to manually export a trained RL policy checkpoint (.pt) to ONNX (.onnx),
attaching the required metadata for ROS 2 deployment.
"""

import argparse
import os
import sys
from dataclasses import asdict

import torch

# Import mjlab and task-specific modules
try:
  import mjlab.tasks
  import pal_mjlab.tasks
except ImportError:
  print("Error: Could not import mjlab or pal_mjlab. Make sure you are running in the correct environment (e.g. uv run).")
  sys.exit(1)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata, attach_metadata_to_onnx
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
  parser = argparse.ArgumentParser(description="Export a .pt checkpoint to .onnx with metadata")
  parser.add_argument(
      "--task",
      type=str,
      required=True,
      help="Task name (e.g., Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0)"
  )
  parser.add_argument(
      "--checkpoint",
      type=str,
      required=True,
      help="Path to the .pt checkpoint file"
  )
  parser.add_argument(
      "--output",
      type=str,
      default=None,
      help="Optional custom output path for the .onnx file (defaults to same folder/name as checkpoint)"
  )
  parser.add_argument(
      "--device",
      type=str,
      default="cpu",
      help="Device to load the model on (default: cpu)"
  )
  args = parser.parse_args()

  # Check if checkpoint exists
  if not os.path.exists(args.checkpoint):
    print(f"Error: Checkpoint path '{args.checkpoint}' does not exist.")
    sys.exit(1)

  # Load configurations
  print(f"Loading configurations for task: {args.task}...")
  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)

  # Only need 1 environment for exporting
  env_cfg.scene.num_envs = 1

  # Initialize environment
  print("Initializing environment...")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Load runner
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device=args.device)

  print(f"Loading checkpoint weights from {args.checkpoint}...")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)

  # Determine output paths
  if args.output is None:
    output_path = os.path.splitext(args.checkpoint)[0] + ".onnx"
  else:
    output_path = args.output

  export_dir = os.path.dirname(os.path.abspath(output_path))
  filename = os.path.basename(output_path)

  print(f"Exporting model to {output_path}...")
  runner.export_policy_to_onnx(export_dir, filename)

  # Extract and attach metadata
  print("Extracting metadata...")
  # get_base_metadata may have been patched by pal_mjlab.tasks.__init__
  metadata = get_base_metadata(env, "manual_export")
  
  print("Attaching metadata to ONNX model...")
  attach_metadata_to_onnx(output_path, metadata)

  print(f"Successfully exported policy to {output_path}!")
  env.close()


if __name__ == "__main__":
  main()
