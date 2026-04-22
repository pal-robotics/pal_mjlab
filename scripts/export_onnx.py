"""Standalone script to export a MJLab policy checkpoint to ONNX."""

import argparse
import os
import torch
from pathlib import Path

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

# Import tasks to populate the registry (this allows us to find Palestinian Kangaroo tasks)
import pal_mjlab.tasks  # noqa: F401


def export_checkpoint(task_id: str, checkpoint_path: str, motion_file: str | None = None, output_path: str | None = None):
    # 1. Load configurations
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id)

    # Apply motion file if provided (required for tracking tasks)
    if motion_file and "motion" in env_cfg.commands:
        print(f"[INFO] Using motion file: {motion_file}")
        env_cfg.commands["motion"].motion_file = motion_file

    if output_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        output_path = checkpoint_dir

    print(f"[INFO] Exporting task: {task_id}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Runner class: {runner_cls.__name__}")

    # 2. Initialize environment (needed for metadata and observation shapes)
    # We use CPU for export to keep it lightweight
    device = "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env)

    # 3. Initialize runner
    # We need to pass the dict version of agent_cfg to standard rsl-rl based runners
    import dataclasses
    agent_dict = dataclasses.asdict(agent_cfg)
    
    runner = runner_cls(env, agent_dict, output_path, device)
    
    # 4. Load weights
    print(f"[INFO] Loading weights...")
    runner.load(checkpoint_path)
    
    # 5. Export to ONNX
    # The filename usually follows the pattern: <run_timestamp>.onnx
    # But we can just use "policy.onnx" for simplicity
    filename = "policy.onnx"
    print(f"[INFO] Exporting to {os.path.join(output_path, filename)}...")
    
    runner.export_policy_to_onnx(output_path, filename=filename, verbose=True)
    
    print(f"[SUCCESS] ONNX export complete.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model checkpoint to ONNX.")
    parser.add_argument("--task", type=str, required=True, help="Task ID (e.g., Mjlab-Tracking-Flat-Pal-Kangaroo-History)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file")
    parser.add_argument("--motion-file", type=str, default=None, help="Path to the .npz motion file (required for tracking tasks)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (defaults to checkpoint directory)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        # Try to find it in logs if not an absolute path
        potential_path = os.path.join("logs", "rsl_rl", "exp1", args.checkpoint)
        if os.path.exists(potential_path):
            args.checkpoint = potential_path
        else:
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
            exit(1)

    export_checkpoint(args.task, args.checkpoint, args.motion_file, args.output)
