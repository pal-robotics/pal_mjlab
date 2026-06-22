#!/usr/bin/env python3
"""
Script to collect and display the current basic info of rewards for registered tasks.
"""

import argparse
import sys
import dataclasses
from typing import Any, List, Dict

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, list_tasks


def format_value(v: Any) -> str:
    """Formats values nicely, formatting floats, dataclasses, lists, slices, and functions."""
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, list):
        return "[" + ", ".join(format_value(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(format_value(x) for x in v) + ")"
    if isinstance(v, slice):
        if v.start is None and v.stop is None and v.step is None:
            return "slice(None)"
        parts = []
        if v.start is not None:
            parts.append(f"start={v.start}")
        if v.stop is not None:
            parts.append(f"stop={v.stop}")
        if v.step is not None:
            parts.append(f"step={v.step}")
        return f"slice({', '.join(parts)})"
    if dataclasses.is_dataclass(v):
        fields = dataclasses.fields(v)
        parts = []
        for f in fields:
            val = getattr(v, f.name)
            # Skip empty or default-looking values to keep print clean
            if val is not None and val != [] and val != () and val != {}:
                # Skip default slices like slice(None, None, None)
                if isinstance(val, slice) and val.start is None and val.stop is None and val.step is None:
                    continue
                parts.append(f"{f.name}={format_value(val)}")
        return f"{v.__class__.__name__}({', '.join(parts)})"
    if hasattr(v, "__name__"):
        return v.__name__
    return str(v)


def collect_rewards(task_name: str, play: bool = False) -> List[Dict[str, Any]]:
    """Loads a task configuration and parses its rewards."""
    try:
        cfg = load_env_cfg(task_name, play=play)
    except Exception as e:
        print(f"Error loading task '{task_name}': {e}", file=sys.stderr)
        return None

    if not hasattr(cfg, "rewards") or not cfg.rewards:
        return []

    rewards_info = []
    for name, reward in cfg.rewards.items():
        weight = reward.weight
        
        # Extract std if present, otherwise default to N/A
        std = reward.params.get("std", "N/A") if reward.params else "N/A"
        if std != "N/A":
            std = format_value(std)
            
        # Collect other params
        other_params = {}
        if reward.params:
            for k, v in reward.params.items():
                if k != "std":
                    other_params[k] = format_value(v)
                    
        rewards_info.append({
            "name": name,
            "weight": weight,
            "std": std,
            "other_params": other_params
        })
    return rewards_info


def display_rewards(task_name: str, rewards: List[Dict[str, Any]], output_format: str = "table") -> str:
    """Formats and returns the rewards information as a string."""
    lines = []
    lines.append(f"### Task: {task_name}")
    
    if not rewards:
        lines.append("No rewards configured or task configuration does not have active rewards.\n")
        return "\n".join(lines)
        
    if output_format == "table":
        lines.append("| Reward Name | Weight | Std | Other Params |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for r in rewards:
            other_params_str = ", ".join([f"{k}: {v}" for k, v in r["other_params"].items()])
            lines.append(f"| {r['name']} | {r['weight']} | {r['std']} | {other_params_str} |")
        lines.append("")
    elif output_format == "csv":
        lines.append("reward_name,weight,std,other_params")
        for r in rewards:
            other_params_str = "; ".join([f"{k}={v}" for k, v in r["other_params"].items()])
            lines.append(f'"{r["name"]}",{r["weight"]},{r["std"]},"{other_params_str}"')
        lines.append("")
    elif output_format == "json":
        import json
        lines.append(json.dumps(rewards, indent=2))
        lines.append("")
        
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect basic info of rewards for registered tasks.")
    parser.add_argument(
        "--task",
        type=str,
        default="Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0",
        help="Task ID to inspect (default: Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect rewards for all registered tasks"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Load the play configuration instead of train configuration"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all registered tasks and exit"
    )

    args = parser.parse_args()

    # Handle --list-tasks first
    if args.list_tasks:
        tasks = list_tasks()
        print("Registered Tasks:")
        for task in sorted(tasks):
            print(f"  - {task}")
        sys.exit(0)

    # Get the list of tasks to process
    if args.all:
        tasks = sorted(list_tasks())
    else:
        tasks = [args.task]

    output_blocks = []
    for task in tasks:
        rewards = collect_rewards(task, play=args.play)
        if rewards is None:
            continue
        output_blocks.append(display_rewards(task, rewards, args.format))

    full_output = "\n".join(output_blocks)

    if args.output:
        with open(args.output, "w") as f:
            f.write(full_output)
        print(f"Results successfully written to {args.output}")
    else:
        print(full_output)


if __name__ == "__main__":
    main()
