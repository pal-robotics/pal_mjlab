#!/usr/bin/env python3
"""
Script to collect and display the current basic info of rewards for registered tasks.
"""

import argparse
import dataclasses
import sys
from typing import Any, Dict, List

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg


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
        if (
          isinstance(val, slice)
          and val.start is None
          and val.stop is None
          and val.step is None
        ):
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

    rewards_info.append(
      {"name": name, "weight": weight, "std": std, "other_params": other_params}
    )
  return rewards_info


def _format_stages(stages: list) -> str:
  """Render a list of curriculum stage dicts as a compact string."""
  parts = []
  for s in stages:
    step = s.get("step", "?")
    params = {k: format_value(v) for k, v in s.items() if k != "step"}
    # Flatten single-key param dicts like {"params": {"p_drop": 0.3}}
    if list(params.keys()) == ["params"] and isinstance(s["params"], dict):
      inner = ", ".join(f"{k}={format_value(v)}" for k, v in s["params"].items())
      parts.append(f"step={step} → {inner}")
    elif "weight" in params:
      parts.append(f"step={step} → weight={params['weight']}")
    else:
      parts.append(f"step={step} → {params}")
  return " | ".join(parts)


def collect_curriculum(task_name: str, play: bool = False) -> List[Dict[str, Any]]:
  """Loads a task configuration and parses its curriculum terms."""
  try:
    cfg = load_env_cfg(task_name, play=play)
  except Exception as e:
    print(f"Error loading task '{task_name}': {e}", file=sys.stderr)
    return None

  if not hasattr(cfg, "curriculum") or not cfg.curriculum:
    return []

  curriculum_info = []
  for name, term in cfg.curriculum.items():
    func_name = term.func.__name__ if hasattr(term.func, "__name__") else str(term.func)
    params = term.params or {}
    stages_str = _format_stages(params.get("stages", []))
    target = ""
    if "reward_name" in params:
      target = f"reward: {params['reward_name']}"
    elif "term_name" in params:
      group = params.get("group_name", "?")
      target = f"{group}/{params['term_name']}"
    curriculum_info.append(
      {
        "name": name,
        "func": func_name,
        "target": target,
        "stages": stages_str,
      }
    )
  return curriculum_info


def collect_terminations(task_name: str, play: bool = False) -> List[Dict[str, Any]]:
  """Loads a task configuration and parses its termination terms."""
  try:
    cfg = load_env_cfg(task_name, play=play)
  except Exception as e:
    print(f"Error loading task '{task_name}': {e}", file=sys.stderr)
    return None

  if not hasattr(cfg, "terminations") or not cfg.terminations:
    return []

  terminations_info = []
  for name, term in cfg.terminations.items():
    func_name = term.func.__name__ if hasattr(term.func, "__name__") else str(term.func)
    params = term.params or {}
    other_params = {k: format_value(v) for k, v in params.items()}
    time_out = getattr(term, "time_out", False)
    terminations_info.append(
      {
        "name": name,
        "func": func_name,
        "time_out": time_out,
        "other_params": other_params,
      }
    )
  return terminations_info


def collect_noise(task_name: str, play: bool = False) -> List[Dict[str, Any]]:
  """Loads a task configuration and parses observation noise settings."""
  try:
    cfg = load_env_cfg(task_name, play=play)
  except Exception as e:
    print(f"Error loading task '{task_name}': {e}", file=sys.stderr)
    return None

  if not hasattr(cfg, "observations") or not cfg.observations:
    return []

  noise_info = []
  for group_name, group in cfg.observations.items():
    if not hasattr(group, "terms") or not group.terms:
      continue
    for term_name, term in group.terms.items():
      noise = getattr(term, "noise", None)
      if noise is None:
        noise_str = "None"
      elif dataclasses.is_dataclass(noise):
        noise_str = format_value(noise)
      else:
        noise_str = str(noise)

      # Extract p_drop if present in term params
      params = term.params or {}
      p_drop = params.get("p_drop", None)
      p_drop_str = format_value(p_drop) if p_drop is not None else "None"

      noise_info.append(
        {
          "group": group_name,
          "term": term_name,
          "noise": noise_str,
          "p_drop": p_drop_str,
        }
      )
  return noise_info


def display_rewards(
  task_name: str, rewards: List[Dict[str, Any]], output_format: str = "table"
) -> str:
  """Formats and returns the rewards information as a string."""
  lines = []
  lines.append(f"### Task: {task_name}")

  if not rewards:
    lines.append(
      "No rewards configured or task configuration does not have active rewards.\n"
    )
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


def display_terminations(
  task_name: str,
  terminations: List[Dict[str, Any]],
  output_format: str = "table",
) -> str:
  """Formats and returns the termination information as a string."""
  lines = []
  lines.append(f"### Terminations: {task_name}")

  if not terminations:
    lines.append("No termination terms configured.\n")
    return "\n".join(lines)

  if output_format == "table":
    lines.append("| Term Name | Function | Type | Params |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for t in terminations:
      type_str = "truncation" if t["time_out"] else "termination"
      params_str = ", ".join(f"{k}: {v}" for k, v in t["other_params"].items())
      lines.append(f"| {t['name']} | {t['func']} | {type_str} | {params_str} |")
    lines.append("")
  elif output_format == "csv":
    lines.append("term_name,func,time_out,params")
    for t in terminations:
      params_str = "; ".join(f"{k}={v}" for k, v in t["other_params"].items())
      lines.append(f'"{t["name"]}",{t["func"]},{t["time_out"]},"{params_str}"')
    lines.append("")
  elif output_format == "json":
    import json

    lines.append(json.dumps(terminations, indent=2))
    lines.append("")

  return "\n".join(lines)


def display_noise(
  task_name: str,
  noise: List[Dict[str, Any]],
  output_format: str = "table",
) -> str:
  """Formats and returns the observation noise information as a string."""
  lines = []
  lines.append(f"### Observation Noise: {task_name}")

  if not noise:
    lines.append("No observation noise configured.\n")
    return "\n".join(lines)

  if output_format == "table":
    lines.append("| Group | Term | Noise | P_Drop |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for n in noise:
      lines.append(f"| {n['group']} | {n['term']} | {n['noise']} | {n['p_drop']} |")
    lines.append("")
  elif output_format == "csv":
    lines.append("group,term,noise,p_drop")
    for n in noise:
      lines.append(f'{n["group"]},{n["term"]},"{n["noise"]}",{n["p_drop"]}')
    lines.append("")
  elif output_format == "json":
    import json

    lines.append(json.dumps(noise, indent=2))
    lines.append("")

  return "\n".join(lines)


def display_curriculum(
  task_name: str,
  curriculum: List[Dict[str, Any]],
  output_format: str = "table",
) -> str:
  """Formats and returns the curriculum information as a string."""
  lines = []
  lines.append(f"### Curriculum: {task_name}")

  if not curriculum:
    lines.append("No curriculum terms configured.\n")
    return "\n".join(lines)

  if output_format == "table":
    lines.append("| Term Name | Function | Target | Stages |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for c in curriculum:
      lines.append(f"| {c['name']} | {c['func']} | {c['target']} | {c['stages']} |")
    lines.append("")
  elif output_format == "csv":
    lines.append("term_name,func,target,stages")
    for c in curriculum:
      lines.append(f'"{c["name"]}",{c["func"]},"{c["target"]}","{c["stages"]}"')
    lines.append("")
  elif output_format == "json":
    import json

    lines.append(json.dumps(curriculum, indent=2))
    lines.append("")

  return "\n".join(lines)


def main():
  parser = argparse.ArgumentParser(
    description="Collect basic info of rewards for registered tasks."
  )
  parser.add_argument(
    "--task",
    type=str,
    default="Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0",
    help="Task ID to inspect (default: Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0)",
  )
  parser.add_argument(
    "--all", action="store_true", help="Collect rewards for all registered tasks"
  )
  parser.add_argument(
    "--play",
    action="store_true",
    help="Load the play configuration instead of train configuration",
  )
  parser.add_argument(
    "--format",
    type=str,
    choices=["table", "csv", "json"],
    default="table",
    help="Output format (default: table)",
  )
  parser.add_argument(
    "--output", type=str, default=None, help="Path to save the output"
  )
  parser.add_argument(
    "--list-tasks", action="store_true", help="List all registered tasks and exit"
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

    curriculum = collect_curriculum(task, play=args.play)
    if curriculum is not None:
      output_blocks.append(display_curriculum(task, curriculum, args.format))

    terminations = collect_terminations(task, play=args.play)
    if terminations is not None:
      output_blocks.append(display_terminations(task, terminations, args.format))

    noise = collect_noise(task, play=args.play)
    if noise is not None:
      output_blocks.append(display_noise(task, noise, args.format))

  full_output = "\n".join(output_blocks)

  if args.output:
    with open(args.output, "w") as f:
      f.write(full_output)
    print(f"Results successfully written to {args.output}")
  else:
    print(full_output)


if __name__ == "__main__":
  main()
