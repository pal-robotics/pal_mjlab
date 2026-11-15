"""Evaluate a controller on the safe-fall task and report damage metrics.

Runs one episode per environment with a fixed seed, so different controllers
face the same starts and kicks, and prints the metrics per kick-speed bin.

Baselines that need no training:

  uv run python -m pal_mjlab.scripts.eval_fall --controller zero
  uv run python -m pal_mjlab.scripts.eval_fall --controller damping

A walking policy with zero velocity command, or a trained fall policy:

  uv run python -m pal_mjlab.scripts.eval_fall --controller policy \
      --checkpoint /path/to/model.pt

Actuator failures are drawn as in training (see the task's
``FAULT_COUNT_PROBABILITIES``) and a second table reports the results by what
failed. ``--fault-counts 1`` runs without failures, ``--fault-counts 0 1`` with
exactly one per episode, and ``--fault-joints 'leg_.*_length_joint'`` limits
the failures to those joints.

The task's random mid-episode shoves are switched off, so that the kick bins
mean what they say. ``--shove-at 3.0`` instead shoves every robot at that
time, to test the ones that have settled by then: how many go down after all,
and how well the chest is protected when they do.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Literal

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from pal_mjlab.tasks.safe_fall import mdp
from pal_mjlab.tasks.safe_fall.mdp.events import (
  FAMILY_MOVING,
  STATE_JOINT_DOWN,
  STATE_TRIPPED,
)

GRAVITY = 9.81
# The position actuators cannot be built with zero stiffness.
DAMPING_STIFFNESS_SCALE = 1e-4


@dataclass
class Args:
  controller: Literal["zero", "damping", "policy"] = "zero"
  """zero: hold the default pose (zero action). damping: position gains
  scaled to almost zero, only joint damping acts. policy: an rsl_rl actor checkpoint, which
  can be a walking policy (it receives a zero velocity command) or a trained
  fall policy."""
  checkpoint: str | None = None
  task: str = "Mjlab-SafeFall-Flat-Pal-Kangaroo"
  num_envs: int = 2048
  max_kick_speed: float = 10.0
  kick_bin_width: float = 2.0
  state_buffer: str | None = None
  """Walker state buffer (.npz) to use instead of the task's default one."""
  moving_fraction: float | None = None
  """Share of starts taken from the walker state buffer. None keeps the task's."""
  forward_fraction: float | None = None
  """Share of episodes kicked (and tipped) the way the robot faces. None keeps
  the task's."""
  as_recorded_fraction: float | None = None
  """Share of the tripped and joint-down walker states that get no kick beyond
  a small one. None keeps the task's."""
  shove_at: float | None = None
  """Time [s] at which every robot gets a second shove. None: no shoves."""
  shove_speed: tuple[float, float] | None = None
  """Speed range [m/s] of that shove. None keeps the task's."""
  fault_counts: tuple[float, ...] | None = None
  """Probability of 0, 1, 2, ... failed actuators per episode. None keeps the
  task's; ``1`` means no failures and ``0 1`` exactly one."""
  fault_joints: str | None = None
  """Regex of the joints whose actuator can fail, all equally likely. None
  keeps the task's weights."""
  seed: int = 0
  device: str = "cuda:0"
  output: str | None = None
  """Optional JSON file for the per-bin results."""


def make_eval_cfg(args: Args) -> ManagerBasedRlEnvCfg:
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = load_env_cfg(args.task).episode_length_s
  params = cfg.events["reset_fall_start"].params
  params["max_kick_speed"] = args.max_kick_speed
  if args.state_buffer is not None:
    params["state_buffer_path"] = args.state_buffer
  if args.moving_fraction is not None:
    params["moving_fraction"] = args.moving_fraction
  if args.forward_fraction is not None:
    params["forward_fraction"] = args.forward_fraction
  if args.as_recorded_fraction is not None:
    params["as_recorded_fraction"] = args.as_recorded_fraction
  if args.shove_at is None:
    cfg.events.pop("shove", None)
  else:
    cfg.events["shove"].interval_range_s = (args.shove_at, args.shove_at)
    if args.shove_speed is not None:
      cfg.events["shove"].params["speed_range"] = args.shove_speed
  if args.fault_counts is not None or args.fault_joints is not None:
    _override_faults(cfg, args)
  if args.controller == "damping":
    robot = cfg.scene.entities["robot"]
    assert robot.articulation is not None
    robot.articulation = replace(
      robot.articulation,
      actuators=tuple(
        replace(a, stiffness=a.stiffness * DAMPING_STIFFNESS_SCALE)
        for a in robot.articulation.actuators
      ),
    )
  return cfg


def _override_faults(cfg: ManagerBasedRlEnvCfg, args: Args) -> None:
  changes: dict = {}
  if args.fault_counts is not None:
    changes["count_probabilities"] = tuple(args.fault_counts)
  if args.fault_joints is not None:
    changes["joint_weights"] = {args.fault_joints: 1.0, r".*": 0.0}
  mdp.replace_actuator_fault_cfg(cfg, **changes)


# What failed in an episode, for the table by failure. Single failures are
# split by the kind of joint (regex on its name).
FAULT_JOINT_GROUPS = (
  ("leg length", r"leg_.*_length_joint"),
  ("hip", r"leg_.*_[123]_joint"),
  ("ankle", r"leg_.*_[45]_joint"),
  ("arm", r"arm_.*"),
  ("waist", r"pelvis_.*"),
)


def load_actor_policy(
  env: RslRlVecEnvWrapper, task: str, checkpoint: str, device: str
) -> Callable[[dict], torch.Tensor]:
  """Inference policy from the actor of an rsl_rl checkpoint."""
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(load_rl_cfg(task)), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


def run_episode(
  env: RslRlVecEnvWrapper, policy, shove_at: float | None = None
) -> dict[str, torch.Tensor]:
  """Roll one episode per env and return per-env results.

  With ``shove_at`` [s] the state just before that time is kept, and what
  happens from then on is reported separately (keys starting with
  ``settled_``, meant for the robots that were at rest before the shove).
  """
  base: ManagerBasedRlEnv = env.unwrapped
  # The strict safe-state definition is the one the task's metrics use.
  flag_params = {
    k: v for k, v in base.cfg.metrics["squat_success"].params.items() if k != "kind"
  }
  flag_params["asset_cfg"] = SceneEntityCfg(
    flag_params["asset_cfg"].name, joint_names=flag_params["asset_cfg"].joint_names
  )
  flag_params["asset_cfg"].resolve(base.scene)
  impact_cfg: mdp.ImpactCfg = flag_params["impact_cfg"]
  reset_term = base.event_manager.get_term_cfg("reset_fall_start").func

  obs, _ = env.reset()
  # Direction the start pushes the body in, in its own frame: 0 = forwards.
  robot = base.scene["robot"]
  start_vel_b = quat_apply_inverse(
    robot.data.root_link_quat_w, robot.data.root_link_lin_vel_w
  )
  push_angle = torch.atan2(start_vel_b[:, 1], start_vel_b[:, 0])
  start_speed = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
  # Which actuators fail in this episode (all of them within the first second).
  fault_count = torch.zeros(base.num_envs, device=base.device)
  fault_group = torch.full((base.num_envs,), -1.0, device=base.device)
  fault_leg_length = torch.zeros(base.num_envs, device=base.device)
  if "actuator_faults" in base.cfg.events:
    faults = mdp.get_actuator_faults(
      base, base.cfg.events["actuator_faults"].params["fault_cfg"]
    )
    planned = faults.planned()
    fault_count = planned.sum(dim=-1).float()
    for index, (_, regex) in enumerate(FAULT_JOINT_GROUPS):
      in_group = torch.tensor(
        [re.fullmatch(regex, name) is not None for name in faults.joint_names],
        device=base.device,
      )
      only = (fault_count == 1) & planned[:, in_group].any(dim=-1)
      fault_group = torch.where(only, torch.full_like(fault_group, index), fault_group)
      if index == 0:
        fault_leg_length = planned[:, in_group].any(dim=-1).float()
  tracker = mdp.get_impact_tracker(base, impact_cfg)
  groups = {
    "torso": tracker.group_mask(r"pelvis_[12]_link"),
    "pelvis_hips": tracker.group_mask(r"(base_link|leg_(left|right)_[123]_link)"),
    "forearms": tracker.group_mask(r"arm_(left|right)_4_link"),
    "arms": tracker.group_mask(r"arm_(left|right)_[1-4]_link"),
    "femur": tracker.group_mask(r"leg_(left|right)_femur_link"),
    "knee_shank": tracker.group_mask(r"leg_(left|right)_(knee|4)_link"),
  }
  n = base.num_envs
  zeros = torch.zeros(n, device=base.device)
  group_impulse = {g: zeros.clone() for g in groups}
  group_contact = {g: zeros.clone().bool() for g in groups}
  imu_peak = zeros.clone()
  last_unsafe_step = zeros.clone()
  squat = rest = zeros.bool()
  imu = base.scene["robot/imu_lin_acc"]
  arm_cfg = base.cfg.metrics["arm_tip_distance"].params["arm_cfg"]
  arm_peak = zeros.clone()
  # Shielding: did a forearm reach the ground before the torso, and was it
  # pressed between the torso and the ground?
  shield_params = base.cfg.metrics["forearm_shield"].params
  never = torch.full((n,), float("inf"), device=base.device)
  first_torso_step = never.clone()
  first_forearm_step = never.clone()
  shield = zeros.clone().bool()
  # The same for the pelvis and the legs: did a knee, shank or femur reach the
  # ground before the pelvis or a hip housing?
  first_pelvis_step = never.clone()
  first_leg_step = never.clone()
  step_cfg = base.cfg.metrics["steps_taken"].params["step_cfg"]
  # Chest guard: how well the forearms cover the chest when it matters.
  guard_params = base.cfg.metrics["guard_coverage"].params
  guard_at_torso_hit = zeros.clone()
  # Shove test. The state is taken a few steps early, so that it is certainly
  # from before the shove.
  shove_step = None if shove_at is None else round(shove_at / base.step_dt) - 3
  settled = zeros.clone().bool()
  guard_before_shove = zeros.clone()
  fall_after_shove = zeros.clone().bool()
  torso_impulse_after_shove = zeros.clone()

  # One step short of the time limit, so the final state is still the episode's.
  num_steps = base.max_episode_length - 1
  for step in range(num_steps):
    with torch.inference_mode():
      obs = env.step(policy(obs))[0]
      tracker = mdp.get_impact_tracker(base, impact_cfg)
      for g, mask in groups.items():
        group_impulse[g] = torch.maximum(
          group_impulse[g], (tracker.step_impulse * mask).max(dim=-1).values
        )
        touching = (tracker.touching() & mask).any(dim=-1)
        group_contact[g] |= touching
        if g == "torso":
          guard_now = mdp.guard_coverage(base, **guard_params)
          guard_at_torso_hit = torch.where(
            touching & first_torso_step.isinf(), guard_now, guard_at_torso_hit
          )
          first_torso_step = torch.where(
            touching & first_torso_step.isinf(), zeros + step, first_torso_step
          )
        elif g == "forearms":
          first_forearm_step = torch.where(
            touching & first_forearm_step.isinf(), zeros + step, first_forearm_step
          )
        elif g == "pelvis_hips":
          first_pelvis_step = torch.where(
            touching & first_pelvis_step.isinf(), zeros + step, first_pelvis_step
          )
        if g in ("femur", "knee_shank"):
          first_leg_step = torch.where(
            touching & first_leg_step.isinf(), zeros + step, first_leg_step
          )
      shield |= mdp.contact_force_above(base, **shield_params).bool()
      imu_peak = torch.maximum(imu_peak, torch.linalg.vector_norm(imu.data, dim=-1))
      arm_peak = torch.maximum(
        arm_peak, mdp.arm_axis_distances(base, arm_cfg)[1].max(dim=-1).values
      )
      steps = mdp.get_step_counter(base, step_cfg).count.clone()
      squat, rest = mdp.safe_state_flags(base, **flag_params)
      fallen = group_contact["torso"] | group_contact["pelvis_hips"]
      if shove_step is not None:
        if step == shove_step:
          settled = rest & ~fallen
          guard_before_shove = guard_now.clone()
        elif step > shove_step:
          fall_after_shove |= settled & fallen
          torso_impulse_after_shove = torch.maximum(
            torso_impulse_after_shove,
            (tracker.step_impulse * groups["torso"]).max(dim=-1).values,
          )
      last_unsafe_step = torch.where(rest, last_unsafe_step, zeros + step + 1)

  # How the episode ends, from the direction of gravity in the torso frame
  # (x forwards, y to the left): lying on the front, the back or a side, or
  # still upright, either sitting on the pelvis or with it off the ground.
  torso_id = robot.find_bodies(("pelvis_2_link",))[0][0]
  down = torch.zeros(n, 3, device=base.device)
  down[:, 2] = -1.0
  gravity_b = quat_apply_inverse(robot.data.body_link_quat_w[:, torso_id], down)
  on_front = gravity_b[:, 0] > 0.5
  on_back = gravity_b[:, 0] < -0.5
  on_side = (gravity_b[:, 1].abs() > 0.5) & ~on_front & ~on_back
  upright = ~on_front & ~on_back & ~on_side
  touching = tracker.touching()
  pelvis_down = (touching & groups["pelvis_hips"]).any(dim=-1)
  torso_down = (touching & groups["torso"]).any(dim=-1)
  arm_down = (touching & groups["arms"]).any(dim=-1)
  fragile_down = pelvis_down | torso_down
  # Resting on knees and feet alone.
  legs_only = ~fragile_down & ~arm_down
  endings = {
    "end_kneel": legs_only & upright,
    "end_folded": legs_only & ~upright,
    "end_arms": arm_down & ~fragile_down,
    "end_seated": fragile_down & upright,
    "end_front": fragile_down & on_front,
    "end_back": fragile_down & on_back,
    "end_side": fragile_down & on_side,
  }

  shove_results = {}
  if shove_step is not None:
    shove_results = {
      "settled": settled.float(),
      # The next three are averaged over the settled episodes only.
      "settled_guard": guard_before_shove,
      "settled_fall_after": fall_after_shove.float(),
      "settled_torso_after": torso_impulse_after_shove,
    }
  return {
    "kick_speed": reset_term.kick_speed.clone(),
    "moving": (reset_term.family == FAMILY_MOVING).float(),
    "tipped": (reset_term.tip_angle > 0.0).float(),
    "push_angle": push_angle,
    "start_speed": start_speed,
    "state_kind": reset_term.state_kind.float(),
    "as_recorded": reset_term.as_recorded.float(),
    "fault_count": fault_count,
    "fault_group": fault_group,
    "fault_leg_length": fault_leg_length,
    **{k: v.float() for k, v in endings.items()},
    "steps": steps,
    "max_weighted_impulse": tracker.running_max.clone(),
    "squat": squat.float(),
    "rest": rest.float(),
    "time_to_rest": last_unsafe_step * base.step_dt,
    "imu_peak_g": imu_peak / GRAVITY,
    "arm_tip_peak": arm_peak,
    "fall": (group_contact["torso"] | group_contact["pelvis_hips"]).float(),
    # The next two only mean something for episodes in which the torso hit the
    # ground; summarize() averages them over those episodes.
    "torso_hit_arm_first": (first_forearm_step <= first_torso_step).float(),
    "torso_hit_shielded": shield.float(),
    "torso_hit_guard": guard_at_torso_hit,
    "guard_end": guard_now.clone(),
    **shove_results,
    # Likewise for episodes in which the pelvis or a hip housing hit the ground.
    "pelvis_hit_leg_first": (first_leg_step <= first_pelvis_step).float(),
    **{f"impulse_{g}": v for g, v in group_impulse.items()},
    **{f"contact_{g}": v.float() for g, v in group_contact.items()},
  }


# Per-episode labels that the tables split by rather than average.
_NOT_AVERAGED = (
  "kick_speed",
  "moving",
  "tipped",
  "push_angle",
  "fault_group",
  "start_speed",
  "state_kind",
  "as_recorded",
)


def summarize(results: dict[str, torch.Tensor], args: Args) -> list[dict]:
  rows = []
  edges = torch.arange(0.0, args.max_kick_speed + 1e-6, args.kick_bin_width)
  upright = results["tipped"] == 0
  families = [
    ("standing", upright & (results["moving"] == 0)),
    ("moving", upright & (results["moving"] == 1)),
    ("tipped", ~upright),
  ]
  for family, family_mask in families:
    for lo in edges[:-1].tolist():
      hi = lo + args.kick_bin_width
      mask = family_mask & (results["kick_speed"] >= lo) & (results["kick_speed"] < hi)
      if not mask.any():
        continue
      row: dict = {"start": family, "kick": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum())}
      torso_hit = mask & (results["contact_torso"] > 0)
      pelvis_hit = mask & (results["contact_pelvis_hips"] > 0)
      settled = mask & (results.get("settled", torch.zeros_like(mask)) > 0)
      for key, value in results.items():
        if key in _NOT_AVERAGED:
          continue
        if key.startswith("torso_hit_"):
          row[key] = float(value[torso_hit].mean()) if torso_hit.any() else None
        elif key.startswith("settled_"):
          row[key] = float(value[settled].mean()) if settled.any() else None
        elif key.startswith("pelvis_hit_"):
          row[key] = float(value[pelvis_hit].mean()) if pelvis_hit.any() else None
        else:
          row[key] = float(value[mask].mean())
      row["max_weighted_impulse_p95"] = float(
        torch.quantile(results["max_weighted_impulse"][mask], 0.95)
      )
      rows.append(row)
  return rows


ENDING_KEYS = (
  "end_kneel",
  "end_folded",
  "end_arms",
  "end_seated",
  "end_front",
  "end_back",
  "end_side",
)


def summarize_endings(results: dict[str, torch.Tensor], args: Args) -> list[dict]:
  """How the episodes end, by the direction and speed of the push."""
  angle = torch.rad2deg(results["push_angle"]).abs()
  upright = results["tipped"] == 0
  sectors = [
    ("forwards", upright & (angle < 45.0)),
    ("sideways", upright & (angle >= 45.0) & (angle <= 135.0)),
    ("backwards", upright & (angle > 135.0)),
  ]
  third = args.max_kick_speed / 3.0
  groups = [
    (name, f"{i * third:.1f}-{(i + 1) * third:.1f}", sector, i * third, (i + 1) * third)
    for name, sector in sectors
    for i in range(3)
  ]
  groups.append(("tipped", "any", ~upright, 0.0, float("inf")))
  rows = []
  for name, kick, sector, lo, hi in groups:
    mask = sector & (results["kick_speed"] >= lo) & (results["kick_speed"] < hi)
    if not mask.any():
      continue
    row: dict = {"push": name, "kick": kick, "n": int(mask.sum())}
    for key in (
      *ENDING_KEYS,
      "contact_pelvis_hips",
      "impulse_pelvis_hips",
      "impulse_torso",
      "guard_end",
    ):
      row[key] = float(results[key][mask].mean())
    rows.append(row)
  return rows


def _summarize_cases(
  results: dict[str, torch.Tensor], cases: list[tuple[str, torch.Tensor]], label: str
) -> list[dict]:
  rows = []
  for name, mask in cases:
    if not mask.any():
      continue
    row: dict = {label: name, "n": int(mask.sum())}
    for key in (
      "fall",
      "contact_torso",
      "contact_pelvis_hips",
      "impulse_torso",
      "impulse_pelvis_hips",
      "impulse_forearms",
      "max_weighted_impulse",
      *ENDING_KEYS,
      "guard_end",
      "rest",
      "imu_peak_g",
    ):
      row[key] = float(results[key][mask].mean())
    row["max_weighted_impulse_p95"] = float(
      torch.quantile(results["max_weighted_impulse"][mask], 0.95)
    )
    rows.append(row)
  return rows


def summarize_walker_events(results: dict[str, torch.Tensor]) -> list[dict]:
  """Results of the starts recorded after the walker tripped or lost a joint."""
  kind, speed = results["state_kind"], results["start_speed"]
  as_recorded = results["as_recorded"] > 0
  cases = []
  for name, value in (("tripped", STATE_TRIPPED), ("joint down", STATE_JOINT_DOWN)):
    this = kind == value
    cases += [
      (f"{name}, below 1 m/s", this & as_recorded & (speed < 1.0)),
      (f"{name}, 1 m/s and above", this & as_recorded & (speed >= 1.0)),
      (f"{name}, kicked on top", this & ~as_recorded),
    ]
  cases.append(
    ("every other start", (kind != STATE_TRIPPED) & (kind != STATE_JOINT_DOWN))
  )
  return _summarize_cases(results, cases, "start")


def summarize_faults(results: dict[str, torch.Tensor]) -> list[dict]:
  """Results by what failed: nothing, one actuator by kind of joint, or more."""
  count = results["fault_count"]
  cases = [("none", count == 0)]
  cases += [
    (f"1: {name}", results["fault_group"] == index)
    for index, (name, _) in enumerate(FAULT_JOINT_GROUPS)
  ]
  leg_length = results["fault_leg_length"] > 0
  cases += [
    ("2-3 with a leg length", (count >= 2) & leg_length),
    ("2-3 without", (count >= 2) & ~leg_length),
    ("any failure", count >= 1),
    ("all episodes", count >= 0),
  ]
  return _summarize_cases(results, cases, "failed")


def print_faults(rows: list[dict]) -> None:
  _print_cases(
    rows,
    "failed",
    "failed actuators",
    "By what failed (limp actuators; all kicks and starts together)",
    "'1: ...' rows have a single failed actuator, of that kind of joint "
    "(hip: the three hip joints; ankle: both ankle joints). The first failure "
    "is there from the start, further ones begin within the first second.",
  )


def print_walker_events(rows: list[dict]) -> None:
  _print_cases(
    rows,
    "start",
    "start",
    "Starts from what the walker does when something goes wrong",
    "'tripped': a swinging foot of the walking robot was caught. 'joint down': "
    "an actuator of the walking robot went limp. Both start a detection time "
    "(0.1 to 0.3 s) after that, where the fallback takes over, at the speed "
    "the walker had then; 'kicked on top' adds the usual kick to such a state.",
  )


def _print_cases(
  rows: list[dict], label: str, header: str, title: str, note: str
) -> None:
  columns = [
    (label, header, "{}"),
    ("n", "n", "{}"),
    ("fall", "fall", "{:.0%}"),
    ("contact_torso", "torso hit", "{:.0%}"),
    ("contact_pelvis_hips", "pelvis hit", "{:.0%}"),
    ("impulse_torso", "torso", "{:.3f}"),
    ("impulse_pelvis_hips", "pelvis", "{:.3f}"),
    ("impulse_forearms", "forearm", "{:.3f}"),
    ("max_weighted_impulse", "max imp", "{:.3f}"),
    ("max_weighted_impulse_p95", "p95", "{:.2f}"),
    ("end_kneel", "kneel", "{:.0%}"),
    ("end_folded", "folded", "{:.0%}"),
    ("end_arms", "on arms", "{:.0%}"),
    ("end_front", "front", "{:.0%}"),
    ("end_side", "side", "{:.0%}"),
    ("end_back", "back", "{:.0%}"),
    ("end_seated", "seated", "{:.0%}"),
    ("guard_end", "guard end", "{:.2f}"),
    ("rest", "rest", "{:.0%}"),
    ("imu_peak_g", "imu g", "{:.1f}"),
  ]
  cells = [[fmt.format(row[key]) for key, _, fmt in columns] for row in rows]
  widths = [
    max(len(header), *(len(c[i]) for c in cells))
    for i, (_, header, _) in enumerate(columns)
  ]
  print(f"\n{title}")
  print("  ".join(h.rjust(w) for (_, h, _), w in zip(columns, widths, strict=True)))
  for line in cells:
    print("  ".join(c.rjust(w) for c, w in zip(line, widths, strict=True)))
  print(f"\n{note}")


def print_endings(rows: list[dict]) -> None:
  columns = [
    ("push", "pushed", "{}"),
    ("kick", "kick m/s", "{}"),
    ("n", "n", "{}"),
    ("end_kneel", "kneel", "{:.0%}"),
    ("end_folded", "folded", "{:.0%}"),
    ("end_arms", "on arms", "{:.0%}"),
    ("end_front", "front", "{:.0%}"),
    ("end_side", "side", "{:.0%}"),
    ("end_back", "back", "{:.0%}"),
    ("end_seated", "seated", "{:.0%}"),
    ("contact_pelvis_hips", "pelvis hit", "{:.0%}"),
    ("impulse_pelvis_hips", "pelvis", "{:.3f}"),
    ("impulse_torso", "torso", "{:.3f}"),
    ("guard_end", "guard end", "{:.2f}"),
  ]
  cells = [[fmt.format(row[key]) for key, _, fmt in columns] for row in rows]
  widths = [
    max(len(header), *(len(c[i]) for c in cells))
    for i, (_, header, _) in enumerate(columns)
  ]
  print("\nHow the episodes end, by the direction the start pushes the body in")
  print("  ".join(h.rjust(w) for (_, h, _), w in zip(columns, widths, strict=True)))
  for line in cells:
    print("  ".join(c.rjust(w) for c, w in zip(line, widths, strict=True)))
  print(
    "\nBy what is on the ground at the end. 'kneel': knees and feet only, torso "
    "upright (within 30 deg). 'folded': knees and feet only, torso folded "
    "forwards or to a side. 'on arms': an arm as well, torso and pelvis off the "
    "ground. With the torso or the pelvis on the ground: 'front', 'side', "
    "'back' by the side of the torso that faces the ground, 'seated' if it is "
    "upright. The first three rows of each direction are upright starts; "
    "'tipped' starts already tip over."
  )


def print_table(rows: list[dict], title: str) -> None:
  columns = [
    ("start", "start", "{}"),
    ("kick", "kick m/s", "{}"),
    ("n", "n", "{}"),
    ("squat", "squat", "{:.0%}"),
    ("rest", "rest", "{:.0%}"),
    ("time_to_rest", "t_rest s", "{:.2f}"),
    ("steps", "steps", "{:.1f}"),
    ("fall", "fall", "{:.0%}"),
    ("max_weighted_impulse", "max imp", "{:.3f}"),
    ("impulse_torso", "torso", "{:.3f}"),
    ("impulse_pelvis_hips", "pelvis", "{:.3f}"),
    ("impulse_forearms", "forearm", "{:.3f}"),
    ("impulse_knee_shank", "knee", "{:.3f}"),
    ("contact_torso", "torso hit", "{:.0%}"),
    ("torso_hit_arm_first", "arm 1st", "{:.0%}"),
    ("torso_hit_shielded", "shield", "{:.0%}"),
    ("torso_hit_guard", "guard@hit", "{:.2f}"),
    ("guard_end", "guard end", "{:.2f}"),
    ("contact_pelvis_hips", "pelvis hit", "{:.0%}"),
    ("pelvis_hit_leg_first", "leg 1st", "{:.0%}"),
    ("imu_peak_g", "imu g", "{:.1f}"),
    ("arm_tip_peak", "arm m", "{:.2f}"),
  ]
  if rows and "settled" in rows[0]:
    columns += [
      ("settled", "settled", "{:.0%}"),
      ("settled_guard", "s.guard", "{:.2f}"),
      ("settled_fall_after", "s.fall", "{:.0%}"),
      ("settled_torso_after", "s.torso", "{:.3f}"),
    ]
  cells = [
    ["-" if row[key] is None else fmt.format(row[key]) for key, _, fmt in columns]
    for row in rows
  ]
  widths = [
    max(len(header), *(len(c[i]) for c in cells))
    for i, (_, header, _) in enumerate(columns)
  ]
  print(f"\n{title}")
  print("  ".join(h.rjust(w) for (_, h, _), w in zip(columns, widths, strict=True)))
  for line in cells:
    print("  ".join(c.rjust(w) for c, w in zip(line, widths, strict=True)))
  print(
    "\nImpulses are per 20 ms control step, normalized by robot mass x 1 m/s "
    "(a body statically carrying the whole robot reads 0.196).\n"
    "'max imp' is the fragility-weighted maximum the policy is trained on; "
    "the per-body columns are unweighted.\n"
    "'fall': torso or pelvis touched the ground. Of the episodes in which the "
    "torso touched the ground, 'arm 1st' is the share where a forearm touched "
    "it first and 'shield' the share where a forearm was pressed against the "
    "torso. Likewise 'leg 1st' is the share of the pelvis hits in which a femur, "
    "knee or shank touched the ground before the pelvis or a hip housing.\n"
    "'start': 'tipped' episodes start already tipping over (from standing or "
    "from walking); the others start upright. 'steps' is the number of foot "
    "touchdowns while no other body was on the ground.\n"
    "'guard' is how well the forearms cover the chest (1 = both forearms in "
    "front of it): 'guard@hit' when the torso first touches the ground, "
    "'guard end' at the end of the episode.\n"
    "With --shove-at: 'settled' is the share at rest without a fall just "
    "before the shove; of those, 's.guard' is the guard at that moment, "
    "'s.fall' the share that fell after the shove and 's.torso' the largest "
    "torso impulse after it.\n"
    "'arm m' is the largest distance of a forearm tip from the body axis during "
    "the episode (about 0.4 with the arms at the sides, 1.06 fully extended)."
  )


def main(args: Args) -> None:
  if args.controller == "policy" and args.checkpoint is None:
    raise ValueError("--controller policy needs --checkpoint.")
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=make_eval_cfg(args), device=args.device)
  )
  if args.controller == "policy":
    assert args.checkpoint is not None
    policy = load_actor_policy(env, args.task, args.checkpoint, args.device)
  else:
    action = torch.zeros(env.num_envs, env.num_actions, device=args.device)
    policy = lambda obs: action  # noqa: E731

  results = run_episode(env, policy, args.shove_at)
  rows = summarize(results, args)
  ending_rows = summarize_endings(results, args)
  label = args.controller if args.checkpoint is None else Path(args.checkpoint).name
  print_table(rows, f"Safe-fall evaluation: {label}, {args.num_envs} episodes")
  print_endings(ending_rows)
  fault_rows = summarize_faults(results)
  print_faults(fault_rows)
  walker_event_rows = summarize_walker_events(results)
  if len(walker_event_rows) > 1:
    print_walker_events(walker_event_rows)
  if args.output is not None:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
      json.dumps(
        {
          "args": asdict(args),
          "rows": rows,
          "endings": ending_rows,
          "faults": fault_rows,
          "walker_events": walker_event_rows,
        }
      )
    )
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
