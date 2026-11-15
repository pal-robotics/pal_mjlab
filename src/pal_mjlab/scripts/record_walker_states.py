"""Record states of a walking policy as start states for the safe-fall task.

The fallback policy takes over from a walking policy, usually mid-gait and
with momentum. This script rolls the walking policy out with random velocity
commands on the safe-fall robot model and saves snapshots of the full state.
``reset_fall_start`` then samples its "moving" starts from the resulting file.

Three kinds of state are recorded:

- walking: the walking policy is undisturbed.
- tripped: a swinging foot is caught (held where it is by a horizontal
  spring and damper for a short time), and the walking policy carries on.
- joint down: an actuator goes limp, and the walking policy carries on with it.

The last two are saved a detection time after the event: the moment a
supervisor would notice and hand over to the fallback policy.

  uv run python -m pal_mjlab.scripts.record_walker_states \
      --checkpoint /path/to/walker/model.pt \
      --output artifacts/kangaroo_walker_states.npz
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from pal_mjlab.scripts.eval_fall import load_actor_policy
from pal_mjlab.tasks.safe_fall import mdp
from pal_mjlab.tasks.safe_fall.mdp.events import (
  STATE_JOINT_DOWN,
  STATE_TRIPPED,
  STATE_WALKING,
)

FOOT_BODIES = ("leg_left_5_link", "leg_right_5_link")
_KIND_NAMES = {
  STATE_WALKING: "walking",
  STATE_TRIPPED: "tripped",
  STATE_JOINT_DOWN: "joint down",
}
# Onset step of an actuator that stays healthy.
_NEVER = 1_000_000


@dataclass
class Args:
  checkpoint: str
  """rsl_rl checkpoint of the walking policy."""
  output: str = "artifacts/kangaroo_walker_states.npz"
  task: str = "Mjlab-SafeFall-Flat-Pal-Kangaroo"
  num_envs: int = 4096
  num_states: int = 200_000
  """Number of plain walking states."""
  num_trip_states: int = 50_000
  num_fault_states: int = 110_000
  """Number of states with an actuator down, spread evenly over the actuators."""
  lin_vel_x: tuple[float, float] = (-0.3, 2.0)
  """Forward command range [m/s]. The walker of 2026-11 follows commands up to
  about 2 m/s and stops advancing above."""
  lin_vel_y: tuple[float, float] = (-0.4, 0.4)
  ang_vel_z: tuple[float, float] = (-1.0, 1.0)
  standing_fraction: float = 0.05
  """Share of environments commanded to stand still."""
  warmup_s: float = 2.5
  """Time after a reset before an environment may be sampled or disturbed."""
  event_window_s: float = 2.5
  """The trip or the actuator failure comes at a random time within this
  window after the warm-up; until then the environment is walking."""
  sample_probability: float = 0.02
  """Chance per control step that a walking environment is sampled. Random
  rather than periodic, so the snapshots cover every gait phase."""
  detection_time_s: tuple[float, float] = (0.1, 0.3)
  """Time from the trip or the failure to the saved state: how long the
  walking policy carries on before the fallback takes over."""
  trip_env_fraction: float = 0.3
  """Share of the environments that are tripped; an actuator fails in the rest."""
  trip_time_s: tuple[float, float] = (0.1, 0.25)
  """How long the caught foot is held, at most until the state is saved."""
  trip_stiffness: float = 4000.0
  """Horizontal force that holds the foot where it was caught, per unit of
  distance from there [N/m] ..."""
  trip_damping: float = 120.0
  """... and per unit of horizontal foot speed [N s/m]. The force is updated
  once per control step, which limits how stiff the hold can be."""
  trip_max_force: float = 1500.0
  trip_min_foot_speed: float = 1.0
  """A foot is caught while it is off the ground and faster than this [m/s]."""
  trip_give_up_s: float = 2.0
  """An environment with no such foot for this long (it stands, or walks
  slowly) starts over."""
  max_tilt_deg: float = 60.0
  """A disturbed state is saved only if the base is tilted less than this."""
  seed: int = 0
  device: str = "cuda:0"


def make_recording_cfg(args: Args) -> ManagerBasedRlEnvCfg:
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.curriculum = {}
  cfg.episode_length_s = 20.0

  # The walker walks: restore velocity commands and remove the kick.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = args.lin_vel_x
  twist.ranges.lin_vel_y = args.lin_vel_y
  twist.ranges.ang_vel_z = args.ang_vel_z
  twist.rel_standing_envs = args.standing_fraction
  twist.resampling_time_range = (3.0, 8.0)
  params = cfg.events["reset_fall_start"].params
  params["max_kick_speed"] = 0.0
  params["max_kick_ang_speed"] = 0.0
  params["moving_fraction"] = 0.0
  params["tip_fraction"] = 0.0
  params["state_buffer_path"] = None
  # Nothing else disturbs the walker: no shoves and no failed actuators.
  cfg.events.pop("shove", None)
  mdp.replace_actuator_fault_cfg(cfg, count_probabilities=(1.0,))

  # Restart environments in which the walker fell, so none of them is sampled.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
  )
  return cfg


def main(args: Args) -> None:
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=make_recording_cfg(args), device=args.device)
  )
  base: ManagerBasedRlEnv = env.unwrapped
  policy = load_actor_policy(env, args.task, args.checkpoint, args.device)
  robot = base.scene["robot"]
  command = base.command_manager.get_term("twist")
  feet = base.scene["feet_ground_contact"]
  foot_ids = robot.find_bodies(FOOT_BODIES, preserve_order=True)[0]
  n, device, dt = base.num_envs, base.device, base.step_dt
  rows = torch.arange(n, device=device)

  def steps(seconds: float) -> int:
    return int(round(seconds / dt))

  def random_steps(range_s: tuple[float, float], count: int) -> torch.Tensor:
    return torch.randint(
      steps(range_s[0]), steps(range_s[1]) + 1, (count,), device=device
    )

  obs, _ = env.reset()
  faults = mdp.get_actuator_faults(
    base, base.cfg.events["actuator_faults"].params["fault_cfg"]
  )
  num_actuators = len(faults.joint_names)
  can_fail = (faults.weights > 0).nonzero().flatten()

  # Per environment: 0 walking, 1 disturbed and waiting for the snapshot,
  # 2 done and waiting for its restart.
  phase = torch.zeros(n, device=device, dtype=torch.long)
  trips = rows < int(n * args.trip_env_fraction)
  event_step = torch.zeros(n, device=device, dtype=torch.long)
  snapshot_step = torch.zeros(n, device=device, dtype=torch.long)
  release_step = torch.zeros(n, device=device, dtype=torch.long)
  delay = torch.zeros(n, device=device, dtype=torch.long)
  hold = torch.zeros(n, device=device, dtype=torch.long)
  joint = torch.zeros(n, device=device, dtype=torch.long)
  foot = torch.zeros(n, device=device, dtype=torch.long)
  anchor = torch.zeros(n, 2, device=device)

  def plan(ids: torch.Tensor) -> None:
    phase[ids] = 0
    event_step[ids] = steps(args.warmup_s) + random_steps(
      (0.0, args.event_window_s), len(ids)
    )
    delay[ids] = random_steps(args.detection_time_s, len(ids))
    hold[ids] = random_steps(args.trip_time_s, len(ids))
    joint[ids] = can_fail[torch.randint(0, len(can_fail), (len(ids),), device=device)]

  plan(rows)
  keys = (
    "root_height",
    "root_quat",
    "root_vel",
    "joint_pos",
    "joint_vel",
    "command",
    "kind",
    "fault_joint",
  )
  chunks: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
  wanted = {
    STATE_WALKING: args.num_states,
    STATE_TRIPPED: args.num_trip_states,
    STATE_JOINT_DOWN: args.num_fault_states,
  }
  saved = dict.fromkeys(wanted, 0)
  # Disturbed robots that were past saving at the detection time.
  lost = dict.fromkeys(wanted, 0)

  def save(pick: torch.Tensor, kind: int) -> None:
    count = int(pick.sum())
    if count == 0 or saved[kind] >= wanted[kind]:
      return
    height = robot.data.root_link_pos_w[:, 2] - base.scene.env_origins[:, 2]
    chunks["root_height"].append(height[pick].cpu())
    chunks["root_quat"].append(robot.data.root_link_quat_w[pick].cpu())
    chunks["root_vel"].append(robot.data.root_link_vel_w[pick].cpu())
    chunks["joint_pos"].append(robot.data.joint_pos[pick].cpu())
    chunks["joint_vel"].append(robot.data.joint_vel[pick].cpu())
    chunks["command"].append(command.command[pick].cpu())
    chunks["kind"].append(torch.full((count,), kind, dtype=torch.long))
    fault_joint = joint[pick].cpu() if kind == STATE_JOINT_DOWN else None
    chunks["fault_joint"].append(
      fault_joint if fault_joint is not None else torch.full((count,), -1)
    )
    saved[kind] += count

  force = torch.zeros(n, len(foot_ids), 3, device=device)
  torque = torch.zeros_like(force)
  previous_step = base.episode_length_buf.clone()
  min_up = math.cos(math.radians(args.max_tilt_deg))
  while any(saved[k] < wanted[k] for k in wanted):
    with torch.inference_mode():
      obs = env.step(policy(obs))[0]
      step = base.episode_length_buf.clone()
      restarted = step != previous_step + 1
      previous_step = step
      for kind, mask in ((STATE_TRIPPED, trips), (STATE_JOINT_DOWN, ~trips)):
        # The walker fell over before the fallback could take over.
        lost[kind] += int((restarted & (phase == 1) & mask).sum())
      if bool(restarted.any()):
        plan(restarted.nonzero().flatten())
      up = -robot.data.projected_gravity_b[:, 2]

      # -- Walking states.
      walking = (phase == 0) & (step >= steps(args.warmup_s)) & (up > 0.9)
      save(
        walking & (torch.rand(n, device=device) < args.sample_probability),
        STATE_WALKING,
      )

      # -- An actuator goes limp.
      due = (phase == 0) & (step >= event_step)
      fail = (due & ~trips).nonzero().flatten()
      if len(fail) > 0:
        onset = torch.full(
          (len(fail), num_actuators), _NEVER, device=device, dtype=torch.long
        )
        onset[torch.arange(len(fail), device=device), joint[fail]] = 0
        faults.set_onsets(fail, onset)

      # -- A swinging foot is caught.
      foot_vel = robot.data.body_link_lin_vel_w[:, foot_ids, :2]
      foot_speed = foot_vel.norm(dim=-1)
      swinging = (feet.data.found == 0) & (foot_speed > args.trip_min_foot_speed)
      trip = due & trips & swinging.any(dim=-1)
      foot[trip] = torch.where(swinging, foot_speed, -1.0).argmax(dim=-1)[trip]
      foot_pos = robot.data.body_link_pos_w[:, foot_ids, :2]
      anchor[trip] = foot_pos[trip, foot[trip]]
      gave_up = due & trips & ~trip & (step >= event_step + steps(args.trip_give_up_s))

      disturbed = trip | (due & ~trips)
      phase[disturbed] = 1
      snapshot_step[disturbed] = step[disturbed] + delay[disturbed]
      release_step[disturbed] = step[disturbed] + hold[disturbed]

      holding = (phase == 1) & trips & (step < release_step)
      pull = (
        -args.trip_stiffness * (foot_pos[rows, foot] - anchor)
        - args.trip_damping * foot_vel[rows, foot]
      )
      pull *= (
        args.trip_max_force / pull.norm(dim=-1, keepdim=True).clamp(min=1e-6)
      ).clamp(max=1.0)
      force.zero_()
      force[holding, foot[holding], :2] = pull[holding]
      robot.write_external_wrench_to_sim(force, torque, body_ids=foot_ids)

      # -- The supervisor notices: this is where the fallback takes over.
      snapshot = (phase == 1) & (step >= snapshot_step)
      for kind, mask in ((STATE_TRIPPED, trips), (STATE_JOINT_DOWN, ~trips)):
        save(snapshot & mask & (up > min_up), kind)
        lost[kind] += int((snapshot & mask & (up <= min_up)).sum())
      done = snapshot | gave_up
      phase[done] = 2
      # Their episode times out at the next step, which restarts them.
      base.episode_length_buf[done] = base.max_episode_length

  data = {}
  for key in keys:
    values = torch.cat(chunks[key]).numpy()
    kind = torch.cat(chunks["kind"]).numpy()
    data[key] = np.concatenate([values[kind == k][: wanted[k]] for k in wanted], axis=0)
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output,
    joint_names=np.array(robot.joint_names),
    fault_joint_names=np.array(faults.joint_names),
    **data,
  )

  print(f"\nSaved {len(data['kind'])} walker states to {output}")
  speed = np.linalg.norm(data["root_vel"][:, :2], axis=-1)
  quat = torch.as_tensor(data["root_quat"])
  # Tilt of the base from upright, from the z component of its z axis.
  tilt = np.degrees(
    np.arccos(np.clip(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2).numpy(), -1, 1))
  )
  turn = np.linalg.norm(data["root_vel"][:, 3:5], axis=-1)
  for kind, name in _KIND_NAMES.items():
    m = data["kind"] == kind
    if not m.any():
      continue
    total = saved[kind] + lost[kind]
    print(
      f"{name:10s} n={int(m.sum()):6d}  speed [m/s] mean {speed[m].mean():.2f} "
      f"p95 {np.percentile(speed[m], 95):.2f} max {speed[m].max():.2f} | "
      f"tilt [deg] mean {tilt[m].mean():.1f} p95 {np.percentile(tilt[m], 95):.1f} | "
      f"roll/pitch rate [rad/s] mean {turn[m].mean():.2f} "
      f"p95 {np.percentile(turn[m], 95):.2f} | "
      f"base height [m] min {data['root_height'][m].min():.2f}"
      + (
        f" | already fallen at the handover: {lost[kind] / total:.1%}"
        if lost[kind]
        else ""
      )
    )
  counts = np.bincount(
    data["fault_joint"][data["fault_joint"] >= 0], minlength=num_actuators
  )
  print(
    "states per failed actuator:",
    dict(zip(faults.joint_names, counts.tolist(), strict=True)),
  )
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
