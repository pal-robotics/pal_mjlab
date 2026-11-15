"""Unit tests for the safe-fall impact tracker, rewards and curriculum."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from pal_mjlab.tasks.safe_fall.mdp import arms, curriculums, impact, rewards, steps

NUM_ENVS = 3
BODIES = ["pelvis_2_link", "leg_left_knee_link", "leg_left_5_link"]
FRAGILITY = {r"pelvis_[12]_link": 1.0, r"leg_(left|right)_knee_link": 0.2}
TOTAL_MASS = 50.0
PHYSICS_DT = 0.002
DECIMATION = 10
STEP_DT = PHYSICS_DT * DECIMATION


def _make_env():
  """Minimal stand-in for ManagerBasedRlEnv with one impact sensor."""
  sensor = SimpleNamespace(
    cfg=SimpleNamespace(history_length=DECIMATION, fields=("found", "force")),
    primary_names=[f"robot/{b}" for b in BODIES],
    data=SimpleNamespace(
      force_history=torch.zeros(NUM_ENVS, len(BODIES), DECIMATION, 3)
    ),
  )
  asset = Mock()
  asset.find_bodies.return_value = ([0, 1, 2], BODIES)
  asset.indexing.body_ids = torch.tensor([1, 2, 3])
  env = SimpleNamespace(
    num_envs=NUM_ENVS,
    device="cpu",
    physics_dt=PHYSICS_DT,
    step_dt=STEP_DT,
    common_step_counter=0,
    episode_length_buf=torch.zeros(NUM_ENVS, dtype=torch.long),
    scene={"impact": sensor, "robot": asset},
    sim=SimpleNamespace(
      mj_model=SimpleNamespace(body_mass=np.array([0.0, 20.0, 20.0, 10.0]))
    ),
    cfg=SimpleNamespace(scale_rewards_by_dt=True),
  )
  return env, sensor


def _step(env, sensor, body_forces):
  """Advance one control step with a constant force [N] per body and env."""
  forces = torch.as_tensor(body_forces, dtype=torch.float32)  # [B, N]
  history = torch.zeros(NUM_ENVS, len(BODIES), DECIMATION, 3)
  history[..., 2] = forces.unsqueeze(-1)
  sensor.data.force_history = history
  env.common_step_counter += 1
  env.episode_length_buf = env.episode_length_buf + 1


@pytest.fixture
def cfg():
  return impact.ImpactCfg(sensor_name="impact", fragility=FRAGILITY)


def test_fragility_weights_first_match_and_default_zero(cfg):
  env, _ = _make_env()
  tracker = impact.get_impact_tracker(env, cfg)
  assert tracker.weights.tolist() == pytest.approx([1.0, 0.2, 0.0])
  assert tracker.total_mass == pytest.approx(TOTAL_MASS)


def test_step_impulse_is_force_integrated_over_substeps(cfg):
  env, sensor = _make_env()
  _step(env, sensor, [[500.0, 0.0, 0.0]] * NUM_ENVS)
  tracker = impact.get_impact_tracker(env, cfg)
  # 500 N for one 20 ms control step on a 50 kg robot: 10 N s -> 0.2 m/s.
  assert tracker.step_impulse[0, 0].item() == pytest.approx(0.2)
  assert tracker.touching()[0].tolist() == [True, False, False]


def test_max_impulse_rewards_sum_to_the_largest_weighted_impulse(cfg):
  """The episode return of the term is the worst impact, not the total."""
  env, sensor = _make_env()
  torso_forces = [100.0, 800.0, 300.0, 0.0, 600.0, 900.0, 50.0]
  total = torch.zeros(NUM_ENVS)
  for force in torso_forces:
    # Env 0: torso. Env 1: same force on the knee (weight 0.2). Env 2: foot.
    _step(env, sensor, [[force, 0, 0], [0, force, 0], [0, 0, force]])
    total += rewards.max_impulse_increment(env, cfg) * STEP_DT

  worst = max(torso_forces) * STEP_DT / TOTAL_MASS
  assert total.tolist() == pytest.approx([worst, 0.2 * worst, 0.0])
  tracker = impact.get_impact_tracker(env, cfg)
  assert tracker.running_max.tolist() == pytest.approx([worst, 0.2 * worst, 0.0])


def test_tracker_updates_once_per_control_step(cfg):
  env, sensor = _make_env()
  _step(env, sensor, [[400.0, 0.0, 0.0]] * NUM_ENVS)
  first = rewards.max_impulse_increment(env, cfg).clone()
  assert first[0].item() > 0.0
  # Another term reading the tracker in the same step sees the same increment.
  assert torch.equal(rewards.max_impulse_increment(env, cfg), first)


def test_running_max_restarts_with_the_episode(cfg):
  env, sensor = _make_env()
  _step(env, sensor, [[1000.0, 0.0, 0.0]] * NUM_ENVS)
  tracker = impact.get_impact_tracker(env, cfg)
  assert tracker.running_max[0].item() == pytest.approx(0.4)

  # Env 0 is reset: the critic observation is zero before its first step.
  env.episode_length_buf = torch.tensor([0, 1, 1])
  assert tracker.episode_running_max().tolist() == pytest.approx([0.0, 0.4, 0.4])

  # First step of the new episode in env 0 with a small force: the old
  # maximum must not carry over.
  sensor.data.force_history = torch.zeros(NUM_ENVS, len(BODIES), DECIMATION, 3)
  sensor.data.force_history[0, 0, :, 2] = 250.0
  env.common_step_counter += 1
  env.episode_length_buf = torch.tensor([1, 2, 2])
  tracker = impact.get_impact_tracker(env, cfg)
  assert tracker.running_max.tolist() == pytest.approx([0.1, 0.4, 0.4])
  assert tracker.increment.tolist() == pytest.approx([0.1, 0.0, 0.0])


def test_joint_limit_strike_only_counts_motion_into_a_limit():
  asset = Mock()
  asset.data.soft_joint_pos_limits = torch.tensor([[[-1.0, 1.0]] * 4])
  #                               inside, past upper, past upper, past lower
  asset.data.joint_pos = torch.tensor([[0.0, 1.1, 1.1, -1.1]])
  asset.data.joint_vel = torch.tensor([[5.0, 2.0, -2.0, -3.0]])
  env = SimpleNamespace(scene={"robot": asset})
  asset_cfg = SimpleNamespace(name="robot", joint_ids=slice(None))
  value = rewards.joint_limit_strike(env, asset_cfg=asset_cfg)
  # Joint 1 moves further past its upper limit (2^2), joint 3 further past
  # its lower limit (3^2). Joint 0 is inside, joint 2 is moving back.
  assert value.item() == pytest.approx(4.0 + 9.0)


def test_event_param_curriculum_applies_reached_stages():
  term_cfg = SimpleNamespace(params={"max_kick_speed": 0.0, "other": 1})
  env = SimpleNamespace(
    common_step_counter=0,
    event_manager=SimpleNamespace(get_term_cfg=lambda name: term_cfg),
  )
  stages = [
    {"step": 0, "params": {"max_kick_speed": 0.5}},
    {"step": 100, "params": {"max_kick_speed": 1.5}},
    {"step": 200, "params": {"max_kick_speed": 2.5}},
  ]
  cfg = SimpleNamespace(params={"event_name": "reset", "stages": stages})
  term = curriculums.event_param_curriculum(cfg, env)

  for step, expected in ((0, 0.5), (99, 0.5), (100, 1.5), (500, 2.5)):
    env.common_step_counter = step
    logged = term(env, torch.arange(1), "reset", stages)
    assert term_cfg.params["max_kick_speed"] == expected
    assert logged["max_kick_speed"].item() == pytest.approx(expected)
  assert term_cfg.params["other"] == 1


def test_event_param_curriculum_rejects_unknown_params():
  term_cfg = SimpleNamespace(params={"max_kick_speed": 0.0})
  env = SimpleNamespace(event_manager=SimpleNamespace(get_term_cfg=lambda n: term_cfg))
  cfg = SimpleNamespace(
    params={"event_name": "reset", "stages": [{"step": 0, "params": {"typo": 1.0}}]}
  )
  with pytest.raises(KeyError):
    curriculums.event_param_curriculum(cfg, env)


ARM_CFG = arms.ArmSpreadCfg(
  torso_body="torso",
  axis_start=(0.0, 0.0, -0.4),
  axis_end=(0.0, 0.0, 0.4),
  elbow_bodies=("elbow",),
  tip_sites=("tip",),
)


class _Scene(dict):
  env_origins = torch.zeros(1, 3)


def _arm_env(torso_quat, elbow_pos, tip_pos, height=0.9):
  """One env: torso at (0, 0, height), one elbow body and one tip site."""
  asset = Mock()
  asset.find_bodies.side_effect = lambda names, preserve_order=False: (
    [{"torso": 0, "elbow": 1}[n] for n in names],
    list(names),
  )
  asset.find_sites.return_value = ([0], ["tip"])
  torso_pos = torch.tensor([0.0, 0.0, height])
  asset.data.body_link_pos_w = torch.stack(
    [torso_pos, torso_pos + torch.tensor(elbow_pos)]
  ).unsqueeze(0)
  asset.data.body_link_quat_w = torch.tensor([torso_quat, [1.0, 0.0, 0.0, 0.0]])[None]
  asset.data.site_pos_w = (torso_pos + torch.tensor(tip_pos)).reshape(1, 1, 3)
  asset.data.root_link_pos_w = torso_pos.unsqueeze(0)
  asset.data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]])
  return SimpleNamespace(device="cpu", scene=_Scene(robot=asset)), asset


def test_arm_distance_is_measured_from_the_body_axis_segment():
  identity = [1.0, 0.0, 0.0, 0.0]
  # Elbow 0.3 m to the side at mid height; tip 0.5 m above the top of the axis.
  env, _ = _arm_env(identity, elbow_pos=[0.0, 0.3, 0.1], tip_pos=[0.0, 0.0, 0.9])
  elbow, tip = arms.arm_axis_distances(env, ARM_CFG)
  assert elbow.item() == pytest.approx(0.3)
  assert tip.item() == pytest.approx(0.5)


def test_arm_distance_follows_the_torso_orientation():
  # Torso pitched 90 degrees about y: its long axis now lies along world x.
  quat = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
  env, _ = _arm_env(quat, elbow_pos=[0.3, 0.0, 0.0], tip_pos=[0.0, 0.0, 0.6])
  elbow, tip = arms.arm_axis_distances(env, ARM_CFG)
  assert elbow.item() == pytest.approx(0.0, abs=1e-6)  # On the axis.
  assert tip.item() == pytest.approx(0.6, abs=1e-6)  # Straight out from it.


GUARD_CFG = arms.ForearmGuardCfg(
  torso_body="torso",
  tip_box_min=(0.1, -0.15, 0.0),
  tip_box_max=(0.2, 0.15, 0.4),
  mid_box_min=(0.1, -0.15, 0.0),
  mid_box_max=(0.2, 0.15, 0.4),
  elbow_bodies=("elbow",),
  tip_sites=("tip",),
  arm_sides=(1.0,),
)


def test_guard_distance_is_zero_inside_the_zone_and_follows_the_torso():
  identity = [1.0, 0.0, 0.0, 0.0]
  # Forearm across the chest: elbow and tip at either side, in front of it.
  env, _ = _arm_env(identity, elbow_pos=[0.15, -0.15, 0.2], tip_pos=[0.15, 0.15, 0.2])
  assert arms.forearm_guard_distances(env, GUARD_CFG).tolist() == [[0.0, 0.0]]

  # Arm hanging at the side: midpoint and tip are 0.15 m outside the zone.
  env, _ = _arm_env(identity, elbow_pos=[0.15, 0.3, 0.3], tip_pos=[0.15, 0.3, 0.1])
  assert arms.forearm_guard_distances(env, GUARD_CFG)[0].tolist() == pytest.approx(
    [0.15, 0.15]
  )

  # Torso pitched 90 degrees about y: its front (local x) now points down.
  quat = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
  env, _ = _arm_env(quat, elbow_pos=[0.2, -0.1, -0.15], tip_pos=[0.2, 0.1, -0.15])
  assert arms.forearm_guard_distances(env, GUARD_CFG)[0].tolist() == pytest.approx(
    [0.0, 0.0], abs=1e-6
  )


def test_forearm_guard_counts_from_the_start_and_has_a_slope_everywhere():
  identity = [1.0, 0.0, 0.0, 0.0]
  guarding = {"elbow_pos": [0.15, -0.15, 0.2], "tip_pos": [0.15, 0.15, 0.2]}

  # Standing upright counts as much as being low.
  for height in (0.9, 0.5):
    env, _ = _arm_env(identity, height=height, **guarding)
    assert rewards.guard_coverage(env, GUARD_CFG, std=0.2).item() == pytest.approx(1.0)
    assert rewards.forearm_guard_distance(env, GUARD_CFG).item() == pytest.approx(0.0)

  # Arms at the sides: both points 0.15 m out of the zone.
  near, _ = _arm_env(
    identity, height=0.5, elbow_pos=[0.15, 0.3, 0.3], tip_pos=[0.15, 0.3, 0.1]
  )
  assert rewards.guard_coverage(near, GUARD_CFG, std=0.2).item() == pytest.approx(
    math.exp(-(0.15**2) / 0.2**2)
  )
  assert rewards.forearm_guard_distance(near, GUARD_CFG).item() == pytest.approx(0.15)

  # Behind the back the bonus is gone, the distance still tells near from far.
  behind, _ = _arm_env(
    identity, height=0.5, elbow_pos=[-0.5, 0.0, 0.2], tip_pos=[-0.5, 0.0, 0.2]
  )
  further, _ = _arm_env(
    identity, height=0.5, elbow_pos=[-0.7, 0.0, 0.2], tip_pos=[-0.7, 0.0, 0.2]
  )
  assert rewards.guard_coverage(behind, GUARD_CFG, std=0.2).item() < 1e-3
  assert rewards.forearm_guard_distance(behind, GUARD_CFG).item() == pytest.approx(0.6)
  assert rewards.forearm_guard_distance(further, GUARD_CFG).item() == pytest.approx(0.8)


def _step_env():
  """Stand-in env with an impact sensor and a two-foot contact sensor."""
  env, impact_sensor = _make_env()
  feet = Mock()
  feet.data.last_air_time = torch.zeros(NUM_ENVS, 2)
  feet.compute_first_contact.return_value = torch.zeros(NUM_ENVS, 2, dtype=torch.bool)
  env.scene["feet"] = feet
  step_cfg = steps.StepCountCfg(
    feet_sensor_name="feet",
    impact_cfg=impact.ImpactCfg(sensor_name="impact", fragility=FRAGILITY),
    free_steps=2,
  )
  return env, impact_sensor, feet, step_cfg


def _touch_down(env, impact_sensor, feet, landed, air_time=0.2, body_forces=None):
  """Advance one control step in which the given feet [B, 2] land."""
  feet.compute_first_contact.return_value = torch.tensor(landed, dtype=torch.bool)
  feet.data.last_air_time = torch.full((NUM_ENVS, 2), air_time)
  _step(env, impact_sensor, body_forces or [[0.0, 0.0, 0.0]] * NUM_ENVS)


def test_excess_steps_sum_to_the_touchdowns_beyond_the_free_ones():
  env, impact_sensor, feet, step_cfg = _step_env()
  # Env 0 takes 5 steps, env 1 takes 2 (all free), env 2 takes none.
  schedule = [
    [[True, False], [True, False], [False, False]],
    [[False, True], [False, True], [False, False]],
    [[True, True], [False, False], [False, False]],
    [[True, False], [False, False], [False, False]],
  ]
  total = torch.zeros(NUM_ENVS)
  for landed in schedule:
    _touch_down(env, impact_sensor, feet, landed)
    total += rewards.excess_steps(env, step_cfg) * STEP_DT
  assert total.tolist() == pytest.approx([3.0, 0.0, 0.0])
  assert steps.get_step_counter(env, step_cfg).count.tolist() == [5.0, 2.0, 0.0]


def test_steps_ignore_chatter_and_feet_landing_during_a_tumble():
  env, impact_sensor, feet, step_cfg = _step_env()
  all_feet = [[True, True]] * NUM_ENVS

  # A foot that was off the ground for less than a control step: chatter.
  _touch_down(env, impact_sensor, feet, all_feet, air_time=0.004)
  assert steps.get_step_counter(env, step_cfg).count.tolist() == [0.0, 0.0, 0.0]

  # Env 0 has its torso on the ground: its feet landing is not stepping.
  forces = [[400.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 400.0]]
  _touch_down(env, impact_sensor, feet, all_feet, body_forces=forces)
  assert steps.get_step_counter(env, step_cfg).count.tolist() == [0.0, 2.0, 2.0]


def test_step_count_restarts_with_the_episode():
  env, impact_sensor, feet, step_cfg = _step_env()
  for _ in range(2):
    _touch_down(env, impact_sensor, feet, [[True, True]] * NUM_ENVS)
    steps.get_step_counter(env, step_cfg)  # Read every control step, as rewards do.
  assert steps.get_step_counter(env, step_cfg).count.tolist() == [4.0, 4.0, 4.0]

  env.episode_length_buf[1] = 0  # Env 1 was reset.
  _touch_down(env, impact_sensor, feet, [[True, False]] * NUM_ENVS)
  counter = steps.get_step_counter(env, step_cfg)
  assert counter.count.tolist() == [5.0, 1.0, 5.0]
  assert counter.excess_increment.tolist() == [1.0, 0.0, 1.0]


def test_rest_counts_in_full_only_with_the_chest_guarded():
  identity = [1.0, 0.0, 0.0, 0.0]
  params = {
    "impact_cfg": impact.ImpactCfg(sensor_name="impact", fragility=FRAGILITY),
    "guard_cfg": GUARD_CFG,
    "guard_std": 0.2,
    "unguarded_fraction": 0.3,
  }

  def at_rest(elbow_pos, tip_pos):
    env, impact_sensor = _make_env()
    arm_env, asset = _arm_env(identity, elbow_pos, tip_pos, height=0.4)
    asset.data.root_link_pos_w = asset.data.root_link_pos_w.expand(NUM_ENVS, -1)
    asset.data.root_link_lin_vel_w = torch.zeros(NUM_ENVS, 3)
    asset.data.root_link_ang_vel_w = torch.zeros(NUM_ENVS, 3)
    for name in ("body_link_pos_w", "body_link_quat_w", "site_pos_w"):
      value = getattr(asset.data, name)
      setattr(asset.data, name, value.expand(NUM_ENVS, *value.shape[1:]))
    arm_lookup = asset.find_bodies.side_effect
    asset.find_bodies.side_effect = lambda names, preserve_order=False: (
      ([0, 1, 2], BODIES)
      if set(names) & set(BODIES)
      else arm_lookup(names, preserve_order)
    )
    asset.indexing.body_ids = torch.tensor([1, 2, 3])
    env.scene = _Scene(env.scene, robot=asset)
    _step(env, impact_sensor, [[0.0, 0.0, 0.0]] * NUM_ENVS)
    return rewards.safe_state(env, **params)

  guarded = at_rest(elbow_pos=[0.15, -0.15, 0.2], tip_pos=[0.15, 0.15, 0.2])
  behind = at_rest(elbow_pos=[-0.5, 0.0, 0.2], tip_pos=[-0.5, 0.0, 0.2])
  assert guarded[0].item() == pytest.approx(1.0)
  assert behind[0].item() == pytest.approx(0.3, abs=1e-3)


def test_resting_on_the_pelvis_is_worth_a_fraction(cfg):
  def at_rest(body_forces):
    env, impact_sensor = _make_env()
    asset = env.scene["robot"]
    asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.3]] * NUM_ENVS)
    asset.data.root_link_lin_vel_w = torch.zeros(NUM_ENVS, 3)
    asset.data.root_link_ang_vel_w = torch.zeros(NUM_ENVS, 3)
    env.scene = _Scene(env.scene)
    _step(env, impact_sensor, body_forces)
    return rewards.safe_state(
      env,
      impact_cfg=cfg,
      rest_fraction=1.0,
      contact_rest_fractions={
        r"pelvis_[12]_link": 0.2,
        r"(pelvis_[12]|leg_left_knee)_link": 0.5,
      },
    )

  # Env 0 rests on the knee, env 1 on the pelvis (the smaller fraction of the
  # two that match applies), env 2 on the foot.
  value = at_rest([[0.0, 100.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
  assert value.tolist() == pytest.approx([0.5, 0.2, 1.0])


def test_backward_lean_grows_with_the_torso_tipping_back():
  def lean(pitch):
    # Positive pitch about y tips the torso forwards.
    quat = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
    env, _ = _arm_env(quat, elbow_pos=[0.0, 0.0, 0.0], tip_pos=[0.0, 0.0, 0.0])
    return rewards.backward_lean(env, "torso").item()

  assert lean(0.0) == pytest.approx(0.0, abs=1e-6)
  assert lean(math.radians(60.0)) == pytest.approx(0.0, abs=1e-6)
  assert lean(math.radians(-30.0)) == pytest.approx(0.5, abs=1e-5)
  assert lean(math.radians(-90.0)) == pytest.approx(1.0, abs=1e-5)


def test_each_arm_is_measured_against_the_zone_on_its_own_side():
  identity = [1.0, 0.0, 0.0, 0.0]
  one_sided = {
    "torso_body": "torso",
    "tip_box_min": (0.1, 0.0, 0.0),
    "tip_box_max": (0.5, 0.3, 0.4),
    "mid_box_min": (0.0, 0.1, 0.0),
    "mid_box_max": (0.4, 0.4, 0.4),
    "elbow_bodies": ("elbow",),
    "tip_sites": ("tip",),
  }
  left = arms.ForearmGuardCfg(**one_sided, arm_sides=(1.0,))
  right = arms.ForearmGuardCfg(**one_sided, arm_sides=(-1.0,))

  # Forearm on the left pointing forwards: midpoint (0.2, 0.2, 0.2).
  env, _ = _arm_env(identity, elbow_pos=[0.0, 0.2, 0.2], tip_pos=[0.4, 0.2, 0.2])
  assert arms.forearm_guard_distances(env, left)[0].tolist() == pytest.approx(
    [0.0, 0.0]
  )
  # For a right arm that is the wrong side: 0.3 m and 0.2 m from its zones.
  env, _ = _arm_env(identity, elbow_pos=[0.0, 0.2, 0.2], tip_pos=[0.4, 0.2, 0.2])
  assert arms.forearm_guard_distances(env, right)[0].tolist() == pytest.approx(
    [0.3, 0.2]
  )
  # The mirror image is where a right arm belongs.
  env, _ = _arm_env(identity, elbow_pos=[0.0, -0.2, 0.2], tip_pos=[0.4, -0.2, 0.2])
  assert arms.forearm_guard_distances(env, right)[0].tolist() == pytest.approx(
    [0.0, 0.0]
  )


def test_elbow_first_only_counts_an_elbow_below_its_forearm_near_the_ground():
  identity = [1.0, 0.0, 0.0, 0.0]

  def value(elbow_height, tip_height):
    env, _ = _arm_env(
      identity,
      elbow_pos=[0.0, 0.0, elbow_height],
      tip_pos=[0.3, 0.0, tip_height],
      height=0.0,
    )
    return rewards.elbow_first(env, GUARD_CFG, height_scale=0.15, drop_scale=0.2).item()

  # Elbow on the ground with the forearm pointing up: the full penalty.
  assert value(0.0, 0.3) == pytest.approx(1.0)
  assert value(0.0, 0.1) == pytest.approx(0.5)
  # Forearm level or leading: nothing, however low the arm is.
  assert value(0.0, 0.0) == 0.0
  assert value(0.2, 0.0) == 0.0
  # The same raised forearm high above the ground hardly counts.
  assert value(0.9, 1.2) < 0.01


def test_arm_motion_only_counts_once_the_base_is_at_rest():
  def value(base_speed):
    env, _ = _make_env()
    asset = env.scene["robot"]
    asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.3]] * NUM_ENVS)
    asset.data.root_link_lin_vel_w = torch.tensor([[base_speed, 0.0, 0.0]] * NUM_ENVS)
    asset.data.root_link_ang_vel_w = torch.zeros(NUM_ENVS, 3)
    asset.data.joint_vel = torch.tensor([[2.0, 1.0, 5.0]] * NUM_ENVS)
    env.scene = _Scene(env.scene)
    return rewards.motion_at_rest(
      env, asset_cfg=SimpleNamespace(name="robot", joint_ids=[0, 1])
    )[0].item()

  assert value(0.0) == pytest.approx(5.0)
  assert value(3.0) < 1e-3
