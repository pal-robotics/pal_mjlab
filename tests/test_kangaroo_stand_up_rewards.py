"""Unit tests for the stand-up task's contact bookkeeping, rewards and curriculum."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.stand_up import mdp
from pal_mjlab.tasks.stand_up.mdp import contacts

NUM_ENVS = 3
WEIGHT = 100.0 * contacts.GRAVITY
BODIES = ("base_link", "arm_left_4_link", "leg_left_foot_link", "leg_right_foot_link")
DECIMATION = 4


class _Scene:
  def __init__(self, entities: dict):
    self.entities = entities
    self.env_origins = torch.zeros(NUM_ENVS, 3)

  def __getitem__(self, name: str):
    return self.entities[name]


def _forces(rows: list[list[float]]) -> torch.Tensor:
  """Ground force per body in robot weights -> force history in newtons."""
  history = torch.zeros(len(rows), len(BODIES), DECIMATION, 3)
  history[..., 2] = (torch.tensor(rows) * WEIGHT).unsqueeze(-1)
  return history


def _make_env(
  rows: list[list[float]],
  base_height: float | list[float] = 0.9,
  tilt: float | list[float] = 0.0,
  episode_length: int = 10,
):
  """A fake environment whose robot presses on the ground as given by ``rows``."""
  sensor = Mock()
  sensor.primary_names = [f"robot/{name}" for name in BODIES]
  sensor.data = SimpleNamespace(force_history=_forces(rows))

  heights = torch.as_tensor(base_height, dtype=torch.float32).expand(NUM_ENVS)
  tilts = torch.as_tensor(tilt, dtype=torch.float32).expand(NUM_ENVS).tolist()
  robot = Mock()
  robot.indexing = SimpleNamespace(
    body_ids=torch.tensor([1]), ctrl_ids=torch.tensor([0, 1])
  )
  robot.find_bodies = Mock(return_value=([0], ["pelvis_2_link"]))
  robot.data = SimpleNamespace(
    root_link_pos_w=torch.stack(
      [torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), heights], dim=-1
    ),
    # Tilted about the y axis.
    body_link_quat_w=torch.tensor(
      [[[math.cos(t / 2), 0.0, math.sin(t / 2), 0.0]] for t in tilts]
    ),
    gravity_vec_w=torch.tensor([[0.0, 0.0, -1.0]]).expand(NUM_ENVS, -1),
    root_link_lin_vel_w=torch.zeros(NUM_ENVS, 3),
    root_link_ang_vel_w=torch.zeros(NUM_ENVS, 3),
  )
  env = SimpleNamespace(
    num_envs=NUM_ENVS,
    device="cpu",
    common_step_counter=1,
    episode_length_buf=torch.full((NUM_ENVS,), episode_length),
    scene=_Scene({"robot": robot, "ground": sensor}),
    sim=SimpleNamespace(
      mj_model=SimpleNamespace(
        body_mass=torch.tensor([0.0, 100.0]),
        # Two position actuators, kp = 100 and kd = 10, limits 50 and 200, on
        # joints 1 and 2 (joint 0 is the free joint).
        actuator_trnid=np.array([[1, -1], [2, -1]]),
        actuator_gainprm=np.array([[100.0, 0.0, 0.0]] * 2),
        actuator_biasprm=np.array([[0.0, -100.0, -10.0]] * 2),
        actuator_forcerange=np.array([[-50.0, 50.0], [-200.0, 200.0]]),
        jnt_qposadr=np.array([0, 7, 8]),
        jnt_dofadr=np.array([0, 6, 7]),
      )
    ),
  )
  return env, robot


CONTACT_CFG = mdp.GroundContactCfg(
  sensor_name="ground", foot_body_regex=r"leg_(left|right)_foot_link"
)
STAND_CFG = mdp.StandCfg(
  contact_cfg=CONTACT_CFG,
  torso_body="pelvis_2_link",
  target_height=0.9,
  min_height=0.2,
  stand_height=0.8,
)
ON_FEET = [[0.0, 0.0, 0.5, 0.5]]
LYING = [[0.9, 0.1, 0.0, 0.0]]
# Standing on both feet; lying on the base; airborne.
STANDING_LYING_AIRBORNE = ON_FEET + LYING + [[0.0, 0.0, 0.0, 0.0]]


def test_ground_contact_separates_feet_from_the_body():
  env, _ = _make_env(STANDING_LYING_AIRBORNE)
  contact = mdp.get_ground_contact(env, CONTACT_CFG)
  assert contact.is_foot.tolist() == [False, False, True, True]
  assert contact.weight == pytest.approx(WEIGHT)
  assert contact.feet_on_ground().tolist() == [True, False, False]
  assert contact.body_on_ground().tolist() == [False, True, False]
  assert contact.feet_load_fraction().tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_standing_needs_height_an_upright_torso_and_only_the_feet_down():
  # Up; crouching; leaning by 30 degrees.
  env, _ = _make_env(
    ON_FEET * 3, base_height=[0.9, 0.5, 0.9], tilt=[0.0, 0.0, math.radians(30.0)]
  )
  assert mdp.is_standing(env, STAND_CFG).tolist() == [True, False, False]
  # At height and upright, but with a hand on the ground.
  env, _ = _make_env([[0.0, 0.2, 0.4, 0.4]] * 3)
  assert not bool(mdp.is_standing(env, STAND_CFG).any())


def test_progress_terms_grow_towards_standing():
  env, _ = _make_env(ON_FEET * 3, base_height=[0.2, 0.55, 0.9])
  assert mdp.base_height_progress(env, STAND_CFG).tolist() == pytest.approx(
    [0.0, 0.5, 1.0]
  )
  # Upright, horizontal, upside down.
  env, _ = _make_env(ON_FEET * 3, tilt=[0.0, math.pi / 2, math.pi])
  expected = pytest.approx([1.0, 0.25, 0.0], abs=1e-6)
  assert mdp.torso_upright(env, STAND_CFG).tolist() == expected
  assert mdp.base_height_progress(env, STAND_CFG).tolist() == expected
  assert mdp.feet_load(env, STAND_CFG).tolist() == expected


def test_ground_force_excess_only_counts_what_exceeds_resting():
  # Resting on the base; a hit of 3 weights on the base; hopping off the feet.
  rows = [[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
  env, _ = _make_env(rows)
  cost = mdp.ground_force_excess(
    env, CONTACT_CFG, body_threshold=1.0, feet_threshold=1.3
  )
  assert cost.tolist() == pytest.approx([0.0, 2.0, 0.7])
  # Not in the step an episode starts with.
  env, _ = _make_env(rows, episode_length=1)
  cost = mdp.ground_force_excess(
    env, CONTACT_CFG, body_threshold=1.0, feet_threshold=1.3
  )
  assert cost.tolist() == [0.0, 0.0, 0.0]


def test_actuator_overload_keeps_growing_with_a_target_out_of_reach():
  env, robot = _make_env(STANDING_LYING_AIRBORNE)
  # Targets: at the joints; 0.45 rad away (90 % of the first limit); 2 rad
  # away on the first (4 limits) and -2 rad on the second (1 limit).
  ctrl = torch.tensor([[0.0, 0.0], [0.45, 0.0], [2.0, -2.0]])
  env.sim.data = SimpleNamespace(
    ctrl=ctrl, qpos=torch.zeros(NUM_ENVS, 9), qvel=torch.zeros(NUM_ENVS, 8)
  )
  cost = mdp.actuator_overload(env, threshold=0.8)
  assert cost.tolist() == pytest.approx([0.0, 0.1, 3.2 + 0.2])

  robot.data.actuator_force = torch.tensor([[0.0, 0.0], [45.0, 0.0], [-50.0, 200.0]])
  assert mdp.actuator_effort_l2(env).tolist() == pytest.approx([0.0, 0.405, 1.0])


def test_joint_speed_excess_is_free_up_to_the_limit():
  env, robot = _make_env(STANDING_LYING_AIRBORNE)
  robot.find_joints = Mock(return_value=([0, 1], ["leg_length_joint", "hip_joint"]))
  robot.data.joint_vel = torch.tensor([[0.5, -2.0], [1.0, 0.0], [-0.75, 6.0]])
  limits = {"leg_length_joint": 0.5, "hip_joint": 2.0}
  asset_cfg = SceneEntityCfg("robot", joint_names=(".*",))
  cfg = SimpleNamespace(params={"speed_limits": limits, "asset_cfg": asset_cfg})
  cost = mdp.joint_speed_excess(cfg, env)(env, limits, asset_cfg)
  assert cost.tolist() == pytest.approx([0.0, 1.0, 2.5])


def test_action_smoothness_starts_from_the_action_that_holds_the_start_pose():
  env, _ = _make_env(STANDING_LYING_AIRBORNE)
  env.action_manager = SimpleNamespace(
    total_action_dim=2,
    action=torch.ones(NUM_ENVS, 2),
    prev_action=torch.zeros(NUM_ENVS, 2),
    prev_prev_action=torch.zeros(NUM_ENVS, 2),
  )
  mdp.get_start_action(env)[:] = 1.0
  # First, second and a later step of an episode.
  env.episode_length_buf = torch.tensor([1, 2, 10])
  assert mdp.episode_action_rate_l2(env).tolist() == [0.0, 2.0, 2.0]
  assert mdp.episode_action_acc_l2(env).tolist() == [0.0, 8.0, 2.0]

  # Before the first step the policy is shown the start action as its last.
  env.episode_length_buf = torch.tensor([0, 1, 10])
  env.action_manager.action = torch.full((NUM_ENVS, 2), 3.0)
  observed = mdp.last_action_from_start(env)
  assert observed.tolist() == [[1.0, 1.0], [3.0, 3.0], [3.0, 3.0]]


def _make_curriculum(env):
  """A stand-up curriculum on two fake care rewards and a fake lift."""
  rewards = {"a": SimpleNamespace(weight=-1.0), "b": SimpleNamespace(weight=-0.5)}
  lift = SimpleNamespace(params={"level": 0.0})
  env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: rewards[name])
  env.event_manager = SimpleNamespace(get_term_cfg=lambda name: lift)
  params = {
    # Both follow the care scale, "a" up to 4 and "b" up to 2.
    "care_rewards": {"a": 4.0, "b": 2.0},
    "stand_cfg": STAND_CFG,
    "assist_event_name": "lift",
    "assist_level": 0.2,
    "assist_step": 0.1,
    "care_step_factor": 2.0,
    "raise_above": 0.8,
    "lower_below": 0.6,
    "update_interval": 10,
    "window": 1,
  }
  term = mdp.stand_up_curriculum(SimpleNamespace(params=params), env)
  return term, params, rewards, lift


def _run_curriculum(env, term, params, standing: bool, updates: int = 1) -> None:
  """Let episodes end with the robots standing or lying, ``updates`` times."""
  rows = (ON_FEET if standing else LYING) * NUM_ENVS
  env.scene["ground"].data.force_history = _forces(rows)
  for _ in range(updates):
    env.common_step_counter += params["update_interval"]
    term(env, torch.arange(NUM_ENVS), **params)


def test_curriculum_removes_the_lift_then_raises_the_care_rewards():
  env, _ = _make_env(STANDING_LYING_AIRBORNE)
  term, params, rewards, lift = _make_curriculum(env)
  assert lift.params["level"] == pytest.approx(0.2)

  # The robots stand up: first the lift goes ...
  _run_curriculum(env, term, params, standing=True)
  assert lift.params["level"] == pytest.approx(0.1)
  _run_curriculum(env, term, params, standing=True)
  assert lift.params["level"] == 0.0
  assert (rewards["a"].weight, rewards["b"].weight) == (-1.0, -0.5)
  # ... then the care rewards grow, each up to its own maximum.
  _run_curriculum(env, term, params, standing=True, updates=5)
  assert (rewards["a"].weight, rewards["b"].weight) == (-4.0, -1.0)

  # They stop standing up: the care rewards give way, the lift stays away.
  _run_curriculum(env, term, params, standing=False)
  assert (rewards["a"].weight, rewards["b"].weight) == (-2.0, -1.0)
  assert lift.params["level"] == 0.0


def test_curriculum_waits_while_the_robots_fail_with_less_lift():
  env, _ = _make_env(STANDING_LYING_AIRBORNE)
  term, params, rewards, lift = _make_curriculum(env)
  _run_curriculum(env, term, params, standing=True)
  _run_curriculum(env, term, params, standing=False, updates=3)
  assert lift.params["level"] == pytest.approx(0.1)
  assert rewards["a"].weight == -1.0


def test_curriculum_state_round_trip():
  env, _ = _make_env(STANDING_LYING_AIRBORNE)
  term, params, _, _ = _make_curriculum(env)
  _run_curriculum(env, term, params, standing=True, updates=3)
  state = term.state_dict()

  env, _ = _make_env(STANDING_LYING_AIRBORNE)
  term, params, rewards, lift = _make_curriculum(env)
  term.load_state_dict(state)
  assert lift.params["level"] == 0.0
  assert rewards["a"].weight == -2.0
