"""Tests for reward manager functionality."""

import functools
import math
import re
from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device
from mjlab.envs.mdp import rewards as mjlab_r
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.mdp import rewards as mjlab_vel_r
from pal_mjlab.robots import (
  KANGAROO_ANKLE_ENVELOPE_LIMITS,
  KANGAROO_HIP_ENVELOPE_LIMITS,
  KANGAROO_KNEE_TO_JOINT_4_OFFSET,
)
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  KANGAROO_LOWER_BODY_XML,
  KANGAROO_XML,
)
from pal_mjlab.tasks.velocity.mdp import rewards as pal_mjlab_r

PARTIALLY_ACTUATED_ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.1 0.05" mass="1.0"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="actuated_joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="link1_geom" type="box" size="0.05 0.05 0.2" mass="0.5"/>
        <body name="link2" pos="0 0 0.4">
          <joint name="passive_joint" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom name="link2_geom" type="box" size="0.05 0.05 0.15" mass="0.3"/>
          <body name="link3" pos="0 0 0.3">
            <joint name="actuated_joint2" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
            <geom name="link3_geom" type="box" size="0.05 0.05 0.1" mass="0.2"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


class SimpleTestReward:
  """A simple class-based reward for testing that tracks state."""

  def __init__(self, cfg: RewardTermCfg, env):
    self.num_envs = env.num_envs
    self.device = env.device
    self.current_air_time = torch.zeros((self.num_envs, 1), device=self.device)

  def __call__(self, env, **kwargs):
    self.current_air_time += 0.01
    return torch.ones(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | None = None, env=None):
    if env_ids is not None and len(env_ids) > 0:
      self.current_air_time[env_ids] = 0


class StatelessReward:
  """A stateless class-based reward without reset method."""

  def __init__(self, cfg: RewardTermCfg, env):
    pass

  def __call__(self, env, **kwargs):
    return torch.ones(env.num_envs)


@pytest.fixture
def mock_env():
  """Create a mock environment for testing."""
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  env.step_dt = 0.01
  env.max_episode_length_s = 10.0
  robot = Mock()
  env.scene = {"robot": robot}
  env.command_manager.get_command = Mock(
    return_value=torch.tensor([[1.0, 0.0, 0.0]] * 4)
  )
  return env


@pytest.fixture
def mock_asset_cfg():
  asset_cfg = Mock()
  asset_cfg.name = "robot"

  return asset_cfg


@pytest.fixture
def class_reward_config():
  """Config with a class-based reward."""
  return {
    "term": RewardTermCfg(
      func=SimpleTestReward,
      weight=1.0,
      params={},
    )
  }


@pytest.fixture
def function_reward_config():
  """Config with a function-based reward."""
  return {
    "term": RewardTermCfg(
      func=lambda env: torch.ones(env.num_envs),
      weight=1.0,
      params={},
    )
  }


@pytest.fixture
def stateless_reward_config():
  """Config with a stateless class-based reward."""
  return {
    "term": RewardTermCfg(
      func=StatelessReward,
      weight=1.0,
      params={},
    )
  }


# ---------------------------------------------------------------------------------------------------------#
#                                                                                                         #
#                                              TEST  REWARDS                                              #
#                                                                                                         #
# ---------------------------------------------------------------------------------------------------------#


# -------------------------------------#
#                                     #
#            Dummy helpers            #
#                                     #
# -------------------------------------#


class command_manager:
  def __init__(self, env):
    dummy_vel_command = torch.zeros((env.num_envs, 3), device=env.device)

    self.active_terms = {
      "twist": dummy_vel_command,
    }

  def get_command(self, command_name):
    if command_name not in self.active_terms.keys():
      return None

    return self.active_terms[command_name]


def tensor_shape_error_message(test_name: str, expected, actual):
  return f"{test_name} returned tensor of wrong shape, expected {expected} got {actual}"


def tensor_value_error_message(test_name: str):
  return f"{test_name} returned incorrect value"


# -------------------------------------#
#                                     #
#                TESTS                #
#                                     #
# -------------------------------------#


def test_track_linear_velocity_reward(mock_env, mock_asset_cfg):
  env = mock_env

  env.scene["robot"].data.root_link_lin_vel_b = torch.ones(
    (env.num_envs, 3), device=env.device
  )

  env.command_manager = command_manager(env)

  value = mjlab_vel_r.track_linear_velocity(
    env, std=1.0, command_name="twist", asset_cfg=mock_asset_cfg
  )

  test_name = "Track linear velocity"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(math.exp(-3.0), abs=1e-6), (
    tensor_value_error_message(test_name)
  )


def test_track_angular_velocity_reward(mock_env, mock_asset_cfg):
  env = mock_env

  env.scene["robot"].data.root_link_ang_vel_b = torch.ones(
    (env.num_envs, 3), device=env.device
  )

  env.command_manager = command_manager(env)

  value = mjlab_vel_r.track_angular_velocity(
    env, std=1.0, command_name="twist", asset_cfg=mock_asset_cfg
  )

  test_name = "Track angular velocity"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(math.exp(-3.0), abs=1e-6), (
    tensor_value_error_message(test_name)
  )


def test_upright_no_terrain_sensors_reward(mock_env, mock_asset_cfg):
  env = mock_env

  asset = env.scene[mock_asset_cfg.name]

  cfg = RewardTermCfg(func=None, weight=1.0)

  upright_term = mjlab_vel_r.upright(cfg, env)

  # NO BODY IDS

  mock_asset_cfg.body_ids = []
  asset.data.root_link_quat_w = torch.zeros((env.num_envs, 4), device=env.device)
  asset.data.root_link_quat_w[:, 0] += 1.0
  asset.data.gravity_vec_w = (
    torch.tensor([0.0, 0.0, -1.0], device=env.device)
    .unsqueeze(0)
    .expand(env.num_envs, -1)
  )

  value = upright_term(
    env=env, std=1.0, asset_cfg=mock_asset_cfg, terrain_sensor_names=None
  )

  test_name = "Upright (no terrain sensor, without body ids)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(1.0, abs=1e-6), tensor_value_error_message(test_name)

  # WITH BODY IDS

  mock_asset_cfg.body_ids = [2]
  asset.data.body_link_quat_w = torch.zeros((env.num_envs, 6, 4), device=env.device)
  asset.data.body_link_quat_w[:, :, 0] += 1.0
  asset.data.gravity_vec_w = (
    torch.tensor([0.0, 0.0, -1.0], device=env.device)
    .unsqueeze(0)
    .expand(env.num_envs, -1)
  )

  value = upright_term(
    env=env, std=1.0, asset_cfg=mock_asset_cfg, terrain_sensor_names=None
  )

  test_name = "Upright (no terrain sensor, with body ids)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(1.0, abs=1e-6), tensor_value_error_message(test_name)


def test_pose_reward(mock_env, mock_asset_cfg):
  env = mock_env

  mock_asset_cfg.joint_ids = (0, 1, 2, 3)
  mock_asset_cfg.joint_names = (
    "joint_left_1",
    "joint_right_1",
    "joint_left_2",
    "joint_right_1",
  )

  asset = env.scene[mock_asset_cfg.name]

  asset.data.default_joint_pos = torch.tensor(
    [[0.1, 0.1, 0.2, 0.2]] * env.num_envs, device=env.device
  )

  asset.data.joint_pos = torch.zeros((env.num_envs, 4), device=env.device)

  mock_asset_cfg.joint_names = (
    "joint_left_1",
    "joint_right_1",
    "joint_left_2",
    "joint_right_2",
  )

  asset.find_joints = lambda names: ([0, 1, 2, 3], list(mock_asset_cfg.joint_names))

  cfg = RewardTermCfg(
    func=None,
    weight=1.0,
    params={
      "std_standing": {
        r"joint_.*_1": 0.1,
        r"joint_.*_2": 0.1,
      },
      "std_walking": {
        r"joint_.*_1": 0.2,
        r"joint_.*_2": 0.2,
      },
      "std_running": {
        r"joint_.*_1": 0.4,
        r"joint_.*_2": 0.4,
      },
      "asset_cfg": mock_asset_cfg,
    },
  )

  env.command_manager = command_manager(env)

  pose_term = mjlab_vel_r.variable_posture(cfg, env)

  # Standing

  value = pose_term(
    env=env,
    std_standing=None,
    std_walking=None,
    std_running=None,
    asset_cfg=mock_asset_cfg,
    command_name="twist",
    walking_threshold=0.5,
    running_threshold=1.5,
  )

  test_name = "Pose (standing)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(math.exp(-2.5), abs=1e-6), (
    tensor_value_error_message(test_name)
  )

  # Walking

  env.command_manager.active_terms["twist"][:, 0] += 0.6

  value = pose_term(
    env=env,
    std_standing=None,
    std_walking=None,
    std_running=None,
    asset_cfg=mock_asset_cfg,
    command_name="twist",
    walking_threshold=0.5,
    running_threshold=1.5,
  )

  test_name = "Pose (walking)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(math.exp(-0.625), abs=1e-6), (
    tensor_value_error_message(test_name)
  )

  # Running

  env.command_manager.active_terms["twist"][:, 0] += 2.0

  value = pose_term(
    env=env,
    std_standing=None,
    std_walking=None,
    std_running=None,
    asset_cfg=mock_asset_cfg,
    command_name="twist",
    walking_threshold=0.5,
    running_threshold=1.5,
  )

  test_name = "Pose (running)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(math.exp(-0.15625), abs=1e-6), (
    tensor_value_error_message(test_name)
  )


def test_body_ang_vel_penalty(mock_env, mock_asset_cfg):
  env = mock_env

  mock_asset_cfg.body_ids = 1

  asset = env.scene[mock_asset_cfg.name]
  asset.data.body_link_ang_vel_w = torch.zeros((env.num_envs, 5, 3), device=env.device)

  asset.data.body_link_ang_vel_w[:, mock_asset_cfg.body_ids, 0] += 1.0
  asset.data.body_link_ang_vel_w[:, mock_asset_cfg.body_ids, 2] += 1.0

  value = mjlab_vel_r.body_angular_velocity_penalty(mock_env, mock_asset_cfg)

  test_name = "Body angular velocity penalty"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(1.0, abs=1e-6), tensor_value_error_message(test_name)


def test_angular_momentum_penalty(mock_env, mock_asset_cfg):
  env = mock_env

  env.extras = {"log": {}}

  env.scene["sensor_angmom"] = Mock()
  env.scene["sensor_angmom"].data = torch.zeros((env.num_envs, 3), device=env.device)
  env.scene["sensor_angmom"].data[:, 0] += 1.0

  value = mjlab_vel_r.angular_momentum_penalty(env, "sensor_angmom")

  test_name = "Angular momentum penalty"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(1.0, abs=1e-6), tensor_value_error_message(test_name)


def test_dofs_pos_limits_penalty(mock_env, mock_asset_cfg):
  env = mock_env

  mock_asset_cfg.joint_ids = [0, 1, 2]

  asset = env.scene[mock_asset_cfg.name]

  asset.data.joint_pos = torch.zeros((env.num_envs, 3), device=env.device)
  asset.data.soft_joint_pos_limits = torch.ones((env.num_envs, 3, 2), device=env.device)
  asset.data.soft_joint_pos_limits[:, :, 0] *= -1.0

  value = mjlab_r.joint_pos_limits(env, mock_asset_cfg)

  test_name = "Dofs pos limits"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  asset.data.joint_pos[:, 1] += 1.5
  asset.data.joint_pos[:, 2] -= 1.5

  value = mjlab_r.joint_pos_limits(env, mock_asset_cfg)

  test_name = "Dofs pos limits"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(1.0, abs=1e-6), tensor_value_error_message(test_name)


def test_action_rate_l2_penalty(mock_env):
  env = mock_env

  num_joints = 5

  env.action_manager.action = torch.ones((env.num_envs, num_joints), device=env.device)
  env.action_manager.prev_action = torch.zeros(
    (env.num_envs, num_joints), device=env.device
  )

  value = mjlab_r.action_rate_l2(env)

  test_name = "Action rate l2"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(num_joints, abs=1e-6), tensor_value_error_message(
    test_name
  )


def test_air_time_reward(mock_env):
  env = mock_env

  env.extras = {"log": {}}

  env.scene["sensor"] = Mock()

  env.scene["sensor"].data.current_air_time = torch.zeros(
    (env.num_envs, 2), device=env.device
  )
  env.scene["sensor"].data.current_air_time[:, 1] += 0.25

  env.command_manager = command_manager(env)

  # Inactive

  value = mjlab_vel_r.feet_air_time(env, "sensor", command_name="twist")

  test_name = "Feet air time (inactive)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Active

  env.command_manager.active_terms["twist"][:, 0] += 0.6

  value = mjlab_vel_r.feet_air_time(env, "sensor", command_name="twist")

  test_name = "Feet air time (active)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(1.0, abs=1e-6), tensor_value_error_message(test_name)


def test_foot_clearance_penalty(mock_env, mock_asset_cfg):
  env = mock_env

  env.scene["height_sensor"] = Mock(spec=mjlab_vel_r.TerrainHeightSensor)
  env.scene["height_sensor"].data.heights = torch.zeros(
    (env.num_envs, 2), device=env.device
  )
  env.scene["height_sensor"].data.heights[:, 0] += 0.05

  mock_asset_cfg.site_ids = [0, 1]

  asset = env.scene[mock_asset_cfg.name]
  asset.data.site_lin_vel_w = torch.zeros((env.num_envs, 2, 3), device=env.device)
  asset.data.site_lin_vel_w[:, 0, 0] = 0.2

  env.command_manager = command_manager(env)

  # Inactive

  value = mjlab_vel_r.feet_clearance(
    env, 0.1, "height_sensor", command_name="twist", asset_cfg=mock_asset_cfg
  )

  test_name = "Feet clearance (inactive)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Active

  env.command_manager.active_terms["twist"][:, 0] += 0.6

  value = mjlab_vel_r.feet_clearance(
    env, 0.1, "height_sensor", command_name="twist", asset_cfg=mock_asset_cfg
  )

  test_name = "Feet clearance (active)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.01, abs=1e-6), tensor_value_error_message(
    test_name
  )


def test_foot_swing_height_penalty(mock_env):
  env = mock_env

  env.extras = {"log": {}}

  env.scene["contact_sensor"] = Mock()
  env.scene["contact_sensor"].data.found = torch.zeros(
    (env.num_envs, 2), device=env.device
  )

  env.scene["contact_sensor"].compute_first_contact = lambda dt: torch.zeros(
    (env.num_envs, 2), device=env.device, dtype=torch.bool
  )

  env.scene["height_sensor"] = Mock(spec=mjlab_vel_r.TerrainHeightSensor)
  env.scene["height_sensor"].num_frames = 2
  env.scene["height_sensor"].data.heights = torch.zeros(
    (env.num_envs, 2), device=env.device
  )
  env.scene["height_sensor"].data.heights[:, 0] += 0.05

  cfg = RewardTermCfg(
    func=None,
    weight=1.0,
    params={
      "height_sensor_name": "height_sensor",
    },
  )

  env.command_manager = command_manager(env)

  fsh_term = mjlab_vel_r.feet_swing_height(cfg, env)

  # Inactive

  value = fsh_term(
    env,
    "contact_sensor",
    "height_sensor",
    0.1,
    command_name="twist",
    command_threshold=0.05,
  )

  test_name = "Feet swing height (inactive)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Active (no first contact)

  env.command_manager.active_terms["twist"][:, 0] += 0.6

  value = fsh_term(
    env,
    "contact_sensor",
    "height_sensor",
    0.1,
    command_name="twist",
    command_threshold=0.05,
  )

  test_name = "Feet swing height (active, no first contact)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Active (first contact)

  env.scene["height_sensor"].data.heights[:, 0] = 0.0

  env.scene["contact_sensor"].compute_first_contact = lambda dt: torch.ones(
    (env.num_envs, 2), device=env.device, dtype=torch.bool
  )

  value = fsh_term(
    env,
    "contact_sensor",
    "height_sensor",
    0.1,
    command_name="twist",
    command_threshold=0.05,
  )

  test_name = "Feet swing height (first contact)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(1.25, abs=1e-6), tensor_value_error_message(
    test_name
  )


def test_foot_slip_penalty(mock_env, mock_asset_cfg):
  env = mock_env

  env.extras = {"log": {}}

  mock_asset_cfg.site_ids = [0, 1]

  asset = env.scene[mock_asset_cfg.name]
  asset.data.site_lin_vel_w = torch.zeros((env.num_envs, 2, 3), device=env.device)
  asset.data.site_lin_vel_w[:, 0, 0] = 0.2

  env.command_manager = command_manager(env)

  env.scene["contact_sensor"] = Mock()
  env.scene["contact_sensor"].data.found = torch.zeros(
    (env.num_envs, 2), device=env.device
  )

  # Inactive

  value = mjlab_vel_r.feet_slip(
    env, "contact_sensor", "twist", asset_cfg=mock_asset_cfg
  )

  test_name = "Feet slip (inactive)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Active (no contacts)

  env.command_manager.active_terms["twist"][:, 0] += 0.6

  value = mjlab_vel_r.feet_slip(
    env, "contact_sensor", "twist", asset_cfg=mock_asset_cfg
  )

  test_name = "Feet slip (no contacts)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Active (contacts)

  env.scene["contact_sensor"].data.found = torch.ones(
    (env.num_envs, 2), device=env.device
  )

  value = mjlab_vel_r.feet_slip(
    env, "contact_sensor", "twist", asset_cfg=mock_asset_cfg
  )

  test_name = "Feet slip (contacts)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.04, abs=1e-6), tensor_value_error_message(
    test_name
  )


def test_soft_landing_penalty(mock_env):
  env = mock_env

  env.extras = {"log": {}}

  env.command_manager = command_manager(env)

  env.scene["contact_sensor"] = Mock()
  env.scene["contact_sensor"].data.found = torch.zeros(
    (env.num_envs, 2), device=env.device
  )

  env.scene["contact_sensor"].data.force = torch.zeros(
    (env.num_envs, 2, 3), device=env.device
  )
  env.scene["contact_sensor"].data.force[:, 0, 2] += 10.0
  env.scene["contact_sensor"].data.force[:, 1, 2] += 5.0

  env.scene["contact_sensor"].compute_first_contact = lambda dt: torch.zeros(
    (env.num_envs, 2), device=env.device, dtype=torch.bool
  )

  # Inactive

  value = mjlab_vel_r.soft_landing(env, "contact_sensor", "twist")

  test_name = "Soft landing (inactive)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Active (no first contacts)

  env.command_manager.active_terms["twist"][:, 0] += 0.6

  value = mjlab_vel_r.soft_landing(env, "contact_sensor", "twist")

  test_name = "Soft landing (active)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Active (first contacts)

  env.scene["contact_sensor"].compute_first_contact = lambda dt: torch.ones(
    (env.num_envs, 2), device=env.device, dtype=torch.bool
  )

  value = mjlab_vel_r.soft_landing(env, "contact_sensor", "twist")

  test_name = "Soft landing (active)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(15.0, abs=1e-6), tensor_value_error_message(
    test_name
  )


def test_self_collisions_penalty(mock_env):
  env = mock_env

  n_contact_sites = 6

  env.scene["contact_sensor"] = Mock()
  env.scene["contact_sensor"].data.found = torch.zeros(
    (env.num_envs, n_contact_sites), device=env.device
  )

  # No history (no contacts)

  env.scene["contact_sensor"].data.force_history = None

  value = mjlab_vel_r.self_collision_cost(env, "contact_sensor")

  test_name = "Self collisions (no force history)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # No history (contacts)

  env.scene["contact_sensor"].data.force_history = None
  env.scene["contact_sensor"].data.found[:, 0] = 1.0
  env.scene["contact_sensor"].data.found[:, 1] = 1.0

  value = mjlab_vel_r.self_collision_cost(env, "contact_sensor")

  test_name = "Self collisions (no force history)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(2.0, abs=1e-6), tensor_value_error_message(test_name)

  # With history

  env.scene["contact_sensor"].data.force_history = (
    torch.ones((env.num_envs, n_contact_sites, 5, 3)) * 11.0
  )

  value = mjlab_vel_r.self_collision_cost(env, "contact_sensor")

  test_name = "Self collisions (force history)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(5, abs=1e-6), tensor_value_error_message(test_name)


def test_joint_vel_limits_penalty(mock_env, mock_asset_cfg):
  env = mock_env

  env.extras = {"log": {}}

  mock_asset_cfg.joint_ids = [0, 1, 2, 3]

  mock_asset_cfg.joint_names = (
    "leg_left_1",
    "leg_right_1",
    "leg_left_2",
    "leg_right_2",
  )

  asset = env.scene[mock_asset_cfg.name]

  asset.data.joint_vel = torch.zeros(
    (
      env.num_envs,
      4,
    ),
    device=env.device,
  )

  asset.find_joints = lambda names: ([0, 1, 2, 3], list(mock_asset_cfg.joint_names))

  cfg = RewardTermCfg(
    func=None,
    weight=1.0,
    params={
      "asset_cfg": mock_asset_cfg,
      "velocity_limits": {
        r"leg_.*_1": (-1.0, 1.7),
        r"leg_.*_2": (-2.5, 2.0),
      },
    },
  )

  reward_term = pal_mjlab_r.joint_vel_limits(cfg, env)

  value = reward_term(env, {}, mock_asset_cfg)

  test_name = "Joint vel limits (within the limits)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  asset.data.joint_vel[:, 1] = 1.5
  asset.data.joint_vel[:, 2] = -2.5

  asset.find_joints = lambda names: ([0, 1, 2, 3], list(mock_asset_cfg.joint_names))

  cfg = RewardTermCfg(
    func=None,
    weight=1.0,
    params={
      "asset_cfg": mock_asset_cfg,
      "velocity_limits": {
        r"leg_.*_1": (-1.0, 1.0),
        r"leg_.*_2": (-2.0, 2.0),
      },
    },
  )

  reward_term = pal_mjlab_r.joint_vel_limits(cfg, env)

  value = reward_term(env, {}, mock_asset_cfg)

  test_name = "Joint vel limits (Outside the limits)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  # 0.5 is contribution from leg_right_1 and 0.8 is from leg_left_2
  assert value[0] == pytest.approx(1.3, abs=1e-6), tensor_value_error_message(test_name)


@functools.cache
def _mjcf_joint_ranges() -> dict[str, tuple[float, float]]:
  """Joint ranges of the Kangaroo lower body MJCF."""
  spec = mujoco.MjSpec.from_file(str(KANGAROO_LOWER_BODY_XML))
  # Remove collision exclusions that refer to bodies absent from the lower-body model.
  for exclude in list(spec.excludes):
    spec.delete(exclude)
  model = spec.compile()
  return {
    model.joint(i).name: tuple(model.jnt_range[i].tolist()) for i in range(model.njnt)
  }


def _envelope_asset(env, joint_names: list[str], joint_pos: torch.Tensor):
  asset = env.scene["robot"]
  asset.find_joints = lambda names, preserve_order=False: (
    [joint_names.index(n) for n in names],
    list(names),
  )
  asset.data.joint_pos = joint_pos
  ranges = _mjcf_joint_ranges()
  asset.data.default_joint_pos_limits = torch.tensor(
    [[ranges.get(n, (0.0, 0.0)) for n in joint_names]] * joint_pos.shape[0]
  )
  asset.data.soft_joint_pos_limits = asset.data.default_joint_pos_limits.clone()
  return asset


def test_mjcf_joint_ranges_hold_envelope_limits():
  # Fitted envelope bounds stored as MJCF joint ranges (read by the envelope rewards).
  expected = {
    "2": (-0.712242, 0.747401),  # hip joint 2 min / max
    "3": (-0.437830, 0.437830),  # hip joint 3, symmetric
    "4": (-0.756443, 0.724851),  # ankle j4 extent at j5 = 0 (actuator max / min)
    "5": (-0.424645, 0.424645),  # ankle joint 5, symmetric
  }
  for path in KANGAROO_XML.parent.glob("kangaroo*.xml"):
    content = path.read_text()
    for side in ("left", "right"):
      for joint, (low, high) in expected.items():
        assert re.search(
          rf'<joint name="leg_{side}_{joint}_joint"[^>]*range="{low:.6f} {high:.6f}"',
          content,
        ), f"{path.name}: leg_{side}_{joint}_joint range != {low} {high}"


_ANKLE_JOINTS = [
  "leg_left_4_joint",
  "leg_right_4_joint",
  "leg_left_5_joint",
  "leg_right_5_joint",
  "leg_left_length_joint",
  "leg_right_length_joint",
  "leg_left_knee_joint",
  "leg_right_knee_joint",
]


def _knee_from_tibia(tibia_pitch: torch.Tensor) -> torch.Tensor:
  """Invert tibia_pitch = offset - knee / 2; input and output are radians."""
  return 2.0 * (KANGAROO_KNEE_TO_JOINT_4_OFFSET - tibia_pitch)


def _ankle_asset(env, j4, j5, tibia_pitch, leg_length):
  """Build an ankle mock from per-leg tensors, each with shape (num_envs, 2)."""
  knee = _knee_from_tibia(tibia_pitch)
  return _envelope_asset(
    env, _ANKLE_JOINTS, torch.cat([j4, j5, leg_length, knee], dim=1)
  )


def _ankle_cfg(mock_asset_cfg, shrink: float = 0.0) -> RewardTermCfg:
  return RewardTermCfg(
    func=pal_mjlab_r.ankle_envelope_limits,
    weight=1.0,
    params={
      "asset_cfg": mock_asset_cfg,
      # Per leg: [joint 4, joint 5, leg length, knee].
      "joint_names_group": [
        [
          "leg_left_4_joint",
          "leg_left_5_joint",
          "leg_left_length_joint",
          "leg_left_knee_joint",
        ],
        [
          "leg_right_4_joint",
          "leg_right_5_joint",
          "leg_right_length_joint",
          "leg_right_knee_joint",
        ],
      ],
      "limits": KANGAROO_ANKLE_ENVELOPE_LIMITS,
      "shrink": shrink,
    },
  )


def _ankle_unfolded(j4, j5, tibia_pitch, leg_length):
  """The ankle limits written out per mirror side (no |j5| fold)."""
  limits = KANGAROO_ANKLE_ENVELOPE_LIMITS
  foot_tibia_collision, tibia_bars_crank_collision = (
    limits["foot_tibia_collision"],
    limits["tibia_bars_crank_collision"],
  )
  actuator_min, actuator_max = limits["actuator_min"], limits["actuator_max"]
  ranges = _mjcf_joint_ranges()
  j4_min, j4_max = ranges["leg_right_4_joint"]
  j5_max = ranges["leg_right_5_joint"][1]
  # Normalize joint 4 to the knee using the equivalent angular offset.
  j4_knee_rel = j4 - tibia_pitch
  ok = j4_knee_rel >= foot_tibia_collision["j4_lim"]
  for s in (1, -1):
    ok &= (
      j4_knee_rel
      >= foot_tibia_collision["offset"] + foot_tibia_collision["slope"] * s * j5
    )
    ok &= (
      j4
      <= tibia_bars_crank_collision["offset"]
      + tibia_bars_crank_collision["ll"] * leg_length
      + tibia_bars_crank_collision["slope"] * s * j5
    )
    ok &= j4 <= j4_max + actuator_min["slope"] * s * j5
    ok &= j4 >= j4_min + actuator_max["slope"] * s * j5
    ok &= s * j5 <= j5_max
  return ok


def test_ankle_envelope_limits(mock_env, mock_asset_cfg):
  env = mock_env
  env.extras = {"log": {}}
  j5_max = _mjcf_joint_ranges()["leg_right_5_joint"][1]

  def set_pose(left, right):
    # (j4, j5, tibia_pitch, leg_length) per leg.
    legs = torch.tensor([left, right], dtype=torch.float32).T  # (4, 2)
    rows = legs.unsqueeze(1).expand(4, env.num_envs, 2)
    return _ankle_asset(env, *rows)

  # Tibia pitch 0.32 rad corresponds to knee angle approximately 1.8 rad.
  inside = (-0.05, 0.40, 0.32, 0.6)
  set_pose(inside, inside)
  cfg = _ankle_cfg(mock_asset_cfg)
  term = pal_mjlab_r.ankle_envelope_limits(cfg, env)

  test_name = "Ankle envelope (inside)"
  value = term(env, **cfg.params)
  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Right leg past the joint 5 limit (the only violated limit at this pose).
  set_pose(inside, (-0.05, -0.45, 0.32, 0.6))
  test_name = "Ankle envelope (joint 5 violated on one leg)"
  value = term(env, **cfg.params)
  assert value[0] == pytest.approx((0.45 - j5_max) ** 2, abs=1e-6), (
    tensor_value_error_message(test_name)
  )
  assert env.extras["log"]["Metrics/joint_envelope_ankle"] == pytest.approx(
    0.45 - j5_max, abs=1e-6
  )

  # Change the left knee configuration so normalized joint 4 falls below its limit:
  # j4_knee_rel = j4 - tibia_pitch = -0.05 - 1.0 = -1.05 < j4_lim.
  j4_lim = KANGAROO_ANKLE_ENVELOPE_LIMITS["foot_tibia_collision"]["j4_lim"]
  set_pose((-0.05, 0.0, 1.0, 0.6), inside)
  test_name = "Ankle envelope (joint 4 normalized to the knee)"
  value = term(env, **cfg.params)
  assert value[0] == pytest.approx((j4_lim - (-1.05)) ** 2, abs=1e-5), (
    tensor_value_error_message(test_name)
  )

  # Shrinking by 10 % pulls the joint 5 limit in to 0.9 * j5_max.
  set_pose(inside, inside)
  cfg_shrink = _ankle_cfg(mock_asset_cfg, shrink=0.1)
  term_shrink = pal_mjlab_r.ankle_envelope_limits(cfg_shrink, env)
  test_name = "Ankle envelope (shrink)"
  value = term_shrink(env, **cfg_shrink.params)
  assert value[0] == pytest.approx(2 * (0.40 - 0.9 * j5_max) ** 2, abs=1e-6), (
    tensor_value_error_message(test_name)
  )


def test_ankle_envelope_matches_unfolded_limits(mock_env, mock_asset_cfg):
  env = mock_env
  zeros = torch.zeros(env.num_envs, 2)
  _ankle_asset(env, zeros, zeros, zeros, zeros)
  term = pal_mjlab_r.ankle_envelope_limits(_ankle_cfg(mock_asset_cfg), env)

  gen = torch.Generator().manual_seed(0)
  n = 200_000

  def uniform(lo, hi):
    return lo + (hi - lo) * torch.rand(n, generator=gen)

  # (n / 2, 2): one column per leg. Knee over its MJCF range.
  j4, j5 = uniform(-1.0, 1.0).view(-1, 2), uniform(-0.6, 0.6).view(-1, 2)
  knee, leg_length = (
    uniform(0.349066, 2.44346).view(-1, 2),
    uniform(0.148, 0.704).view(-1, 2),
  )
  tibia_pitch = (
    KANGAROO_KNEE_TO_JOINT_4_OFFSET - 0.5 * knee
  )  # Same arithmetic as the reward.
  inside = term.violation(j4, j5, knee, leg_length, shrink=0.0) <= 0
  assert 0.05 < inside.float().mean() < 0.95
  assert torch.equal(inside, _ankle_unfolded(j4, j5, tibia_pitch, leg_length))
  # Shrinking only removes configurations.
  shrunk = term.violation(j4, j5, knee, leg_length, shrink=0.05) <= 0
  assert not (shrunk & ~inside).any()
  assert shrunk.sum() < inside.sum()


_HIP_JOINTS = [
  "leg_left_2_joint",
  "leg_left_3_joint",
  "leg_right_2_joint",
  "leg_right_3_joint",
]


def _hip_cfg(
  mock_asset_cfg, shrink: float | tuple[float, float] = 0.0
) -> RewardTermCfg:
  return RewardTermCfg(
    func=pal_mjlab_r.hip_envelope_limits,
    weight=1.0,
    params={
      "asset_cfg": mock_asset_cfg,
      "joint_names_group": [
        ["leg_left_2_joint", "leg_left_3_joint"],
        ["leg_right_2_joint", "leg_right_3_joint"],
      ],
      "limits": KANGAROO_HIP_ENVELOPE_LIMITS,
      "shrink": shrink,
    },
  )


def test_hip_envelope_limits(mock_env, mock_asset_cfg):
  env = mock_env
  env.extras = {"log": {}}

  def pose(*row):
    return torch.tensor([row] * env.num_envs, dtype=torch.float32)

  _envelope_asset(env, _HIP_JOINTS, pose(0.0, 0.0, 0.0, 0.0))
  cfg = _hip_cfg(mock_asset_cfg)
  cfg.params["limits"] = _HIP_ACTUATOR_TEST_LIMITS
  term = pal_mjlab_r.hip_envelope_limits(cfg, env)

  test_name = "Hip envelope (inside)"
  value = term(env, **cfg.params)
  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  # Both actuator residuals are clipped to zero at q=0.
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Left past joint 2 max, right past joint 3 (either sign of joint 3).
  env.scene["robot"].data.joint_pos = pose(0.8, 0.0, 0.0, -0.5)
  test_name = "Hip envelope (both legs violated)"
  value = term(env, **cfg.params)
  # Both joint and normalized actuator residuals contribute.
  expected = math.sqrt(
    (0.8 - 0.747401) ** 2 + 2 * (0.0453125 / math.hypot(0.125, 0.0625)) ** 2
  ) + (0.5 - 0.437830)
  assert value[0] == pytest.approx(expected, abs=1e-6), tensor_value_error_message(
    test_name
  )


_HIP_ACTUATOR_TEST_LIMITS = {
  "b0": 0.0078125,
  "J": [[0.125, 0.0625], [0.125, -0.0625]],
  "A": 0.0625,
}


@pytest.mark.parametrize("actuator_index", [0, 1])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("shrink", [0.0, 0.02])
@pytest.mark.parametrize("distance", [-0.01, 0.0, 0.01])
def test_hip_envelope_actuator_boundary_distance(
  mock_env, mock_asset_cfg, actuator_index, sign, shrink, distance
):
  # Move along the unit normal from an original actuator boundary. The
  # reported excess must equal distance + shrink on either side/actuator.
  env = mock_env
  env.extras = {"log": {}}
  cfg = _hip_cfg(mock_asset_cfg, shrink=shrink)
  affine = _HIP_ACTUATOR_TEST_LIMITS
  cfg.params["limits"] = affine
  jacobian = torch.tensor(affine["J"])
  normal = sign * jacobian[actuator_index]
  norm = normal.norm()
  boundary = normal * ((affine["A"] - sign * affine["b0"]) / norm.square())
  q = boundary + distance * normal / norm
  asset = _envelope_asset(
    env, _HIP_JOINTS, torch.cat((q, torch.zeros(2))).repeat(env.num_envs, 1)
  )
  # Isolate actuator constraints for this geometric check.
  asset.data.default_joint_pos_limits[..., 0] = -10
  asset.data.default_joint_pos_limits[..., 1] = 10
  original_q = asset.data.joint_pos.clone()
  term = pal_mjlab_r.hip_envelope_limits(cfg, env)
  expected = max(0.0, distance + shrink)
  value = term(env, **cfg.params)
  torch.testing.assert_close(
    value, torch.full((env.num_envs,), expected), atol=1e-7, rtol=1e-5
  )
  assert env.extras["log"]["Metrics/joint_envelope_hip"] == pytest.approx(
    expected, abs=1e-7
  )
  torch.testing.assert_close(asset.data.joint_pos, original_q)


@pytest.mark.parametrize("shrink", [0.02, (0.02, 0.02), (0.0, 0.02), (0.01, 0.03)])
def test_hip_envelope_actuator_margin(mock_env, mock_asset_cfg, shrink):
  env = mock_env
  env.extras = {"log": {}}
  _envelope_asset(env, _HIP_JOINTS, torch.zeros(env.num_envs, 4))
  cfg = _hip_cfg(mock_asset_cfg, shrink=shrink)
  cfg.params["limits"] = _HIP_ACTUATOR_TEST_LIMITS
  term = pal_mjlab_r.hip_envelope_limits(cfg, env)
  s2, s3 = (shrink, shrink) if isinstance(shrink, float) else shrink
  expected = 0.0625 - math.hypot(0.125 * s2, 0.0625 * s3)
  torch.testing.assert_close(term._actuator_max, torch.full((2,), expected))
  assert torch.all(term._actuator_max <= 0.0625)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_hip_envelope_joint_only_violation(mock_env, mock_asset_cfg, sign):
  env = mock_env
  env.extras = {"log": {}}
  _envelope_asset(
    env, _HIP_JOINTS, torch.tensor([[0.0, sign * 0.5, 0.0, 0.0]] * env.num_envs)
  )
  cfg = _hip_cfg(mock_asset_cfg, shrink=0.02)
  term = pal_mjlab_r.hip_envelope_limits(cfg, env)
  torch.testing.assert_close(
    term(env, **cfg.params), torch.full((env.num_envs,), 0.5 - (0.437830 - 0.02))
  )


def test_hip_envelope_precomputes_joint_and_actuator_limits(mock_env, mock_asset_cfg):
  env = mock_env
  env.extras = {"log": {}}
  joint_pos = torch.tensor([[0.4, 0.2, -0.2, -0.1]] * env.num_envs)
  original_joint_pos = joint_pos.clone()
  asset = _envelope_asset(env, _HIP_JOINTS, joint_pos)
  asset.data.soft_joint_pos_limits = None
  model_limits = asset.data.default_joint_pos_limits
  model_limits[1, 0] = torch.tensor([-0.4, 0.5])
  model_limits[2, 3] = torch.tensor([-0.2, 0.3])
  original_limits = model_limits.clone()
  cfg = _hip_cfg(mock_asset_cfg, shrink=(0.02, 0.03))
  cfg.params["limits"] = _HIP_ACTUATOR_TEST_LIMITS
  term = pal_mjlab_r.hip_envelope_limits(cfg, env)

  margin = torch.tensor([0.02, 0.03])
  expected_lower = original_limits[..., 0].reshape(env.num_envs, 2, 2) + margin
  expected_upper = original_limits[..., 1].reshape(env.num_envs, 2, 2) - margin
  expected_actuator = torch.full((2,), 0.0625 - math.hypot(0.125 * 0.02, 0.0625 * 0.03))
  torch.testing.assert_close(term._joint_min, expected_lower)
  torch.testing.assert_close(term._joint_max, expected_upper)
  torch.testing.assert_close(term._actuator_max, expected_actuator)
  torch.testing.assert_close(model_limits, original_limits)
  baseline = term(env, **cfg.params)
  assert torch.all(baseline > 0)

  # All bounds remain fixed after initialization, including call-time shrink.
  model_limits.zero_()
  cfg.params["shrink"] = (0.2, 0.1)
  cfg.params["limits"] = {**_HIP_ACTUATOR_TEST_LIMITS, "A": 1.0}
  torch.testing.assert_close(term._joint_min, expected_lower)
  torch.testing.assert_close(term._joint_max, expected_upper)
  torch.testing.assert_close(term._actuator_max, expected_actuator)
  torch.testing.assert_close(term(env, **cfg.params), baseline)
  assert torch.equal(asset.data.joint_pos, original_joint_pos)


def test_hip_envelope_without_soft_limits(mock_env, mock_asset_cfg):
  env = mock_env
  env.extras = {"log": {}}
  asset = _envelope_asset(env, _HIP_JOINTS, torch.zeros(env.num_envs, 4))
  asset.data.soft_joint_pos_limits = None
  cfg = _hip_cfg(mock_asset_cfg, shrink=0.02)
  term = pal_mjlab_r.hip_envelope_limits(cfg, env)
  torch.testing.assert_close(term(env, **cfg.params), torch.zeros(env.num_envs))


def test_hip_envelope_asymmetric_joint3_range(mock_env, mock_asset_cfg):
  env = mock_env
  env.extras = {"log": {}}
  asset = _envelope_asset(env, _HIP_JOINTS, torch.zeros(env.num_envs, 4))
  for joint_id in (1, 3):
    asset.data.default_joint_pos_limits[:, joint_id] = torch.tensor([-0.1, 0.3])
  original_limits = asset.data.default_joint_pos_limits.clone()
  cfg = _hip_cfg(mock_asset_cfg, shrink=(0.01, 0.02))
  term = pal_mjlab_r.hip_envelope_limits(cfg, env)

  # Fixed radian margins move each endpoint equally, without recentering.
  torch.testing.assert_close(
    term._joint_min[..., 1], torch.full((env.num_envs, 2), -0.08)
  )
  torch.testing.assert_close(
    term._joint_max[..., 1], torch.full((env.num_envs, 2), 0.28)
  )
  torch.testing.assert_close(asset.data.default_joint_pos_limits, original_limits)


@pytest.mark.parametrize(
  "affine",
  [
    KANGAROO_HIP_ENVELOPE_LIMITS,
    {"b0": 0.001, "J": [[0.06, 0.025], [0.06, -0.025]], "A": 0.03},
    {"b0": 0.001, "J": [[0.06, 0.025], [0.04, -0.035]], "A": 0.03},
    {"b0": 0.001, "J": [[0.0, 0.025], [0.04, -0.035]], "A": 0.03},
  ],
)
@pytest.mark.parametrize("shrink", [0.0, 0.02, (0.01, 0.03)])
def test_hip_envelope_matches_joint_space_halfplanes(
  mock_env, mock_asset_cfg, affine, shrink
):
  env = mock_env
  env.extras = {"log": {}}
  gen = torch.Generator().manual_seed(0)
  n = 10_000
  joint_pos = torch.stack(
    [
      -0.9 + 1.8 * torch.rand(n, generator=gen),
      -0.6 + 1.2 * torch.rand(n, generator=gen),
      -0.9 + 1.8 * torch.rand(n, generator=gen),
      -0.6 + 1.2 * torch.rand(n, generator=gen),
    ],
    dim=1,
  )
  asset = _envelope_asset(env, _HIP_JOINTS, joint_pos)
  cfg = _hip_cfg(mock_asset_cfg, shrink=shrink)
  cfg.params["limits"] = affine
  term = pal_mjlab_r.hip_envelope_limits(cfg, env)
  value = term(env, **cfg.params)

  # Independent oracle: eight explicit affine halfplanes in joint space.
  # Normalize first, then shift each boundary by its joint-space clearance.
  jacobian = joint_pos.new_tensor(affine["J"])
  normals = torch.cat((torch.eye(2), -torch.eye(2), jacobian, -jacobian))
  norms = normals.norm(dim=-1)
  unit_normals = normals / norms[:, None]
  bounds = asset.data.default_joint_pos_limits.reshape(n, 2, 2, 2)
  offsets = torch.cat(
    (
      bounds[..., 1],
      -bounds[..., 0],
      torch.full((n, 2, 2), affine["A"] - affine["b0"]),
      torch.full((n, 2, 2), affine["A"] + affine["b0"]),
    ),
    dim=-1,
  )
  margin = joint_pos.new_tensor(shrink).expand(2)
  clearance = (unit_normals * margin).norm(dim=-1)
  residual = (
    joint_pos.reshape(n, 2, 2) @ unit_normals.T - offsets / norms + clearance
  ).clamp(min=0)
  expected = residual.norm(dim=-1).sum(dim=1)
  torch.testing.assert_close(value, expected, atol=2e-6, rtol=1e-5)
  assert torch.all(value >= 0)


@pytest.mark.parametrize("shrink", [-0.01, (0.01, -0.02), float("nan"), float("inf")])
def test_hip_envelope_rejects_invalid_shrink(mock_env, mock_asset_cfg, shrink):
  _envelope_asset(mock_env, _HIP_JOINTS, torch.zeros(mock_env.num_envs, 4))
  with pytest.raises(ValueError, match="shrink must be finite and nonnegative"):
    pal_mjlab_r.hip_envelope_limits(_hip_cfg(mock_asset_cfg, shrink), mock_env)


def test_track_body_ang_vel_z_exp_reward(mock_env, mock_asset_cfg):
  env = mock_env

  asset = env.scene[mock_asset_cfg.name]
  asset.data.root_link_ang_vel_b = torch.ones((env.num_envs, 3), device=env.device)

  env.command_manager = command_manager(env)

  value = pal_mjlab_r.track_body_ang_vel_z_exp(
    env, std=1.0, command_name="twist", asset_cfg=mock_asset_cfg
  )

  # Only the yaw error counts: the coupled mjlab reward would give exp(-3.0) here.
  test_name = "Track body angular velocity z (decoupled from roll/pitch)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(math.exp(-1.0), abs=1e-6), (
    tensor_value_error_message(test_name)
  )

  # Perfect yaw tracking is fully rewarded even while roll/pitch rates are non-zero.

  env.command_manager.active_terms["twist"][:, 2] = 1.0

  value = pal_mjlab_r.track_body_ang_vel_z_exp(
    env, std=1.0, command_name="twist", asset_cfg=mock_asset_cfg
  )

  test_name = "Track body angular velocity z (perfect yaw tracking)"

  assert value[0] == pytest.approx(1.0, abs=1e-6), tensor_value_error_message(test_name)

  # The kernel width scales with std**2, not std.

  env.command_manager.active_terms["twist"][:, 2] = 0.0

  value = pal_mjlab_r.track_body_ang_vel_z_exp(
    env, std=0.5, command_name="twist", asset_cfg=mock_asset_cfg
  )

  test_name = "Track body angular velocity z (std scaling)"

  assert value[0] == pytest.approx(math.exp(-4.0), abs=1e-6), (
    tensor_value_error_message(test_name)
  )

  # A missing command is caught instead of silently propagating None.

  with pytest.raises(AssertionError):
    pal_mjlab_r.track_body_ang_vel_z_exp(
      env, std=1.0, command_name="nonexistent", asset_cfg=mock_asset_cfg
    )


def test_body_ang_vel_xy_l2_penalty(mock_env, mock_asset_cfg):
  env = mock_env

  asset = env.scene[mock_asset_cfg.name]
  asset.data.root_link_ang_vel_b = torch.tensor(
    [[1.0, 2.0, 10.0]] * env.num_envs, device=env.device
  )

  value = pal_mjlab_r.body_ang_vel_xy_l2_penalty(env, mock_asset_cfg)

  # Only roll/pitch are penalised: the large yaw rate must not leak in.
  test_name = "Body angular velocity xy penalty (yaw excluded)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(5.0, abs=1e-6), tensor_value_error_message(test_name)

  # No angular velocity, no penalty.

  asset.data.root_link_ang_vel_b = torch.zeros((env.num_envs, 3), device=env.device)

  value = pal_mjlab_r.body_ang_vel_xy_l2_penalty(env, mock_asset_cfg)

  test_name = "Body angular velocity xy penalty (zero rates)"

  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)
