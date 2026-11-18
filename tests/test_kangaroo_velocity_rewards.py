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
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import rewards as mjlab_vel_r
from mjlab.utils.lab_api.string import resolve_matching_names
from pal_mjlab.robots import (
  KANGAROO_ANKLE_ENVELOPE_LIMITS,
  KANGAROO_HIP_ENVELOPE_LIMITS,
  KANGAROO_TIBIA_PITCH_OFFSET,
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
  """Mock asset with the given joints, resolving names like ``Entity.find_joints``."""
  asset = env.scene["robot"]
  asset.joint_names = joint_names
  asset.num_joints = len(joint_names)
  asset.find_joints = lambda names, preserve_order=False: resolve_matching_names(
    names, joint_names, preserve_order
  )
  asset.data.joint_pos = joint_pos
  ranges = _mjcf_joint_ranges()
  asset.data.default_joint_pos_limits = torch.tensor(
    [[ranges.get(n, (0.0, 0.0)) for n in joint_names]] * joint_pos.shape[0]
  )
  return asset


def test_mjcf_joint_ranges_hold_envelope_limits():
  # The envelope rewards read these ranges from the model; the fit assumed them.
  expected = {
    "2": (-0.712242, 0.747401),  # hip joint 2
    "3": (-0.437830, 0.437830),  # hip joint 3
    "5": (-0.424645, 0.424645),  # ankle joint 5
  }
  for path in KANGAROO_XML.parent.glob("kangaroo*.xml"):
    content = path.read_text()
    for side in ("left", "right"):
      for joint, (low, high) in expected.items():
        assert re.search(
          rf'<joint name="leg_{side}_{joint}_joint"[^>]*range="{low:.6f} {high:.6f}"',
          content,
        ), f"{path.name}: leg_{side}_{joint}_joint range != {low} {high}"


# Ankle roles, in the order the reward expects them.
_ANKLE_ROLES = ("4", "5", "length", "femur", "knee")
_ANKLE_PATTERNS = tuple(rf"leg_.*_{role}_joint" for role in _ANKLE_ROLES)
# Mock joints in model order: one leg after the other, as in the MJCF.
_ANKLE_JOINTS = [
  f"leg_{side}_{role}_joint" for side in ("left", "right") for role in _ANKLE_ROLES
]
# Margins of the velocity and tracking tasks.
_ANKLE_MARGIN = {"joint": 0.02, "actuator": 0.0008, "crank": 0.02}
# (j4, j5, leg length, femur, knee) well inside the ankle envelope:
# tibia = 70 deg + 0.7 - 1.6 = 0.32 rad, so j4n = -0.32 rad.
_ANKLE_INSIDE = (0.0, 0.0, 0.4, 0.7, 1.6)


class _AnkleReward:
  """The ankle reward on a mock asset whose pose is set per call."""

  def __init__(self, env, shrink=None, joint_ranges=None):
    env.extras = {"log": {}}
    self.env = env
    self.asset = _envelope_asset(
      env, _ANKLE_JOINTS, torch.zeros(env.num_envs, len(_ANKLE_JOINTS))
    )
    for name, joint_range in (joint_ranges or {}).items():
      self.asset.data.default_joint_pos_limits[:, _ANKLE_JOINTS.index(name)] = (
        torch.tensor(joint_range)
      )
    asset_cfg = SceneEntityCfg(
      "robot", joint_names=_ANKLE_PATTERNS, preserve_order=True
    )
    asset_cfg.resolve(env.scene)  # As the reward manager does.
    self.params = {
      "asset_cfg": asset_cfg,
      "limits": KANGAROO_ANKLE_ENVELOPE_LIMITS,
      "shrink": shrink or {},
    }
    cfg = RewardTermCfg(func=None, weight=1.0, params=self.params)
    self.term = pal_mjlab_r.ankle_envelope_limits(cfg, env)

  def __call__(self, left, right=_ANKLE_INSIDE):
    """Reward for per-leg (j4, j5, leg length, femur, knee); each value a scalar
    or a tensor of shape (n,). Returns shape (n,)."""
    left = torch.broadcast_tensors(*(torch.as_tensor(q) for q in left))
    right = [torch.full_like(left[0], q) for q in right]
    self.asset.data.joint_pos = torch.stack([*left, *right], dim=-1).reshape(
      -1, len(_ANKLE_JOINTS)
    )
    return self.term(self.env, **self.params)


def _ankle_inside(j4, j5, leg_length, femur, knee, shrink=None):
  """Independent oracle: the ankle limits, with margins, written out as inequalities."""
  shrink = shrink or {}
  margin_joint = shrink.get("joint", 0.0)
  margin_act = shrink.get("actuator", 0.0)
  margin_crank = shrink.get("crank", 0.0)
  limits = KANGAROO_ANKLE_ENVELOPE_LIMITS
  foot_tibia = limits["foot_tibia_collision"]
  act, crank = limits["actuator"], limits["crank_bar_collision"]
  j5_min, j5_max = _mjcf_joint_ranges()["leg_left_5_joint"]

  j4n = j4 - (KANGAROO_TIBIA_PITCH_OFFSET + femur - knee)
  ft_line = foot_tibia["b0j5"] + foot_tibia["dj4ndj5"] * j5.abs()
  ft_margin = margin_joint * (1 + foot_tibia["dj4ndj5"] ** 2) ** 0.5
  cb_margin = margin_crank * (1 + crank["dj4dj5"] ** 2) ** 0.5
  ok = j4n >= foot_tibia["j4n_min"] + margin_joint
  ok &= (j5 >= j5_min + margin_joint) & (j5 <= j5_max - margin_joint)
  ok &= j4n >= ft_line + ft_margin
  ok &= j4 <= (
    crank["b0l"] + crank["dj4dl"] * leg_length - crank["dj4dj5"] * j5.abs() - cb_margin
  )
  for sign in (1.0, -1.0):
    stroke = act["b0j5"] + act["dadj4"] * j4 + sign * act["dadj5"] * j5
    ok &= (stroke >= act["min"] + margin_act) & (stroke <= act["max"] - margin_act)
  return ok


@pytest.mark.parametrize("shrink", [None, _ANKLE_MARGIN])
def test_ankle_envelope_matches_unfolded_limits(mock_env, shrink):
  # Left leg random, right leg inside: the reward is 0 iff the left leg is inside.
  reward = _AnkleReward(mock_env, shrink)
  gen = torch.Generator().manual_seed(0)
  n = 100_000
  low = torch.tensor([-1.0, -0.6, 0.148, 0.174533, 0.349066])  # Femur, knee: MJCF.
  high = torch.tensor([1.0, 0.6, 0.704, 1.22173, 2.44346])
  q = (low + (high - low) * torch.rand(n, 5, generator=gen)).unbind(-1)
  penalty = reward(q)
  inside = _ankle_inside(*q, shrink)
  assert 0.05 < inside.float().mean() < 0.95
  assert torch.all(penalty[inside] == 0)
  # Points within float32 rounding of an edge may land on either side.
  assert not inside[penalty > 1e-5].any()


def test_ankle_envelope_penalty(mock_env):
  reward = _AnkleReward(mock_env)
  j5_max = _mjcf_joint_ranges()["leg_left_5_joint"][1]
  foot_tibia = KANGAROO_ANKLE_ENVELOPE_LIMITS["foot_tibia_collision"]

  value = reward(_ANKLE_INSIDE)
  assert value.shape == (1,)
  assert value[0] == pytest.approx(0.0, abs=1e-6)

  # Right leg past joint 5 min only: the excess [rad].
  value = reward(_ANKLE_INSIDE, (0.0, -0.45, 0.4, 0.7, 1.6))
  assert value[0] == pytest.approx(0.45 - j5_max, abs=1e-6)

  # Knee bent so j4n = -70 deg: past j4n_min and past the foot-tibia line, as a
  # distance in the (j4, j5) plane. At j5 = 0 both rows of that |j5| pair are
  # crossed equally; the pair counts once.
  j4n = -math.radians(70.0)
  j4n_excess = foot_tibia["j4n_min"] - j4n
  line_excess = (foot_tibia["b0j5"] - j4n) / (1 + foot_tibia["dj4ndj5"] ** 2) ** 0.5
  value = reward((0.0, 0.0, 0.4, 0.7, 0.7))
  assert value[0] == pytest.approx(j4n_excess + line_excess, abs=1e-5)
  # The metrics use the largest single violation, not the sum.
  log = mock_env.extras["log"]
  for stat in ("mean", "max"):
    assert log[f"Metrics/joint_envelope_ankle_violation_{stat}"] == pytest.approx(
      max(j4n_excess, line_excess), abs=1e-5
    )

  # Two envs, one inside and one past joint 5: mean and max across envs differ.
  reward((0.0, torch.tensor([0.0, 0.45]), 0.4, 0.7, 1.6))
  assert log["Metrics/joint_envelope_ankle_violation_mean"] == pytest.approx(
    0.5 * (0.45 - j5_max), abs=1e-6
  )
  assert log["Metrics/joint_envelope_ankle_violation_max"] == pytest.approx(
    0.45 - j5_max, abs=1e-6
  )

  # A joint margin of 0.02 rad pulls the joint 5 limit in, on both legs.
  pose = (0.0, 0.41, 0.4, 0.7, 1.6)
  value = _AnkleReward(mock_env, {"joint": 0.02})(pose, pose)
  assert value[0] == pytest.approx(2 * (0.41 - (j5_max - 0.02)), abs=1e-6)


def _ankle_edge_pose(edge: str, distance: float):
  """A pose at a signed distance [rad] past one ankle edge, moved in by its
  _ANKLE_MARGIN, along the edge's normal in the (j4, j5) plane. Every other limit
  stays inactive. Returns (j4, j5, leg length, femur, knee)."""
  limits = KANGAROO_ANKLE_ENVELOPE_LIMITS
  if edge in ("actuator +j5", "actuator -j5"):
    # a4 / a5 max: a = b0j5 + dadj4 j4 +- dadj5 j5 <= max - margin [m].
    act, sign = limits["actuator"], 1.0 if edge.endswith("+j5") else -1.0
    norm = (act["dadj4"] ** 2 + act["dadj5"] ** 2) ** 0.5
    along = (act["max"] - _ANKLE_MARGIN["actuator"] - act["b0j5"]) / norm + distance
    j4, j5 = along * act["dadj4"] / norm, along * sign * act["dadj5"] / norm
    return j4, j5, 0.6, 0.0, KANGAROO_TIBIA_PITCH_OFFSET + 0.3  # Tibia -0.3 rad.
  if edge == "crank-bar":
    # j4 <= b0l + dj4dl L - dj4dj5 j5 - margin * norm, at j5 > 0.
    crank, leg_length, j5 = limits["crank_bar_collision"], 0.3, 0.1
    norm = (1 + crank["dj4dj5"] ** 2) ** 0.5
    j4 = crank["b0l"] + crank["dj4dl"] * leg_length - crank["dj4dj5"] * j5
    j4 -= _ANKLE_MARGIN["crank"] * norm
    j4, j5 = j4 + distance / norm, j5 + distance * crank["dj4dj5"] / norm
    return j4, j5, leg_length, 0.0, KANGAROO_TIBIA_PITCH_OFFSET - 0.3  # Tibia 0.3.
  # Foot-tibia: j4n >= b0j5 + dj4ndj5 j5 + margin * norm, at j5 > 0.
  ft, tibia, j5 = limits["foot_tibia_collision"], 0.9, 0.3
  norm = (1 + ft["dj4ndj5"] ** 2) ** 0.5
  j4n = ft["b0j5"] + ft["dj4ndj5"] * j5 + _ANKLE_MARGIN["joint"] * norm
  j4n, j5 = j4n - distance / norm, j5 + distance * ft["dj4ndj5"] / norm
  return j4n + tibia, j5, 0.4, 0.0, KANGAROO_TIBIA_PITCH_OFFSET - tibia


@pytest.mark.parametrize(
  "edge", ["actuator +j5", "actuator -j5", "crank-bar", "foot-tibia"]
)
@pytest.mark.parametrize("distance", [-0.01, 0.01])
def test_ankle_envelope_edge_distance(mock_env, edge, distance):
  # Along an edge's normal, the penalty is the distance [rad] past the edge moved
  # in by its margin: in metres for the actuators (converted by the row norm),
  # in radians for the collision lines (scaled by their slope).
  reward = _AnkleReward(mock_env, _ANKLE_MARGIN)
  value = reward(_ankle_edge_pose(edge, distance))
  assert value[0] == pytest.approx(max(0.0, distance), abs=1e-6)


def test_ankle_envelope_joint5_range_from_model(mock_env):
  reward = _AnkleReward(mock_env, joint_ranges={"leg_left_5_joint": (-0.1, 0.2)})
  assert reward((0.0, 0.15, 0.4, 0.7, 1.6))[0] == pytest.approx(0.0, abs=1e-6)
  assert reward((0.0, 0.25, 0.4, 0.7, 1.6))[0] == pytest.approx(0.05, abs=1e-6)
  assert reward((0.0, -0.15, 0.4, 0.7, 1.6))[0] == pytest.approx(0.05, abs=1e-6)


_HIP_ROLES = ("2", "3")
_HIP_PATTERNS = tuple(rf"leg_.*_{role}_joint" for role in _HIP_ROLES)
_HIP_JOINTS = [
  f"leg_{side}_{role}_joint" for side in ("left", "right") for role in _HIP_ROLES
]
# Margins of the velocity and tracking tasks.
_HIP_MARGIN = {"joint": 0.02, "actuator": 0.0015}


class _HipReward:
  """The hip reward on a mock asset whose pose is set per call."""

  def __init__(self, env, shrink=None, joint_ranges=None):
    env.extras = {"log": {}}
    self.env = env
    self.asset = _envelope_asset(env, _HIP_JOINTS, torch.zeros(env.num_envs, 4))
    for name, joint_range in (joint_ranges or {}).items():
      self.asset.data.default_joint_pos_limits[:, _HIP_JOINTS.index(name)] = (
        torch.tensor(joint_range)
      )
    asset_cfg = SceneEntityCfg("robot", joint_names=_HIP_PATTERNS, preserve_order=True)
    asset_cfg.resolve(env.scene)  # As the reward manager does.
    self.params = {
      "asset_cfg": asset_cfg,
      "limits": KANGAROO_HIP_ENVELOPE_LIMITS,
      "shrink": shrink or {},
    }
    cfg = RewardTermCfg(func=None, weight=1.0, params=self.params)
    self.term = pal_mjlab_r.hip_envelope_limits(cfg, env)

  def __call__(self, left, right=(0.0, 0.0)):
    """Reward for per-leg (j2, j3); each value a scalar or a tensor of shape (n,).
    Returns shape (n,)."""
    j2, j3 = torch.broadcast_tensors(*(torch.as_tensor(q) for q in left))
    columns = [j2, j3, torch.full_like(j2, right[0]), torch.full_like(j2, right[1])]
    self.asset.data.joint_pos = torch.stack(columns, dim=-1).reshape(-1, 4)
    return self.term(self.env, **self.params)


@pytest.mark.parametrize("shrink", [None, _HIP_MARGIN])
def test_hip_envelope_matches_halfplanes(mock_env, shrink):
  reward = _HipReward(mock_env, shrink)
  margin_joint = (shrink or {}).get("joint", 0.0)
  margin_act = (shrink or {}).get("actuator", 0.0)
  act = KANGAROO_HIP_ENVELOPE_LIMITS["actuator"]
  ranges = _mjcf_joint_ranges()
  j2_min, j2_max = ranges["leg_left_2_joint"]
  j3_min, j3_max = ranges["leg_left_3_joint"]
  gen = torch.Generator().manual_seed(0)
  j2 = -0.9 + 1.8 * torch.rand(100_000, generator=gen)
  j3 = -0.6 + 1.2 * torch.rand(100_000, generator=gen)

  # Independent oracle: summed excess over the eight limits, each in rad.
  norm = (act["dadj2"] ** 2 + act["dadj3"] ** 2) ** 0.5
  excess = [
    j2_min + margin_joint - j2,
    j2 - (j2_max - margin_joint),
    j3_min + margin_joint - j3,
    j3 - (j3_max - margin_joint),
  ]
  for sign in (1.0, -1.0):
    stroke = act["b0j3"] + act["dadj2"] * j2 + sign * act["dadj3"] * j3
    excess += [
      (stroke - (act["max"] - margin_act)) / norm,
      (act["min"] + margin_act - stroke) / norm,
    ]
  expected = torch.stack(excess, dim=-1).clamp(min=0.0).sum(-1)
  torch.testing.assert_close(reward((j2, j3)), expected, atol=2e-6, rtol=1e-5)


def test_hip_envelope_penalty(mock_env):
  reward = _HipReward(mock_env)
  j3_max = _mjcf_joint_ranges()["leg_left_3_joint"][1]

  value = reward((0.0, 0.0))
  assert value.shape == (1,)
  assert value[0] == pytest.approx(0.0, abs=1e-6)

  # Right leg past joint 3 min only: the excess [rad].
  value = reward((0.4, 0.0), (0.0, -0.45))
  assert value[0] == pytest.approx(0.45 - j3_max, abs=1e-6)

  # Both legs past joint 3: the legs add up; the metric is the largest single
  # violation.
  value = reward((0.0, 0.5), (0.0, -0.45))
  assert value[0] == pytest.approx((0.5 - j3_max) + (0.45 - j3_max), abs=1e-6)
  log = mock_env.extras["log"]
  for stat in ("mean", "max"):
    assert log[f"Metrics/joint_envelope_hip_violation_{stat}"] == pytest.approx(
      0.5 - j3_max, abs=1e-6
    )

  # Two envs, one inside and one past joint 3: mean and max across envs differ.
  reward((0.0, torch.tensor([0.0, 0.5])))
  assert log["Metrics/joint_envelope_hip_violation_mean"] == pytest.approx(
    0.5 * (0.5 - j3_max), abs=1e-6
  )
  assert log["Metrics/joint_envelope_hip_violation_max"] == pytest.approx(
    0.5 - j3_max, abs=1e-6
  )


def test_hip_envelope_joint_ranges_from_model(mock_env):
  reward = _HipReward(
    mock_env,
    joint_ranges={"leg_left_2_joint": (-0.2, 0.3), "leg_left_3_joint": (-0.1, 0.2)},
  )
  assert reward((0.25, 0.15))[0] == pytest.approx(0.0, abs=1e-6)
  assert reward((0.35, 0.0))[0] == pytest.approx(0.05, abs=1e-6)
  assert reward((-0.25, 0.0))[0] == pytest.approx(0.05, abs=1e-6)
  assert reward((0.0, 0.25))[0] == pytest.approx(0.05, abs=1e-6)
  assert reward((0.0, -0.15))[0] == pytest.approx(0.05, abs=1e-6)


def test_envelope_patterns_resolve_per_role_on_model():
  # On the real joint order, one pattern per role gives the IDs grouped by role,
  # legs in model order (left before right) within each role.
  model_joints = list(_mjcf_joint_ranges())
  for patterns, roles in ((_ANKLE_PATTERNS, _ANKLE_ROLES), (_HIP_PATTERNS, _HIP_ROLES)):
    _, names = resolve_matching_names(patterns, model_joints, preserve_order=True)
    assert names == [
      f"leg_{side}_{role}_joint" for role in roles for side in ("left", "right")
    ]


@pytest.mark.parametrize(
  "reward, joints, patterns",
  [
    (
      pal_mjlab_r.hip_envelope_limits,
      _HIP_JOINTS,
      (*_HIP_PATTERNS[:1], "leg_left_3_joint"),
    ),
    (pal_mjlab_r.ankle_envelope_limits, _ANKLE_JOINTS, _ANKLE_PATTERNS[:4]),
  ],
)
def test_envelope_rejects_wrong_joint_count(mock_env, reward, joints, patterns):
  _envelope_asset(mock_env, joints, torch.zeros(mock_env.num_envs, len(joints)))
  asset_cfg = SceneEntityCfg("robot", joint_names=patterns, preserve_order=True)
  asset_cfg.resolve(mock_env.scene)
  cfg = RewardTermCfg(func=reward, weight=1.0, params={"asset_cfg": asset_cfg})
  with pytest.raises(ValueError, match="same number of joints"):
    reward(cfg, mock_env)


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
