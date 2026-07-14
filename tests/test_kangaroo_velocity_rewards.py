"""Tests for reward manager functionality."""

import math
from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device
from mjlab.envs.mdp import rewards as mjlab_r
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.mdp import rewards as mjlab_vel_r
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


def test_convex_hull_joint_limnits(mock_env, mock_asset_cfg):
  env = mock_env

  env.extras = {"log": {}}

  mock_asset_cfg.joint_ids = [0, 1, 2, 3]

  mock_asset_cfg.joint_names = (
    "joint_left_1",
    "joint_right_1",
    "joint_left_2",
    "joint_right_2",
  )

  asset = env.scene[mock_asset_cfg.name]

  asset.find_joints = lambda names: (
    [0, 2] if names == cfg.params["joint_names_group"][0] else [1, 3],
    list(names),
  )

  cfg = RewardTermCfg(
    func=None,
    weight=1.0,
    params={
      "asset_cfg": mock_asset_cfg,
      "metrics_suffix": "dummy_suffix",
      "joint_names_group": [
        [r"joint_left_1", r"joint_left_2"],
        [r"joint_right_1", r"joint_right_2"],
      ],
      "margin": 0.0,
      "hull_points": torch.tensor(
        [[2.3, 7.1], [5.8, 1.4], [9.2, 6.5], [1.0, 3.8], [6.4, 9.7]]
      ),
    },
  )

  asset.data.joint_pos = torch.tensor(
    [[5.0, 5.0, 5.0, 5.0]] * env.num_envs,
    device=env.device,
    dtype=torch.float32,
  )

  reward_term = pal_mjlab_r.joint_limits_convex_hull(cfg, env, mock_asset_cfg)

  # All inside

  value = reward_term(env, **cfg.params)

  test_name = "Convex hull (all inside)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # All outside

  asset.data.joint_pos = torch.tensor(
    [[5.8, 5.8, 0.4, 0.4]] * env.num_envs,
    device=env.device,
    dtype=torch.float32,
  )

  value = reward_term(env, **cfg.params)

  test_name = "Convex hull (all outside)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(1.6, abs=1e-6), tensor_value_error_message(test_name)

  # On the hull boundary (both groups exactly at vertex (5.8, 1.4))

  asset.data.joint_pos = torch.tensor(
    [[5.8, 5.8, 1.4, 1.4]] * env.num_envs,
    device=env.device,
    dtype=torch.float32,
  )

  value = reward_term(env, **cfg.params)

  test_name = "Convex hull (on the boundary)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Mixed: left group inside at (5.0, 5.0), right group outside at (5.8, 0.4)
  # Only the right group contributes: 0.8

  asset.data.joint_pos = torch.tensor(
    [[5.0, 5.8, 5.0, 0.4]] * env.num_envs,
    device=env.device,
    dtype=torch.float32,
  )

  value = reward_term(env, **cfg.params)

  test_name = "Convex hull (one group inside, one outside)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.8, abs=1e-5), tensor_value_error_message(test_name)

  # Asymmetric violations: left group at (5.8, -0.6) contributes 3.2,
  # right group at (5.8, 0.4) contributes 0.8

  asset.data.joint_pos = torch.tensor(
    [[5.8, 5.8, -0.6, 0.4]] * env.num_envs,
    device=env.device,
    dtype=torch.float32,
  )

  value = reward_term(env, **cfg.params)

  test_name = "Convex hull (asymmetric violations)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(4.0, abs=1e-5), tensor_value_error_message(test_name)

  # Far outside: both groups at the origin (0.0, 0.0)
  # Max facet violation is 3.8460369213 per group -> 2 * 3.8460369213**2

  asset.data.joint_pos = torch.tensor(
    [[0.0, 0.0, 0.0, 0.0]] * env.num_envs,
    device=env.device,
    dtype=torch.float32,
  )

  value = reward_term(env, **cfg.params)

  test_name = "Convex hull (far outside)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(29.584, abs=1e-4), tensor_value_error_message(
    test_name
  )

  # With margin: the hull is shrunk by 0.5, so a point exactly on the original
  # boundary now violates by the margin -> 2 * 0.5**2 = 0.5

  cfg_margin = RewardTermCfg(
    func=None,
    weight=1.0,
    params={**cfg.params, "margin": 0.5, "metrics_suffix": "dummy_suffix_margin"},
  )

  asset.data.joint_pos = torch.tensor(
    [[5.8, 5.8, 1.4, 1.4]] * env.num_envs,
    device=env.device,
    dtype=torch.float32,
  )

  reward_term_margin = pal_mjlab_r.joint_limits_convex_hull(
    cfg_margin, env, mock_asset_cfg
  )

  value = reward_term_margin(env, **cfg_margin.params)

  test_name = "Convex hull (margin, on original boundary)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.5, abs=1e-5), tensor_value_error_message(test_name)

  # With margin: a point deep inside stays penalty-free

  asset.data.joint_pos = torch.tensor(
    [[5.0, 5.0, 5.0, 5.0]] * env.num_envs,
    device=env.device,
    dtype=torch.float32,
  )

  value = reward_term_margin(env, **cfg_margin.params)

  test_name = "Convex hull (margin, deep inside)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)
