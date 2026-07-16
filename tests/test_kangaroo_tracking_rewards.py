import math
from unittest.mock import Mock

import pytest
import torch
from pal_mjlab.tasks.tracking.mdp import rewards as pal_mjlab_r

# -------------------------------------#
#                                      #
#              UTILITIES               #
#                                      #
# -------------------------------------#


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
  return env


@pytest.fixture
def mock_asset_cfg():
  asset_cfg = Mock()
  asset_cfg.name = "robot"

  return asset_cfg


# -------------------------------------#
#                                      #
#            Dummy helpers             #
#                                      #
# -------------------------------------#


class command_manager:
  def __init__(self, env):
    dummy_motion_command = Mock()

    dummy_motion_command.anchor_lin_vel_w = torch.zeros(
      (env.num_envs, 3), device=env.device
    )
    dummy_motion_command.anchor_lin_vel_w[:, 2] += 2.0

    dummy_motion_command.robot_anchor_lin_vel_w = torch.zeros(
      (env.num_envs, 3), device=env.device
    )

    self.active_terms = {
      "motion": dummy_motion_command,
    }

  def get_term(self, command_name):
    if command_name not in self.active_terms.keys():
      return None

    return self.active_terms[command_name]


def tensor_shape_error_message(test_name: str, expected, actual):
  return f"{test_name} returned tensor of wrong shape, expected {expected} got {actual}"


def tensor_value_error_message(test_name: str):
  return f"{test_name} returned incorrect value"


# -------------------------------------#
#                                      #
#                TESTS                 #
#                                      #
# -------------------------------------#


def test_motion_global_anchor_velocity_z_error_exp_reward(mock_env):
  env = mock_env

  env.scene["robot"].data.root_link_lin_vel_b = torch.ones(
    (env.num_envs, 3), device=env.device
  )

  env.command_manager = command_manager(env)

  value = pal_mjlab_r.motion_global_anchor_velocity_z_error_exp(
    env, std=1.0, command_name="motion"
  )

  test_name = "Motion global anchor vel z"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(math.exp(-4.0), abs=1e-6), (
    tensor_value_error_message(test_name)
  )


def test_tracking_feet_air_time_reward(mock_env):
  env = mock_env

  env.scene["contact_sensor"] = Mock()

  # Both air times below threshold

  env.scene["contact_sensor"].data.current_air_time = torch.zeros(
    (
      env.num_envs,
      2,
    ),
    device=env.device,
  )
  env.scene["contact_sensor"].data.current_air_time[:, 0] += 0.2
  env.scene["contact_sensor"].data.current_air_time[:, 1] += 0.4

  value = pal_mjlab_r.feet_air_time(env, sensor_name="contact_sensor", threshold=0.5)

  test_name = "Tracking _ feet air time (both below threshold)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # One air time below threshold

  value = pal_mjlab_r.feet_air_time(env, sensor_name="contact_sensor", threshold=0.3)

  test_name = "Tracking _ feet air time (one below threshold)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.0, abs=1e-6), tensor_value_error_message(test_name)

  # Both air times above threshold

  value = pal_mjlab_r.feet_air_time(env, sensor_name="contact_sensor", threshold=0.1)

  test_name = "Tracking _ feet air time (one below threshold)"

  assert value.shape == (env.num_envs,), tensor_shape_error_message(
    test_name, (env.num_envs,), value.shape
  )
  assert value[0] == pytest.approx(0.3, abs=1e-6), tensor_value_error_message(test_name)
