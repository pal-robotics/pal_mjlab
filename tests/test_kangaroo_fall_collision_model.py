"""Tests for the mesh-accurate collision model used by the safe-fall task."""

import math
import re

import mujoco
import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Resolves the mjlab/pal_mjlab import order.
import pytest
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg
from pal_mjlab.robots import (
  FALL_COLLISION_MESH_SUFFIX,
  FALL_TERRAIN_COLLISION,
  get_kangaroo_fall_robot_cfg,
)

SELF_BIT = 1
GROUND_BIT = 2
SOLE_REGEX = re.compile(r"robot/(left|right)_foot\d+_collision$")
ANKLE_REGEX = re.compile(r"robot/leg_(left|right)_ankle(_bar)?_collision$")
FOOT_AVOIDANCE_REGEX = re.compile(r"robot/(left|right)_foot_avoidance_collision_\d+$")

# Bodies whose visual mesh must be able to touch the ground. The feet are
# represented by their sole and ankle capsules instead.
MESH_BODIES = (
  ["base_link", "pelvis_1_link", "pelvis_2_link"]
  + [f"arm_{s}_{i}_link" for s in ("left", "right") for i in (1, 2, 3, 4)]
  + [
    f"leg_{s}_{part}_link"
    for s in ("left", "right")
    for part in (1, 2, 3, 4, "femur", "knee")
  ]
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  cfg = SceneCfg(
    num_envs=1,
    terrain=TerrainEntityCfg(
      terrain_type="plane", collisions=(FALL_TERRAIN_COLLISION,)
    ),
    entities={"robot": get_kangaroo_fall_robot_cfg()},
  )
  model = Scene(cfg, device="cpu").compile()
  model.opt.timestep = 0.002
  return model


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
  return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def _is_mesh(name: str) -> bool:
  return name.endswith(FALL_COLLISION_MESH_SUFFIX)


def _touches_ground(name: str) -> bool:
  return bool(_is_mesh(name) or SOLE_REGEX.match(name) or ANKLE_REGEX.match(name))


def _reset_to_default(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  height: float,
  quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
  """Default pose at the given base pose, held by the position actuators."""
  init = get_kangaroo_fall_robot_cfg().init_state
  mujoco.mj_resetData(model, data)
  data.qpos[0:3] = (0.0, 0.0, height)
  data.qpos[3:7] = quat
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    for pattern, value in init.joint_pos.items():
      if re.fullmatch(pattern, name.split("/")[-1]):
        data.qpos[model.jnt_qposadr[joint_id]] = value
  for act_id in range(model.nu):
    joint_id = model.actuator_trnid[act_id, 0]
    data.ctrl[act_id] = data.qpos[model.jnt_qposadr[joint_id]]
  mujoco.mj_forward(model, data)


def _terrain_contact_geoms(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
  names = []
  for contact in data.contact[: data.ncon]:
    pair = (_geom_name(model, contact.geom1), _geom_name(model, contact.geom2))
    if "terrain" in pair:
      names.append(pair[0] if pair[1] == "terrain" else pair[1])
  return names


def _lowest_visual_vertex(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  """Lowest world z over the vertices of every visual link mesh."""
  lowest = math.inf
  for geom_id in range(model.ngeom):
    is_visual_mesh = (
      model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
      and model.geom_group[geom_id] == 2
    )
    if not is_visual_mesh:
      continue
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    world_z = (vertices @ rotation.T + data.geom_xpos[geom_id])[:, 2]
    lowest = min(lowest, float(world_z.min()))
  return lowest


def test_every_link_has_a_collision_mesh(model):
  for body in MESH_BODIES:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{body}")
    assert body_id >= 0, body
    meshes = [
      g
      for g in range(model.ngeom)
      if model.geom_bodyid[g] == body_id and _is_mesh(_geom_name(model, g))
    ]
    assert meshes, f"{body} has no collision mesh"
    for g in meshes:
      assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
      assert model.mesh_graphadr[model.geom_dataid[g]] >= 0  # Convex hull built.


def test_collision_layers(model):
  terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  assert model.geom_contype[terrain] == GROUND_BIT
  assert model.geom_conaffinity[terrain] == 0

  for g in range(model.ngeom):
    name = _geom_name(model, g)
    contype, conaffinity = model.geom_contype[g], model.geom_conaffinity[g]
    if g == terrain or not (contype or conaffinity):
      continue
    if _is_mesh(name):
      # Link shell: touches the terrain and the other links.
      assert (contype, conaffinity) == (SELF_BIT, SELF_BIT | GROUND_BIT), name
      assert model.geom_condim[g] == 3, name
    elif SOLE_REGEX.match(name) or ANKLE_REGEX.match(name):
      # Foot: touches the terrain only.
      assert (contype, conaffinity) == (0, GROUND_BIT), name
      assert model.geom_condim[g] == 3, name
    elif FOOT_AVOIDANCE_REGEX.match(name):
      # Touches the robot only, never the terrain.
      assert (contype, conaffinity) == (SELF_BIT, SELF_BIT), name
    else:
      pytest.fail(f"Unexpected active collision geom '{name}'")


def test_default_pose_stands_on_the_soles_only(model):
  data = mujoco.MjData(model)
  _reset_to_default(model, data, height=0.9)
  touching = _terrain_contact_geoms(model, data)
  assert touching
  assert all(SOLE_REGEX.match(name) for name in touching), touching
  # No self-collision in the default pose either.
  assert len(touching) == data.ncon


@pytest.mark.parametrize(
  "axis, angle",
  [
    ((0.0, 1.0, 0.0), math.pi / 2),  # Face down.
    ((0.0, 1.0, 0.0), -math.pi / 2),  # On the back.
    ((1.0, 0.0, 0.0), -math.pi / 2),  # Left side.
    ((1.0, 0.0, 0.0), math.pi / 2),  # Right side.
  ],
)
def test_lying_robot_rests_on_its_mesh_surface(model, axis, angle):
  """The visual shell neither sinks into the floor nor hovers above it."""
  data = mujoco.MjData(model)
  quat = (math.cos(angle / 2), *(np.array(axis) * math.sin(angle / 2)))
  _reset_to_default(model, data, height=0.8, quat=quat)
  for _ in range(2500):  # 5 s: fall and settle.
    mujoco.mj_step(model, data)

  touching = _terrain_contact_geoms(model, data)
  assert touching
  assert all(_touches_ground(name) for name in touching), touching
  assert any(_is_mesh(name) for name in touching)

  lowest = _lowest_visual_vertex(model, data)
  assert -0.003 < lowest < 0.003, f"lowest visual vertex at {lowest:.4f} m"


def test_forearm_can_press_against_the_torso(model):
  """A forearm folded across the chest collides with the torso shell.

  The safe-fall task relies on this: a forearm between the chest and the
  ground can only shield the torso if the two actually collide.
  """
  data = mujoco.MjData(model)
  _reset_to_default(model, data, height=1.5)
  folded = {
    "arm_left_1_joint": -1.482,
    "arm_left_2_joint": 1.357,
    "arm_left_3_joint": -2.585,
    "arm_left_4_joint": 2.259,
  }
  for name, value in folded.items():
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
    data.qpos[model.jnt_qposadr[joint_id]] = value
  mujoco.mj_forward(model, data)

  def body(geom_id: int) -> str:
    return mujoco.mj_id2name(
      model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]
    )

  pairs = {frozenset((body(c.geom1), body(c.geom2))) for c in data.contact[: data.ncon]}
  assert frozenset(("robot/arm_left_4_link", "robot/pelvis_2_link")) in pairs
