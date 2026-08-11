from pathlib import Path

import mujoco

TALOS_XML = (
  Path(__file__).parents[1]
  / "src"
  / "pal_mjlab"
  / "robots"
  / "pal_talos"
  / "xmls"
  / "talos.xml"
)


def test_talos_uses_primitive_collision_geometries():
  model = mujoco.MjModel.from_xml_path(str(TALOS_XML))
  active_collision_geoms = [
    geom_id
    for geom_id in range(model.ngeom)
    if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
  ]

  assert active_collision_geoms
  assert all(
    model.geom_type[geom_id]
    in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE)
    for geom_id in active_collision_geoms
  )


def test_talos_capsules_do_not_collide_in_home_pose():
  model = mujoco.MjModel.from_xml_path(str(TALOS_XML))
  data = mujoco.MjData(model)

  mujoco.mj_forward(model, data)

  assert data.ncon == 0
