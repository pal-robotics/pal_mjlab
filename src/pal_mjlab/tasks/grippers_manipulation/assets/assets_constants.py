# THIS FILE SHOULD MANAGE SPAWNING AND CONFIGURATION OF SMALL AND TABLE BOX FOR MULTIPLE ENVS

from pathlib import Path

import mujoco
from mjlab.entity import  EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from pal_mjlab import PAL_MJLAB_SRC_PATH

from mjlab.entity.entity import EntityCfg

ASSETS_PATH = PAL_MJLAB_SRC_PATH / "tasks" / "grippers_manipulation" / "assets"
SMALL_BOX_XML = ASSETS_PATH / "small_box.xml"
TABLE_XML = ASSETS_PATH / "table.xml"

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={".*_collision": 3},
    priority={".*_collision": 1},
    friction={".*_collision": (0.8,)},
)

def _load_spec(xml_path: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  return spec


def get_small_box_spec() -> mujoco.MjSpec:
  return _load_spec(SMALL_BOX_XML)

def get_table_spec() -> mujoco.MjSpec:
  return _load_spec(TABLE_XML)
    
INIT_STATE_BOX = EntityCfg.InitialStateCfg(
  pos=(0.75, 0.0, 1.026)
)

INIT_STATE_TABLE = EntityCfg.InitialStateCfg(
  pos=(0.75, 0.0, 0.0)
)

def get_small_box_cfg() -> EntityCfg:
   spec_fn =  get_small_box_spec
   articulation = None
   collision = FULL_COLLISION
   return EntityCfg(
    init_state=INIT_STATE_BOX,
    collisions=(collision,),
    spec_fn=spec_fn,
    articulation=articulation,
    )

def get_table_cfg() -> EntityCfg:
   spec_fn =  get_table_spec
   articulation = None
   collision = FULL_COLLISION
   return EntityCfg(
    init_state=INIT_STATE_TABLE,
    collisions=(collision,),
    spec_fn=spec_fn,
    articulation=articulation,
    )