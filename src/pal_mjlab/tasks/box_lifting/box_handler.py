# THIS FILE SHOULD MANAGE SPAWNING AND CONFIGURATION OF BOX FOR MULTIPLE ENVS

from pathlib import Path

import mujoco
from mjlab.entity import  EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from pal_mjlab import PAL_MJLAB_SRC_PATH

from mjlab.entity.entity import EntityCfg

BOX_PATH = PAL_MJLAB_SRC_PATH / "tasks" / "box_lifting"
BOX_XML = BOX_PATH / "box.xml"


BOX_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={".*_collision": 3},
    priority={".*_collision": 1},
    friction={".*_collision": (0.8,)},
)

def _load_spec(xml_path: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  return spec


def get_box_spec() -> mujoco.MjSpec:
  return _load_spec(BOX_XML)
    
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(1.0, 0.0, 0.0)
)

def get_box_cfg() -> EntityCfg:
   spec_fn =  get_box_spec
   articulation = None
   collision = BOX_COLLISION
   return EntityCfg(
    init_state=INIT_STATE,
    collisions=(collision,),
    spec_fn=spec_fn,
    articulation=articulation,
    )
