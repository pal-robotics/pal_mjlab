"""Pass contacts between mujoco_warp and Kamino, both ways.

mjw keeps doing collision detection, so geom filtering, margins and friction mixing
stay exactly what the current simulator does. Kamino's solved reactions come back in
mjw's pyramidal ``efc.force`` encoding, which ``sensor_acc`` then decodes unchanged.
"""

import warp as wp
from mujoco_warp._src.types import ContactType

from pal_mjlab.third_party.kamino._src.geometry.contacts import (
  make_contact_frame_znorm,
  reserve_contact_capacity,
)
from pal_mjlab.third_party.kamino._src.geometry.keying import build_pair_key2


@wp.kernel
def _from_mjw(
  # mjw contacts:
  nacon: wp.array[int],
  dist: wp.array[float],
  pos: wp.array[wp.vec3],
  frame: wp.array[wp.mat33],
  includemargin: wp.array[float],
  friction: wp.array(dtype=wp.types.vector(5, float)),
  geom: wp.array[wp.vec2i],
  worldid: wp.array[int],
  ctype: wp.array[int],
  dim: wp.array[int],
  # Layout:
  geom_body: wp.array[int],
  nbody: int,
  ngeom: int,
  model_max: int,
  world_max: int,
  # Kamino contacts:
  model_active: wp.array[wp.int32],
  world_active: wp.array[wp.int32],
  overflow: wp.array[wp.int32],
  wid_out: wp.array[wp.int32],
  cid_out: wp.array[wp.int32],
  gid_ab: wp.array[wp.vec2i],
  bid_ab: wp.array[wp.vec2i],
  position_a: wp.array[wp.vec3f],
  position_b: wp.array[wp.vec3f],
  gapfunc: wp.array[wp.vec4f],
  frame_out: wp.array[wp.quatf],
  material: wp.array[wp.vec2f],
  key: wp.array[wp.uint64],
  remap: wp.array[wp.int32],
):
  c = wp.tid()
  if c >= nacon[0] or not (ctype[c] & int(ContactType.CONSTRAINT)):
    return
  gap = dist[c] - includemargin[c]
  if gap > 0.0:
    return

  w = worldid[c]
  n = wp.vec3f(frame[c][0, 0], frame[c][0, 1], frame[c][0, 2])  # geom0 to geom1.
  half = 0.5 * dist[c] * n
  g0, g1 = geom[c][0], geom[c][1]
  b0, b1 = geom_body[g0], geom_body[g1]
  a, b = wp.vec2i(b0, b1), wp.vec2i(g0, g1)
  if b1 < 0:
    # Kamino wants the moving body as B, with the normal still pointing A to B.
    a, b = wp.vec2i(b1, b0), wp.vec2i(g1, g0)
    n = -n
    half = -half
  if a[1] < 0:
    return

  slot = reserve_contact_capacity(
    model_max, world_max, w, 1, model_active, world_active, overflow
  )
  if slot[0] == 0:
    return
  m = slot[2]
  offset = w * nbody
  wid_out[m] = w
  cid_out[m] = slot[1]
  gid_ab[m] = b
  bid_ab[m] = wp.vec2i(a[0] + offset if a[0] >= 0 else -1, a[1] + offset)
  position_a[m] = pos[c] - half
  position_b[m] = pos[c] + half
  gapfunc[m] = wp.vec4f(n[0], n[1], n[2], gap)
  frame_out[m] = wp.quat_from_matrix(make_contact_frame_znorm(n))
  material[m] = wp.vec2f(
    0.0 if dim[c] == 1 else friction[c][0], 0.0
  )  # condim 1 slides.
  # Keys match contacts for warm starting, so they must be unique across worlds.
  geoms = w * ngeom
  key[m] = build_pair_key2(wp.uint32(geoms + b[0]), wp.uint32(geoms + b[1]))
  remap[m] = c


@wp.kernel
def _to_mjw(
  # Kamino contacts:
  model_active: wp.array[wp.int32],
  gid_ab: wp.array[wp.vec2i],
  frame_in: wp.array[wp.quatf],
  reaction: wp.array[wp.vec3f],
  wid: wp.array[wp.int32],
  remap: wp.array[wp.int32],
  # mjw contacts:
  frame: wp.array[wp.mat33],
  friction: wp.array(dtype=wp.types.vector(5, float)),
  geom: wp.array[wp.vec2i],
  dim: wp.array[int],
  njmax: int,
  # Out:
  rows_used: wp.array[int],
  efc_address: wp.array2d[int],
  efc_force: wp.array2d[float],
):
  m = wp.tid()
  if m >= model_active[0]:
    return
  c = remap[m]
  w = wid[m]
  # Kamino's reaction is the force on its B; mjw's convention is the force on geom1.
  f = wp.quat_rotate(frame_in[m], reaction[m])
  if gid_ab[m][0] == geom[c][1]:
    f = -f
  fn = wp.dot(f, wp.vec3f(frame[c][0, 0], frame[c][0, 1], frame[c][0, 2]))
  rows = 1 if dim[c] == 1 else 4
  adr = wp.atomic_add(rows_used, w, rows)
  if adr + rows > njmax:
    return
  efc_address[c, 0] = adr
  if rows == 1:
    efc_force[w, adr] = fn
    return
  t1 = wp.dot(f, wp.vec3f(frame[c][1, 0], frame[c][1, 1], frame[c][1, 2]))
  t2 = wp.dot(f, wp.vec3f(frame[c][2, 0], frame[c][2, 1], frame[c][2, 2]))
  mu = wp.max(friction[c][0], 1e-6)
  efc_force[w, adr] = 0.25 * fn + 0.5 * t1 / mu
  efc_force[w, adr + 1] = 0.25 * fn - 0.5 * t1 / mu
  efc_force[w, adr + 2] = 0.25 * fn + 0.5 * t2 / mu
  efc_force[w, adr + 3] = 0.25 * fn - 0.5 * t2 / mu


def read_mjw_contacts(mjw_data, geom_body, nbody, ngeom, contacts) -> None:
  """Fill ``contacts`` from what ``mjwarp.collision`` just found."""
  c, k = mjw_data.contact, contacts.data
  contacts.clear()
  wp.launch(
    _from_mjw,
    dim=mjw_data.naconmax,
    inputs=[
      mjw_data.nacon,
      c.dist,
      c.pos,
      c.frame,
      c.includemargin,
      c.friction,
      c.geom,
      c.worldid,
      c.type,
      c.dim,
      geom_body,
      nbody,
      ngeom,
      k.model_max_contacts_host,
      k.world_max_contacts_host[0],
      k.model_active_contacts,
      k.world_active_contacts,
      k._contact_overflow_warning_emitted,
      k.wid,
      k.cid,
      k.gid_AB,
      k.bid_AB,
      k.position_A,
      k.position_B,
      k.gapfunc,
      k.frame,
      k.material,
      k.key,
      k.remap,
    ],
  )


def write_mjw_forces(mjw_data, njmax, rows_used, contacts) -> None:
  """Encode Kamino's reactions as the rows ``contact_force_fn`` reads."""
  c, k = mjw_data.contact, contacts.data
  rows_used.zero_()
  wp.launch(
    _to_mjw,
    dim=k.model_max_contacts_host,
    inputs=[
      k.model_active_contacts,
      k.gid_AB,
      k.frame,
      k.reaction,
      k.wid,
      k.remap,
      c.frame,
      c.friction,
      c.geom,
      c.dim,
      njmax,
      rows_used,
      c.efc_address,
      mjw_data.efc.force,
    ],
  )
