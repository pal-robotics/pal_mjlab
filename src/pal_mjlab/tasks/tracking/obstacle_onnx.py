"""ONNX export of the box-drop tracking task: the motion retimed in the graph.

The plain tracking export bundles the motion and looks it up by ``time_step``. Here the
graph holds the reference motion once and also takes ``obstacle_height`` (m, [1, 1]):
it retimes the reference for that height frame by frame, as ``make_height_variant``
does in training, so any height works, not only the trained ones. Inputs and outputs
are otherwise the plain export's:

  inputs   obs [1, n], time_step [1, 1], obstacle_height [1, 1]
  outputs  actions, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w,
           body_ang_vel_w

``time_step`` counts frames from the start of the (retimed) motion; past its end the
last frame is held. Training covers the heights of ``obstacle_heights_trained``
(metadata); outside them the motion is still well defined, the policy is not trained.

Patch an export of the old format (``time_step`` only):

  python -m pal_mjlab.tasks.tracking.obstacle_onnx IN.onnx OUT.onnx MOTION.npz
"""

from __future__ import annotations

import argparse
import os
from typing import cast

import onnx
import torch
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from torch import nn

from pal_mjlab.tasks.tracking.mdp.obstacle_commands import (
  DescentParams,
  ObstacleDropMotionCommand,
)

MOTION_OUTPUTS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


class ObstacleDropMotion(nn.Module):
  """``(time_step, obstacle_height) -> reference frame``, the reference retimed.

  Frame by frame what ``make_height_variant`` builds as a whole: before the apex, the
  reference shifted by ``dh``; over the descent, the reference time-warped onto the
  parabola for the new drop; after the touchdown, the reference unchanged.
  """

  def __init__(
    self,
    reference: dict[str, torch.Tensor],
    descent: DescentParams,
    reference_obstacle_height: float,
    dt: float,
  ) -> None:
    super().__init__()
    for name in MOTION_OUTPUTS:
      self.register_buffer(name, reference[name].detach().float().cpu().clone())
    self.n_frames: int = reference["joint_pos"].shape[0]
    self.d = descent
    self.h_ref = reference_obstacle_height
    self.dt = dt

  def forward(self, time_step: torch.Tensor, obstacle_height: torch.Tensor):
    d, dt = self.d, self.dt
    t0, t1 = float(d.apex), float(d.touchdown)
    j = time_step.squeeze(-1).floor().clamp(min=0.0)  # [B], whole frames.
    dh = obstacle_height.squeeze(-1) - self.h_ref

    # Descent duration for the new drop (make_height_variant).
    drop = d.z_a + dh - d.z_b
    disc = (d.v_a**2 + 2.0 * d.acc0 * drop).clamp(min=0.0)
    n_flight = torch.round((d.v_a + disc.sqrt()) / d.acc0 / dt).clamp(min=1.0)
    dur = n_flight * dt
    acc = 2.0 * (drop + d.v_a * dur) / dur**2
    speed = d.dur0 / dur

    pre = j < t0
    flight = (j >= t0) & (j < t0 + n_flight)
    jj = j - t0
    src = torch.where(
      pre,
      j,
      torch.where(flight, t0 + jj * (t1 - t0) / n_flight, j - t0 - n_flight + t1),
    ).clamp(max=float(self.n_frames - 1))
    i0 = src.floor().long().clamp(max=self.n_frames - 1)
    i1 = (i0 + 1).clamp(max=self.n_frames - 1)
    a = src - i0.float()
    vel_scale = torch.where(flight, speed, torch.ones_like(speed))

    tau = jj * dt
    t_src = (src - t0) * dt
    dz_flight = (d.z_a + dh + d.v_a * tau - 0.5 * acc * tau**2) - (
      d.z_a + d.v_a * t_src - 0.5 * d.acc0 * t_src**2
    )
    dz = torch.where(pre, dh, torch.where(flight, dz_flight, torch.zeros_like(dh)))
    dvz_flight = (d.v_a - acc * tau) - (d.v_a - d.acc0 * t_src) * speed
    dvz = torch.where(flight, dvz_flight, torch.zeros_like(dh))

    def lerp(x: torch.Tensor) -> torch.Tensor:
      w = a.view(-1, *([1] * (x.dim() - 1)))
      return x[i0] * (1.0 - w) + x[i1] * w

    def per_env(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
      return v.view(-1, *([1] * (x.dim() - 1)))

    jp = cast(torch.Tensor, self.joint_pos)
    jv = cast(torch.Tensor, self.joint_vel)
    bp = cast(torch.Tensor, self.body_pos_w)
    bq = cast(torch.Tensor, self.body_quat_w)
    blv = cast(torch.Tensor, self.body_lin_vel_w)
    bav = cast(torch.Tensor, self.body_ang_vel_w)

    body_pos = lerp(bp)
    body_pos = body_pos + torch.stack(
      [
        torch.zeros_like(body_pos[..., 0]),
        torch.zeros_like(body_pos[..., 0]),
        per_env(dz, body_pos[..., 0]).expand_as(body_pos[..., 0]),
      ],
      dim=-1,
    )
    lin_vel = lerp(blv) * per_env(vel_scale, blv[i0])
    lin_vel = lin_vel + torch.stack(
      [
        torch.zeros_like(lin_vel[..., 0]),
        torch.zeros_like(lin_vel[..., 0]),
        per_env(dvz, lin_vel[..., 0]).expand_as(lin_vel[..., 0]),
      ],
      dim=-1,
    )

    # Quaternions: nlerp over the descent, the frame as is elsewhere.
    q0, q1 = bq[i0], bq[i1]
    q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0.0, -q1, q1)
    w = per_env(a, q0)
    q = q0 * (1.0 - w) + q1 * w
    q = q / q.norm(dim=-1, keepdim=True)
    quat = torch.where(per_env(flight, q0), q, q0)

    return (
      lerp(jp),
      lerp(jv) * per_env(vel_scale, jv[i0]),
      body_pos,
      quat,
      lin_vel,
      lerp(bav) * per_env(vel_scale, bav[i0]),
    )


class _OnnxObstacleDropModel(nn.Module):
  def __init__(self, actor, motion: ObstacleDropMotion) -> None:
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.motion = motion

  def forward(self, x, time_step, obstacle_height):
    return (self.policy(x), *self.motion(time_step, obstacle_height))


def motion_from_command(cmd: ObstacleDropMotionCommand) -> ObstacleDropMotion:
  ref = cmd.reference_motion
  reference = {name: getattr(ref, name) for name in MOTION_OUTPUTS}
  root = cmd.motion_anchor_body_index
  descent = DescentParams.fit(
    reference["body_pos_w"],
    reference["body_lin_vel_w"],
    cmd.reference_flight_frames,
    cmd._env.step_dt,
    root,
  )
  return ObstacleDropMotion(
    reference, descent, cmd.cfg.reference_obstacle_height, cmd._env.step_dt
  )


def obstacle_metadata(motion: ObstacleDropMotion, heights) -> dict:
  return {
    "obstacle_heights_trained": [float(h) for h in heights],
    "reference_obstacle_height": float(motion.h_ref),
  }


def export_motion_only(motion: ObstacleDropMotion, path: str) -> None:
  """The motion graph alone (inputs time_step, obstacle_height)."""
  torch.onnx.export(
    motion.eval(),
    (torch.zeros(1, 1), torch.full((1, 1), motion.h_ref)),
    path,
    export_params=True,
    opset_version=18,
    input_names=["time_step", "obstacle_height"],
    output_names=list(MOTION_OUTPUTS),
    dynamic_axes={},
    dynamo=False,
  )


class ObstacleDropMotionTrackingRunner(MotionTrackingOnPolicyRunner):
  """The tracking runner, exporting the box-drop graph (see module doc)."""

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    cmd = self.env.unwrapped.command_manager.get_term("motion")
    assert isinstance(cmd, ObstacleDropMotionCommand)
    motion = motion_from_command(cmd)
    model = _OnnxObstacleDropModel(self.alg.get_policy(), motion).to("cpu").eval()
    torch.onnx.export(
      model,
      (
        torch.zeros(1, model.policy.input_size),
        torch.zeros(1, 1),
        torch.full((1, 1), motion.h_ref),
      ),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "time_step", "obstacle_height"],
      output_names=["actions", *MOTION_OUTPUTS],
      dynamic_axes={},
      dynamo=False,
    )
    # save() attaches the base metadata after this.
    attach_metadata_to_onnx(
      os.path.join(path, filename), obstacle_metadata(motion, cmd.cfg.obstacle_heights)
    )


# -- Patching an export of the old format.


def patch_onnx(
  src: str, dst: str, motion_file: str, flight_frames: tuple[int, int] | None = None
) -> None:
  """Swap the bundled bank of ``src`` for the in-graph retiming; the policy is kept
  node for node. ``motion_file`` is the task's reference motion (the npz)."""
  import tempfile

  import numpy as np

  import pal_mjlab.tasks  # noqa: F401  (import order of pal_mjlab.robots)
  from pal_mjlab.robots import get_kangaroo_robot_cfg
  from pal_mjlab.tasks.tracking.kangaroo.env_cfgs import (
    DROP_N_LAND_OBSTACLE_HEIGHT,
    OBSTACLE_HEIGHTS,
  )
  from pal_mjlab.tasks.tracking.mdp.obstacle_commands import (
    ObstacleDropMotionCommandCfg,
    detect_flight_frames,
  )

  old = onnx.load(src)
  meta = {p.key: p.value for p in old.metadata_props}
  if "obstacle_height" in [i.name for i in old.graph.input]:
    raise ValueError(f"{src} already takes obstacle_height.")

  # The reference motion, on the exported bodies (MotionLoader's selection).
  spec = get_kangaroo_robot_cfg().spec_fn()  # Keep it alive while reading names.
  robot_bodies = [b.name for b in spec.bodies][1:]
  bodies = [robot_bodies.index(b) for b in meta["body_names"].split(",")]
  data = np.load(motion_file)
  full = {k: torch.tensor(data[k], dtype=torch.float32) for k in MOTION_OUTPUTS}
  reference = {
    k: (v[:, bodies] if k.startswith("body_") else v) for k, v in full.items()
  }
  dt = 1.0 / float(np.asarray(data["fps"]).reshape(-1)[0])
  if flight_frames is None:
    defaults = ObstacleDropMotionCommandCfg.__dataclass_fields__
    feet = [robot_bodies.index(b) for b in defaults["foot_body_names"].default]
    tol = defaults["contact_tolerance"].default
    flight_frames = detect_flight_frames(full["body_pos_w"][:, feet, 2], tol)
  root = meta["body_names"].split(",").index(meta["anchor_body_name"])
  descent = DescentParams.fit(
    reference["body_pos_w"], reference["body_lin_vel_w"], flight_frames, dt, root
  )
  motion = ObstacleDropMotion(reference, descent, DROP_N_LAND_OBSTACLE_HEIGHT, dt)

  with tempfile.TemporaryDirectory() as tmp:
    motion_path = os.path.join(tmp, "motion.onnx")
    export_motion_only(motion, motion_path)
    mot = onnx.load(motion_path)
  mot = onnx.compose.add_prefix(
    mot, "motion/", rename_inputs=False, rename_outputs=False
  )
  pol = onnx.utils.Extractor(old).extract_model(["obs"], ["actions"])

  g = old.graph
  inputs = {i.name: i for i in list(g.input) + list(mot.graph.input)}
  outputs = {o.name: o for o in list(g.output) + list(mot.graph.output)}
  graph = onnx.helper.make_graph(
    nodes=list(pol.graph.node) + list(mot.graph.node),
    name=g.name,
    inputs=[inputs[n] for n in ("obs", "time_step", "obstacle_height")],
    outputs=[outputs[o.name] for o in g.output],
    initializer=list(pol.graph.initializer) + list(mot.graph.initializer),
  )
  new = onnx.helper.make_model(graph, opset_imports=old.opset_import)
  new.ir_version = old.ir_version
  new.producer_name, new.producer_version = old.producer_name, old.producer_version
  for p in old.metadata_props:
    new.metadata_props.append(p)
  onnx.checker.check_model(new)
  onnx.save(new, dst)
  attach_metadata_to_onnx(dst, obstacle_metadata(motion, OBSTACLE_HEIGHTS))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("src", help="ONNX export taking time_step only")
  parser.add_argument("dst", help="output ONNX (may not be src)")
  parser.add_argument("motion_file", help="the task's reference motion (npz)")
  args = parser.parse_args()
  if os.path.abspath(args.src) == os.path.abspath(args.dst):
    raise SystemExit("dst must differ from src.")
  patch_onnx(args.src, args.dst, args.motion_file)


if __name__ == "__main__":
  main()
