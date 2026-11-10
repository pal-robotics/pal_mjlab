"""World-model pieces: state round trip, loss, gradients, the termination head's inputs."""

import pytest
import torch
from pal_mjlab.tasks.supermimic.mdp.tracking import projected_gravity_z
from pal_mjlab.tasks.supermimic.rl.world_model import (
  WorldModel,
  WorldModelCfg,
  window_loss,
)
from pal_mjlab.tasks.supermimic.state import (
  heading_pose,
  heading_state,
  integrate_pose,
  pose_delta,
  state_to_qpos_qvel,
)

NJ, NUM_ACTIONS, CONTEXT = 28, 28, 32


def _random_qpos_qvel(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
  g = torch.Generator().manual_seed(seed)
  qpos = torch.randn(n, 7 + NJ, generator=g, dtype=torch.float64)
  qpos[:, 3:7] /= qpos[:, 3:7].norm(
    dim=-1, keepdim=True
  )  # any orientation, upside down too
  return qpos, torch.randn(n, 6 + NJ, generator=g, dtype=torch.float64)


@pytest.mark.parametrize("seed", range(3))
def test_state_round_trip(seed):
  qpos, qvel = _random_qpos_qvel(256, seed)
  qpos2, qvel2 = state_to_qpos_qvel(heading_state(qpos, qvel), heading_pose(qpos))
  sign = torch.sign((qpos2[:, 3:7] * qpos[:, 3:7]).sum(-1, keepdim=True))
  qpos2 = torch.cat([qpos2[:, :3], qpos2[:, 3:7] * sign, qpos2[:, 7:]], dim=-1)
  torch.testing.assert_close(qpos2, qpos, atol=1e-10, rtol=0)
  torch.testing.assert_close(qvel2, qvel, atol=1e-10, rtol=0)
  pose, other = heading_pose(qpos), heading_pose(qpos.roll(1, 0))
  torch.testing.assert_close(
    integrate_pose(pose, pose_delta(pose, other)), other, atol=1e-10, rtol=0
  )


def _tiny_model(dtype=torch.float32) -> WorldModel:
  torch.manual_seed(0)
  cfg = WorldModelCfg(
    state_dim=13 + 2 * NJ,
    action_dim=NUM_ACTIONS,
    context_dim=CONTEXT,
    hidden=(32, 32),
    termination_hidden=(16,),
  )
  return WorldModel(cfg).to(dtype)


def _batch(n: int, k: int, dtype=torch.float32):
  g = torch.Generator().manual_seed(1)
  qpos, qvel = _random_qpos_qvel(n * (k + 1), 2)
  states = heading_state(qpos, qvel).to(dtype).reshape(n, k + 1, -1)
  poses = heading_pose(qpos).to(dtype).reshape(n, k + 1, -1)
  return dict(
    states=states,
    poses=poses,
    actions=torch.randn(n, k, NUM_ACTIONS, generator=g, dtype=dtype),
    context=torch.randn(n, CONTEXT, generator=g, dtype=dtype),
    ref_root=torch.randn(n, k, 2, generator=g, dtype=dtype),
    failure=torch.rand(n, k, generator=g) < 0.1,
  )


def test_model_overfits_a_batch():
  model = _tiny_model()
  b = _batch(16, 4)
  opt = torch.optim.Adam(model.parameters(), lr=3e-3)
  first = None
  for _ in range(300):
    loss, _ = window_loss(model, **b)
    first = first if first is not None else loss.item()
    opt.zero_grad()
    loss.backward()
    opt.step()
  assert loss.item() < 0.2 * first


def test_rollout_gradient_matches_finite_differences():
  """BPTT through the model: d(objective)/d(actions) equals finite differences (float64)."""
  model = _tiny_model(torch.float64)
  b = _batch(2, 3, torch.float64)
  actions = b["actions"].clone().requires_grad_()

  def objective(acts: torch.Tensor) -> torch.Tensor:
    s, p = b["states"][:, 0], b["poses"][:, 0]
    total = 0.0
    for j in range(acts.shape[1]):
      s, p = model.step(s, p, acts[:, j], b["context"])
      total = total + s.square().sum() + p.square().sum()
    return total

  objective(actions).backward()
  eps = 1e-6
  for idx in [(0, 0, 0), (1, 2, 5), (0, 1, 27)]:
    plus, minus = actions.detach().clone(), actions.detach().clone()
    plus[idx] += eps
    minus[idx] -= eps
    fd = (objective(plus) - objective(minus)) / (2 * eps)
    assert torch.isclose(actions.grad[idx], fd, rtol=1e-5, atol=1e-7), idx


def test_rule_feature_head_sees_the_rules_errors():
  """With the reference at the robot's own root height and gravity-z, both inputs of the
  termination head are zero (sign and frame agree with the env's rule)."""
  g = torch.Generator().manual_seed(0)
  q = torch.randn(32, 4, generator=g)
  q = q / q.norm(dim=-1, keepdim=True)
  qpos = torch.cat(
    [torch.randn(32, 3, generator=g), q, torch.randn(32, NJ, generator=g)], -1
  )
  state = heading_state(qpos, torch.randn(32, 6 + NJ, generator=g))
  ref_root = torch.stack([qpos[:, 2], projected_gravity_z(q)], dim=-1)
  wm = _tiny_model()
  seen = []
  wm.termination.register_forward_hook(lambda m, inp, out: seen.append(inp[0]))
  wm.termination_logit(state, ref_root)
  torch.testing.assert_close(seen[0], torch.zeros(32, 2), atol=1e-5, rtol=0)
