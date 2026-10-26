import os
from pathlib import Path

import torch
import torch.nn as nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

_ACTIVATIONS = {
  "elu": nn.ELU,
  "relu": nn.ReLU,
  "tanh": nn.Tanh,
  "selu": nn.SELU,
  "silu": nn.SiLU,
  "gelu": nn.GELU,
  "lrelu": nn.LeakyReLU,
}


class _ProjectionHead(nn.Module):
  """Always-present trainable head: base(x) + f(x).

  base is a learnable rectangular-identity Linear when in_dim != out_dim, or a
  plain skip connection when dims match. f is a small MLP whose last layer is
  zero-initialised, so the head starts as an identity map (padding / truncating
  if dims differ). The MLP residual is created in both cases."""

  def __init__(self, in_dim: int, out_dim: int, hidden: int):
    super().__init__()
    self.proj = None
    if in_dim != out_dim:
      self.proj = nn.Linear(in_dim, out_dim, bias=False)
      with torch.no_grad():
        self.proj.weight.zero_()
        k = min(in_dim, out_dim)
        self.proj.weight[:k, :k] = torch.eye(k)
    self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ELU(), nn.Linear(hidden, out_dim))
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, x):
    base = x if self.proj is None else self.proj(x)
    return base + self.net(x)


class _FrozenBackbone(nn.Module):
  """Pretrained obs normalizer + MLP rebuilt from the base policy's own
  state_dict, so its dimensions are independent of the current env."""

  def __init__(self, actor_sd: dict, activation: str = "elu", eps: float = 1e-2):
    super().__init__()
    self.eps = eps
    self.normalize = "obs_normalizer._mean" in actor_sd
    if self.normalize:
      self.register_buffer("mean", actor_sd["obs_normalizer._mean"].clone())
      self.register_buffer("std", actor_sd["obs_normalizer._std"].clone())

    idxs = sorted(
      {int(k.split(".")[1]) for k in actor_sd if k.startswith("mlp.") and k.endswith(".weight")}
    )
    act = _ACTIVATIONS[activation]
    layers = []
    for n, i in enumerate(idxs):
      w, b = actor_sd[f"mlp.{i}.weight"], actor_sd[f"mlp.{i}.bias"]
      lin = nn.Linear(w.shape[1], w.shape[0])
      with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(b)
      layers.append(lin)
      if n < len(idxs) - 1:
        layers.append(act())
    self.mlp = nn.Sequential(*layers)
    self.in_dim = layers[0].in_features
    self.out_dim = layers[-1].out_features

    for p in self.parameters():
      p.requires_grad_(False)

  def forward(self, x):
    if self.normalize:
      x = (x - self.mean) / (self.std + self.eps)
    return self.mlp(x)


class _ProjectedMLP(nn.Module):
  """head_in -> frozen pretrained backbone -> head_out."""

  def __init__(self, backbone: _FrozenBackbone, obs_dim: int, act_dim: int, hidden: int = 128):
    super().__init__()
    self.backbone = backbone
    self.head_in = _ProjectionHead(obs_dim, backbone.in_dim, hidden)
    self.head_out = _ProjectionHead(backbone.out_dim, act_dim, hidden)

  def train(self, mode: bool = True):
    super().train(mode)
    self.backbone.eval()
    return self

  def forward(self, x):
    return self.head_out(self.backbone(self.head_in(x)))


class VelocityProjectorOnPolicyRunner(VelocityOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    ckpt_path = train_cfg.get("pretrained_checkpoint") or os.environ.get("PRETRAINED_CKPT")
    if not ckpt_path:
      raise ValueError(
        "VelocityProjectorOnPolicyRunner needs the base policy to build the frozen "
        "backbone (set train_cfg['pretrained_checkpoint'] or PRETRAINED_CKPT)."
      )

    actor = self.alg.get_policy()

    # Dims of the *current* env, read from the freshly built actor.
    linears = [m for m in actor.mlp.modules() if isinstance(m, nn.Linear)]
    obs_dim, act_dim = linears[0].in_features, linears[-1].out_features

    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
    actor_cfg = train_cfg.get("actor", {}) or {}
    activation = (
      actor_cfg.get("activation", "elu") if isinstance(actor_cfg, dict) else "elu"
    )
    backbone = _FrozenBackbone(
      ckpt["actor_state_dict"],
      activation=activation,
      eps=getattr(actor.obs_normalizer, "eps", 1e-2),
    )
    actor.mlp = _ProjectedMLP(backbone, obs_dim, act_dim).to(self.device)

    def _n(params):
        return sum(p.numel() for p in params)

    def _head_desc(name, head, i, o):
        if head.proj is None:
            return f"{name:<9}: {i} -> {o}  (dims match: identity skip + residual MLP)"
        k = min(i, o)
        return (
            f"{name:<9}: {i} -> {o}  (identity-init proj on first {k} dims, "
            f"{'zero-padded' if o > i else 'truncated'}, + residual MLP)"
        )

    m = actor.mlp
    print(
      "[INFO] Projector runner\n"
      f"  base policy : {ckpt_path}\n"
      f"  {_head_desc('head_in', m.head_in, obs_dim, backbone.in_dim)}\n"
      f"  backbone  : {backbone.in_dim} -> {backbone.out_dim}  (frozen, "
      f"{_n(backbone.parameters()):,} params)\n"
      f"  {_head_desc('head_out', m.head_out, backbone.out_dim, act_dim)}\n"
      f"  trainable : heads {_n(list(m.head_in.parameters()) + list(m.head_out.parameters())):,}"
      f" | std {_n(actor.distribution.parameters()):,}"
      f" | critic {_n(self.alg.critic.parameters()):,}"
    )

    # The backbone owns the (pretrained) normalizer; the actor's own one is bypassed.
    actor.obs_normalizer = nn.Identity()
    if hasattr(actor, "update_normalization"):
      actor.update_normalization = lambda obs: None

    # The critic's obs space differs from the base policy's, so it trains from scratch.
    params = [p for p in actor.parameters() if p.requires_grad]
    params += list(self.alg.critic.parameters())
    self.alg.optimizer = torch.optim.Adam(params, lr=self.alg.learning_rate)

  # ---------------------------------------------------------------- save
  @staticmethod
  def _heads_path(path: str) -> str:
    p = Path(path)
    return str(p.parent / "heads" / p.name)

  def _heads_state(self) -> dict:
    actor = self.alg.get_policy()
    return {
      "head_in": actor.mlp.head_in.state_dict(),
      "head_out": actor.mlp.head_out.state_dict(),
      "distribution": actor.distribution.state_dict(),
    }

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    heads_path = self._heads_path(path)
    os.makedirs(os.path.dirname(heads_path), exist_ok=True)
    torch.save(
      {
        "projector_heads": self._heads_state(),
        "iter": self.current_learning_iteration,
        "infos": infos,
      },
      heads_path,
    )

  # ---------------------------------------------------------------- load
  def load(self, path, load_cfg=None, strict=True, map_location=None):
    if Path(path).parent.name != "heads":
      return super().load(path, load_cfg, strict, map_location)

    loaded = torch.load(path, map_location=map_location, weights_only=False)
    state = loaded["projector_heads"]
    actor = self.alg.get_policy()
    actor.mlp.head_in.load_state_dict(state["head_in"], strict=strict)
    actor.mlp.head_out.load_state_dict(state["head_out"], strict=strict)
    actor.distribution.load_state_dict(state["distribution"], strict=strict)
    return loaded.get("infos")