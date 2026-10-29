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


# --------------------------------------------------------------------------- #
# Index-map helpers
# --------------------------------------------------------------------------- #
def build_index_map(cur_terms, base_terms, term_maps=None):
  """Match observation terms by name.

  Equal-dim terms are matched 1:1. An explicit per-term map
  {name: [(cur_local, base_local), ...]} overrides. Same-name terms with
  different dims and no explicit map are left unmapped (learned by the proj).
  Returns [(cur_idx, base_idx), ...].
  """
  term_maps = term_maps or {}

  def offsets(terms):
    out, o = {}, 0
    for name, dim in terms:
      out[name] = (o, int(dim))
      o += int(dim)
    return out

  cur, base = offsets(cur_terms), offsets(base_terms)
  pairs = []
  for name, (co, cd) in cur.items():
    if name not in base:
      continue
    bo, bd = base[name]
    if name in term_maps:
      for c, b in term_maps[name]:
        if not (0 <= c < cd and 0 <= b < bd):
          raise ValueError(
            f"projector_term_maps['{name}']: pair ({c}, {b}) out of range "
            f"(cur dim {cd}, base dim {bd})"
          )
        pairs.append((co + c, bo + b))
    elif cd == bd:
      pairs += [(co + j, bo + j) for j in range(cd)]
    else:
      print(
        f"[INFO] Term '{name}' dim {cd} (current) vs {bd} (base): "
        f"not index-mapped, learned by the projection layer."
      )
  return pairs

def _validate_index_map(index_map, in_dim, out_dim, name="index_map"):
  """Normalise to list of (src, dst) int tuples and sanity check."""
  pairs = [(int(s), int(d)) for s, d in index_map]
  for s, d in pairs:
    if not (0 <= s < in_dim):
      raise ValueError(f"{name}: src index {s} out of range [0, {in_dim})")
    if not (0 <= d < out_dim):
      raise ValueError(f"{name}: dst index {d} out of range [0, {out_dim})")
  dsts = [d for _, d in pairs]
  if len(set(dsts)) != len(dsts):
    raise ValueError(f"{name}: duplicate dst indices (contributions would sum)")
  return pairs


def _build_fill(backbone, index_map, in_dim, out_dim):
  """Constant offset for backbone inputs that nothing maps onto.

  The backbone normalises as (x - mean) / (std + eps), so a raw 0 in an unmatched
  slot is NOT neutral. Filling it with the pretrained mean makes it normalise to 0.
  Matched slots get their value from the projection, so their fill is 0.
  """
  if not backbone.normalize:
    return None
  if index_map is None:
    k = min(in_dim, out_dim)
    index_map = [(i, i) for i in range(k)]
  fill = backbone.mean.clone().flatten()
  for _, dst in index_map:
    fill[dst] = 0.0
  if not torch.any(fill != 0):
    return None
  return fill


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #
class _ProjectionHead(nn.Module):
  def __init__(
    self,
    in_dim: int,
    out_dim: int,
    hidden: int | tuple[int, ...],
    index_map: list[tuple[int, int]] | None = None,  # (src idx in x, dst idx in output)
    fill: torch.Tensor | None = None,  # constant for unmatched outputs, shape (out_dim,)
    freeze_proj: bool = False,
  ):
    super().__init__()
    hidden = (hidden,) if isinstance(hidden, int) else hidden

    self.n_mapped = None
    self.proj = None
    if in_dim != out_dim or index_map is not None:
      self.proj = nn.Linear(in_dim, out_dim, bias=False)
      with torch.no_grad():
        self.proj.weight.zero_()
        if index_map is None:
          k = min(in_dim, out_dim)
          self.proj.weight[:k, :k] = torch.eye(k)
        else:
          index_map = _validate_index_map(index_map, in_dim, out_dim)
          self.n_mapped = len(index_map)
          for src, dst in index_map:
            self.proj.weight[dst, src] = 1.0
      if freeze_proj:
        self.proj.weight.requires_grad_(False)

    # persistent=False: recomputed from the backbone, so heads checkpoints stay
    # interchangeable regardless of whether a fill was used.
    if fill is not None:
      self.register_buffer("fill", fill.clone(), persistent=False)
    else:
      self.fill = None

    layers, d = [], in_dim
    for h in hidden:
      layers += [nn.Linear(d, h), nn.ELU()]
      d = h
    layers.append(nn.Linear(d, out_dim))
    self.net = nn.Sequential(*layers)
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, x):
    base = x if self.proj is None else self.proj(x)
    if self.fill is not None:
      base = base + self.fill
    return base + self.net(x)


class _FrozenBackbone(nn.Module):
  """Pretrained obs normalizer + MLP rebuilt from the base policy's own
  state_dict, so its dimensions are independent of the current env."""

  def __init__(self, actor_sd: dict, activation: str = "elu", eps: float = 1e-2, normalize: bool = True):
    super().__init__()
    self.eps = eps
    self.normalize = normalize and "obs_normalizer._mean" in actor_sd
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
  def __init__(
    self,
    backbone,
    obs_dim,
    act_dim,
    hidden_in=(128,),
    hidden_out=(128,),
    obs_map=None,
    act_map=None,
    freeze_proj=False,
  ):
    super().__init__()
    self.backbone = backbone
    self.head_in = _ProjectionHead(
      obs_dim,
      backbone.in_dim,
      hidden_in,
      index_map=obs_map,
      fill=_build_fill(backbone, obs_map, obs_dim, backbone.in_dim),
      freeze_proj=freeze_proj,
    )
    # Output side: no normalizer on actions, so no fill needed.
    self.head_out = _ProjectionHead(
      backbone.out_dim,
      act_dim,
      hidden_out,
      index_map=act_map,
      fill=None,
      freeze_proj=freeze_proj,
    )

  def train(self, mode: bool = True):
    super().train(mode)
    self.backbone.eval()
    return self

  def forward(self, x):
    return self.head_out(self.backbone(self.head_in(x)))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
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
    activation = actor_cfg.get("activation", "elu") if isinstance(actor_cfg, dict) else "elu"
    env_norm = os.environ.get("PROJECTOR_BACKBONE_NORMALIZE")
    backbone_normalize = (
      env_norm == "1" if env_norm is not None
      else bool(train_cfg.get("projector_backbone_normalize", True))
    )
    backbone = _FrozenBackbone(
      ckpt["actor_state_dict"],
      activation=activation,
      eps=getattr(actor.obs_normalizer, "eps", 1e-2),
      normalize=backbone_normalize,
    )
    ckpt["actor_state_dict"].keys()
    dist_sd = {k.split("distribution.", 1)[1]: v
                for k, v in ckpt["actor_state_dict"].items()
                if k.startswith("distribution.")}
    if dist_sd:
      actor.distribution.load_state_dict(dist_sd, strict=False)

    hidden_in = tuple(train_cfg.get("projector_hidden_in", (256, 256)))
    hidden_out = tuple(train_cfg.get("projector_hidden_out", (128,)))
    freeze_proj = bool(train_cfg.get("projector_freeze_remap", False))

    term_maps = train_cfg.get("projector_term_maps")

    obs_map = train_cfg.get("projector_obs_map")
    if (
      obs_map is None
      and train_cfg.get("projector_cur_terms")
      and train_cfg.get("projector_base_terms")
    ):
      obs_map = build_index_map(
        train_cfg["projector_cur_terms"],
        train_cfg["projector_base_terms"],
        term_maps,
      )

    act_map = train_cfg.get("projector_act_map")
    if act_map is None and term_maps and "actions" in term_maps:
      # Obs pairs are (cur, base); the output head wants (backbone_idx, cur_idx).
      act_map = [(b, c) for c, b in term_maps["actions"]]

    actor.mlp = _ProjectedMLP(
      backbone,
      obs_dim,
      act_dim,
      hidden_in=hidden_in,
      hidden_out=hidden_out,
      obs_map=obs_map,
      act_map=act_map,
      freeze_proj=freeze_proj,
    ).to(self.device)

    def _n(params):
      return sum(p.numel() for p in params)

    def _mlp_dims(net):
      lins = [l for l in net.modules() if isinstance(l, nn.Linear)]
      return " -> ".join([str(lins[0].in_features)] + [str(l.out_features) for l in lins])

    def _head_block(name, head):
      tot = _n(head.parameters())
      trn = _n(p for p in head.parameters() if p.requires_grad)
      lines = [f"  {name}: {head.net[0].in_features} -> {head.net[-1].out_features}"]
      if head.proj is None:
        lines.append("    proj  : none (identity skip)")
      else:
        kind = (
          f"{head.n_mapped} dims remapped by index"
          if head.n_mapped is not None
          else f"identity-init on first {min(head.proj.in_features, head.proj.out_features)} dims"
        )
        state = "trainable" if head.proj.weight.requires_grad else "frozen"
        lines.append(
          f"    proj  : Linear({head.proj.in_features} -> {head.proj.out_features}, "
          f"bias=False), {kind}, {state}"
        )
      lines.append(
        f"    fill  : {'normalizer-mean offset' if head.fill is not None else 'none'}"
      )
      lines.append(f"    mlp   : {_mlp_dims(head.net)}  (residual, zero-init last layer)")
      lines.append(f"    params: {tot:,} total, {trn:,} trainable")
      return "\n".join(lines)

    m = actor.mlp
    print(
      "[INFO] Projector runner\n"
      f"  base policy : {ckpt_path}\n"
      f"{_head_block('head_in', m.head_in)}\n"
      f"  backbone: {backbone.in_dim} -> {backbone.out_dim}  (frozen, "
      f"{_n(backbone.parameters()):,} params)\n"
      f"    mlp   : {_mlp_dims(backbone.mlp)}\n"
      f"{_head_block('head_out', m.head_out)}\n"
      f"  trainable : heads "
      f"{_n([p for p in list(m.head_in.parameters()) + list(m.head_out.parameters()) if p.requires_grad]):,}"
      f" | std {_n(actor.distribution.parameters()):,}"
      f" | critic {_n(self.alg.critic.parameters()):,}"
    )

    if backbone.normalize:
      # The backbone owns the (pretrained) normalizer; the actor's own one is bypassed.
      actor.obs_normalizer = nn.Identity()
      if hasattr(actor, "update_normalization"):
        actor.update_normalization = lambda obs: None
    # else: the actor's own normalizer stays active and keeps updating.

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
    state = {
      "head_in": actor.mlp.head_in.state_dict(),
      "head_out": actor.mlp.head_out.state_dict(),
      "distribution": actor.distribution.state_dict(),
    }
    if not actor.mlp.backbone.normalize:
      state["obs_normalizer"] = actor.obs_normalizer.state_dict()
    return state

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
    if "obs_normalizer" in state:
      actor.obs_normalizer.load_state_dict(state["obs_normalizer"])
    return loaded.get("infos")