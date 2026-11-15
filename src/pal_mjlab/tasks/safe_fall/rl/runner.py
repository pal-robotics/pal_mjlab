"""Runner for the safe-fall task: optional warm start from a walking policy."""

from __future__ import annotations

import math

import torch
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.lab_api.string import resolve_matching_names_values
from rsl_rl.env import VecEnv

from pal_mjlab.robots import KANGAROO_ACTION_SCALE

_ACTION_TERM = "joint_pos"
_LAST_ACTION_OBS = "actions"
_ACTOR_OBS_GROUP = "actor"


class SafeFallOnPolicyRunner(VelocityOnPolicyRunner):
  """PPO runner that can initialize the actor from another policy.

  Config keys (see ``SafeFallRunnerCfg``):

  - ``init_actor_checkpoint``: rsl_rl checkpoint whose actor (weights and
    observation normalizer) initializes this run's actor. The critic, the
    optimizer and the iteration counter start fresh, because the reward is
    different from the one the checkpoint was trained on.
  - ``init_action_std``: action std to restart exploration from.
  - ``actor_freeze_iterations``: number of initial iterations in which only the
    critic learns, so the first policy updates do not follow the advantages of
    an untrained critic.

  The task's action scale (joint target shift per unit of policy output) is
  saved with every checkpoint. A checkpoint whose scale differs from this
  task's, such as a walking policy or a fall policy from before the arms got
  a larger scale, is converted when it is loaded, so that it commands exactly
  the same joint targets as before. This holds for warm starts and for
  ``load`` (playing and evaluating).

  The same goes for a checkpoint whose actor lacks the observations the task
  appends to the walking policy's (which actuators are down): its actor is
  widened with zero weights for them, so it acts exactly as before.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self._actor_freeze_iterations = int(train_cfg.get("actor_freeze_iterations", 0))
    checkpoint = train_cfg.get("init_actor_checkpoint")
    if checkpoint:
      self._init_actor_from(checkpoint, float(train_cfg.get("init_action_std", 0.5)))
    if self._actor_freeze_iterations > 0:
      self._wrap_update_with_actor_freeze()

  def _init_actor_from(self, path: str, action_std: float) -> None:
    loaded = torch.load(path, map_location=self.device, weights_only=False)
    actor_state = loaded["actor_state_dict"]
    actor = self.alg.get_policy()
    actor_state = self._widen_actor_inputs(actor_state)
    own_state = actor.state_dict()
    for key, value in actor_state.items():
      if key not in own_state:
        raise ValueError(f"Checkpoint actor has unexpected parameter '{key}'.")
      if own_state[key].shape != value.shape:
        raise ValueError(
          f"Actor layout mismatch on '{key}': checkpoint {tuple(value.shape)} vs "
          f"task {tuple(own_state[key].shape)}. The fallback task must use the "
          "same actor observations and actions as the source policy."
        )
    actor.load_state_dict(actor_state, strict=True)
    self._convert_actor_action_scale(loaded.get("infos"))

    with torch.no_grad():
      state = actor.state_dict()
      if "distribution.std_param" in state:
        state["distribution.std_param"].fill_(action_std)
      elif "distribution.log_std_param" in state:
        state["distribution.log_std_param"].fill_(math.log(action_std))
    print(
      f"[INFO] Actor initialized from {path} (iteration {loaded.get('iter')}); "
      f"action std reset to {action_std}, critic starts fresh."
    )

  def save(self, path: str, infos=None) -> None:
    infos = {**(infos or {}), "action_scale": self._task_action_scale()}
    super().save(path, infos)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    # The parent reads the file and hands the dict to the algorithm: widen the
    # actor of a checkpoint from before the task had its last observations.
    alg_load = self.alg.load

    def load_widened(loaded_dict: dict, cfg: dict | None, strict_: bool) -> bool:
      if cfg is None or cfg.get("actor"):
        loaded_dict["actor_state_dict"] = self._widen_actor_inputs(
          loaded_dict["actor_state_dict"]
        )
      return alg_load(loaded_dict, cfg, strict_)

    self.alg.load = load_widened  # type: ignore[method-assign]
    try:
      infos = super().load(path, load_cfg, strict, map_location)
    finally:
      self.alg.load = alg_load  # type: ignore[method-assign]
    if load_cfg is None or load_cfg.get("actor"):
      self._convert_actor_action_scale(infos)
    return infos

  def _widen_actor_inputs(self, actor_state: dict) -> dict:
    """Fit an actor trained without this task's last observations.

    The task appends observations to the walking policy's (which actuators
    are down). An actor from before that has fewer inputs: its first layer
    gets zero weights for the new ones and their normalizer entries start
    fresh, so it acts exactly as it did, whatever the new inputs say.
    """
    own_state = self.alg.get_policy().state_dict()
    weight_key = "mlp.0.weight"
    own_inputs = own_state[weight_key].shape[1]
    inputs = actor_state[weight_key].shape[1]
    if inputs == own_inputs:
      return actor_state
    manager = self.env.unwrapped.observation_manager
    terms = manager.active_terms[_ACTOR_OBS_GROUP]
    dims = [math.prod(d) for d in manager.group_obs_term_dim[_ACTOR_OBS_GROUP]]
    # Widths the actor had when the task's last 0, 1, ... terms did not exist.
    earlier = {sum(dims[:i]): terms[i:] for i in range(1, len(dims))}
    if inputs not in earlier:
      raise ValueError(
        f"Actor layout mismatch: the checkpoint's actor has {inputs} inputs and "
        f"the task's {own_inputs}, which is not the same actor with observations "
        "added at the end."
      )
    widened = dict(actor_state)
    for key, value in actor_state.items():
      if (
        value.dim() == 2
        and value.shape[1] == inputs
        and (key == weight_key or key.startswith("obs_normalizer."))
      ):
        new = own_state[key].detach().clone()
        if key == weight_key:
          new.zero_()
        new[:, :inputs] = value.to(new.device)
        widened[key] = new
    print(
      f"[INFO] Actor widened from {inputs} to {own_inputs} inputs; it ignores "
      f"the new observations {earlier[inputs]}."
    )
    return widened

  def _task_action_scale(self) -> dict[str, float]:
    """Action scale of this task, per joint in action order."""
    term = self.env.unwrapped.action_manager.get_term(_ACTION_TERM)
    names = term.target_names
    scale = term.scale
    if isinstance(scale, torch.Tensor):
      return dict(zip(names, scale[0].tolist(), strict=True))
    return dict.fromkeys(names, float(scale))

  def _convert_actor_action_scale(self, infos: dict | None) -> None:
    """Make an actor trained with another action scale act the same here.

    With scale ratio k = this task's / the checkpoint's for an action, the
    output is divided by k, so the joint target is unchanged. The policy also
    observes its last output, which is now k times smaller: the mean and
    the divisor of that observation's normalizer are divided by k as well, and
    the network sees the same normalized input as before.
    """
    own = self._task_action_scale()
    names = list(own)
    source = (infos or {}).get("action_scale")
    if source is None:
      # Checkpoints from before the scale was saved: the walking task's.
      _, matched, values = resolve_matching_names_values(KANGAROO_ACTION_SCALE, names)
      source = dict(zip(matched, values, strict=True))
    if set(source) != set(own):
      raise ValueError("The checkpoint's action scale is for other joints.")
    ratio = torch.tensor([own[n] / source[n] for n in names], device=self.device)
    if torch.allclose(ratio, torch.ones_like(ratio)):
      return

    env = self.env.unwrapped
    terms = env.observation_manager.active_terms[_ACTOR_OBS_GROUP]
    dims = [
      math.prod(d) for d in env.observation_manager.group_obs_term_dim[_ACTOR_OBS_GROUP]
    ]
    start = sum(dims[: terms.index(_LAST_ACTION_OBS)])
    columns = slice(start, start + len(names))

    actor = self.alg.get_policy()
    state = actor.state_dict()
    last = max(int(k.split(".")[1]) for k in state if k.startswith("mlp."))
    with torch.no_grad():
      state[f"mlp.{last}.weight"] /= ratio.unsqueeze(1)
      state[f"mlp.{last}.bias"] /= ratio
      if "obs_normalizer._mean" in state:
        # The normalizer divides by (std + eps): keep that whole divisor in
        # proportion, so that the conversion is exact.
        eps = float(actor.obs_normalizer.eps)
        std = state["obs_normalizer._std"][:, columns]
        new_std = ((std + eps) / ratio - eps).clamp(min=1e-6)
        state["obs_normalizer._mean"][:, columns] /= ratio
        state["obs_normalizer._std"][:, columns] = new_std
        state["obs_normalizer._var"][:, columns] = torch.square(new_std)
    changed = {
      n: round(r, 2)
      for n, r in zip(names, ratio.tolist(), strict=True)
      if abs(r - 1) > 1e-6
    }
    print(f"[INFO] Actor converted to this task's action scale (ratio): {changed}")

  def _wrap_update_with_actor_freeze(self) -> None:
    actor = self.alg.get_policy()
    original_update = self.alg.update
    start_iteration = self.current_learning_iteration

    def update(*args, **kwargs):
      done = self.current_learning_iteration - start_iteration
      if done >= self._actor_freeze_iterations:
        return original_update(*args, **kwargs)
      # The optimizer is shared, so let it step and then put the actor back.
      # The observation normalizer is a buffer and is not touched by the update.
      frozen = [p.detach().clone() for p in actor.parameters()]
      result = original_update(*args, **kwargs)
      with torch.no_grad():
        for param, value in zip(actor.parameters(), frozen, strict=True):
          param.copy_(value)
      return result

    self.alg.update = update  # type: ignore[method-assign]
