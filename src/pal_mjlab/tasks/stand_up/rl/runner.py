from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

_STATE_KEY = "curriculum_state"


class StandUpOnPolicyRunner(VelocityOnPolicyRunner):
  """Saves the state of the curriculum terms with every checkpoint, so that a
  resumed run continues where the checkpoint was."""

  def _stateful_curriculum_terms(self) -> dict:
    manager = self.env.unwrapped.curriculum_manager
    terms = {name: manager.get_term_cfg(name).func for name in manager.active_terms}
    return {name: term for name, term in terms.items() if hasattr(term, "state_dict")}

  def save(self, path: str, infos=None) -> None:
    state = {n: t.state_dict() for n, t in self._stateful_curriculum_terms().items()}
    super().save(path, {**(infos or {}), _STATE_KEY: state})

  def load(self, path: str, *args, **kwargs) -> dict:
    infos = super().load(path, *args, **kwargs)
    state = (infos or {}).get(_STATE_KEY, {})
    for name, term in self._stateful_curriculum_terms().items():
      if name in state:
        term.load_state_dict(state[name])
    return infos
