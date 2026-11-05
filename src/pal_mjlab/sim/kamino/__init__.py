"""Step mjlab with Kamino instead of mujoco_warp's solver."""

from dataclasses import dataclass
from typing import Literal

import mjlab.envs.manager_based_rl_env as _env
from mjlab.sim.sim import Simulation, SimulationCfg


@dataclass(kw_only=True)
class KaminoSimulationCfg(SimulationCfg):
  """``SimulationCfg`` that selects Kamino.

  Defaults are the fastest setting found that stays stable on kangaroo_full. Fewer inner
  or outer iterations diverge on foot contact, as does Kamino's 9-iteration RL setting.
  """

  alpha: float = 0.1
  """Baumgarte factor on joint constraints."""
  max_joint_error: float = 0.02
  """Largest joint or loop error the Baumgarte term corrects for, in m or rad. Unbounded,
  a loop opened at foot contact gets a correction that grows with the gap and blows up."""
  max_limit_error: float = 0.05
  """The same for joint limit violations, which reach several rad once a loop has opened."""
  rho_0: float = 0.05
  """PADMM penalty."""
  tolerance: float = 1e-3
  """PADMM primal, dual and complementarity tolerance."""
  max_iterations: int = 60
  linear_solver: Literal["LLTB", "CRF"] = "CRF"
  """Matrix-free conjugate residual, or dense LLT. CRF can't stop on convergence inside
  a CUDA graph, so it always runs ``max_iterations``, and its kernels are GPU only."""
  crf_iterations: int = 30
  """Inner linear solver iterations per PADMM iteration."""
  min_body_mass: float = 0.01
  """Floor for the generated 1e-05 kg placeholder bodies, in kg."""
  min_frictionloss: float = 0.05
  """Joint friction below this gets no constraint row. The friction event adds
  +/-0.008 N m to every linkage joint, otherwise adding about 35 rows per world."""
  max_contacts: int = 48
  """Contacts per world that the solver can take. Standing alone can reach 32."""
  contact_warmstart: Literal[
    "geom_pair_net_force", "key_and_position_with_net_force_backup"
  ] = "key_and_position_with_net_force_backup"
  """Kamino's RL setting, or the same with position matching first. Equally fast, and
  the benchmark can't tell their loop errors apart."""


def _make_simulation(*, cfg, **kwargs) -> Simulation:
  # mjlab has no backend hook: ManagerBasedRlEnv constructs Simulation directly.
  if isinstance(cfg, KaminoSimulationCfg):
    from pal_mjlab.sim.kamino.simulation import KaminoSimulation

    return KaminoSimulation(cfg=cfg, **kwargs)
  return Simulation(cfg=cfg, **kwargs)


_env.Simulation = _make_simulation
