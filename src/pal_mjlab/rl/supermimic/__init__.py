"""SuperMimic: training through a learned differentiable world model (arXiv 2610.28480)."""

from .config import SuperMimicRunnerCfg
from .runner import SuperMimicRunner

__all__ = ["SuperMimicRunner", "SuperMimicRunnerCfg"]
