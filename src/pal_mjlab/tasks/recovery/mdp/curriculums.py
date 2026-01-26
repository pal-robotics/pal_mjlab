from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict
import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class PresetProbabilityStage(TypedDict):
    step: int
    preset_probability: float


def preset_probability_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    probability_stages: list[PresetProbabilityStage],
) -> torch.Tensor:
    """Update preset_probability parameter in reset event based on training steps.
    
    Gradually increases the probability of using preset fallen configurations
    as training progresses, starting with more default poses (easier) and
    moving toward more diverse preset poses (harder).
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs (unused, kept for API consistency).
        event_name: Name of the event term to update (e.g., "reset_robot_qpos").
        probability_stages: List of stages defining when to update the probability.
            Each stage has 'step' (training step threshold) and 'preset_probability'
            (value between 0.0 and 1.0).
    
    Returns:
        Current preset_probability as a tensor for logging.
    
    Example:
```python
        curriculum = {
            "preset_difficulty": CurriculumTermCfg(
                func=mdp.preset_probability_curriculum,
                params={
                    "event_name": "reset_robot_qpos",
                    "probability_stages": [
                        {"step": 0, "preset_probability": 0.3},           # Start easy
                        {"step": 3000 * 24, "preset_probability": 0.6},   # Increase
                        {"step": 6000 * 24, "preset_probability": 0.9},   # Final value
                    ],
                },
            ),
        }
```
    """
    del env_ids  # Unused.
    
    # Get the event term configuration
    event_term_cfg = env.event_manager.get_term_cfg(event_name)
    
    # Update probability based on current training step
    for stage in probability_stages:
        if env.common_step_counter > stage["step"]:
            event_term_cfg.params["preset_probability"] = stage["preset_probability"]
    
    # Return current value for logging
    current_prob = event_term_cfg.params["preset_probability"]
    return torch.tensor([current_prob])