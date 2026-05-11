import os
import torch
from typing import Any, Dict
from dataclasses import is_dataclass, asdict

from pal_mjlab.tasks.manipulation.tiago_pro.env_cfgs import lift_vision_env_cfg
from pal_mjlab.tasks.manipulation.tiago_pro.rl_cfg import lift_vision_ppo_runner_cfg

def format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return str(v)
    if hasattr(v, "__name__"):
        return v.__name__
    return str(v)

def generate_report(output_path: str = "training_report.md"):
    # Load configurations
    # We assume depth as default for the report
    env_cfg = lift_vision_env_cfg(cam_type="depth")
    rl_cfg = lift_vision_ppo_runner_cfg()

    lines = []
    lines.append("# Training Information Report")
    lines.append(f"\n## 1. Environment: {env_cfg.__class__.__name__}")
    
    # 1. Rewards
    lines.append("\n### Rewards")
    lines.append("| Name | Weight | Parameters |")
    lines.append("| :--- | :--- | :--- |")
    for name, reward in env_cfg.rewards.items():
        params_str = ", ".join([f"{k}: {format_value(v)}" for k, v in reward.params.items() if k != "asset_cfg" and k != "command_name"])
        lines.append(f"| {name} | {reward.weight} | {params_str} |")

    # 2. Observations
    lines.append("\n### Observation Groups")
    for group_name, group in env_cfg.observations.items():
        lines.append(f"\n#### Group: {group_name}")
        lines.append("| Term | Function | Parameters |")
        lines.append("| :--- | :--- | :--- |")
        for term_name, term in group.terms.items():
            params_str = ", ".join([f"{k}: {format_value(v)}" for k, v in term.params.items() if k != "asset_cfg" and k != "command_name"])
            func_name = term.func.__name__ if hasattr(term.func, "__name__") else str(term.func)
            lines.append(f"| {term_name} | {func_name} | {params_str} |")

    # 3. Terminations
    lines.append("\n### Terminations")
    lines.append("| Name | Function | Parameters |")
    lines.append("| :--- | :--- | :--- |")
    for name, term in env_cfg.terminations.items():
        params_str = ", ".join([f"{k}: {format_value(v)}" for k, v in term.params.items()] if term.params else [])
        func_name = term.func.__name__ if hasattr(term.func, "__name__") else str(term.func)
        lines.append(f"| {name} | {func_name} | {params_str} |")

    # 4. RL Configuration (PPO & CNN)
    lines.append("\n## 2. RL Configuration (PPO)")
    lines.append(f"\n- **Experiment Name**: {rl_cfg.experiment_name}")
    lines.append(f"- **Max Iterations**: {rl_cfg.max_iterations}")
    lines.append(f"- **Steps per Env**: {rl_cfg.num_steps_per_env}")
    lines.append(f"- **Learning Rate**: {rl_cfg.algorithm.learning_rate}")
    lines.append(f"- **Entropy Coef**: {rl_cfg.algorithm.entropy_coef}")
    
    # Distribution info
    dist = rl_cfg.actor.distribution_cfg
    if dist:
        lines.append(f"- **Distribution**: {dist.get('class_name', 'N/A')}")
        lines.append(f"- **Initial Std**: {dist.get('init_std', 'N/A')}")

    lines.append("\n### CNN Architecture")
    actor_cfg = rl_cfg.actor
    if hasattr(actor_cfg, "cnn_cfg") and actor_cfg.cnn_cfg:
        cnn = actor_cfg.cnn_cfg
        lines.append(f"- **Output Channels**: {cnn.get('output_channels')}")
        lines.append(f"- **Kernel Sizes**: {cnn.get('kernel_size')}")
        lines.append(f"- **Strides**: {cnn.get('stride')}")
        lines.append(f"- **CNN Activation**: {cnn.get('activation')}")
        lines.append(f"- **Spatial Softmax**: {cnn.get('spatial_softmax')}")
        if cnn.get('spatial_softmax'):
            lines.append(f"- **Spatial Softmax Temperature**: {cnn.get('spatial_softmax_temperature')}")
    else:
        lines.append("No CNN configuration found in actor.")

    # 5. Actor/Critic MLP
    lines.append("\n### MLP Architecture")
    lines.append(f"- **Actor Hidden Dims**: {rl_cfg.actor.hidden_dims}")
    lines.append(f"- **Critic Hidden Dims**: {rl_cfg.critic.hidden_dims}")
    lines.append(f"- **MLP Activation**: {rl_cfg.actor.activation}")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Report generated at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_report()
