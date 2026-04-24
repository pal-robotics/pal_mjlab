import os
import torch
from typing import Any

def generate_env_summary(env_cfg: Any, rl_cfg: Any) -> str:
    """Generate a detailed Markdown summary of the environment and RL configuration."""
    
    lines = []
    lines.append("# Kangaroo Tracking Environment — Dynamic Run Summary")
    lines.append("")
    
    # 1. History Configuration
    has_history = "actor_history" in env_cfg.observations
    lines.append("## 1. General Configuration")
    lines.append(f"- **History Enabled**: {has_history}")
    if has_history:
        h_len = env_cfg.observations["actor_history"].history_length
        lines.append(f"- **History Length**: {h_len} frames")
    lines.append(f"- **Control DT**: {env_cfg.sim.mujoco.timestep * env_cfg.decimation * 1000:.1f} ms (MuJoCo dt={env_cfg.sim.mujoco.timestep * 1000:.1f} ms, decimation={env_cfg.decimation})")
    if hasattr(env_cfg, 'commands') and "motion" in env_cfg.commands:
        m_cmd = env_cfg.commands["motion"]
        resample = getattr(m_cmd, "resampling_time_range", "—")
        lines.append(f"- **Command Resampling**: {resample} s")
    lines.append("")
    
    # 2. Observations
    lines.append("## 2. Observation Space")
    # Dynamically find all observation groups (actor, critic, actor_history, etc.)
    all_groups = sorted(env_cfg.observations.keys())
    for gname in all_groups:
        lines.append(f"### {gname.capitalize()} Group")
        lines.append("| Term | Noise / Scale | Parameters |")
        lines.append("| :--- | :--- | :--- |")
        group = env_cfg.observations[gname]
        for tname, term in group.terms.items():
            noise_str = "None"
            if hasattr(term, 'noise') and term.noise is not None:
                if hasattr(term.noise, 'n_min'):
                     noise_str = f"U({term.noise.n_min}, {term.noise.n_max})"
                else:
                     noise_str = str(term.noise)
            
            scale_str = f" x{term.scale}" if hasattr(term, 'scale') and term.scale is not None else ""
            params_str = str(term.params) if term.params else "—"
            lines.append(f"| `{tname}` | {noise_str}{scale_str} | {params_str} |")
        lines.append("")

    # 3. Rewards
    # ... (no changes needed for rewards)
    lines.append("## 3. Reward Structure")
    lines.append("| Reward | Weight | Specifics |")
    lines.append("| :--- | :--- | :--- |")
    for rname, term in env_cfg.rewards.items():
        # Try to extract 'std' or relevant params
        specs = []
        if term.params:
            for key in ["std", "threshold", "sigma"]:
                if key in term.params:
                    specs.append(f"{key}={term.params[key]}")
        
        specs_str = ", ".join(specs) if specs else "—"
        lines.append(f"| `{rname}` | {term.weight} | {specs_str} |")
    lines.append("")

    # 4. Domain Randomization (Events)
    lines.append("## 4. Domain Randomization (Events)")
    lines.append("| Event | Mode | Function | Ranges / Params |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for ename, term in env_cfg.events.items():
        # Try to find range-like parameters dynamically
        found_range = None
        for rkey in ["ranges", "ranges_dict", "delay_range", "kp_range", "range"]:
            if rkey in term.params:
                found_range = term.params[rkey]
                break
        
        ranges_str = str(found_range) if found_range is not None else "—"
        func_name = getattr(term.func, "__name__", str(term.func))
        lines.append(f"| `{ename}` | {term.mode} | {func_name} | {ranges_str} |")
    lines.append("")

    # 5. Terminations
    lines.append("## 5. Terminations")
    lines.append("| Termination | Condition / Threshold |")
    lines.append("| :--- | :--- | :--- |")
    for tname, term in env_cfg.terminations.items():
        threshold = term.params.get('threshold', '—')
        lines.append(f"| `{tname}` | {threshold} |")
    lines.append("")

    # 6. PPO Configuration
    lines.append("## 6. RL Hyperparameters (PPO)")
    if hasattr(rl_cfg, 'algorithm'):
        alg = rl_cfg.algorithm
        lines.append(f"- **Learning Rate**: {getattr(alg, 'learning_rate', '—')}")
        lines.append(f"- **Entropy Coef**: {getattr(alg, 'entropy_coef', '—')}")
        lines.append(f"- **Gamma / Lambda**: {getattr(alg, 'gamma', '—')} / {getattr(alg, 'lam', '—')}")
    # 7. Curriculum
    if hasattr(env_cfg, 'curriculum') and env_cfg.curriculum:
        active_cur = {k: v for k, v in env_cfg.curriculum.items()}
        if active_cur:
            lines.append("## 7. Curriculum")
            lines.append("| Term | Parameters |")
            lines.append("| :--- | :--- |")
            for cname, term in active_cur.items():
                lines.append(f"| `{cname}` | {term.params} |")
            lines.append("")
    
    return "\n".join(lines)

def log_summary_as_artifact(env_cfg, rl_cfg, run_name: str = "run"):
    """Starts a background thread to log the MD summary as a WandB Artifact.
    
    This avoids blocking if WandB is initialized later in the training loop.
    """
    import threading
    import time
    
    def _deferred_log():
        timeout = 30  # seconds
        start_time = time.time()
        print("[INFO] Summary artifact logger waiting for WandB to initialize...")
        
        while time.time() - start_time < timeout:
            import wandb
            if wandb.run:
                try:
                    summary_md = generate_env_summary(env_cfg, rl_cfg)
                    
                    # Save locally for reference
                    local_path = os.path.join(wandb.run.dir, "env_summary.md")
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, "w") as f:
                        f.write(summary_md)
                        
                    # Log as artifact
                    artifact = wandb.Artifact(
                        name=f"env_config_summary",
                        type="documentation",
                        description="Dynamically generated environment configuration summary"
                    )
                    artifact.add_file(local_path)
                    wandb.run.log_artifact(artifact)
                    print(f"[INFO] Environment summary successfully logged to WandB run: {wandb.run.name}")
                    return
                except Exception as e:
                    print(f"[WARN] Failed to log summary artifact: {e}")
                    return
            time.sleep(1.0)
        print("[WARN] Summary artifact logger timed out after 30s. No WandB run detected.")

    thread = threading.Thread(target=_deferred_log, daemon=True)
    thread.start()
    print("[INFO] Summary artifact will be logged asynchronously once WandB starts.")
