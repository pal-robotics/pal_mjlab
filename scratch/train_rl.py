import os

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.runner import MjlabOnPolicyRunner as RslRlOnPolicyRunner
from pal_mjlab.tasks.manipulation.tiago_pro.env_cfgs import lift_vision_env_cfg
from pal_mjlab.tasks.manipulation.tiago_pro.rl_cfg import lift_vision_ppo_runner_cfg


def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 1. Setup Environment and Config
  print("Initializing Environment...")
  env_cfg = lift_vision_env_cfg(cam_type="depth")
  # env_cfg.scene.num_envs = 64 # You can adjust this
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")

  print("Setting up RL Runner...")
  train_cfg = lift_vision_ppo_runner_cfg()

  # Ensure log directory exists
  log_dir = os.path.join("logs", "rsl_rl", "lift_depth_curriculum")
  os.makedirs(log_dir, exist_ok=True)

  # 2. Instantiate Runner
  runner = RslRlOnPolicyRunner(env, train_cfg, log_dir, device=str(device))

  # 3. Load Pre-trained Backbone
  backbone_path = "pretrained_backbone.pth"
  if os.path.exists(backbone_path):
    print(f"Loading pre-trained backbone from {backbone_path}...")
    weights = torch.load(backbone_path, map_location=device)

    # Map keys: 'cnn.X.*' -> 'cnns.camera.cnn.X.*'
    mapped_weights = {}
    for k, v in weights.items():
      if k.startswith("cnn."):
        mapped_k = f"cnns.camera.{k}"
        mapped_weights[mapped_k] = v

    # Load into both actor and critic robustly
    for model_attr in ["actor_critic", "actor", "critic"]:
      if hasattr(runner.alg, model_attr):
        obj = getattr(runner.alg, model_attr)
        if model_attr == "actor_critic":
          for sub_model in [obj.actor, obj.critic]:
            sub_model.load_state_dict(mapped_weights, strict=False)
        else:
          obj.load_state_dict(mapped_weights, strict=False)
  else:
    print("Warning: No pre-trained backbone found. Starting from scratch.")

  # 4. INITIAL FREEZE
  print("Freezing CNN backbone for the first 300 iterations...")
  # We freeze both actor and critic CNNs
  for model in [runner.alg.actor_critic.actor, runner.alg.actor_critic.critic]:
    for param in model.cnns.parameters():
      param.requires_grad = False

  # 5. Training Loop with Custom Curriculum
  # We'll run the training in chunks to check the iteration count
  total_iterations = train_cfg.runner.max_iterations
  unfreeze_iteration = 30000
  fine_tune_lr = 1e-5

  print(f"Starting training loop (Total iterations: {total_iterations})...")

  # rsl_rl Runner.learn handles the loop, so we'll need to use a hook
  # if it supports it, OR manually run the iterations.
  # Since rsl_rl's Runner.learn is a big loop, we can't easily break into it
  # without subclassing. Let's do it the clean way: subclassing the Runner.

  class CurriculumRunner(RslRlOnPolicyRunner):
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
      # This is a simplified version of the rsl_rl loop logic
      # to allow for the unfreezing hook.
      self.unfrozen = False

      # Start the standard learn but we'll monitor current_learning_iteration
      # Actually, rsl_rl doesn't have an easy "per-step" hook in the public API.
      # So we will just run the standard learn and hope we can catch it,
      # but better yet, let's just implement the unfreeze at the start
      # if we are resuming, or use the 'alg' update logic.

      # Since rsl_rl calls alg.update() every iteration,
      # we can wrap the algorithm!
      return super().learn(num_learning_iterations, init_at_random_ep_len)

  # Actually, the most robust way to do this in rsl_rl is to
  # set the requires_grad property and then Re-Initialize the optimizer
  # when we want to unfreeze, OR use an optimizer that ignores grad=False.

  # Let's use a simpler approach: Run 300 iterations, then unfreeze, then run the rest.

  print(f"Phase 1: Training MLP only for {unfreeze_iteration} iterations...")
  runner.learn(num_learning_iterations=unfreeze_iteration, init_at_random_ep_len=True)

  print(f"Phase 2: Unfreezing CNN and setting LR to {fine_tune_lr}...")
  for model in [runner.alg.actor_critic.actor, runner.alg.actor_critic.critic]:
    for param in model.cnns.parameters():
      param.requires_grad = True

  # Update Learning Rate in the optimizer
  # rsl_rl optimizer is runner.alg.optimizer
  for param_group in runner.alg.optimizer.param_groups:
    param_group["lr"] = fine_tune_lr

  print(
    f"Resuming training for the remaining {total_iterations - unfreeze_iteration} iterations..."
  )
  runner.learn(
    num_learning_iterations=total_iterations - unfreeze_iteration,
    init_at_random_ep_len=False,
  )

  print("Training complete!")


if __name__ == "__main__":
  train()
