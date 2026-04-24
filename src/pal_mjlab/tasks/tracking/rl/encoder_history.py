"""Custom RL Runner for Kangaroo Tracking with History Encoder Support."""

import os
from typing import cast

import torch
from torch import nn
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)


class _PalOnnxMotionHistoryModel(nn.Module):
    """ONNX-exportable model that handles current obs AND history buffer."""

    def __init__(self, actor_critic, motion, main_obs_set="actor", history_obs_set="actor_history"):
        super().__init__()
        # as_onnx() returns a wrapper that includes normalizers and the actor network
        self.policy = actor_critic.as_onnx(verbose=False)
        self.main_obs_set = main_obs_set
        self.history_obs_set = history_obs_set
        
        # Motion reference buffers (standard tracking metadata)
        self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
        self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
        self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
        self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
        self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
        self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
        self.time_step_total: int = int(self.joint_pos.shape[0])

    def forward(self, obs, obs_history, time_step):
        # Clamp time_step to [0, total-1]
        time_step_clamped = torch.clamp(
            time_step.long().squeeze(-1), max=self.time_step_total - 1
        )
        
        # Package into dictionary for the HistoryEncoderModel
        # The actor_critic.as_onnx() wrapper's forward expects a dict if the model does
        obs_dict = {
            self.main_obs_set: obs,
            self.history_obs_set: obs_history
        }
        
        return (
            self.policy(obs_dict),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )


class PalStandardOnPolicyRunner(MotionTrackingOnPolicyRunner):
    """Standard tracking runner that logs the environment summary to WandB."""

    def __init__(self, env, runner_cfg, log_dir, device="cuda:0", **kwargs):
        super().__init__(env, runner_cfg, log_dir, device, **kwargs)
        from pal_mjlab.utils.wandb_utils import log_summary_as_artifact
        log_summary_as_artifact(self.env.unwrapped.cfg, self.cfg)


class PalMotionTrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
    """Tracking runner with dynamic ONNX export and WandB summary support."""

    def __init__(self, env, runner_cfg, log_dir, device="cuda:0", **kwargs):
        super().__init__(env, runner_cfg, log_dir, device, **kwargs)
        from pal_mjlab.utils.wandb_utils import log_summary_as_artifact
        log_summary_as_artifact(self.env.unwrapped.cfg, self.cfg)

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        from pal_mjlab.tasks.tracking.kangaroo.custom_models import HistoryEncoderModel
        
        # In this RSL-RL version, get_policy() returns the actor model directly
        model = self.alg.get_policy()
        
        if isinstance(model, HistoryEncoderModel):
            print(f"[INFO] HistoryEncoderModel detected. Using multi-input ONNX export logic.")
            os.makedirs(path, exist_ok=True)
            cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
            
            # Use custom multi-input wrapper
            # IMPORTANT: Capture the original device so we can restore it after export.
            # onnx_wrapper.to("cpu") moves the underlying model in-place.
            original_device = next(model.parameters()).device
            
            onnx_wrapper = _PalOnnxMotionHistoryModel(
                model, 
                cmd.motion,
                main_obs_set=model.main_obs_set,
                history_obs_set=model.history_obs_set
            )
            onnx_wrapper.to("cpu")
            onnx_wrapper.eval()
            
            # Create dummy inputs for tracing
            # current frame: [1, obs_dim]
            obs = torch.zeros(1, model.actor_obs_dim)
            
            # Get history length from the environment group configuration
            obs_mgr = self.env.unwrapped.observation_manager
            group_cfg = obs_mgr.cfg.get(model.history_obs_set)
            history_length = group_cfg.history_length if group_cfg and group_cfg.history_length else 15
            
            obs_history = torch.zeros(1, history_length, model.history_obs_dim)
            
            time_step = torch.zeros(1, 1)
            
            torch.onnx.export(
                onnx_wrapper,
                (obs, obs_history, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=18,
                verbose=verbose,
                # Explicitly name inputs matching the C++ deployment node expectations
                input_names=["actor", "actor_history", "time_step"],
                output_names=[
                    "actions",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ],
                dynamic_axes={},
                dynamo=False,
            )
            
            # Restore original device and training mode
            model.to(original_device)
            model.train()
            
            # Attach metadata (mandatory for C++ deployment)
            try:
                import wandb
                run_name = wandb.run.name if wandb.run else "local"
                
                env = self.env.unwrapped
                metadata = get_base_metadata(env, run_name)
                
                # Add motion-specific metadata
                metadata.update({
                    "anchor_body_name": cmd.cfg.anchor_body_name,
                    "body_names": list(cmd.cfg.body_names),
                })
                
                attach_metadata_to_onnx(os.path.join(path, filename), metadata)
                print(f"[INFO] Metadata attached successfully to {filename}")
            except Exception as e:
                print(f"[WARN] Failed to attach metadata: {e}")

        else:
            # Fallback to standard tracking export logic
            print(f"[INFO] Standard model detected. Using default ONNX export.")
            super().export_policy_to_onnx(path, filename, verbose)
