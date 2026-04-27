"""Custom RL Runner for A-RMA 3-phase training orchestration."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner
from pal_mjlab.tasks.tracking.rl.encoder_history import PalMotionTrackingOnPolicyRunner
from mjlab.utils.os import get_checkpoint_path


class ArmaOnPolicyRunner(PalMotionTrackingOnPolicyRunner):
    """Orchestrates Phase 1 (PPO), Phase 2 (DAgger), and Phase 3 (Finetune) sequentially
    within a single standard `mjlab` CLI execution.
    """

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the full 3-phase A-RMA curriculum."""
        # Use custom config properties if available, fallback to defaults
        p1_iters = self.cfg.get("p1_iterations", 25000)
        p2_iters = self.cfg.get("p2_iterations", 5000)
        p3_iters = self.cfg.get("p3_iterations", 10000)

        # Prevent RSL-RL from closing WandB at the end of Phase 1 or 3
        # We manually manage the closure after all phases complete.
        original_stop = self.logger.stop_logging_writer
        self.logger.stop_logging_writer = lambda: None

        # ---------------------------------------------------------------------
        # Phase 1: Privileged Training (Standard PPO)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("[INFO] STARTING A-RMA PHASE 1: PRIVILEGED TRAINING")
        print("=" * 80 + "\n")
        
        super().learn(num_learning_iterations=p1_iters, init_at_random_ep_len=init_at_random_ep_len)
        
        # Explicit intermediate save to protect progress if later phases crash
        intermediate_p1_path = os.path.join(self.logger.log_dir, f"model_phase1_end.pt")
        self.save(intermediate_p1_path)
        print(f"[INFO] Phase 1 complete. Saved checkpoint to {intermediate_p1_path}")

        # ---------------------------------------------------------------------
        # Phase 2: DAgger Adaptation Module (TCN)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("[INFO] STARTING A-RMA PHASE 2: ADAPTATION MODULE (DAgger)")
        print("=" * 80 + "\n")
        
        self._run_dagger(p2_iters)
        
        # Intermediate save for TCN
        intermediate_p2_path = os.path.join(self.logger.log_dir, f"tcn_phase2_end.pt")
        torch.save(self.tcn.state_dict(), intermediate_p2_path)
        print(f"[INFO] Phase 2 complete. Saved TCN to {intermediate_p2_path}")

        # ---------------------------------------------------------------------
        # Phase 3: Fine-tuning for Robustness
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("[INFO] STARTING A-RMA PHASE 3: FINE-TUNING")
        print("=" * 80 + "\n")
        
        self._inject_tcn()
        super().learn(num_learning_iterations=p3_iters, init_at_random_ep_len=False)
        
        # Save final policy
        final_path = os.path.join(self.logger.log_dir, f"model_final.pt")
        self.save(final_path)

        print("\n" + "=" * 80)
        print("[INFO] A-RMA PIPELINE: ALL PHASES SUCCESSFULLY COMPLETED")
        print("=" * 80 + "\n")
        
        # Cleanly shut down logging
        self.logger.stop_logging_writer = original_stop
        if getattr(self.logger, "writer", None) is not None:
            self.logger.stop_logging_writer()


    def _run_dagger(self, num_iterations: int):
        """Train the TCN using Supervised Learning against the Privileged Encoder."""
        from pal_mjlab.tasks.tracking.kangaroo.custom_models import AdaptationModule
        
        # Phase 2 & 3: Disable RSI and force static starts (v=0)
        self._disable_rsi()
        
        # Gather environment geometry
        obs = self.env.get_observations()
        input_dim = obs["actor"].shape[-1]
        
        # Safely inspect history length from env_cfg
        obs_mgr = self.env.unwrapped.observation_manager
        group_cfg = obs_mgr.cfg.get("actor")
        history_length = group_cfg.history_length if group_cfg and group_cfg.history_length else 75
        latent_dim = 8
        
        learning_rate = self.cfg.get("p2_learning_rate", 1e-3)
        batch_size = self.cfg.get("p2_batch_size", 8192)

        self.tcn = AdaptationModule(input_dim, history_length, latent_dim).to(self.device)
        optimizer = optim.Adam(self.tcn.parameters(), lr=learning_rate)
        mse_loss = nn.MSELoss()

        actor = self.alg.actor
        privileged_encoder = actor.privileged_encoder
        num_steps_per_rollout = self.cfg.get("num_steps_per_env", 24)

        # Freeze network parameters for execution
        for param in self.alg.actor.parameters(): param.requires_grad = False
        for param in self.alg.critic.parameters(): param.requires_grad = False
        self.alg.actor.eval()
        self.alg.critic.eval()
        self.tcn.train()

        start_it = self.current_learning_iteration
        pbar = tqdm(range(num_iterations))
        
        for logic_it in pbar:
            it = start_it + logic_it
            
            # --- Collection ---
            histories = []
            latents_gt = []
            obs = self.env.get_observations()

            for _ in range(num_steps_per_rollout):
                with torch.inference_mode():
                    history_t = obs["actor"]
                    z_hat_t = self.tcn(history_t)
                    actions = actor(obs, z_hat=z_hat_t)
                    
                    e_t = obs["privileged"]
                    e_t_norm = actor.priv_normalizer(e_t)
                    z_gt_t = privileged_encoder(e_t_norm)
                    
                    obs, rewards, dones, _ = self.env.step(actions.to(self.device))
                    obs = {k: v.to(self.device) for k, v in obs.items()}
                    dones = dones.to(self.device)

                    # Filter terminal frames
                    active_mask = ~dones.bool()
                    if active_mask.any():
                        histories.append(history_t[active_mask])
                        latents_gt.append(z_gt_t[active_mask])

            # --- Update ---
            if not histories:
                self.current_learning_iteration += 1
                continue

            batch_histories = torch.cat(histories, dim=0)
            batch_latents_gt = torch.cat(latents_gt, dim=0)
            
            indices = torch.randperm(batch_histories.size(0))
            batch_histories = batch_histories[indices]
            batch_latents_gt = batch_latents_gt[indices]

            total_samples = batch_histories.size(0)
            bs = min(batch_size, total_samples)
            num_mini_batches = max(1, total_samples // bs)

            epoch_loss = 0
            for i in range(num_mini_batches):
                start = i * bs
                end = start + bs
                x = batch_histories[start:end]
                y_gt = batch_latents_gt[start:end]

                y_pred = self.tcn(x)
                loss = mse_loss(y_pred, y_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_mini_batches
            pbar.set_description(f"MSE Loss: {avg_loss:.6f}")

            # --- Logging ---
            if getattr(self.logger, 'writer', None) is not None and getattr(self.logger.writer, 'add_scalar', None):
                self.logger.writer.add_scalar("Phase2/mse_loss", avg_loss, global_step=it)
                self.logger.writer.add_scalar("Phase2/learning_rate", optimizer.param_groups[0]["lr"], global_step=it)
                
                with torch.inference_mode():
                    last_x = batch_histories[-bs:]
                    last_y_gt = batch_latents_gt[-bs:]
                    last_y_pred = self.tcn(last_x)
                    per_dim_mae = (last_y_pred - last_y_gt).abs().mean(dim=0)
                
                for d, mae_d in enumerate(per_dim_mae.tolist()):
                    self.logger.writer.add_scalar(f"Phase2/z_mae_dim{d}", mae_d, global_step=it)

            # Manually increment the shared learning step counter
            self.current_learning_iteration += 1


    def _inject_tcn(self):
        """Freeze TCN, inject it into the Actor, unfreeze Actor and Critic to resume PPO."""
        for param in self.tcn.parameters():
            param.requires_grad = False
        self.tcn.eval()
        
        self.alg.actor.adaptation_module = self.tcn

        for param in self.alg.actor.parameters():
            param.requires_grad = True
        for param in self.alg.critic.parameters():
            param.requires_grad = True

        self.alg.actor.train()
        self.alg.critic.train()

    def _disable_rsi(self):
        """Disable RSI and force stable static starts for Phase 2 and 3."""
        print("[INFO] Disabling RSI and enforcing static starts (v=0).")
        cmd = self.env.unwrapped.command_manager.get_term("motion")
        cmd.cfg.sampling_mode = "start"
        cmd.cfg.pose_range = {}
        cmd.cfg.velocity_range = {}
        cmd.cfg.joint_position_range = (0.0, 0.0)
        
        if hasattr(cmd.cfg, "joint_position_ranges"):
            cmd.cfg.joint_position_ranges = {}

        # Hardcode initial frame velocities to 0 for a true static start
        cmd.motion.joint_vel[0] = 0.0
        cmd.motion.body_lin_vel_w[0] = 0.0
        cmd.motion.body_ang_vel_w[0] = 0.0
