# A-RMA: Adaptive Rapid Motor Adaptation for Kangaroo

This document explains the 3-phase training pipeline implemented for the Kangaroo robot to achieve robust locomotion across varying physical environments.
This pipeline is fully integrated into the standard `mjlab` runner ecosystem.

## Phase 1: Privileged Training
**Goal:** Train a base policy that can walk perfectly given exact physical knowledge.

- **Architecture:** 
    - **Actor:** Takes `[obs, z]`, where `z` is an 8D latent vector compressed from the privileged physics `e`.
    - **Critic:** Takes `[obs, e]`, seeing the full raw physics.
    - **Encoder:** An MLP trained via the Actor's PPO loss to extract relevant features from `e`.

## Phase 2: Adaptation Module (DAgger)
**Goal:** Train a TCN to "guess" the latent vector `z` from history alone.

- **Architecture:** 
    - **TCN (Adaptation Module):** A 1D Convolutional Network that maps a 75-frame history of observations to an estimate `z_hat`.
- **Training (DAgger):**
    - Executed internally by the runner. The robot walks using `z_hat` predicted by the TCN.
    - We calculate the **perfect** `z` using the Phase 1 Encoder.
    - The TCN is updated via Supervised MSE loss.

## Phase 3: Fine-Tuning for Robustness
**Goal:** Teach the Actor to handle the TCN's estimation errors.

- **Procedure:**
    1. The Phase 1 Encoder is removed.
    2. The TCN is wired into the Actor and **frozen**.
    3. The Actor and Critic weights are unfrozen and PPO training resumes.

## Usage

The entire 3-phase A-RMA pipeline is completely encapsulated within the standard `mjlab` runner structure via the `ArmaOnPolicyRunner` class.

### Running the automated pipeline
Because A-RMA is built directly into the `Mjlab-Tracking-Flat-Pal-Kangaroo` task, you can launch the entire 3-phase training natively:

```bash
uv run train Mjlab-Tracking-Flat-Pal-Kangaroo \
    --registry-name l_barbieri-sapienza-universit-di-roma/csv_to_npz/jump_retarget \
    --env.scene.num-envs 4096 \
    --agent.logger wandb \
    --agent.wandb-project kangaroo-arma \
    --agent.run-name my_experiment
```

### Modifying Phase Iterations
All hyper-parameters for phases are automatically parsed by `tyro`. You can modify how long each phase trains using CLI flags:

```bash
uv run train Mjlab-Tracking-Flat-Pal-Kangaroo \
    --agent.p1-iterations 500 \
    --agent.p2-iterations 50 \
    --agent.p3-iterations 100 \
    ...
```

### State Preservation & Fail-safes
If a particular phase fails (e.g. system crash), the `ArmaOnPolicyRunner` automatically executes intermediate saves (`model_phase1_end.pt`, `tcn_phase2_end.pt`) so you have fallback weights.

### Deployment
The policy exported automatically at the end of Phase 3 is a single ONNX file (`policy.onnx`) that includes the **TCN-History Buffer** inside. It takes a 75-frame tensor as input and outputs actions directly.
