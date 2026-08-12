# TIAGo Pro Manipulation Implementation

This directory contains the reinforcement learning (RL) manipulation environment setup for the **PAL TIAGo Pro** robot in `mjlab`.

The primary registered task is:
- **Task ID**: `Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0`

---

## 📁 File & Directory Overview

```
src/pal_mjlab/tasks/manipulation/
├── README.md               # Task documentation (this file)
├── __init__.py             # Task module initializer
├── tiago_pro/              # Task definitions and configuration for PAL TIAGo Pro
│   ├── __init__.py         # Registers task with mjlab registry
│   ├── env_cfgs.py         # Full environment configuration (scene, actions, obs, rewards, events, terminations)
│   └── rl_cfg.py           # PPO runner and neural network architecture config (RSL-RL)
└── mdp/
    ├── __init__.py         # Exports all MDP modules
    ├── commands.py         # Lifting command generator, procedural table/box XML specs, metric tracking
    ├── contact_sensor.py   # Fingertip proximity and contact helper functions
    ├── curriculums.py      # Curriculum learning utilities
    ├── events.py           # Custom resets and domain randomization (table height, joint resets)
    ├── metrics.py          # Metric logging terms (object height, position error, success rate, etc.)
    ├── observations.py     # Relative frame transformations (object pose, EE pos, reached flag, contact)
    ├── rewards.py          # Shaped rewards and penalties (reaching, lifting, goal tracking, stability, collision)
    ├── terminations.py     # Failure and success termination conditions
    └── utils.py            # Utility decorators (e.g., `nan_safe`)
```

### Detailed File Descriptions

#### Core Task Registration & Configs (`tiago_pro/`)
- [`tiago_pro/__init__.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/tiago_pro/__init__.py): Registers `Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0` into the `mjlab` task registry using `ManipulationOnPolicyRunner`.
- [`tiago_pro/env_cfgs.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/tiago_pro/env_cfgs.py): Defines `lift_env_cfg()`, configuring simulation parameters ($dt = 0.005\text{ s}$, decimation = 4 $\rightarrow 50\text{ Hz}$ control rate, 4s episodes), scene entities (robot, table, box), sensor suite, asymmetric actor/critic observations, reward terms, domain randomizations, and terminations.
- [`tiago_pro/rl_cfg.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/tiago_pro/rl_cfg.py): Defines `lift_ppo_runner_cfg()`, providing hyperparameter settings for PPO training via RSL-RL (Actor/Critic MLPs: `512 x 256 x 128`, ELU activations, adaptive LR schedule).

#### MDP Infrastructure (`mdp/`)
- [`mdp/commands.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/mdp/commands.py): Implements `LiftingCommand` and `LiftingCommandCfg`. Generates random box spawn poses and target goal 3D positions in space, procedural MuJoCo XML specs for table (`get_table_spec`) and box (`get_box_spec`), and manages episode metrics (`reached`, `at_goal_time`, `grasped_distance`).
- [`mdp/contact_sensor.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/mdp/contact_sensor.py): Provides `site_contact_both_fingers()`, which checks proximity and contact between both right fingertips and the target object.
- [`mdp/events.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/mdp/events.py): Contains custom domain randomization terms including `randomize_table_height` (shifts table top while keeping it grounded) and `reset_joints_mixed` (initializes arm joints near default or goal configurations).
- [`mdp/observations.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/mdp/observations.py): Features spatial observation calculators relative to the robot root frame (`object_position_in_robot_root_frame`, `object_yaw_in_robot_root_frame`, `ee_position_in_robot_base_frame`, `reached_flag`, `object_both__contact_fingers`).
- [`mdp/rewards.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/mdp/rewards.py): Collection of shaped reward terms (`reaching_object`, `gripper_open_during_approach`, `lifting_object`, `object_goal_tracking`, `post_reached_ee_stability`, `post_reached_gripper_open`) and penalty terms (`top_surface_penetration_penalty`, `object_table_sliding_penalty`, `fingertip_cube_alignment_reward_adaptive`, `arm_right_1_joint_limit_penalty`, `self_collisions`).
- [`mdp/terminations.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/mdp/terminations.py): Defines episode termination conditions (`object_released_on_floor_term`, `cube_contact_with_table_after_reached_term`, `cube_fell_off_table_term`, `top_surface_penetration_term`, `nan_term`).
- [`mdp/utils.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/mdp/utils.py): Provides `@nan_safe` wrapper decorator to sanitize potential NaNs/Infs in reward outputs.

---

## 🦾 Controlled Joints & Actuation Model

### 1. Controlled Active Joints (8 DOFs)
The policy directly outputs relative joint position commands for **8 degrees of freedom**:

| Joint Name | Actuator Category | Target Match Expression | Action Scaling Factor | Description |
| :--- | :--- | :--- | :--- | :--- |
| `arm_right_1_joint` | **S_PLUS** | `arm_right_1_joint` | $0.05 \times \frac{\text{effort}}{\text{stiffness}} \approx 0.0051$ | Right shoulder pitch |
| `arm_right_2_joint` | **S_PLUS** | `arm_right_2_joint` | $0.05 \times \frac{\text{effort}}{\text{stiffness}} \approx 0.0051$ | Right shoulder roll |
| `arm_right_3_joint` | **S_MINUS** | `arm_right_3_joint` | $0.05 \times \frac{\text{effort}}{\text{stiffness}} \approx 0.0047$ | Right shoulder yaw |
| `arm_right_4_joint` | **S_MINUS** | `arm_right_4_joint` | $0.05 \times \frac{\text{effort}}{\text{stiffness}} \approx 0.0047$ | Right elbow pitch |
| `arm_right_5_joint` | **S_MINUS** | `arm_right_5_joint` | $0.05 \times \frac{\text{effort}}{\text{stiffness}} \approx 0.0047$ | Right wrist roll |
| `arm_right_6_joint` | **XS** | `arm_right_6_joint` | $0.05 \times \frac{\text{effort}}{\text{stiffness}} \approx 0.0047$ | Right wrist pitch |
| `arm_right_7_joint` | **XS** | `arm_right_7_joint` | $0.05 \times \frac{\text{effort}}{\text{stiffness}} \approx 0.0047$ | Right wrist yaw |
| `gripper_right_finger_joint` | **GRIPPER** | `gripper_right_finger_joint` | $0.01$ (tuned) | Active right parallel gripper finger |

### 2. Uncontrolled / Fixed Joints
- **Torso**: `torso_lift_joint` is held at `0.0` (fixed elevation).
- **Left Arm**: `arm_left_1_joint` through `arm_left_7_joint` are maintained at their default resting joint postures.


## 🎯 Task Specifications (`Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0`)

### Task Workflow
1. **Approach**: Robot arm moves end-effector towards the spawned box on the table while maintaining an open gripper.
2. **Grasp**: Parallel fingertips close around the box without penetrating the top surface or sliding the box on the table.
3. **Lift**: Object is elevated off the table towards the target 3D goal coordinate.
4. **Reach & Hold**: Object is held within $0.05\text{ m}$ (`success_threshold`) of target coordinate for at least $0.1\text{ s}$ (`holding_time`). Once held for this duration, `reached` becomes `True`.
   > **Note on Tuning**: `holding_time` (default `0.1` seconds) and `success_threshold` (default `0.05` m) are parameters of [`LiftingCommandCfg`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/mdp/commands.py#L228) and can be configured in [`tiago_pro/env_cfgs.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/tiago_pro/env_cfgs.py#L116) under `cfg.commands["lift_height"]`.
5. **Release**: Gripper opens after reaching the goal, letting the object settle (`episode_success = True`).

### Observation Space
The environment uses **asymmetric actor-critic observations**:

| Term | Dimension | Description | Noise (Actor, Training) | Noise (Critic) |
| :--- | :---: | :--- | :--- | :--- |
| `joint_pos` | 7 | Relative joint positions of 7 right arm joints | $\mathcal{U}(-0.02, 0.02)$ | Clean |
| `joint_vel` | 7 | Joint velocities of 7 right arm joints | $\mathcal{U}(-0.05, 0.05)$ | Clean |
| `object_position` | 3 | Box center position in robot root frame | $\mathcal{U}(-0.01, 0.01)$ | Clean |
| `object_yaw` | 2 | $[\cos(\psi), \sin(\psi)]$ of box relative yaw | $\mathcal{U}(-0.05, 0.05)$ | Clean |
| `target_object_position` | 3 | Target goal position in robot base frame | None | Clean |
| `gripper_pos` | 1 | Right gripper finger joint position | $\mathcal{U}(-0.003, 0.003)$ | Clean |
| `ee_position` | 3 | End-effector grasping site in base frame | $\mathcal{U}(-0.01, 0.01)$ | Clean |
| `reached_flag` | 1 | Binary flag ($1.0$ once goal held for $\ge 0.1\text{ s}$) | None | Clean |
| `object_both__contact_fingers` | 1 | Binary flag ($1.0$ if both fingertips touch box) | None | Clean |

### Reward Structure
- `reaching_object` (weight: $+3.0$): Distance adaptive reward from end-effector to box.
- `gripper_open_during_approach` (weight: $+1.0$): Encourages keeping gripper open while far from box.
- `lifting_object` (weight: $+1.0$): Rewards lifting box off table surface while maintaining fingertip contact.
- `object_goal_tracking` (weight: $+5.0$): Weighted distance tracking reward ($z$ axis weight 3.0) between box and target.
- `object_contact_both_fingers` (weight: $+1.0$): Dual fingertip contact bonus.
- `post_reached_ee_stability` (weight: $+3.0$): End-effector position stability reward after goal reached.
- `post_reached_gripper_open` (weight: $+5.0$): Exponential reward for opening gripper after `reached`.
- `top_surface_penetration_penalty` (weight: $-5.0$): Penalty for fingertip penetrating top box surface.
- `object_table_sliding_penalty` (weight: $-5.0$): Penalty for sliding box along table surface before lift.
- `fingertip_cube_alignment` (weight: $-5.0$): Penalty for planar yaw misalignment or vertical tilt.
- `arm_right_1_joint_limit_penalty` (weight: $-0.5$): Exponential penalty if `arm_right_1_joint` drops below $-0.35\text{ rad}$.
- `action_rate_l2` (weight: $-0.01$): Action smoothness penalty.
- `joint_torques_l2` (weight: $-0.0005$): Torque minimization penalty.
- `self_collisions` (weight: $-2.0$): Penalty for self-collisions or robot-table collisions.

### Domain Randomization & Sim2Real Features
- **Table Height**: `randomize_table_height` shifts working surface height by $\pm 10\text{ cm}$ while keeping table grounded.
- **Box Dimensions**: Box size randomized independently in X ($1 - 2.5\text{ cm}$), Y ($1 - 2.5\text{ cm}$), Z ($2 - 4\text{ cm}$).
- **Box Mass**: Pseudo-inertia randomization (`alpha_range`: $0.0 - 1.1513$).
- **Joint Resets**: `reset_robot_joints` randomizes arm joints with a $50\%$ probability of initializing near a pre-grasp goal configuration.
- **Encoder Bias**: Random zero-point offset per joint ($\pm 0.01\text{ rad}$) at startup.
- **Actuator Friction**: DOF friction loss variability ($\pm 0.005$) across robot units at startup.

---

## 🔄 Adapting Task for Grasping & Reaching Only (Pick-and-Hold)

To modify the pipeline for a **Pick-and-Hold** task (where the robot's goal is to pick up the object and hold it at the target position without releasing it), make the following adjustments:

1. **Remove `reached_flag` from Observations** ([`tiago_pro/env_cfgs.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/tiago_pro/env_cfgs.py#L186)):
   - Remove or pop `terms["reached_flag"]` from both `actor` and `critic` observation dictionaries so the policy does not observe the phase switch signal.

2. **Remove Post-Reached Rewards** ([`tiago_pro/env_cfgs.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/tiago_pro/env_cfgs.py#L319)):
   - Remove `post_reached_ee_stability` and `post_reached_gripper_open` from `cfg.rewards` to avoid rewarding gripper opening after reaching.

3. **Update Success Condition in Command** ([`mdp/commands.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/mdp/commands.py#L150)):
   - Change the success evaluation in `LiftingCommand._update_metrics()` and `compute_success()` from `self.reached & ~contact_both` (held + fingers released) to simply `self.reached` (held at goal).

4. **Update Success Termination** ([`mdp/terminations.py`](file:///home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/src/pal_mjlab/tasks/manipulation/mdp/terminations.py#L89)):
   - Change `object_released_on_floor_term` (or create a dedicated `object_reached_goal_term`) to trigger when `command.reached` is `True`, rather than requiring fingertip contact release (`command.reached & ~contact_both`).
