# TIAGo Pro: Vision-Based Manipulation Specification

This document details the configuration for the visual-perception training task.

## 1. Task Identification
*   **Task ID**: `Mjlab-Manipulation-Lift-Cube-Vision-Pal-Tiago-Pro-v0`
*   **Modality**: Depth-based (Default) / RGB (Optional)
*   **Resolution**: 128x128 pixels

## 2. Visual Observation Pipeline
The agent receives visual input from the robot's onboard sensors. Unlike the Oracle task, all ground-truth object coordinates are removed.

| Sensor Name | Attached Link | Frame | Data Type |
| :--- | :--- | :--- | :--- |
| **`wrist_realsense_camera`** | `arm_right_7_link` | Right Palm | RGB + Depth |
| **`head_realsense_camera`** | `head_2_link` | Head | RGB + Depth |

### Observation Processing:
*   **Primary Input**: The `wrist_realsense_camera` is typically used as the primary observation for the `camera` group.
*   **Normalization**: Depth values are clipped to a maximum distance (e.g., 0.5m - 1.0m) to emphasize the workspace and ignore the background.
*   **Encoding**: During training, these 128x128 images are compressed into a latent vector (e.g., size 64 or 128) using a VAE or CNN encoder.

## 3. Proprioceptive State (Actor Group)
Even in vision mode, the agent retains knowledge of its own body state:
*   **Joint States**: Positions and velocities of the 7-DOF arm and gripper.
*   **End-Effector Pose**: The 6D location of the palm in the robot's base frame.
*   **Action History**: The previous command sent to the IK solver.
*   **Goal Vector**: The 3D target position (coordinates where the cube *should* end up).

> [!IMPORTANT]
> **Oracle Data Removed**: In this task, the `object_position` and `object_orientation` terms are explicitly removed from the actor observation group. The agent **must** infer the cube's location from the camera feeds.

## 4. Training Rewards
The reward function remains identical to the Oracle task to ensure a smooth transition from coordinate-based learning to vision-based learning:
*   **Reaching**: `5.0` (Encourages moving hand toward visual features of the cube).
*   **Lifting**: `5.0` (Triggered when the cube enters the "lifted" state).
*   **Tracking**: `25.0` + `10.0` (Reward for moving the detected cube toward the goal).

## 5. Visual Testing Command
Use this command to verify the vision feed in the viewer:
```bash
uv run python3 -m mjlab.scripts.play Mjlab-Manipulation-Lift-Cube-Vision-Pal-Tiago-Pro-v0 --num_envs 1 --agent zero --viewer viser
```
