[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![mjlab](https://img.shields.io/badge/mjlab-1.1.1-76B900.svg)](https://mujocolab.github.io/mjlab/v1.1.1/index.html)

# PAL Robotics in mjlab

This repository showcases the implementation of [PAL Robotics](https://pal-robotics.com/)' robots in [mjlab](https://github.com/mujocolab/mjlab).

![Project banner](static/pal_mjlab_viser.png)

## What is mjlab?

mjlab brings the [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) API to [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html). 
It's lightweight, easy to install, and has been validated for sim-to-real transfer on the G1 and Go1 robots for RL locomotion and motion imitation. 
See the [announcement thread](https://x.com/kevin_zakka/status/1972757435707424898) for videos, or read about the [motivation behind mjlab](https://github.com/mujocolab/mjlab/blob/main/docs/motivation.md).

## Installation

Install uv.
```bash
curl -LsSf https://astral.sh/uv/install.sh \| sh
```

Clone the repository.

```bash
git clone https://github.com/pal-robotics/pal_mjlab.git 
cd pal_mjlab
uv sync
```

## Quick Start

List available environments.

```bash
uv run list_envs --keyword pal
```

Test with dummy agents.

```bash
uv run play Mjlab-Velocity-Flat-Pal-Talos --agent zero    # send zero actions
uv run play Mjlab-Velocity-Flat-Pal-Talos --agent random  # send random actions
```


## Velocity Tracking

Train a locomotion policy.

```bash
uv run train Mjlab-Velocity-Flat-Pal-Talos --env.scene.num-envs 4096
```

Evaluate a trained policy.

```bash
uv run play Mjlab-Velocity-Flat-Pal-Talos --wandb-run-path your-org/mjlab/run-id
```


## Motion Imitation

We added support to [GMR](https://github.com/YanjieZe/GMR) to retarget animations from the [LaFAN1 dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) for PAL robots (KANGAROO and Talos).

### Retargeting a new motion

First, use GMR to retarget and convert a motion file. Note that for convenience we provide a few retargeted motions under `motions` folder.

Then convert the CSV to NPZ format.

```bash
uv run -m pal_mjlab.scripts.csv_to_npz \
    --input-file motions/kang_walk.csv \
    --output-name kang_walk \
    --input-fps 30 \
    --output-fps 50 \
    --robot-name kangaroo \
    --render True
```

### Training and evaluation

Train.

```bash
uv run train Mjlab-Tracking-Flat-Pal-Kangaroo \
    --registry-name your-org/csv_to_npz/kang_walk \
    --env.scene.num-envs 4096
```

Evaluate.

```bash
uv run play Mjlab-Tracking-Flat-Pal-Kangaroo --wandb-run-path your-org/mjlab/run-id
```

## Results

<table>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/a378a306-f65d-448b-a6c8-3d40c3f7b3ce" controls muted loop playsinline style="width:100%;"></video>
      <p align="center"><em>Velocity Tracking for Talos</em></p>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/c0d6c444-a0a2-4c93-99a5-895ae7d31317" controls muted loop playsinline style="width:100%;"></video>
      <p align="center"><em>Motion Imitation for Talos</em></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/cce8f9d2-d5e1-4218-828e-88d0ae1e0cdb" controls muted loop playsinline style="width:100%;"></video>
      <p align="center"><em>Velocity Tracking for Kangaroo</em></p>
    </td>
    <td width="50%"></td>
  </tr>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/fef23ffe-f644-4616-9a3c-780b969a4f23" controls muted loop playsinline style="width:100%;"></video>
      <p align="center"><em>Reaching policy for TIAGo Pro (sim)</em></p>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/85da2add-ffe7-485b-b4e7-d3c20a435012" controls muted loop playsinline style="width:100%;"></video>
      <p align="center"><em>Deployed Reaching policy for TIAGo Pro (using interactive marker RViz)</em></p>
    </td>
  </tr>
</table>

## Contributing

Contributions are welcome! 
Please open an issue to discuss proposed changes or report bugs. 
As mjlab is in early development, breaking changes may occur—thank you for your patience.

## Acknowledgements

Thanks to the teams behind [mjlab](https://github.com/mujocolab/mjlab), [PAL Robotics](https://pal-robotics.com/), [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html), [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html), and the [Inria HUCEBOT Team](https://team.inria.fr/hucebot/).

## License

See [LICENSE](LICENSE) for details.
