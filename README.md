# gen_hoi

Flow-matching experiments for generating physically plausible humanoid–object interaction motions. The project models Unitree G1 and object trajectories jointly, with optional contact labels and fingertip–object geometry.

> **Status:** Research prototype under active development. Training data and checkpoints are not included in this repository.

## Overview

The pipeline:

1. preprocesses humanoid and object trajectories into fixed-length motion windows;
2. optionally augments each frame with contact labels, fingertip closest points, and object-surface normals;
3. trains a conditional 1D flow-matching model with either a DiT or UNet backbone;
4. generates future motion from an observed prefix; and
5. evaluates motion consistency and interaction plausibility with geometry-based metrics.

## Motion representation

Each frame can include:

- G1 joint positions and velocities;
- root position, orientation, linear velocity, and angular velocity;
- object position, orientation, linear velocity, and angular velocity;
- optional per-link contact labels;
- optional fingertip–object closest points and surface normals.

Relative coordinates and dataset normalization are supported by the data loader.

## Models

- **Diffusion Transformer (DiT):** transformer-based 1D trajectory generator.
- **Conditional UNet:** convolutional 1D trajectory generator.
- **Linear flow matching:** learns a velocity field between noise and motion trajectories.
- **Prefix conditioning:** keeps the first motion frames fixed while generating the remaining sequence.

## Physical-plausibility metrics

The evaluation utilities include:

- root, joint, and object velocity consistency;
- hand–object distance;
- contact-weighted hand–object sliding;
- foot sliding;
- foot penetration and floating;
- geometry queries based on robot collision points and object meshes.

## Setup

The project requires Python 3.13 and uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The PyTorch configuration currently targets CUDA 12.8.

## Data

Training expects one or more motion `.npz` files containing G1 joint/body states and object states. Contact and fingertip-geometry features can be generated with:

```bash
uv run python scripts/preprocess_hand.py --help
uv run python scripts/preprocess_geometry.py --help
```

The dataset itself is not distributed with this repository.

## Training

Train the default DiT flow-matching model:

```bash
uv run python scripts/train.py --data <motion.npz> --run-name <experiment>
```

To use the optional interaction features:

```bash
uv run python scripts/train.py \
  --data <motion.npz> \
  --use-contact-label \
  --use-closest-points \
  --use-surface-normals
```

Training configurations, checkpoints, metrics, and Weights & Biases logs are stored under `outputs/<run-name>/`.

## Sampling

Generate and evaluate rollouts from a trained checkpoint:

```bash
uv run python scripts/sample.py \
  --run-dir outputs/<run-name> \
  --num-rollouts 6
```

## Repository structure

```text
scripts/
  train.py                  training entry point
  sample.py                 trajectory rollout and evaluation
  preprocess_hand.py        contact-label preprocessing
  preprocess_geometry.py    fingertip–object geometry preprocessing
src/
  dataset.py                motion representation and windowing
  flow.py                   flow-matching objective and sampler
  models/                   DiT, UNet, and PointNet modules
  geometry/                 robot–object geometry queries
  metrics/                  kinematic and physical-plausibility metrics
vis/
  vis_mujoco.py             MuJoCo visualization
  vis_viser.py              interactive trajectory visualization
tests/                      dataset, geometry, and metric tests
```

## Development checks

```bash
uv run pytest -q
```
