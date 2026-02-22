# Semantic Segmentation for Sentinel-1 SAR Data

A semantic segmentation framework for Sentinel-1 Synthetic Aperture Radar (SAR) data processing. This project provides tools for preprocessing point cloud data, inference using deep learning models, and containerized deployment.

## Overview

This repository contains a complete pipeline for semantic segmentation tasks, built on top of the Pointcept framework. It includes:

- **Preprocessing**: Tiling and feature extraction from LAS point cloud files
- **Inference**: Model inference on processed tiles using PyTorch
- **Containerization**: Docker/Podman support for reproducible environments with GPU acceleration

## Project Structure

```
semantic-segmentation-s1/
├── configs/                      # Model configuration files
│   └── s1_inference.py          # Inference configuration (16 semantic classes)
├── data/
│   └── processed_tiles/          # Directory for processed tile data
│       └── tiling_metadata.json  # Metadata from tiling process
├── scripts/                       # Executable scripts
│   ├── run_inference.py          # Main inference script
│   └── inspect_tile.py           # Utility for inspecting tile data
├── src/
│   └── preprocessing/             # Preprocessing utilities
│       ├── tile_s1.py            # Main tiling script for LAS files
│       └── tile_utils.py         # Helper functions for tiling
├── Containerfile                 # Container definition for Podman/Docker
├── Makefile                      # Build and deployment commands
└── requirements.txt              # Python dependencies
```

## Features

### Semantic Classes

The model supports 16 semantic classes:
- barrier, bicycle, bus, car, construction_vehicle
- motorcycle, pedestrian, traffic_cone, trailer, truck
- driveable_surface, other_flat, sidewalk, terrain, manmade
- vegetation

### Key Dependencies

- **PyTorch & PyTorch Geometric**: Deep learning framework
- **Open3D**: Point cloud processing
- **LASpy**: LAS/LAZ point cloud file I/O
- **Pointcept**: Core segmentation framework
- **PEFT**: Parameter-Efficient Fine-Tuning
- **Weights & Biases (wandb)**: Experiment tracking
- **TensorBoard**: Visualization

## Installation

### Local Setup

```bash
pip install -r requirements.txt
```

### Container Setup

Build the container image:

```bash
make build
```

## Usage

### Preprocessing: Tiling Point Cloud Data

Convert raw LAS point cloud files into tiles:

```bash
python src/preprocessing/tile_s1.py \
  --src <path_to_las_file> \
  --output_dir data/processed_tiles/ \
  --block_size 20.0 \
  --stride 10.0
```

**Parameters:**
- `--src`: Path to input LAS/LAZ file
- `--output_dir`: Output directory for tiles
- `--block_size`: Size of tiles in meters (default: 20.0)
- `--stride`: Overlap stride in meters (default: 10.0)

The tiling process generates:
- Voxelized point cloud tiles
- `tiling_metadata.json` with configuration and feature schema

### Inference

Run semantic segmentation inference:

```bash
python scripts/run_inference.py \
  --config configs/s1_inference.py \
  --weights <path_to_model_weights> \
  --data_dir data/processed_tiles/ \
  --output_dir results/
```

**Parameters:**
- `--config`: Path to inference configuration file
- `--weights`: Path to pretrained model weights
- `--data_dir`: Directory containing processed tiles
- `--output_dir`: Directory for inference results

### Inspect Tiles

Examine processed tile data:

```bash
python scripts/inspect_tile.py <tile_file>
```

## Container-Based Workflow

### Development with GPU

Launch an interactive development container with GPU support:

```bash
make dev
```

This mounts:
- Project directory at `/workspace/project`
- Data directory at `/workspace/project/data`
- GPU devices and libraries

### Running Inference in Container

```bash
make infer
```

### Clean Up Containers

```bash
make clean
```

## Configuration

### Model Configuration

Edit [configs/s1_inference.py](configs/s1_inference.py) to customize:
- Model backbone parameters
- Input/output dimensions
- Loss functions
- Batch size and number of workers
- Class names and ignore indices

Example override:

```python
model = dict(
    backbone=dict(in_channels=4),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)]
)
```

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 32GB+ (default container allocation)
- **Storage**: Sufficient space for raw LAS files and processed tiles
- **Docker/Podman**: For containerized workflows

## Development

### Code Formatting

The project uses `yapf` for Python code formatting. Format your code with:

```bash
yapf --in-place <file.py>
```

### Experiment Tracking

Use Weights & Biases for experiment tracking:

```bash
wandb login
# Experiments will be logged automatically during training/inference
```

## License

See [LICENSE](LICENSE) for details.

## References

This project builds upon:
- [Pointcept](https://github.com/Pointcept/Pointcept) - Point cloud segmentation framework
- [LASpy](https://laspy.readthedocs.io/) - LAS/LAZ file handling
- [Open3D](http://www.open3d.org/) - 3D data processing