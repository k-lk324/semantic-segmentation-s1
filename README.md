# Semantic Segmentation for Sentinel-1 SAR Data

Point cloud semantic segmentation pipeline built on [Pointcept](https://github.com/Pointcept/Pointcept). Processes LAS/LAZ files through tiling → inference → reconstruction.

## Quick Start

```bash
# Build container
make build

# Create test data
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace/project \
  s1-segmentation python scripts/create_dummy_pointcloud.py

# Run full pipeline
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace/project \
  s1-segmentation python src/preprocessing/tile_s1.py \
  --src data/dummy_pointcloud.las --dst data/processed_tiles/ --min_points 100

make infer  # Requires model weights
make reconstruct
```

## Pipeline

### 1. Preprocessing (Tiling)
Split large point clouds into overlapping tiles:

```bash
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace/project \
  s1-segmentation python src/preprocessing/tile_s1.py \
  --src <input.las> \
  --dst data/processed_tiles/ \
  --block_size 20.0 \
  --stride 10.0 \
  --min_points 100
```

Outputs:
- Voxelized tiles (`.pth` format)
- `tiling_metadata.json` with coordinate shift and feature schema

### 2. Inference
Run semantic segmentation on tiles:

```bash
make infer
```

Requires model weights at `weights/ptv3_nuscenes.pth`. Configure in [configs/s1_inference.py](configs/s1_inference.py).

### 3. Reconstruction
Merge predicted tiles back into full point cloud:

```bash
make reconstruct
```

Uses majority voting for overlapping regions. Output: `data/reconstructed.las`

## Semantic Classes

16 classes: barrier, bicycle, bus, car, construction_vehicle, motorcycle, pedestrian, traffic_cone, trailer, truck, driveable_surface, other_flat, sidewalk, terrain, manmade, vegetation

## Utilities

**Create dummy data:**
```bash
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace/project \
  s1-segmentation python scripts/create_dummy_pointcloud.py \
  --output data/test.las --num-points 10000
```

**Inspect tiles:**
```bash
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace/project \
  s1-segmentation python scripts/inspect_tile.py data/processed_tiles/tile_0000.pth
```

**Development shell:**
```bash
make dev
```

## Configuration

Edit [configs/s1_inference.py](configs/s1_inference.py):
- `model.backbone.in_channels`: Input feature dimensions (default: 4 for x,y,z,intensity)
- `data.num_classes`: Number of semantic classes (16)
- `batch_size`, `num_worker`: Processing parameters

## Requirements

- Docker/Podman
- NVIDIA GPU with CUDA support (for inference)
- Python dependencies in [requirements.txt](requirements.txt) (laspy, torch, open3d, etc.)

## License

See [LICENSE](LICENSE)