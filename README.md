# LiDAR Point Cloud Semantic Segmentation

3D semantic segmentation pipeline for LAS/LAZ point clouds using [Pointcept](https://github.com/Pointcept/Pointcept) with Point Transformer V3. Pretrained on nuScenes automotive LiDAR dataset (16 classes).

## Quick Start

```bash
make build
make tile
make infer
make reconstruct
```

Outputs:
- data/reconstructed.las: 4-class remapped labels
- data/reconstructed_original_classes.las: original 16-class labels

Both outputs include the semantic_label field for visualization in CloudCompare.

## Pipeline

```text
LAS -> Tiling -> Inference -> Voting/Reconstruction -> Labeled LAS
```

### 1) Tiling

```bash
make tile
```

Default behavior:
- 0.05 m voxel subsampling
- Overlapping tiles (50 percent stride overlap)
- Per-tile coordinate centering
- Saves .pth tiles with metadata (grid_id, global_shift, indices)

Useful overrides:
```bash
make tile SRC_LAS=data/other_scan.las TILE_MIN_POINTS=500
```

### 2) Inference

```bash
make infer FEATURE_MODE=zero
```

Feature modes:
- intensity: uses LAS intensity
- zero: geometry-only, best default for domain shift
- height: absolute elevation as 4th channel
- blend: alpha blend of intensity and height

Output:
- data/predictions/*.pth (tile logits)

### 3) Reconstruction

```bash
make reconstruct
```

What it does:
- Loads per-tile logits
- Applies softmax voting in overlap regions
- Reconstructs scene output with preserved geometry and attributes
- Writes both 4-class remapped and original 16-class LAS files

## 16-Class to 4-Class Remap

To improve out-of-domain stability, predictions are collapsed into superclasses:
- vegetation (0): vegetation
- object (1): barrier, bicycle, bus, car, construction_vehicle, motorcycle, pedestrian, traffic_cone, trailer, truck
- ground (2): driveable_surface, other_flat, sidewalk, terrain
- structure (3): manmade

This remapped output is written to data/reconstructed.las.

## Configuration

Defaults live in Makefile and can be overridden per command.

Key variables:
- SRC_LAS: input LAS path
- TILE_MIN_POINTS: minimum points per tile
- FEATURE_MODE: intensity, zero, height, blend
- BLEND_ALPHA: blend weight for blend mode
- RECON_OUT_REMAPPED: output path for 4-class LAS
- RECON_OUT_ORIGINAL: output path for 16-class LAS

Example:
```bash
make tile SRC_LAS=data/scan2.las
make infer FEATURE_MODE=height
make reconstruct RECON_OUT_REMAPPED=data/out_4class.las RECON_OUT_ORIGINAL=data/out_16class.las
```

## Domain Shift Notes

The model is pretrained on nuScenes automotive LiDAR.

Recommended order:
1. Start with FEATURE_MODE=zero.
2. Use remapped 4-class output for downstream work.
3. Treat small object classes as low-confidence unless fine-tuned.


## Visualization and Utilities

Generate per-tile prediction images:
```bash
make viz_preds
```

Generate a single 3D tile view:
```bash
make viz_3d
```

Export processed tiles to PLY:
```bash
make export_ply
```

Quick tile sanity check:
```bash
make check_tile
```

## Development

Open interactive container shell:
```bash
make dev
```

Create synthetic test LAS:
```bash
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace/project \
  s1-segmentation python scripts/create_dummy_pointcloud.py \
  --output data/test.las --num-points 50000
```

## Model and Requirements

Model config:
- configs/s1_inference.py

Weights:
- weights/ptv3_nuscenes.pth

Runtime requirements:
- Docker or Podman
- NVIDIA GPU (CUDA 12+)
- LAS input with XYZ (RGB and intensity optional)

Dependencies managed via container (see [Containerfile](Containerfile)):
- PyTorch 2.x with CUDA
- spconv (sparse convolution)
- Pointcept framework
- laspy, open3d for point cloud I/O

## License

See [LICENSE](LICENSE)