# LiDAR Point Cloud Semantic Segmentation

3D semantic segmentation pipeline for LAS/LAZ point clouds using [Pointcept](https://github.com/Pointcept/Pointcept) with Point Transformer V3. Pretrained on nuScenes automotive LiDAR dataset (16 classes).

## Quick Start

```bash
# Build Docker container
make build

# Run full pipeline with defaults
make tile     # Split point cloud into overlapping tiles
make infer    # Run semantic segmentation
make reconstruct  # Merge predictions into full point cloud
```

Outputs:
- [data/reconstructed.las](data/reconstructed.las): remapped 4-class labels
- [data/reconstructed_original_classes.las](data/reconstructed_original_classes.las): original 16-class labels

Both files include the `semantic_label` field (visualize in CloudCompare).

## Pipeline Overview

```
LAS file → Tiling (voxel subsample + overlap) → Inference (per tile) → Voting + Reconstruction → Labeled LAS
```

### 1. Tiling

Split large point clouds into 20m×20m tiles with 10m stride (50% overlap):

```bash
make tile
```

**Configuration:**
- `SRC_LAS`: Source LAS file path (default: full-resolution Section_normal.las)
- `TILE_MIN_POINTS`: Minimum points to keep a tile (default: 200)

**Override example:**
```bash
make tile SRC_LAS=data/other_scan.las TILE_MIN_POINTS=500
```

**What it does:**
- Applies 0.05m voxel grid subsampling
- Extracts overlapping blocks (avoids edge effects)
- Centers coordinates per tile
- Saves as `.pth` with metadata (grid_id, global_shift, indices)

### 2. Inference

Run pretrained model on each tile:

```bash
make infer FEATURE_MODE=zero
```

**Configuration:**
- `FEATURE_MODE`: Input feature channel (choices: `intensity`, `zero`, `height`, `blend`)
  - `zero`: Geometry-only (XYZ). **Recommended for domain adaptation.**
  - `intensity`: Use LAS intensity values (sensor-specific, may cause domain shift)
  - `height`: Use absolute elevation as 4th channel
  - `blend`: Weighted mix of height and intensity (controlled by `BLEND_ALPHA`)
- `BLEND_ALPHA`: Blending weight for `blend` mode (0.0=height only, 1.0=intensity only)

**Why feature modes matter:**  
The model was trained on nuScenes (vehicle-mounted automotive LiDAR). Intensity values from different sensors/scenes cause domain shift. Using `FEATURE_MODE=zero` (geometry-only) improves robustness for out-of-domain data.

**Output:** Prediction logits saved to `data/predictions/*.pth`

### 3. Reconstruction

Merge tile predictions into full point cloud with majority voting:

```bash
make reconstruct
```

**What it does:**
- Loads predictions for all tiles
- Applies softmax to logits
- softmax voting for overlapping regions
- Preserves original RGB colors and intensity
- Writes two reconstructed LAS files with identical geometry/attributes:
  - [data/reconstructed.las](data/reconstructed.las): remapped 4-class labels
  - [data/reconstructed_original_classes.las](data/reconstructed_original_classes.las): original 16-class labels

**Visualization in CloudCompare:**
1. Open either [data/reconstructed.las](data/reconstructed.las) or [data/reconstructed_original_classes.las](data/reconstructed_original_classes.las)
2. Select point cloud in DB Tree
3. Edit → Scalar Fields → Select `semantic_label`
4. The segmentation shows as colors

## Semantic Classes

16 nuScenes classes:

| Class ID | Name                  | Typical Objects                     |
|----------|-----------------------|-------------------------------------|
| 0        | barrier               | Jersey barriers, safety barriers    |
| 1        | bicycle               | Bicycles                            |
| 2        | bus                   | Buses                               |
| 3        | car                   | Cars, sedans, SUVs                  |
| 4        | construction_vehicle  | Excavators, loaders, dump trucks    |
| 5        | motorcycle            | Motorcycles, scooters               |
| 6        | pedestrian            | People                              |
| 7        | traffic_cone          | Traffic cones                       |
| 8        | trailer               | Trailers                            |
| 9        | truck                 | Trucks, pickups                     |
| 10       | driveable_surface     | Roads, parking lots, asphalt        |
| 11       | other_flat            | Other flat surfaces                 |
| 12       | sidewalk              | Sidewalks, walkways                 |
| 13       | terrain               | Grass, dirt, natural terrain        |
| 14       | manmade               | Buildings, walls, structures        |
| 15       | vegetation            | Trees, bushes                       |

## Class Remapping for Out-of-Domain Data

Due to domain shift, the model often concentrates predictions in 3-4 classes while leaving 12+ classes nearly empty (<1% each). **Common issue:** Roads/parking lots are frequently misclassified as "truck" (class 9).

### Automatic 4-Class Remapping

Predictions are automatically collapsed from 16 nuScenes classes into 4 semantically meaningful **superclasses**:

| Superclass | ID | Mapped Classes | Notes |
|------------|----|----|-------|
| **ground** | 2 | driveable_surface, other_flat, sidewalk, terrain | Roads, parking lots, surfaces |
| **structure** | 3 | manmade | Buildings, walls, infrastructure |
| **object** | 1 | barrier, bicycle, bus, car, construction_vehicle, motorcycle, pedestrian, traffic_cone, trailer, truck | All vehicles and movable objects |
| **vegetation** | 0 | vegetation | Trees, bushes, plants |

The remapping is automatically applied in [scripts/vote_and_reconstruct.py](scripts/vote_and_reconstruct.py) and written to [data/reconstructed.las](data/reconstructed.las). The full 16-class output is also preserved in [data/reconstructed_original_classes.las](data/reconstructed_original_classes.las) for reference.

**Why 4-class remapping works better:**
- **Compensates for domain shift:** Trucks and vehicles are merged into one "object" class, avoiding misclassification issues
- **Groups functionally similar classes:** All ground surfaces together, structures together
- **Reduces noise:** Most vehicle classes have <0.1% representation in out-of-domain scans
- **Clean visualization:** 4 distinct, reliable categories instead of 16 mostly-empty classes

**Expected distribution after remapping:**
- Ground: ~40-50% (parking lots, roads, sidewalks, terrain)
- Structure: ~45-55% (buildings, walls)
- Object: <1% (sparse detections)
- Vegetation: <5% (sparse detections)

## Configuration Variables

Edit defaults in [Makefile](Makefile) or override per command:

| Variable          | Default              | Description                                      |
|-------------------|----------------------|--------------------------------------------------|
| `SRC_LAS`         | Section_normal.las   | Input LAS file for tiling                        |
| `RECON_OUT_REMAPPED` | data/reconstructed.las | Reconstructed LAS with 4 remapped classes      |
| `RECON_OUT_ORIGINAL` | data/reconstructed_original_classes.las | Reconstructed LAS with original 16 classes |
| `TILE_MIN_POINTS` | 200                  | Minimum points per tile (filters empty regions)  |
| `FEATURE_MODE`    | zero                 | Input features: intensity/zero/height/blend      |
| `BLEND_ALPHA`     | 0.5                  | Blending weight for blend mode (0.0-1.0)         |

**Example usage:**
```bash
# Use intensity mode with different source file
make tile SRC_LAS=data/scan2.las
make infer FEATURE_MODE=intensity
make reconstruct

# Customize output locations
make reconstruct RECON_OUT_REMAPPED=data/out_4class.las RECON_OUT_ORIGINAL=data/out_16class.las

# Try height features with lower threshold
make tile TILE_MIN_POINTS=100
make infer FEATURE_MODE=height
make reconstruct
```

## Domain Adaptation Challenges

**Important:** This model is pretrained on nuScenes (vehicle-mounted automotive LiDAR) but may be applied to other domains (aerial scans, static scanners, parking lots). Common issues:

1. **Intensity Domain Shift:** LiDAR intensity values are sensor-specific. Using `FEATURE_MODE=intensity` may cause systematic bias (e.g., over-predicting trucks). **Solution:** Use `FEATURE_MODE=zero` for geometry-only inference.

2. **Class Distribution Mismatch:** nuScenes focuses on road scenes with vehicles. Static parking lot scans have different geometry (top-down vs. side views). Small vehicles may not be detected reliably.

3. **Point Density Differences:** Automotive LiDAR is typically 10-40 points/m². Higher density scans may oversample; lower density may lose detail. Voxel size (0.05m) is tuned for automotive data.

**Recommendations:**
- Start with `FEATURE_MODE=zero` for new domains
- Expect best performance on: driveable surfaces, buildings (manmade), sidewalks
- Expect poor performance on: small vehicles, pedestrians, small objects (<0.2m)
- For production use, consider fine-tuning or retraining on your specific domain

## Troubleshooting

### Domain Shift Issues
**Symptom:** Model concentrates predictions in 2-3 classes (e.g., >90% ground/structure), other classes <1%  
**Cause:** Model trained on nuScenes automotive scenes; domain gap when applied to different contexts  
**Solution:** 
  1. Use `FEATURE_MODE=zero` (geometry-only) instead of intensity
  2. Rely on 4-class remapped output ([data/reconstructed.las](data/reconstructed.las)) instead of 16-class original
  3. Review troubleshooting section on [Domain Adaptation Challenges](#domain-adaptation-challenges)

### Low Coverage in Reconstruction
**Symptom:** Reconstruction has fewer points than original  
**Cause:** Voxel subsampling (0.05m) reduces point density  
**Expected:** ~70% coverage for typical automotive-density scans  
**Note:** Reconstruction preserves indices to map predictions back to original points; subsampling is intentional for balanced training

### Intensity domain shift causing systematic bias
**Symptom:** Different sensors produce very different predictions on similar geometry  
**Cause:** LiDAR intensity is sensor-specific; model overfits to intensity patterns from nuScenes  
**Solution:** Use `FEATURE_MODE=zero` for geometry-only inference (recommended for new domains)

### Inference Crashes with Grid Errors
**Symptom:** RuntimeError about spatial shape or negative indices  
**Cause:** Coordinate range incompatible with sparse convolution  
**Solution:** Check that coordinates are reasonable (within ±1e5 range). Tiling should handle centering automatically via `coord_shift`.

### Tiles not being processed
**Symptom:** Tiling completes but `data/processed_tiles/` is empty  
**Cause:** All tiles fell below `TILE_MIN_POINTS` threshold  
**Solution:** Lower `TILE_MIN_POINTS` (default 200) or check source LAS file for insufficient point density

## Visualization & Export

### Generate Prediction Visualization
Create 2D visualizations of tile predictions (overlaid on raw point cloud):

```bash
make viz_preds
```

Outputs PNG images to `results_figs/` showing predictions per tile.

### Generate 3D View of Single Tile
Visualize geometry and predictions for a specific tile:

```bash
make viz_3d
```

Default tile: `Fh_parking_outside_2025-09-19-17-08-32_Section_Section_normal_tile_0050.pth`
Output: `raw_tile_view.png`

### Export Tiles to PLY Format
Convert processed tiles to PLY format for external viewers:

```bash
make export_ply
```

Outputs labeled point clouds to `data/labeled_plys/`

### Inspect Tile Metadata
Check tile structure and properties:

```bash
make check_tile
```

Default tile: `Fh_parking_outside_2025-09-19-17-08-32_Section_Section_normal_tile_0050.pth`

## Development

**Interactive shell in container:**
```bash
make dev
```

**Create synthetic test data:**
```bash
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace/project \
  s1-segmentation python scripts/create_dummy_pointcloud.py \
  --output data/test.las --num-points 50000
```

**Run Python scripts interactively:**
```bash
make dev
python scripts/visualize_preds.py --input_dir data/processed_tiles --pred_dir data/predictions --output_dir results_figs
```

## Model Configuration

Edit [configs/s1_inference.py](configs/s1_inference.py):
- `model.backbone.in_channels`: Must match feature mode (4 for xyzf)
- `data.num_classes`: 16 (nuScenes classes)
- `batch_size`, `num_worker`: Adjust for GPU memory

Model weights: `weights/ptv3_nuscenes.pth` (Point Transformer V3 pretrained on nuScenes)

## Requirements

- Docker or Podman
- NVIDIA GPU with CUDA 12+ support
- Model weights file (not included, must be obtained separately)
- Input: LAS/LAZ files with XYZ coordinates (RGB and intensity optional)

Dependencies managed via container (see [Containerfile](Containerfile)):
- PyTorch 2.x with CUDA
- spconv (sparse convolution)
- Pointcept framework
- laspy, open3d for point cloud I/O

## License

See [LICENSE](LICENSE)