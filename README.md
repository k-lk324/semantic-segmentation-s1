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

Output: [data/reconstructed.las](data/reconstructed.las) with `semantic_label` field (visualize in CloudCompare).

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
- Outputs [data/reconstructed.las](data/reconstructed.las) with `semantic_label` field

**Visualization in CloudCompare:**
1. Open `data/reconstructed.las`
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

### Recommended 4-Class Remapping

For parking lot or static scans, collapse the 16 classes into 4 semantically meaningful superclasses:

```python
# Add to vote_and_reconstruct.py after prediction
SUPERCLASS_MAP = {
    # 0: Driveable (roads, parking lots - includes misclassified truck)
    'driveable': [9, 10, 11],  # truck, driveable_surface, other_flat
    
    # 1: Structure (buildings, walls, barriers)
    'structure': [0, 14],  # barrier, manmade
    
    # 2: Walkable (sidewalks, paths, terrain)
    'walkable': [12, 13],  # sidewalk, terrain
    
    # 3: Objects (vehicles, people, cones, vegetation)
    'objects': [1, 2, 3, 4, 5, 6, 7, 8, 15]  # all other classes
}

# Apply remapping (class ID → superclass ID)
superclass_mapping = np.array([1, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 1, 3])
predicted_label = superclass_mapping[predicted_label]
```

**Why this works:**
- **Merges truck (9) into driveable (0):** Compensates for road→truck misclassification
- **Groups functionally similar classes:** All ground surfaces together, all structures together
- **Eliminates noise:** Most vehicle classes have <0.1% representation and are unreliable
- **Clean visualization:** 4 distinct categories instead of 16 mostly-empty ones

**Expected distribution after remapping:**
- Driveable: ~45% (parking lots, roads)
- Structure: ~48% (buildings, walls)
- Walkable: ~6% (sidewalks)
- Objects: <1% (sparse detections)

## Configuration Variables

Edit defaults in [Makefile](Makefile) or override per command:

| Variable          | Default              | Description                                      |
|-------------------|----------------------|--------------------------------------------------|
| `SRC_LAS`         | Section_normal.las   | Input LAS file for tiling                        |
| `TILE_MIN_POINTS` | 200                  | Minimum points per tile (filters empty regions)  |
| `FEATURE_MODE`    | zero                 | Input features: intensity/zero/height/blend      |
| `BLEND_ALPHA`     | 0.5                  | Blending weight for blend mode (0.0-1.0)         |

**Example usage:**
```bash
# Use intensity mode with different source file
make tile SRC_LAS=data/scan2.las
make infer FEATURE_MODE=intensity
make reconstruct

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

### Large Objects Classified as Trucks
**Symptom:** Buildings, walls, or large surfaces predicted as truck (14% or more)  
**Cause:** Intensity domain shift  
**Solution:** Use `FEATURE_MODE=zero` instead of `intensity`

### Low Coverage in Reconstruction
**Symptom:** Reconstruction has fewer points than original  
**Cause:** Voxel subsampling (0.05m) reduces density  
**Expected:** ~70% coverage for typical automotive-density scans  
**Note:** Reconstruction preserves indices to map predictions back to original points

### Classes Concentrated in Few Categories
**Symptom:** 95%+ points in manmade/driveable/truck, other classes <1%  
**Cause:** Domain shift (model trained on road scenes, tested on different domain). Roads often misclassified as truck.  
**Solution:** Use class remapping (see [Class Remapping section](#class-remapping-for-out-of-domain-data)) to collapse 16 classes into 4 meaningful superclasses.

### Inference Crashes with Grid Errors
**Symptom:** RuntimeError about spatial shape or negative indices  
**Cause:** Coordinate range incompatible with sparse convolution  
**Note:** This should be fixed (GRID_OFFSET applied). If still occurs, check coordinate magnitude in tiles.

## Development

**Interactive shell in container:**
```bash
make dev
```

**Inspect tile contents:**
```bash
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace/project \
  s1-segmentation python scripts/inspect_tile.py data/processed_tiles/tile_0000.pth
```

**Create synthetic test data:**
```bash
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace/project \
  s1-segmentation python scripts/create_dummy_pointcloud.py \
  --output data/test.las --num-points 50000
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