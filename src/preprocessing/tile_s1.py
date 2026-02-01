import laspy
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm
from pathlib import Path

from .tile_utils import extract_features, voxel_grid_subsampling, VOXEL_SIZE

DEFAULT_BLOCK_SIZE = 20.0
DEFAULT_STRIDE = 10.0 # 50% of block size
MIN_POINTS = 1000


def save_metadata(output_dir: Path, args: argparse.Namespace,
                  coord_shift: np.ndarray) -> None:
    """Saves run configuration for reproducibility"""
    meta = vars(args)
    meta['coord_shift'] = coord_shift.tolist()
    las = laspy.read(Path(args.src))
    features_schema = ['intensity']
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        features_schema.extend(['red', 'green', 'blue'])
    meta['features_schema'] = features_schema
    with open(output_dir / "tiling_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)


def process_file(file_path: Path,
                 output_dir: Path,
                 block_size: float = DEFAULT_BLOCK_SIZE,
                 stride: float = DEFAULT_STRIDE,
                 min_points: int = MIN_POINTS) -> np.ndarray:

    # Validate tiling parameters to avoid invalid grids or unexpected behavior
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride > block_size:
        raise ValueError(
            f"stride ({stride}) must be less than or equal to block_size ({block_size})"
        )
    filename = file_path.stem
    print(f"Loading {filename}...")
    las = laspy.read(file_path)

    coords = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    features = extract_features(las, len(coords))

    coord_shift = coords.mean(axis=0)
    coords -= coord_shift
    print(f"  [Info] Centered data. Shift: {coord_shift}")

    # Apply voxel filter to standardize density
    coords, features = voxel_grid_subsampling(coords,
                                              features,
                                              voxel_size=VOXEL_SIZE)

    max_coord = coords.max(axis=0)
    min_coord = coords.min(axis=0)  # Re-calc min after shift/subsample

    grid_x = np.arange(min_coord[0], max_coord[0], stride)
    grid_y = np.arange(min_coord[1], max_coord[1], stride)

    print(
        f"Tiling {len(coords)} points into {len(grid_x)}x{len(grid_y)} grid..."
    )

    count = 0

    for x in tqdm(grid_x, desc="Processing Strips", leave=False):
        x_mask = (coords[:, 0] >= x) & (coords[:, 0] < x + block_size)
        if np.sum(x_mask) < min_points:
            continue

        strip_coords = coords[x_mask]
        strip_feats = features[x_mask]

        for y in grid_y:
            y_mask = (strip_coords[:, 1] >= y) & (strip_coords[:, 1]
                                                  < y + block_size)
            if np.sum(y_mask) < min_points:
                continue

            block_coords = strip_coords[y_mask].astype(np.float32)
            block_feats = strip_feats[y_mask]

            block_center_z = float(block_coords[:, 2].mean())
            block_center = np.array(
                [x + block_size / 2, y + block_size / 2, block_center_z],
                dtype=np.float32)
            block_coords -= block_center

            save_dict = {
                "coord": block_coords,
                "features": block_feats,
                "grid_id": np.array([x, y, block_center_z], dtype=np.float32),
                "global_shift": coord_shift.astype(np.float32),
                "name": f"{filename}_tile_{count:04d}"
            }
            torch.save(save_dict,
                       output_dir / f"{filename}_tile_{count:04d}.pth")
            count += 1

    print(f"Finished. Created {count} tiles in {output_dir}")
    return coord_shift


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LiDAR Tiling for Semantic Segmentation")
    parser.add_argument("--src",
                        type=str,
                        required=True,
                        help="Path to input .las file")
    parser.add_argument("--dst", type=str, required=True, help="Output folder")
    parser.add_argument("--block_size",
                        type=float,
                        default=DEFAULT_BLOCK_SIZE,
                        help="Tile size (m)")
    parser.add_argument("--stride",
                        type=float,
                        default=DEFAULT_STRIDE,
                        help="Overlap stride (m)")
    parser.add_argument("--min_points",
                        type=int,
                        default=MIN_POINTS,
                        help="Filter empty tiles")

    args = parser.parse_args()

    dst_path = Path(args.dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    shift = process_file(Path(args.src), dst_path, args.block_size,
                         args.stride, args.min_points)
    save_metadata(dst_path, args, shift)
