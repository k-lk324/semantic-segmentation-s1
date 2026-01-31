import laspy
import numpy as np
import torch
import argparse
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List

DEFAULT_BLOCK_SIZE = 20.0
DEFAULT_STRIDE = 10.0
MIN_POINTS = 1000
VOXEL_SIZE = 0.05  # 5cm


def normalize_color(color: np.ndarray) -> np.ndarray:
    """
    Normalizes color to [-1, 1]. Detects 8-bit vs 16-bit automatically.
    """
    max_val = color.max()
    if max_val > 255:
        # 16-bit case
        return (color.astype(np.float32) / 65535.0 * 2) - 1
    # 8-bit case
    return (color.astype(np.float32) / 255.0 * 2) - 1


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    """
    Normalizes intensity to [-1, 1].
    """
    intensity = intensity.astype(np.float32)
    max_val = np.percentile(intensity, 99)
    min_val = np.min(intensity)

    div = max_val - min_val
    if div == 0: div = 1.0

    norm = (intensity - min_val) / div
    norm = np.clip(norm, 0, 1)
    return (norm * 2) - 1


def extract_features(las: laspy.LasData, point_count: int) -> np.ndarray:
    """
    Extracts and normalizes features (Intensity + RGB) from the LAS object.
    Returns:
        np.ndarray: Matrix of shape (N, 4) -> [Intensity, R, G, B]
    """
    feats_list = []

    if hasattr(las, 'intensity'):
        intensity = np.array(las.intensity).reshape(-1, 1)
        feats_list.append(normalize_intensity(intensity))
    else:
        # Fallback: Use zeros if intensity is missing
        print("  [Warning] No intensity found. Using placeholders.")
        feats_list.append(np.zeros((point_count, 1), dtype=np.float32))

    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        print("  [Info] Found RGB Color. Including in features.")
        r = np.array(las.red).reshape(-1, 1)
        g = np.array(las.green).reshape(-1, 1)
        b = np.array(las.blue).reshape(-1, 1)
        rgb = np.hstack([r, g, b])
        feats_list.append(normalize_color(rgb))
    else:
        print("  [Info] No RGB Color found. Skipping color features.")
        pass

    return np.hstack(feats_list).astype(np.float32)


def voxel_grid_subsampling(
        coords: np.ndarray,
        features: np.ndarray,
        voxel_size: float = VOXEL_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a voxel grid filter (5cm).
    Calculates the barycenter (mean) of points within each voxel.
    """
    print(
        f"  [Processing] Applying {voxel_size}m voxel filter to {len(coords)} points..."
    )

    # 1. Create a DataFrame to leverage optimized GroupBy
    # Combining coords and features into one structure for grouping
    # Features usually 1 (Intensity) or 4 (Int + RGB)
    num_feat_cols = features.shape[1]
    feat_col_names = [f'feat_{i}' for i in range(num_feat_cols)]

    df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    df_feats = pd.DataFrame(features, columns=feat_col_names)
    df = pd.concat([df, df_feats], axis=1)

    # 2. Calculate Voxel Indices
    df['vx'] = (df['x'] / voxel_size).astype(np.int64)
    df['vy'] = (df['y'] / voxel_size).astype(np.int64)
    df['vz'] = (df['z'] / voxel_size).astype(np.int64)

    # 3. Group by Voxel Index and compute Mean (Barycenter)
    grouped = df.groupby(['vx', 'vy', 'vz'], as_index=False).mean()

    # 4. Extract results
    new_coords = grouped[['x', 'y', 'z']].values.astype(np.float32)
    new_feats = grouped[feat_col_names].values.astype(np.float32)

    print(
        f"  [Processing] Reduced to {len(new_coords)} points ({(1 - len(new_coords)/len(coords))*100:.1f}% reduction)."
    )

    return new_coords, new_feats


def save_metadata(output_dir: Path, args: argparse.Namespace,
                  coord_shift: np.ndarray):
    """Saves run configuration for reproducibility"""
    meta = vars(args)
    meta['coord_shift'] = coord_shift.tolist()
    meta['features_schema'] = ['intensity', 'red', 'green', 'blue']
    with open(output_dir / "tiling_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)


def process_file(file_path: Path,
                 output_dir: Path,
                 block_size: float = DEFAULT_BLOCK_SIZE,
                 stride: float = DEFAULT_STRIDE,
                 min_points: int = MIN_POINTS) -> np.ndarray:

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

            block_center = np.array(
                [x + block_size / 2, y + block_size / 2, 0], dtype=np.float32)
            block_coords -= block_center

            save_dict = {
                "coord": block_coords,
                "strength": block_feats,
                "grid_id": np.array([x, y]),
                "global_shift": coord_shift,
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
