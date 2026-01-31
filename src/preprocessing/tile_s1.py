import laspy
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from pathlib import Path


def normalize_intensity(intensity):
    """
    Normalizes FJD Trion S1 intensity (typically 16-bit or arbitrary range)
    to [-1, 1] range for stability in Neural Networks.
    """
    max_val = np.percentile(intensity, 99)
    min_val = np.min(intensity)

    norm = (intensity - min_val) / (max_val - min_val + 1e-6)
    norm = np.clip(norm, 0, 1)

    return (norm * 2) - 1


def process_file(file_path,
                 output_dir,
                 block_size=20.0,
                 stride=10.0,
                 min_points=1000):
    """
    Process a LAS point cloud file into tiled blocks for semantic segmentation.
    This function reads a LAS file, applies global coordinate shifting for precision,
    and tiles the point cloud into overlapping blocks using a sliding window approach.
    Each block is centered locally and saved with associated metadata.
    Args:
        file_path (str): Path to the input LAS file.
        output_dir (str): Directory where output tiles will be saved as .pth files.
        block_size (float, optional): Size of each tile block in XY plane. Defaults to 20.0.
        stride (float, optional): Sliding window stride in XY plane. Defaults to 10.0.
        min_points (int, optional): Minimum number of points required to create a tile.
                                    Tiles with fewer points are skipped. Defaults to 1000.
    Returns:
        None
    Notes:
        - Coordinates are globally shifted to (0,0,0) origin to prevent float32 precision loss.
        - Intensity values are normalized if available; zeros are used as placeholders otherwise.
        - Each tile block is centered locally around its block center.
        - Output files are named as "{filename}_tile_{count:04d}.pth".
        - Saved dictionary contains: coord, strength, grid_id, global_shift, and name keys.
    """
    filename = Path(file_path).stem

    print(f"Loading {filename}...")
    las = laspy.read(file_path)

    coords = np.vstack((las.x, las.y, las.z)).transpose().astype(np.float64)

    if hasattr(las, 'intensity'):
        intensity = np.array(las.intensity).reshape(-1, 1).astype(np.float32)
        intensity = normalize_intensity(intensity)
    else:
        print("Warning: No intensity found. Using placeholder zeros.")
        intensity = np.zeros((coords.shape[0], 1), dtype=np.float32)

    coord_shift = coords.min(axis=0)
    coords -= coord_shift

    max_coord = coords.max(axis=0)

    grid_x = np.arange(0, max_coord[0], stride)
    grid_y = np.arange(0, max_coord[1], stride)

    print(
        f"Tiling {filename} into {len(grid_x) * len(grid_y)} potential blocks..."
    )

    count = 0
    for x in tqdm(grid_x, leave=False):
        for y in grid_y:
            # Define block boundaries
            x_min, x_max = x, x + block_size
            y_min, y_max = y, y + block_size

            mask = (coords[:, 0] >= x_min) & (coords[:, 0] < x_max) & \
                   (coords[:, 1] >= y_min) & (coords[:, 1] < y_max)

            if np.sum(mask) < min_points:
                continue

            # Extract data
            block_coords = coords[mask].astype(np.float32)
            block_feats = intensity[mask]

            # Center the block (Local Coordinates) for the model
            # But we keep the relative position for context if needed
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

            save_path = os.path.join(output_dir,
                                     f"{filename}_tile_{count:04d}.pth")
            torch.save(save_dict, save_path)
            count += 1

    print(f"Finished. Created {count} tiles in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",
                        type=str,
                        required=True,
                        help="Path to raw .las file")
    parser.add_argument("--dst",
                        type=str,
                        required=True,
                        help="Folder to save .pth tiles")
    parser.add_argument("--block_size",
                        type=float,
                        default=20.0,
                        help="Size of tile in meters")
    parser.add_argument("--stride",
                        type=float,
                        default=10.0,
                        help="Step size (overlap = block - stride)")
    parser.add_argument("--min_points",
                        type=int,
                        default=1000,
                        help="Minimum points per tile")

    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    process_file(args.src, args.dst, args.block_size, args.stride,
                 args.min_points)
