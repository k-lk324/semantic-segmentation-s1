import torch
import matplotlib.pyplot as plt
from pathlib import Path


def inspect(tile_path):
    print(f"--- Inspecting: {tile_path} ---")
    data = data = torch.load(tile_path, weights_only=True)

    coords = data['coord']  # (N, 3)
    feats = data['features']  # (N, 4) or (N, 1)
    grid_id = data['grid_id']
    global_shift = data['global_shift']

    print(f"1. Structure Check:")
    print(f"   - Point Count: {coords.shape[0]}")
    print(f"   - Features Shape: {feats.shape}")
    print(f"   - Global Shift used: {global_shift}")

    # Check 1: Normalization
    print(f"\n2. Feature Check (Should be approx [-1, 1]):")
    print(
        f"   - Intensity: Min {feats[:, 0].min():.3f}, Max {feats[:, 0].max():.3f}"
    )
    if feats.shape[1] > 1:
        print(
            f"   - RGB: Min {feats[:, 1:].min():.3f}, Max {feats[:, 1:].max():.3f}"
        )

    # Check 2: Voxel Subsampling
    # Calculate density. If voxel size is 0.05m, points shouldn't be closer than 0.05m roughly.
    # Note: Barycenters can technically be closer, but average density should be lower.
    print(f"\n3. Geometry Check:")
    print(f"   - Tile Center: {coords.mean(axis=0)}")
    print(
        f"   - Bounds X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(
        f"   - Bounds Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")

    # Visual check (2D Projection)
    plt.figure(figsize=(10, 5))
    plt.scatter(coords[:, 0], coords[:, 1], s=1, c=feats[:, 0], cmap='viridis')
    plt.title(f"Tile {grid_id} (Top Down View)\nColor=Intensity")
    plt.axis('equal')
    plt.savefig(tile_path.with_suffix('.png'))
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect a tiled point cloud .pth file.")
    parser.add_argument("tile_path",
                        type=str,
                        help="Path to the .pth tile file to inspect.")
    args = parser.parse_args()

    inspect(Path(args.tile_path))

# python scripts/inspect_tile.py path/to/tile_file.pth

