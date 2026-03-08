import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np

def simulate_velodyne_mask(coords, tolerance=0.2, device="cpu"):
    """The exact same masking function from your inference script."""
    velodyne_angles = torch.linspace(10.67, -30.67, 32, device=device)
    centered_coords = coords - coords.mean(dim=0)
    
    r = torch.norm(centered_coords, dim=1) + 1e-6 
    z = centered_coords[:, 2]
    elevation = torch.asin(z / r) * (180.0 / torch.pi)
    
    angle_diffs = torch.abs(elevation.unsqueeze(1) - velodyne_angles.unsqueeze(0))
    min_diffs, _ = torch.min(angle_diffs, dim=1)
    
    mask = min_diffs < tolerance
    return mask, elevation

def unroll_point_cloud(coords):
    """Calculates Azimuth (horizontal angle) for the X-axis of our image."""
    centered_coords = coords - coords.mean(dim=0)
    x = centered_coords[:, 0]
    y = centered_coords[:, 1]
    # Calculate azimuth angle between -180 and 180 degrees
    azimuth = torch.atan2(y, x) * (180.0 / torch.pi)
    return azimuth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", required=True, help="Path to a single .pth tile")
    parser.add_argument("--tol", type=float, default=0.2, help="Velodyne tolerance")
    parser.add_argument("--output", default="ring_visualization.png", help="Output image name")
    args = parser.parse_args()

    print(f"Loading tile: {args.tile}")
    data = torch.load(args.tile, weights_only=False)
    
    coord = data["coord"]
    if not torch.is_tensor(coord):
        coord = torch.from_numpy(coord)
    coord = coord.float()

    print("Calculating projections and applying mask...")
    azimuth = unroll_point_cloud(coord)
    mask, elevation = simulate_velodyne_mask(coord, tolerance=args.tol)

    # Convert to numpy for matplotlib
    azimuth_np = azimuth.cpu().numpy()
    elevation_np = elevation.cpu().numpy()
    mask_np = mask.cpu().numpy()

    print("Generating high-res image...")
    fig, axs = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True)

    # Plot 1: The Raw FJD Data
    axs[0].scatter(azimuth_np, elevation_np, s=0.1, c='grey', alpha=0.5)
    axs[0].set_title("1. Raw FJD Point Cloud (Dense & Unstructured)", fontsize=14)
    axs[0].set_ylabel("Elevation Angle (Degrees)")
    axs[0].set_facecolor('black')

    # Plot 2: The Simulated Velodyne Data
    axs[1].scatter(azimuth_np[mask_np], elevation_np[mask_np], s=0.1, c='cyan', alpha=0.8)
    axs[1].set_title(f"2. Simulated Velodyne 32-Beam (Tolerance: {args.tol}°)", fontsize=14)
    axs[1].set_xlabel("Azimuth Angle (Degrees)")
    axs[1].set_ylabel("Elevation Angle (Degrees)")
    axs[1].set_facecolor('black')

    # Limit the Y-axis to the Velodyne's field of view to crop out extreme high/low points
    axs[0].set_ylim(-35, 15)
    axs[0].set_xlim(-180, 180)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Done! Saved visualization to {args.output}")

if __name__ == "__main__":
    main()