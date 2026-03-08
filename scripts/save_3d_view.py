import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", required=True, help="Path to a single .pth tile")
    parser.add_argument("--output", default=None, help="Output image path")
    args = parser.parse_args()

    print(f"Loading tile: {args.tile}")
    data = torch.load(args.tile, weights_only=False)
    
    # Extract coordinates
    coord = data["coord"]
    if torch.is_tensor(coord):
        coord = coord.cpu().numpy()

    # --- Downsample for fast Matplotlib rendering ---
    # Matplotlib 3D is slow with millions of points. 
    # Taking a uniform subset of ~25,000 points is perfect for visual checks.
    target_points = 25000
    if coord.shape[0] > target_points:
        step = coord.shape[0] // target_points
        coord = coord[::step]
        
    print(f"Rendering {coord.shape[0]} points (downsampled for speed)...")

    # --- Colorize by height (Z-axis) on a Blue-to-Red gradient ---
    z = coord[:, 2]
    z_normalized = (z - z.min()) / (z.max() - z.min() + 1e-6)
    
    # Create colors (Normalized Red, Green, Blue)
    colors = np.zeros((coord.shape[0], 3))
    colors[:, 0] = z_normalized          # Red for high points
    colors[:, 1] = 0.2                   # Slight Green
    colors[:, 2] = 1.0 - z_normalized    # Blue for low points

    # --- Setup Matplotlib 3D Headless Render ---
    # Use a dark background style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], 
               c=colors, s=1.0, marker='.', alpha=0.8, edgecolors='none')

    # Set a good "Drone" viewing angle (elevation, azimuth)
    ax.view_init(elev=45, azim=45)

    # Hide grid and axes for a cleaner "rendered" look
    ax.axis('off')
    
    # Force equal aspect ratio so your point cloud isn't stretched
    # Matplotlib 3D needs manual bounding box limits to stay proportional
    max_range = np.array([coord[:,0].max()-coord[:,0].min(), 
                          coord[:,1].max()-coord[:,1].min(), 
                          coord[:,2].max()-coord[:,2].min()]).max() / 2.0
    
    mid_x = (coord[:,0].max()+coord[:,0].min()) * 0.5
    mid_y = (coord[:,1].max()+coord[:,1].min()) * 0.5
    mid_z = (coord[:,2].max()+coord[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- Define Output Path and Save ---
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.tile))[0]
        output_path = f"{base_name}_3d.png"
    else:
        output_path = args.output

    print(f"Saving 3D scene to image: {output_path}...")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print("Done!")

if __name__ == "__main__":
    main()