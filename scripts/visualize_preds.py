import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tile", required=True, help="Path to original .pth tile (data/processed_tiles/...)")
    parser.add_argument("--pred_tile", required=True, help="Path to prediction .pth tile (data/predictions/...)")
    parser.add_argument("--output", default="prediction_view.png", help="Output image name")
    args = parser.parse_args()

    print(f"Loading input coordinates from: {args.input_tile}")
    data = torch.load(args.input_tile, weights_only=False)
    coord = data["coord"]
    if torch.is_tensor(coord):
        coord = coord.cpu().numpy()

    print(f"Loading model predictions from: {args.pred_tile}")
    logits = torch.load(args.pred_tile, weights_only=False)
    
    # Convert logits to class IDs
    if logits.ndim > 1:
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    else:
        preds = logits.cpu().numpy()

    # NuScenes typical PTv3 classes: 4 is Car, 11 is Driveable Surface, 16 is Vegetation
    # Let's create an array of gray colors for everything
    colors = np.full((coord.shape[0], 3), 0.5) # Default Gray

    # Highlight specific classes
    cars_mask = (preds == 4) | (preds == 3) | (preds == 10) # Cars, Buses, Trucks
    road_mask = (preds == 11) | (preds == 13) # Road, Sidewalk
    veg_mask = (preds == 16) # Vegetation

    colors[veg_mask] = [0.1, 0.5, 0.1] # Dark Green
    colors[road_mask] = [0.2, 0.2, 0.2] # Dark Gray/Black
    colors[cars_mask] = [1.0, 0.0, 0.0] # BRIGHT RED

    print(f"Found {cars_mask.sum()} points classified as vehicles.")

    # Downsample for matplotlib speed (only if it's huge, 30x30m might be fine, but just in case)
    if coord.shape[0] > 50000:
        step = coord.shape[0] // 50000
        coord = coord[::step]
        colors = colors[::step]

    # Setup Matplotlib BEV (Bird's Eye View)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # Plot X and Y (Top-down view)
    ax.scatter(coord[:, 0], coord[:, 1], c=colors, s=2.0, alpha=0.9, edgecolors='none')
    
    ax.set_title(f"Model Predictions (Red = Vehicle)\nVehicles found: {cars_mask.sum()}", fontsize=14)
    ax.axis('equal') # Keep scale 1:1

    print(f"Saving to {args.output}...")
    plt.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    print("Done!")

if __name__ == "__main__":
    main()