import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def visualize_single_tile(input_tile_path, pred_tile_path, output_path):
    """Visualize predictions for a single tile."""
    data = torch.load(input_tile_path, weights_only=False)
    coord = data["coord"]
    if torch.is_tensor(coord):
        coord = coord.cpu().numpy()

    logits = torch.load(pred_tile_path, weights_only=False)
    
    # Convert logits to class IDs
    if logits.ndim > 1:
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    else:
        preds = logits.cpu().numpy()

    # NuScenes PTv3 classes are 0-based: 3=car, 10=driveable_surface, 15=vegetation
    # Let's create an array of gray colors for everything
    colors = np.full((coord.shape[0], 3), 0.5) # Default Gray

    # Highlight specific classes
    cars_mask = (preds == 3) | (preds == 2) | (preds == 9) # Cars, Buses, Trucks
    road_mask = (preds == 10) | (preds == 12) # Driveable surface, Sidewalk
    veg_mask = (preds == 15) # Vegetation

    colors[veg_mask] = [0.1, 0.5, 0.1] # Dark Green
    colors[road_mask] = [0.2, 0.2, 0.2] # Dark Gray/Black
    colors[cars_mask] = [1.0, 0.0, 0.0] # BRIGHT RED

    # Downsample for matplotlib speed
    if coord.shape[0] > 50000:
        step = coord.shape[0] // 50000
        coord = coord[::step]
        colors = colors[::step]

    # Setup Matplotlib BEV (Bird's Eye View)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # Plot X and Y (Top-down view)
    ax.scatter(coord[:, 0], coord[:, 1], c=colors, s=2.0, alpha=0.9, edgecolors='none')
    
    tile_name = Path(input_tile_path).stem
    ax.set_title(f"{tile_name}\nVehicles: {cars_mask.sum()}", fontsize=12)
    ax.axis('equal') # Keep scale 1:1

    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    return cars_mask.sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with original .pth tiles (data/processed_tiles/)")
    parser.add_argument("--pred_dir", required=True, help="Directory with prediction .pth tiles (data/predictions/)")
    parser.add_argument("--output_dir", required=True, help="Output directory for visualization images")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all input tiles
    input_tiles = sorted(input_dir.glob("*.pth"))
    
    print(f"Found {len(input_tiles)} tiles to visualize")
    print(f"Output directory: {output_dir}")
    
    total_vehicles = 0
    processed = 0
    
    for input_tile_path in tqdm(input_tiles, desc="Visualizing tiles"):
        pred_tile_path = pred_dir / input_tile_path.name
        
        if not pred_tile_path.exists():
            print(f"\nWarning: Prediction not found for {input_tile_path.name}, skipping...")
            continue
        
        output_path = output_dir / f"{input_tile_path.stem}.png"
        
        try:
            vehicle_count = visualize_single_tile(str(input_tile_path), str(pred_tile_path), str(output_path))
            total_vehicles += vehicle_count
            processed += 1
        except Exception as e:
            print(f"\nError processing {input_tile_path.name}: {e}")
            continue
    
    print(f"\nVisualization complete!")
    print(f"  Processed: {processed}/{len(input_tiles)} tiles")
    print(f"  Total vehicles detected: {total_vehicles:,}")
    print(f"  Output saved to: {output_dir}")

if __name__ == "__main__":
    main()