import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", required=True, help="Path to .pth tile")
    args = parser.parse_args()

    data = torch.load(args.tile, weights_only=False)
    coord = data["coord"]
    
    print("\n--- TILE SANITY CHECK ---")
    print(f"Total Points: {coord.shape[0]}")
    
    # Calculate bounding box
    min_x, max_x = coord[:, 0].min().item(), coord[:, 0].max().item()
    min_y, max_y = coord[:, 1].min().item(), coord[:, 1].max().item()
    min_z, max_z = coord[:, 2].min().item(), coord[:, 2].max().item()
    
    print(f"X (Length/Width) Range: [{min_x:.2f}, {max_x:.2f}] -> Span: {max_x - min_x:.2f}")
    print(f"Y (Length/Width) Range: [{min_y:.2f}, {max_y:.2f}] -> Span: {max_y - min_y:.2f}")
    print(f"Z (Height) Range:       [{min_z:.2f}, {max_z:.2f}] -> Span: {max_z - min_z:.2f}")

if __name__ == "__main__":
    main()