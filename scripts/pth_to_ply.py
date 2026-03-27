import torch
import open3d as o3d
import argparse
import os
import numpy as np
from glob import glob

def convert_tile(pth_path, output_dir):
    print(f"Loading {pth_path}...")
    data = torch.load(pth_path, weights_only=False)
    
    # 1. Extract XYZ Coordinates
    coord = data["coord"]
    if torch.is_tensor(coord):
        coord = coord.cpu().numpy()
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)

    # 2. Extract Intensity to use as Grayscale Colors (helps with manual labeling!)
    if "features" in data:
        feat = data["features"]
        if torch.is_tensor(feat):
            feat = feat.cpu().numpy()
            
        # Grab the first channel (intensity)
        intensity = feat[:, 0] if feat.ndim > 1 else feat
        
        # Normalize intensity between 0.0 and 1.0 for Open3D colors
        i_min, i_max = intensity.min(), intensity.max()
        if i_max > i_min:
            i_norm = (intensity - i_min) / (i_max - i_min)
        else:
            i_norm = np.zeros_like(intensity)
            
        # Stack into RGB array (Grayscale)
        colors = np.stack((i_norm, i_norm, i_norm), axis=-1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 3. Save as PLY
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pth_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}.ply")
    
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Pointcept .pth tiles to .ply for CloudCompare")
    parser.add_argument("--input", required=True, help="Path to a single .pth file OR a directory of .pth files")
    parser.add_argument("--output_dir", default="data/labeled_plys", help="Directory to save the .ply files")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        pth_files = sorted(glob(os.path.join(args.input, "*.pth")))
        print(f"Found {len(pth_files)} tiles in directory.")
        for f in pth_files:
            convert_tile(f, args.output_dir)
    else:
        convert_tile(args.input, args.output_dir)

if __name__ == "__main__":
    main()