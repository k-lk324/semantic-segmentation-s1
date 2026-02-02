import torch
import numpy as np
import laspy
import os
import glob
from tqdm import tqdm
import argparse

# NuScenes Color Palette (16 Classes)
# We map these to RGB (0-255) for visualization
COLOR_MAP = {
    0: [255, 120, 50],  # barrier (Orange)
    1: [255, 192, 203],  # bicycle (Pink)
    2: [255, 255, 0],  # bus (Yellow)
    3: [0, 0, 255],  # car (Blue)
    4: [0, 255, 255],  # construction_vehicle (Cyan)
    5: [255, 0, 0],  # motorcycle (Red)
    6: [255, 240, 150],  # pedestrian (Light Yellow)
    7: [135, 60, 0],  # traffic_cone (Brown)
    8: [160, 32, 240],  # trailer (Purple)
    9: [255, 61, 99],  # truck (Red)
    10: [175, 240, 255],  # driveable_surface (Light Blue)
    11: [75, 0, 75],  # other_flat (Dark Purple)
    12: [75, 0, 175],  # sidewalk (Violet)
    13: [150, 240, 80],  # terrain (Grass Green)
    14: [230, 230, 250],  # manmade (Light Gray)
    15: [0, 175, 0],  # vegetation (Dark Green)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        required=True,
                        help="Path to processed_tiles")
    parser.add_argument("--pred_dir",
                        required=True,
                        help="Path to predictions")
    parser.add_argument("--output",
                        required=True,
                        help="Output .las file path")
    args = parser.parse_args()

    # Find matches
    tile_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pth")))

    all_points = []
    all_labels = []
    all_colors = []

    print(f"Reconstructing {len(tile_files)} tiles...")

    for tile_path in tqdm(tile_files):
        tile_name = os.path.basename(tile_path)
        pred_path = os.path.join(args.pred_dir, tile_name)

        if not os.path.exists(pred_path):
            print(f"Warning: No prediction found for {tile_name}")
            continue

        data = torch.load(tile_path, weights_only=False)
        logits = torch.load(pred_path, weights_only=False)  # shape (N, 16)

        local_coord = data["coord"]

        final_coord = local_coord

        if "grid_id" in data:

            grid_id = data["grid_id"]
            if isinstance(grid_id, torch.Tensor): grid_id = grid_id.numpy()

            block_size = getattr(args, "block_size", 20.0)
            block_center = np.array([
                grid_id[0] + block_size / 2, grid_id[1] + block_size / 2,
                grid_id[2]
            ])
            final_coord = final_coord + block_center

        if "global_shift" in data:
            final_coord = final_coord + data["global_shift"]

        all_points.append(final_coord)

        labels = torch.argmax(logits, dim=1).cpu().numpy()
        all_labels.append(labels)

        colors = np.zeros((len(labels), 3), dtype=np.uint8)
        for cls_id, rgb in COLOR_MAP.items():
            colors[labels == cls_id] = rgb
        all_colors.append(colors)

    if not all_points:
        print("No data found!")
        return

    print("Merging tiles...")
    points_cat = np.concatenate(all_points, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    colors_cat = np.concatenate(all_colors, axis=0)

    print(f"Saving {len(points_cat)} points to {args.output}...")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]  # 1mm precision

    offset = np.min(points_cat, axis=0)
    header.offset = offset

    las = laspy.LasData(header)
    las.x = points_cat[:, 0]
    las.y = points_cat[:, 1]
    las.z = points_cat[:, 2]

    las.classification = labels_cat.astype(np.uint8)

    las.red = colors_cat[:, 0].astype(
        np.uint16) * 256  # laspy expects 16-bit color
    las.green = colors_cat[:, 1].astype(np.uint16) * 256
    las.blue = colors_cat[:, 2].astype(np.uint16) * 256

    las.write(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
