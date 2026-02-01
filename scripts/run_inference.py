import torch
import glob
import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import sys

# Add Pointcept to path
sys.path.insert(0, "/workspace/libs/Pointcept")

from pointcept.models import build_model
from pointcept.utils.config import Config


def load_checkpoint(model, path, map_location="cuda"):
    print(f"Loading weights from {path}...")
    checkpoint = torch.load(path,
                            map_location=map_location,
                            weights_only=False)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Load into model
    model.load_state_dict(new_state_dict, strict=False)
    print("Weights loaded successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.05,
        help=
        "Grid size for voxelization. Should match VOXEL_SIZE from preprocessing (default: 0.05)"
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model).to(device)
    load_checkpoint(model, args.weights, map_location=device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    tile_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pth")))

    print(f"Running inference on {len(tile_files)} tiles...")

    # Load preprocessing metadata to validate grid_size consistency
    grid_size = args.grid_size
    metadata_path = os.path.join(args.data_dir, "tiling_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                prep_voxel_size = metadata.get('voxel_size', None)
                if prep_voxel_size is not None and abs(prep_voxel_size -
                                                       grid_size) > 1e-6:
                    print(
                        f"[Warning] grid_size ({grid_size}) does not match preprocessing VOXEL_SIZE ({prep_voxel_size})."
                    )
                else:
                    grid_size = prep_voxel_size
            print(f"[Info] Loaded preprocessing metadata from {metadata_path}")
        except Exception as e:
            print(f"[Warning] Could not read metadata: {e}")
    else:
        print(
            f"[Warning] No preprocessing metadata found at {metadata_path}. "
            f"Using grid_size={grid_size} (ensure it matches preprocessing VOXEL_SIZE)"
        )

    with torch.no_grad():
        for tile_path in tqdm(tile_files):
            data = torch.load(tile_path, weights_only=False)

            coord = data["coord"]
            if not torch.is_tensor(coord):
                coord = torch.from_numpy(coord)
            coord = coord.float().to(device)

            raw_feat = data["features"]
            if not torch.is_tensor(raw_feat):
                raw_feat = torch.from_numpy(raw_feat)
            raw_feat = raw_feat.float().to(device)

            # Extract intensity (first channel) from features
            # Note: tile_s1.py preprocessing may include RGB channels (shape: N x 4 for intensity+RGB),
            # but the model is configured with in_channels=4 expecting only [x, y, z, intensity].
            # We intentionally use only intensity to match the model's input specification.
            if raw_feat.ndim == 1:
                intensity = raw_feat.unsqueeze(1)
            else:
                intensity = raw_feat[:, :
                                     1]  # Take first channel (intensity) only

            input_feat = torch.cat([coord, intensity], dim=1)

            # Use grid_size from preprocessing (validated against metadata if available)
            grid_coord = torch.div(coord, grid_size,
                                   rounding_mode='floor').int()

            offset = torch.IntTensor([coord.shape[0]]).to(device)

            input_dict = dict(coord=coord,
                              grid_coord=grid_coord,
                              feat=input_feat,
                              offset=offset)

            output = model(input_dict)

            if isinstance(output, dict):
                seg_logits = output["seg_logits"]
            else:
                seg_logits = output

            save_path = os.path.join(args.output_dir, Path(tile_path).name)
            torch.save(seg_logits.half().cpu(), save_path)


if __name__ == "__main__":
    main()
