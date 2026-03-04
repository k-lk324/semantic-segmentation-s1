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
        "--feature_mode",
        type=str,
        choices=["intensity", "zero", "height", "blend"],
        default="intensity",
        help=
        "Auxiliary feature mode for channel 4: intensity|zero|height|blend (default: intensity)"
    )
    parser.add_argument(
        "--blend_alpha",
        type=float,
        default=0.5,
        help=
        "Intensity weight for blend mode. Feature = alpha*intensity + (1-alpha)*height (default: 0.5)"
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.05,
        help=
        "Grid size for voxelization. Should match VOXEL_SIZE from preprocessing (default: 0.05)"
    )
    args = parser.parse_args()

    if not 0.0 <= args.blend_alpha <= 1.0:
        raise ValueError(
            f"blend_alpha must be between 0 and 1, got {args.blend_alpha}")

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

    # Use a global offset large enough to ensure non-negative indices across all tiles
    # This offset is applied to grid indices only, not to coordinates passed to the model
    # The model expects tile-relative coordinates (centered around block centers)
    GRID_OFFSET = torch.tensor([1000, 1000, 1000], dtype=torch.int64).to(device)
    print(
        f"[Info] Feature mode: {args.feature_mode}"
        + (f" (blend_alpha={args.blend_alpha:.2f})"
           if args.feature_mode == "blend" else ""))

    # Compute global height bounds for consistent normalization across all tiles
    # This preserves absolute elevation information needed for top-down scans
    global_z_min, global_z_max = None, None
    if args.feature_mode in ["height", "blend"]:
        print("[Info] Computing global height bounds from absolute elevations...")
        z_values = []
        for tile_path in tqdm(tile_files, desc="Scanning heights"):
            data = torch.load(tile_path, weights_only=False)
            coord = data["coord"]
            if not torch.is_tensor(coord):
                coord = torch.from_numpy(coord)

            grid_id = data.get("grid_id")
            global_shift = data.get("global_shift")

            if grid_id is not None and global_shift is not None:
                if not torch.is_tensor(grid_id):
                    grid_id = torch.from_numpy(grid_id)
                if not torch.is_tensor(global_shift):
                    global_shift = torch.from_numpy(global_shift)

                # absolute_z = tile_relative_z + block_center_z + scene_shift_z
                abs_z = coord[:, 2] + grid_id[2] + global_shift[2]
            else:
                # Fallback when metadata is missing
                abs_z = coord[:, 2]

            z_values.append(abs_z.min().item())
            z_values.append(abs_z.max().item())
        global_z_min = min(z_values)
        global_z_max = max(z_values)
        z_range = global_z_max - global_z_min
        print(
            f"[Info] Global absolute Z range: [{global_z_min:.2f}, {global_z_max:.2f}] (span: {z_range:.2f}m)"
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

            # Build alternative auxiliary feature channels for domain adaptation
            # Reconstruct absolute elevation for height-based features
            z_coord = coord[:, 2:3]
            if global_z_min is not None and global_z_max is not None:
                grid_id = data.get("grid_id")
                global_shift = data.get("global_shift")

                if grid_id is not None and global_shift is not None:
                    if not torch.is_tensor(grid_id):
                        grid_id = torch.from_numpy(grid_id)
                    if not torch.is_tensor(global_shift):
                        global_shift = torch.from_numpy(global_shift)

                    grid_id = grid_id.to(device=device, dtype=coord.dtype)
                    global_shift = global_shift.to(device=device,
                                                   dtype=coord.dtype)
                    # absolute_z = tile_relative_z + block_center_z + scene_shift_z
                    abs_z = z_coord + grid_id[2] + global_shift[2]
                else:
                    abs_z = z_coord

                # Use global absolute height bounds for consistent normalization
                z_range = max(global_z_max - global_z_min, 1e-6)
                height = ((abs_z - global_z_min) / z_range) * 2.0 - 1.0
            else:
                # Fallback to per-tile normalization (shouldn't happen with height mode)
                z_min = z_coord.min()
                z_max = z_coord.max()
                z_range = (z_max - z_min).clamp(min=1e-6)
                height = ((z_coord - z_min) / z_range) * 2.0 - 1.0

            if args.feature_mode == "intensity":
                aux_feat = intensity
            elif args.feature_mode == "zero":
                aux_feat = torch.zeros_like(intensity)
            elif args.feature_mode == "height":
                aux_feat = height
            else:  # blend
                aux_feat = args.blend_alpha * intensity + (
                    1.0 - args.blend_alpha) * height
                aux_feat = torch.clamp(aux_feat, min=-1.0, max=1.0)

            # Use tile-relative coordinates (as they appear in the tiles, centered)
            # This is what the model was trained on
            input_feat = torch.cat([coord, aux_feat], dim=1)

            # Compute grid coordinates from relative coordinates with a global offset
            # The offset ensures all indices stay non-negative for sparse convolution
            grid_coord = torch.div(coord, grid_size,
                                   rounding_mode='floor').int() + GRID_OFFSET

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
