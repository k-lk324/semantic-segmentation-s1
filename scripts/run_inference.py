import torch
import glob
import os
import argparse
import numpy as np
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
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model).to(device)
    load_checkpoint(model, args.weights, map_location=device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    tile_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pth")))

    print(f"Running inference on {len(tile_files)} tiles...")

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

            if raw_feat.ndim == 1:
                intensity = raw_feat.unsqueeze(1)
            else:
                intensity = raw_feat[:, :1]  # Take first channel only

            input_feat = torch.cat([coord, intensity], dim=1)

            grid_size = 0.05
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
