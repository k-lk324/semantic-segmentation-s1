import torch
import glob
import os
import argparse
from tqdm import tqdm
from pathlib import Path

from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.checkpoint import load_checkpoint


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
    load_checkpoint(model, args.weights, map_location="cuda")
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    tile_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pth")))

    print(f"Running inference on {len(tile_files)} tiles...")

    with torch.no_grad():
        for tile_path in tqdm(tile_files):
            data = torch.load(tile_path)

            coord = data["coord"].to(device)
            feat = data["features"].to(device)

            # NuScenes weights expect 4 channels (xyz + intensity)
            feat = feat[:, :1]

            offset = torch.IntTensor([coord.shape[0]]).to(device)
            input_dict = dict(coord=coord,
                              feat=feat,
                              offset=offset,
                              grid_size=0.05)

            output = model(input_dict)

            if isinstance(output, dict):
                seg_logits = output["seg_logits"]
            else:
                seg_logits = output

            save_path = os.path.join(args.output_dir, Path(tile_path).name)
            torch.save(seg_logits.half().cpu(), save_path)


if __name__ == "__main__":
    main()
