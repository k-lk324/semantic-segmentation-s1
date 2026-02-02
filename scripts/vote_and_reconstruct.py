import torch
import numpy as np
import laspy
import argparse
from pathlib import Path
from tqdm import tqdm

from src.preprocessing.tile_utils import voxel_grid_subsampling, extract_features, VOXEL_SIZE


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_las", required=True)
    parser.add_argument("--tiles_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--output_las", required=True)
    args = parser.parse_args()

    print("[Load] Reading original LAS...")
    las = laspy.read(args.src_las)

    coords = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    features = extract_features(las, len(coords))
    orig_indices = np.arange(len(coords), dtype=np.int64)

    print("[Rebuild] Applying voxel subsampling...")
    coords, features, subsample_orig_indices = voxel_grid_subsampling(
        coords, features, orig_indices, VOXEL_SIZE)

    N = coords.shape[0]
    C = args.num_classes

    print(f"[Init] Global subsampled points: {N}")

    # Build mapping from original index -> subsampled index (via sorting)
    sorted_order = np.argsort(subsample_orig_indices)
    sorted_orig = subsample_orig_indices[sorted_order]

    global_probs = np.zeros((N, C), dtype=np.float32)
    global_counts = np.zeros(N, dtype=np.int32)

    tile_files = sorted(Path(args.tiles_dir).glob("*.pth"))

    for tile_path in tqdm(tile_files):
        pred_path = Path(args.pred_dir) / tile_path.name
        if not pred_path.exists():
            continue

        tile = torch.load(tile_path, weights_only=False)
        logits = torch.load(pred_path).numpy()

        indices = np.asarray(tile["indices"], dtype=np.int64)
        probs = softmax(logits)

        pos = np.searchsorted(sorted_orig, indices)
        valid = (pos < len(sorted_orig)) & (sorted_orig[pos] == indices)
        if not np.all(valid):
            indices = indices[valid]
            probs = probs[valid]
            pos = pos[valid]

        mapped_indices = sorted_order[pos]

        global_probs[mapped_indices] += probs
        global_counts[mapped_indices] += 1

    valid = global_counts > 0
    global_probs[valid] /= global_counts[valid, None]

    labels = np.full(N, -1, dtype=np.int32)
    labels[valid] = np.argmax(global_probs[valid], axis=1)

    out = laspy.LasData(las.header)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(coords),
                                                   header=las.header)
    out.x = coords[:, 0]
    out.y = coords[:, 1]
    out.z = coords[:, 2]

    out.add_extra_dim(
        laspy.ExtraBytesParams(name="semantic_label", type=np.int32))
    out.semantic_label = labels

    out.write(args.output_las)
    print(f"[Done] Saved voxelized segmented cloud {args.output_las}")


if __name__ == "__main__":
    main()
