import torch
import numpy as np
import laspy
import argparse
import copy
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from class_mappings import CLASS_NAMES, SUPERCLASS_MAPPING, SUPERCLASS_NAMES
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
    parser.add_argument("--output_las", required=True,
                        help="Output LAS path for remapped superclass labels (4 classes)")
    parser.add_argument("--output_las_original", default=None,
                        help="Optional output LAS path for original nuScenes labels (16 classes)")
    args = parser.parse_args()
    coords = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    # Center coordinates before voxel subsampling to match tiling behavior
    coord_shift = coords.mean(axis=0)
    coords -= coord_shift

    features = extract_features(las, len(coords))
    orig_indices = np.arange(len(coords), dtype=np.int64)

    print("[Rebuild] Applying voxel subsampling...")
    coords, features, subsample_orig_indices = voxel_grid_subsampling(
        coords, features, orig_indices, VOXEL_SIZE)
    # Restore original coordinate frame after subsampling
    coords += coord_shift

    N = coords.shape[0]
    C = args.num_classes

    print(f"[Init] Global subsampled points: {N}")

    # Build mapping from original index -> subsampled index (via sorting)
    sorted_order = np.argsort(subsample_orig_indices)
    sorted_orig = subsample_orig_indices[sorted_order]

    global_probs = np.zeros((N, C), dtype=np.float32)
    global_counts = np.zeros(N, dtype=np.int32)

    tile_files = sorted(Path(args.tiles_dir).glob("*.pth"))

    for tile_path in tqdm(tile_files, desc="Accumulating predictions"):
        pred_path = Path(args.pred_dir) / tile_path.name
        if not pred_path.exists():
            print(f"\n[Warning] Prediction not found: {pred_path.name}")
            continue

        tile = torch.load(tile_path, weights_only=False)
        logits = torch.load(pred_path).numpy().astype(np.float32)

        if "indices" not in tile:
            raise KeyError(
                f"Tile {tile_path.name} missing 'indices' key. "
                "Please re-run preprocessing with updated tile_s1.py"
            )
        
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
    
    # Apply superclass remapping (16 classes → 4 superclasses)
    # This merges truck with driveable (truck misclassifies roads)
    labels_remapped = np.full(N, -1, dtype=np.int32)
    labels_remapped[valid] = SUPERCLASS_MAPPING[labels[valid]]

    def build_output_las(semantic_labels):
        out_header = copy.deepcopy(las.header)
        out = laspy.LasData(out_header)
        out.points = laspy.ScaleAwarePointRecord.zeros(len(coords),
                                                       header=out_header)
        out.x = coords[:, 0]
        out.y = coords[:, 1]
        out.z = coords[:, 2]

        # Preserve original (non-normalized) intensity and RGB from source LAS.
        # Use subsample_orig_indices so each reconstructed point gets attributes
        # from the exact representative source point used during voxel subsampling.
        if hasattr(las, 'intensity'):
            out.intensity = np.asarray(las.intensity)[subsample_orig_indices].astype(np.uint16)
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            out.red = np.asarray(las.red)[subsample_orig_indices].astype(np.uint16)
            out.green = np.asarray(las.green)[subsample_orig_indices].astype(np.uint16)
            out.blue = np.asarray(las.blue)[subsample_orig_indices].astype(np.uint16)

        if "semantic_label" not in set(out.point_format.dimension_names):
            out.add_extra_dim(
                laspy.ExtraBytesParams(name="semantic_label", type=np.int32))
        out.semantic_label = semantic_labels
        return out

    out_original = build_output_las(labels)
    out_original.write(output_original_path)

    out_remapped = build_output_las(labels_remapped)
    out_remapped.write(output_remapped_path)
    
    # Print statistics
    print(f"\nReconstruction complete!")
    print(f"  Output (original classes): {output_original_path}")
    print(f"  Output (remapped classes): {output_remapped_path}")
    print(f"  Total points: {len(coords):,}")
    print(f"  Points with predictions: {np.sum(valid):,} ({100*np.sum(valid)/len(coords):.1f}%)")
    
    print(f"\nOriginal class distribution (16 nuScenes classes):")
    unique, counts = np.unique(labels[valid], return_counts=True)
    for cls, count in zip(unique, counts):
        percentage = 100 * count / np.sum(valid)
        class_name = CLASS_NAMES.get(cls, "unknown")
        print(f"  Class {cls:2d} ({class_name:25s}): {count:7,} points ({percentage:5.1f}%)")
    
    print(f"\nRemapped superclass distribution (4 classes in output LAS):")
    unique_super, counts_super = np.unique(labels_remapped[valid], return_counts=True)
    for cls, count in zip(unique_super, counts_super):
        percentage = 100 * count / np.sum(valid)
        class_name = SUPERCLASS_NAMES.get(cls, "unknown")
        print(f"  Class {cls:1d} ({class_name:12s}): {count:8,} points ({percentage:5.1f}%)")


if __name__ == "__main__":
    main()
