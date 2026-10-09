import argparse
import copy
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# Ensure local imports resolve correctly
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from class_mappings import CLASS_NAMES, SUPERCLASS_MAPPING, SUPERCLASS_NAMES


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities along axis 1."""
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def map_tile_indices_to_subsampled(
    sorted_orig: np.ndarray,
    sorted_order: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Safely map tile indices to global subsampled point indices using searchsorted.

    Returns:
        valid: Boolean mask of tile points present in sorted_orig.
        mapped_indices: Indices into global subsampled arrays for valid points.
    """
    if sorted_orig.size == 0 or indices.size == 0:
        return np.zeros(len(indices), dtype=bool), np.array([], dtype=np.int64)

    pos = np.searchsorted(sorted_orig, indices)
    valid = pos < len(sorted_orig)
    matches = np.zeros_like(valid, dtype=bool)
    matches[valid] = (sorted_orig[pos[valid]] == indices[valid])
    valid = matches

    pos_valid = pos[valid]
    mapped_indices = sorted_order[pos_valid]
    return valid, mapped_indices


def validate_tile_prediction(
    logits: np.ndarray,
    expected_num_points: int,
    expected_num_classes: int,
    tile_name: str,
) -> None:
    """Validate prediction tensor dimensions, classes, and point counts."""
    if logits.ndim != 2:
        raise ValueError(
            f"Prediction {tile_name} has invalid dimension {logits.ndim}, expected 2D tensor"
        )
    if logits.shape[1] != expected_num_classes:
        raise ValueError(
            f"Prediction {tile_name} has {logits.shape[1]} classes, "
            f"expected {expected_num_classes}"
        )
    if logits.shape[0] != expected_num_points:
        raise ValueError(
            f"Prediction {tile_name} point count ({logits.shape[0]}) "
            f"does not match tile point count ({expected_num_points})"
        )


def ensure_output_directories(*output_paths: str | Path) -> None:
    """Ensure parent directories exist before writing files."""
    for output_path in output_paths:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)


def build_output_las(
    template_header,
    coords: np.ndarray,
    semantic_labels: np.ndarray,
    subsample_orig_indices: np.ndarray,
    source_las,
):
    """Build output LasData preserving geometry, intensity, and RGB attributes."""
    import laspy

    out_header = copy.deepcopy(template_header)
    out = laspy.LasData(out_header)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(coords), header=out_header)
    out.x = coords[:, 0]
    out.y = coords[:, 1]
    out.z = coords[:, 2]

    if hasattr(source_las, "intensity"):
        out.intensity = np.asarray(source_las.intensity)[subsample_orig_indices].astype(np.uint16)
    if hasattr(source_las, "red") and hasattr(source_las, "green") and hasattr(source_las, "blue"):
        out.red = np.asarray(source_las.red)[subsample_orig_indices].astype(np.uint16)
        out.green = np.asarray(source_las.green)[subsample_orig_indices].astype(np.uint16)
        out.blue = np.asarray(source_las.blue)[subsample_orig_indices].astype(np.uint16)

    if "semantic_label" not in set(out.point_format.dimension_names):
        out.add_extra_dim(laspy.ExtraBytesParams(name="semantic_label", type=np.int32))
    out.semantic_label = semantic_labels
    return out


def print_reconstruction_summary(
    total_points: int,
    valid_points_count: int,
    labels: np.ndarray,
    labels_remapped: np.ndarray,
    valid_mask: np.ndarray,
    output_original_path: str,
    output_remapped_path: str,
) -> None:
    """Print statistical breakdown of reconstruction results."""
    print("\nReconstruction complete!")
    print(f"  Output (original classes): {output_original_path}")
    print(f"  Output (remapped classes): {output_remapped_path}")
    print(f"  Total points: {total_points:,}")
    percentage = (100.0 * valid_points_count / total_points) if total_points else 0.0
    print(f"  Points with predictions: {valid_points_count:,} ({percentage:.1f}%)")

    print("\nOriginal class distribution (16 nuScenes classes):")
    unique, counts = np.unique(labels[valid_mask], return_counts=True)
    for cls, count in zip(unique, counts):
        cls_pct = (100.0 * count / valid_points_count) if valid_points_count else 0.0
        class_name = CLASS_NAMES.get(int(cls), "unknown")
        print(f"  Class {cls:2d} ({class_name:25s}): {count:7,} points ({cls_pct:5.1f}%)")

    print("\nRemapped superclass distribution (4 classes in output LAS):")
    unique_super, counts_super = np.unique(labels_remapped[valid_mask], return_counts=True)
    for cls, count in zip(unique_super, counts_super):
        cls_pct = (100.0 * count / valid_points_count) if valid_points_count else 0.0
        class_name = SUPERCLASS_NAMES.get(int(cls), "unknown")
        print(f"  Class {cls:1d} ({class_name:12s}): {count:8,} points ({cls_pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Vote tile predictions and reconstruct scene-level LAS outputs."
    )
    parser.add_argument("--src_las", required=True)
    parser.add_argument("--tiles_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument(
        "--output_las",
        required=True,
        help="Output LAS path for remapped superclass labels (4 classes)",
    )
    parser.add_argument(
        "--output_las_original",
        default=None,
        help="Optional output LAS path for original nuScenes labels (16 classes)",
    )
    args = parser.parse_args()

    import laspy
    import torch
    from tqdm import tqdm
    from src.preprocessing.tile_utils import voxel_grid_subsampling, extract_features, VOXEL_SIZE

    las = laspy.read(args.src_las)
    output_remapped_path = args.output_las
    output_original_path = args.output_las_original
    if output_original_path is None:
        output_original_path = str(
            Path(output_remapped_path).with_name(
                Path(output_remapped_path).stem + "_original.las"
            )
        )

    coords = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    coord_shift = coords.mean(axis=0)
    coords -= coord_shift

    features = extract_features(las, len(coords))
    orig_indices = np.arange(len(coords), dtype=np.int64)

    print("[Rebuild] Applying voxel subsampling...")
    coords, features, subsample_orig_indices = voxel_grid_subsampling(
        coords, features, orig_indices, VOXEL_SIZE
    )
    coords += coord_shift

    N = coords.shape[0]
    C = args.num_classes

    if N == 0 or subsample_orig_indices.size == 0:
        raise ValueError("No subsampled points available for reconstruction.")

    print(f"[Init] Global subsampled points: {N}")

    sorted_order = np.argsort(subsample_orig_indices)
    sorted_orig = subsample_orig_indices[sorted_order]

    global_probs = np.zeros((N, C), dtype=np.float32)
    global_counts = np.zeros(N, dtype=np.int32)

    tile_files = sorted(Path(args.tiles_dir).glob("*.pth"))
    valid_tiles_count = 0

    for tile_path in tqdm(tile_files, desc="Accumulating predictions"):
        pred_path = Path(args.pred_dir) / tile_path.name
        if not pred_path.exists():
            print(f"\n[Warning] Prediction not found: {pred_path.name}")
            continue

        tile = torch.load(tile_path, weights_only=False)
        logits_raw = torch.load(pred_path, weights_only=False)
        if isinstance(logits_raw, torch.Tensor):
            logits = logits_raw.cpu().numpy().astype(np.float32)
        else:
            logits = np.asarray(logits_raw, dtype=np.float32)

        if "indices" not in tile:
            raise KeyError(
                f"Tile {tile_path.name} missing 'indices' key. "
                "Please re-run preprocessing with updated tile_s1.py"
            )

        indices = np.asarray(tile["indices"], dtype=np.int64)

        validate_tile_prediction(
            logits=logits,
            expected_num_points=len(indices),
            expected_num_classes=C,
            tile_name=tile_path.name,
        )

        probs = softmax(logits)

        valid_mask, mapped_indices = map_tile_indices_to_subsampled(
            sorted_orig=sorted_orig,
            sorted_order=sorted_order,
            indices=indices,
        )

        if not np.all(valid_mask):
            probs = probs[valid_mask]

        global_probs[mapped_indices] += probs
        global_counts[mapped_indices] += 1
        valid_tiles_count += 1

    if valid_tiles_count == 0:
        raise ValueError("Reconstruction received 0 valid prediction tiles.")

    valid_global = global_counts > 0
    global_probs[valid_global] /= global_counts[valid_global, None]

    labels = np.full(N, -1, dtype=np.int32)
    labels[valid_global] = np.argmax(global_probs[valid_global], axis=1)

    labels_remapped = np.full(N, -1, dtype=np.int32)
    labels_remapped[valid_global] = SUPERCLASS_MAPPING[labels[valid_global]]

    ensure_output_directories(output_remapped_path, output_original_path)

    out_original = build_output_las(
        template_header=las.header,
        coords=coords,
        semantic_labels=labels,
        subsample_orig_indices=subsample_orig_indices,
        source_las=las,
    )
    out_original.write(output_original_path)

    out_remapped = build_output_las(
        template_header=las.header,
        coords=coords,
        semantic_labels=labels_remapped,
        subsample_orig_indices=subsample_orig_indices,
        source_las=las,
    )
    out_remapped.write(output_remapped_path)

    print_reconstruction_summary(
        total_points=len(coords),
        valid_points_count=int(np.sum(valid_global)),
        labels=labels,
        labels_remapped=labels_remapped,
        valid_mask=valid_global,
        output_original_path=output_original_path,
        output_remapped_path=output_remapped_path,
    )


if __name__ == "__main__":
    main()
