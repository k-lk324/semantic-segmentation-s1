import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

# Ensure scripts directory is on sys.path for local module resolution
sys.path.append(str(Path(__file__).resolve().parent))

from class_mappings import SUPERCLASS_MAPPING, SUPERCLASS_NAMES

# Five hand-annotated classes from nuScenes.
# Note on labeling convention: nuScenes class 0 is 'barrier', but barrier was not annotated
# in this project. Ground-truth label 0 is treated as unannotated/ignored in 16-class mode.
CLASS_NAMES: dict[int, str] = {
    -1: "Ignore",
    3: "Car",
    9: "Truck",
    10: "Driveable Surface",
    14: "Manmade",
    15: "Vegetation",
}


def load_gt_labels(ply_path: str | Path) -> np.ndarray:
    """Load ground-truth labels from a .ply file."""
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    if "scalar_segmentation" not in plydata.elements[0].data.dtype.names:
        raise ValueError(f"No 'scalar_segmentation' field found in {ply_path}.")
    return np.array(plydata.elements[0].data["scalar_segmentation"], dtype=np.int32)


def load_pred_labels(pth_path: str | Path) -> np.ndarray:
    """Load prediction labels from a .pth file."""
    import torch

    logits = torch.load(str(pth_path), weights_only=False)
    if logits.ndim > 1:
        preds = torch.argmax(logits, dim=1)
    else:
        preds = logits
    return preds.cpu().numpy().astype(np.int32)


def calculate_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_id: int,
) -> tuple[float, float, float]:
    """Calculate IoU, Precision, and Recall for a specific class ID."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Ground truth and predictions must have matching shapes")

    tp = int(np.count_nonzero((y_true == class_id) & (y_pred == class_id)))
    fp = int(np.count_nonzero((y_true != class_id) & (y_pred == class_id)))
    fn = int(np.count_nonzero((y_true == class_id) & (y_pred != class_id)))

    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return iou, precision, recall


def validate_prediction_labels(y_pred: np.ndarray) -> None:
    """Validate that all prediction labels belong to [0, 15]."""
    if y_pred.size > 0 and ((y_pred < 0) | (y_pred > 15)).any():
        invalid_labels = np.unique(y_pred[(y_pred < 0) | (y_pred > 15)])
        raise ValueError(
            f"Prediction labels must belong to [0, 15]. Found invalid labels: {invalid_labels}"
        )


def remap_predictions(y_pred: np.ndarray) -> np.ndarray:
    """Validate and remap 16-class predictions to 4 superclasses."""
    validate_prediction_labels(y_pred)
    return SUPERCLASS_MAPPING[y_pred]


def remap_ground_truth(y_true: np.ndarray) -> np.ndarray:
    """Remap annotated classes to 4 superclasses while preserving unlabeled regions."""
    y_true_remapped = np.full_like(y_true, -1)
    label_map = {3: 1, 9: 1, 10: 2, 14: 3, 15: 0}
    for orig_class, superclass in label_map.items():
        mask = (y_true == orig_class)
        y_true_remapped[mask] = superclass
    # Preserve original unlabeled/ignore regions
    y_true_remapped[y_true < 0] = y_true[y_true < 0]
    return y_true_remapped


def load_evaluation_dataset(
    gt_dir: str | Path,
    pred_dir: str | Path,
    gt_loader: Callable[[str | Path], np.ndarray] = load_gt_labels,
    pred_loader: Callable[[str | Path], np.ndarray] = load_pred_labels,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load, validate, and concatenate ground truth and prediction files."""
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)

    gt_files = sorted(glob.glob(str(gt_path / "*.ply")))
    if not gt_files:
        raise FileNotFoundError(f"No .ply files found in {gt_dir}")

    all_gt: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    total_points = 0

    for gt_file in gt_files:
        stem = Path(gt_file).stem
        pred_file = pred_path / f"{stem}.pth"

        if not pred_file.exists():
            raise FileNotFoundError(
                f"Missing prediction file for ground truth '{stem}': {pred_file}"
            )

        gt_labels = gt_loader(gt_file)
        pred_labels = pred_loader(pred_file)

        if len(gt_labels) != len(pred_labels):
            raise ValueError(
                f"Mismatched point counts for '{stem}': "
                f"GT has {len(gt_labels)}, Pred has {len(pred_labels)}"
            )

        validate_prediction_labels(pred_labels)

        all_gt.append(gt_labels)
        all_pred.append(pred_labels)
        total_points += len(gt_labels)

    if not all_gt or total_points == 0:
        raise ValueError("No valid evaluation samples found.")

    y_true = np.concatenate(all_gt)
    y_pred = np.concatenate(all_pred)
    total_files = len(gt_files)

    return y_true, y_pred, total_files, total_points


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    remap: bool = False,
) -> tuple[list[tuple[str, float, float, float]], float]:
    """Compute per-class IoU, precision, recall, and overall mean IoU."""
    if remap:
        y_pred = remap_predictions(y_pred)
        y_true = remap_ground_truth(y_true)
        # In remap mode, class 0 (vegetation) is valid, unlabeled regions are < 0
        valid_mask = (y_true >= 0)
    else:
        # In original mode, label 0 is treated as unannotated/ignored
        valid_mask = (y_true > 0)

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    present_classes = np.unique(y_true_valid)
    metrics_list: list[tuple[str, float, float, float]] = []
    iou_list: list[float] = []

    for class_id in present_classes:
        iou, precision, recall = calculate_class_metrics(
            y_true_valid,
            y_pred_valid,
            int(class_id),
        )
        if remap:
            name = SUPERCLASS_NAMES.get(int(class_id), f"Class {class_id}")
        else:
            name = CLASS_NAMES.get(int(class_id), f"Class {class_id}")

        metrics_list.append((name, iou, precision, recall))
        iou_list.append(iou)

    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    return metrics_list, mean_iou


def resolve_output_filename(pred_dir: str, requested_filename: str | None) -> str:
    """Resolve destination CSV path using run_config.json if not specified."""
    if requested_filename:
        return requested_filename

    config_path = os.path.join(pred_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        sim = "velodyne" if config.get("sim_velodyne") else "raw"
        tol = config.get("velodyne_tol", "none")
        feat = config.get("feature_mode", "standard")
        return f"results_mode_{sim}_tol_{tol}_feat_{feat}.csv"

    return "results_manual_run.csv"


def save_metrics_to_csv(
    output_filename: str,
    metrics: list[tuple[str, float, float, float]],
    mean_iou: float,
) -> None:
    """Save evaluation metrics to CSV file."""
    with open(output_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class Name", "IoU (%)", "Precision (%)", "Recall (%)"])
        for name, iou, precision, recall in metrics:
            writer.writerow([name, f"{iou*100:.2f}", f"{precision*100:.2f}", f"{recall*100:.2f}"])
        writer.writerow([])
        writer.writerow(["Mean IoU", f"{mean_iou*100:.2f}", "", ""])


def main():
    parser = argparse.ArgumentParser(description="Calculate mIoU against hand-labeled FJD data.")
    parser.add_argument("--gt_dir", required=True, help="Folder containing your hand-labeled .ply files")
    parser.add_argument("--pred_dir", required=True, help="Folder containing the model's prediction .pth files")
    parser.add_argument("--output_csv", type=str, default=None, help="Force a specific CSV name (optional)")
    parser.add_argument("--remap", action="store_true", help="Remap 16 classes to 4 superclasses before evaluation")
    args = parser.parse_args()

    output_filename = resolve_output_filename(args.pred_dir, args.output_csv)
    print(f"\n[Info] Metrics will be saved to: {output_filename}\n")

    y_true, y_pred, total_files, total_points = load_evaluation_dataset(args.gt_dir, args.pred_dir)
    print(f"[Info] Evaluated {total_files} file(s) with {total_points} total points.")

    if args.remap:
        print("[Info] Applying 4-class superclass remapping...\n")

    metrics, mean_iou = compute_evaluation_metrics(y_true, y_pred, remap=args.remap)

    print(f"{'Class Name':<20} | {'IoU (%)':<8} | {'Precision (%)':<13} | {'Recall (%)':<10}")
    print("-" * 50)
    for name, iou, precision, recall in metrics:
        print(f"{name:<20} | {iou*100:>7.2f}% | {precision*100:>12.2f}% | {recall*100:>9.2f}%")
    print("-" * 50)
    print(f"Mean IoU (mIoU): {mean_iou*100:.2f}%\n")

    save_metrics_to_csv(output_filename, metrics, mean_iou)
    print(f"Successfully saved results to {output_filename}")


if __name__ == "__main__":
    main()
