import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure scripts directory is available for importing calculate_miou and class_mappings
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "scripts"))

from calculate_miou import (
    calculate_class_metrics,
    validate_prediction_labels,
    remap_predictions,
    remap_ground_truth,
    load_evaluation_dataset,
    compute_evaluation_metrics,
)
from class_mappings import SUPERCLASS_MAPPING


def test_perfect_predictions():
    """Verify IoU = 100%, Precision = 100%, and Recall = 100% on identical arrays."""
    y_true = np.array([3, 3, 10, 10, 15])
    y_pred = np.array([3, 3, 10, 10, 15])

    iou, precision, recall = calculate_class_metrics(y_true, y_pred, class_id=3)
    assert iou == 1.0
    assert precision == 1.0
    assert recall == 1.0


def test_all_predictions_incorrect():
    """Verify IoU = 0%, Precision = 0%, and Recall = 0% when no predictions match."""
    y_true = np.array([3, 3, 10, 10])
    y_pred = np.array([14, 14, 15, 15])

    iou, precision, recall = calculate_class_metrics(y_true, y_pred, class_id=3)
    assert iou == 0.0
    assert precision == 0.0
    assert recall == 0.0


def test_prediction_outside_annotated_classes():
    """Verify regression test where incorrect prediction outside present class counts as false negative."""
    y_true = np.array([3, 3, 10, 10])
    y_pred = np.array([3, 9, 10, 9])

    iou, precision, recall = calculate_class_metrics(y_true, y_pred, class_id=3)
    assert iou == 0.5
    assert recall == 0.5


def test_four_class_remapping():
    """Verify correct superclass assignments for both predictions and ground truth."""
    # Test predictions remapping across all 16 nuScenes classes
    preds = np.arange(16)
    remapped_preds = remap_predictions(preds)
    np.testing.assert_array_equal(remapped_preds, SUPERCLASS_MAPPING)

    # Test ground truth remapping for the 5 hand-annotated classes:
    # 3 (Car) -> 1 (object), 9 (Truck) -> 1 (object)
    # 10 (Driveable Surface) -> 2 (ground)
    # 14 (Manmade) -> 3 (structure)
    # 15 (Vegetation) -> 0 (vegetation)
    # negative values (unlabeled/ignore) -> preserved
    gt = np.array([-1, 3, 9, 10, 14, 15])
    expected_remapped_gt = np.array([-1, 1, 1, 2, 3, 0])
    remapped_gt = remap_ground_truth(gt)
    np.testing.assert_array_equal(remapped_gt, expected_remapped_gt)


def test_missing_prediction_file(tmp_path):
    """Verify explicit FileNotFoundError when corresponding prediction file is absent."""
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    # Create dummy ground-truth file
    (gt_dir / "sample_001.ply").write_text("dummy")

    with pytest.raises(FileNotFoundError, match="Missing prediction file"):
        load_evaluation_dataset(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            gt_loader=lambda p: np.array([3, 10]),
            pred_loader=lambda p: np.array([3, 10]),
        )


def test_mismatched_point_counts(tmp_path):
    """Verify explicit ValueError when ground truth and prediction point counts mismatch."""
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    (gt_dir / "sample_001.ply").write_text("dummy")
    (pred_dir / "sample_001.pth").write_text("dummy")

    with pytest.raises(ValueError, match="Mismatched point counts"):
        load_evaluation_dataset(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            gt_loader=lambda p: np.array([3, 10, 15]),
            pred_loader=lambda p: np.array([3, 10]),
        )


def test_invalid_prediction_id():
    """Verify explicit ValueError when prediction label is outside [0, 15]."""
    invalid_preds_negative = np.array([3, -1, 10])
    invalid_preds_exceed = np.array([3, 16, 10])

    with pytest.raises(ValueError, match="Prediction labels must belong to"):
        validate_prediction_labels(invalid_preds_negative)

    with pytest.raises(ValueError, match="Prediction labels must belong to"):
        validate_prediction_labels(invalid_preds_exceed)

    with pytest.raises(ValueError, match="Prediction labels must belong to"):
        remap_predictions(invalid_preds_negative)


def test_calculate_class_metrics_shape_mismatch():
    """Verify ValueError when y_true and y_pred shapes differ."""
    y_true = np.array([3, 10])
    y_pred = np.array([3, 10, 15])

    with pytest.raises(ValueError, match="matching shapes"):
        calculate_class_metrics(y_true, y_pred, class_id=3)


def test_compute_evaluation_metrics_unannotated_zero():
    """Verify that in original mode label 0 is ignored and metrics average over annotated classes."""
    # Label 0 in y_true is ignored in original mode
    y_true = np.array([0, 3, 3, 10])
    y_pred = np.array([0, 3, 3, 10])

    metrics, mean_iou = compute_evaluation_metrics(y_true, y_pred, remap=False)
    # Only classes 3 and 10 should be evaluated
    class_names = [m[0] for m in metrics]
    assert "Car" in class_names
    assert "Driveable Surface" in class_names
    assert "barrier" not in class_names
    assert mean_iou == 1.0
