import torch
import numpy as np
import argparse
from pathlib import Path
from plyfile import PlyData
from sklearn.metrics import confusion_matrix
import glob
import os
import json
import csv

from class_mappings import SUPERCLASS_MAPPING, SUPERCLASS_NAMES

# The classes we expect you to have labeled based on nuScenes
CLASS_NAMES = {
    -1: "Ignore",
    3: "Car",
    10: "Driveable Surface",
    15: "Vegetation",
    14: "Manmade",
    9: "Truck",
}

def load_gt_labels(ply_path):
    plydata = PlyData.read(ply_path)
    if 'scalar_segmentation' not in plydata.elements[0].data.dtype.names:
        print(f"[Warning] No 'scalar_segmentation' field found in {ply_path}.")
        return None
    return np.array(plydata.elements[0].data['scalar_segmentation'], dtype=np.int32)

def load_pred_labels(pth_path):
    logits = torch.load(pth_path, weights_only=False)
    if logits.ndim > 1:
        preds = torch.argmax(logits, dim=1)
    else:
        preds = logits
    return preds.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Calculate mIoU against hand-labeled FJD data.")
    parser.add_argument("--gt_dir", required=True, help="Folder containing your hand-labeled .ply files")
    parser.add_argument("--pred_dir", required=True, help="Folder containing the model's prediction .pth files")
    parser.add_argument("--output_csv", type=str, default=None, help="Force a specific CSV name (optional)")
    parser.add_argument("--remap", action="store_true", help="Remap 16 classes to 4 superclasses before evaluation")
    args = parser.parse_args()

    # --- AUTO-DETECT CONFIGURATION FOR CSV NAMING ---
    output_filename = args.output_csv
    config_path = os.path.join(args.pred_dir, "run_config.json")
    
    if not output_filename:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            
            sim = "velodyne" if config.get("sim_velodyne") else "raw"
            tol = config.get("velodyne_tol", "none")
            feat = config.get("feature_mode", "standard")
            
            output_filename = f"results_mode_{sim}_tol_{tol}_feat_{feat}.csv"
        else:
            output_filename = "results_manual_run.csv"
            
    print(f"\n[Info] Metrics will be saved to: {output_filename}\n")

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.ply")))
    if not gt_files:
        print(f"No .ply files found in {args.gt_dir}")
        return

    all_gt, all_pred = [], []

    for gt_file in gt_files:
        filename = Path(gt_file).stem
        pred_file = os.path.join(args.pred_dir, f"{filename}.pth")

        if not os.path.exists(pred_file):
            continue

        gt_labels = load_gt_labels(gt_file)
        if gt_labels is None: continue
            
        pred_labels = load_pred_labels(pred_file)

        if len(gt_labels) != len(pred_labels):
            continue

        all_gt.append(gt_labels)
        all_pred.append(pred_labels)

    if not all_gt:
        print("No valid data to evaluate.")
        return

    y_true = np.concatenate(all_gt)
    y_pred = np.concatenate(all_pred)

    # Apply remapping if requested
    if args.remap:
        print("[Info] Applying 4-class superclass remapping...\n")
        # Remap predictions (0-15 → 0-3)
        y_pred_remapped = np.zeros_like(y_pred)
        for i in range(16):
            mask = (y_pred == i)
            y_pred_remapped[mask] = SUPERCLASS_MAPPING[i]
        y_pred = y_pred_remapped
        
        # Remap ground truth (handle only labeled classes)
        # Use -1 to mark unlabeled regions initially
        y_true_remapped = np.full_like(y_true, -1)
        label_map = {3: 1, 9: 1, 10: 2, 14: 3, 15: 0}  # car, truck→object; driveable→ground; manmade→structure; vegetation→vegetation
        for orig_class, superclass in label_map.items():
            mask = (y_true == orig_class)
            y_true_remapped[mask] = superclass
        # Keep original unlabeled/ignore regions as-is
        y_true_remapped[y_true < 0] = y_true[y_true < 0]
        y_true = y_true_remapped
    
    # In remap mode, class 0 (vegetation) is valid, so filter out only negative values
    # In original mode, filter out class 0 and negative values
    if args.remap:
        valid_mask = (y_true >= 0)
    else:
        valid_mask = (y_true > 0)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    present_classes = np.unique(y_true_valid)
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=present_classes)

    print(f"{'Class Name':<20} | {'IoU (%)':<8} | {'Precision (%)':<13} | {'Recall (%)':<10}")
    print("-" * 50)

    iou_list, precision_list, recall_list, name_list = [], [], [], []

    for i, class_id in enumerate(present_classes):
        if args.remap:
            name = SUPERCLASS_NAMES.get(class_id, f"Class {class_id}")
        else:
            name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        iou_list.append(iou)
        precision_list.append(precision)
        recall_list.append(recall)
        name_list.append(name)

        print(f"{name:<20} | {iou*100:>7.2f}% | {precision*100:>12.2f}% | {recall*100:>9.2f}%")

    mean_iou = np.mean(iou_list)
    print("-" * 50)
    print(f"Mean IoU (mIoU): {mean_iou*100:.2f}%\n")

    # --- SAVE TO CSV ---
    with open(output_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Class Name", "IoU (%)", "Precision (%)", "Recall (%)"])
        for i in range(len(present_classes)):
            writer.writerow([name_list[i], f"{iou_list[i]*100:.2f}", f"{precision_list[i]*100:.2f}", f"{recall_list[i]*100:.2f}"])
        writer.writerow([])
        writer.writerow(["Mean IoU", f"{mean_iou*100:.2f}", "", ""])
    print(f"Successfully saved results to {output_filename}")

if __name__ == "__main__":
    main()