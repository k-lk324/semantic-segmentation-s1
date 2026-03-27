#!/bin/bash

set -e  # Exit on any error

# Create results directory if it doesn't exist
mkdir -p results_ablation

echo "====================================="
echo "  STARTING ZERO-SHOT ABLATION STUDY  "
echo "====================================="

# Config 1: Baseline (Raw dense points, standard FJD intensity)
echo -e "\n---> Running Config 1: Baseline (Raw + Intensity)..."
make infer SIM_VELODYNE=0 FEATURE_MODE=intensity
cp -r data/predictions results_ablation/config1_baseline
echo "  Evaluating original 16 classes..."
docker run --rm --user 1003:1003 \
  -v $(pwd):/workspace/project \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir results_ablation/config1_baseline \
  --output_csv results_ablation/config1_baseline.csv
echo "  Evaluating remapped 4 classes..."
docker run --rm --user 1003:1003 \
  -v $(pwd):/workspace/project \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir results_ablation/config1_baseline \
  --output_csv results_ablation/config1_baseline_remapped.csv \
  --remap

# Config 2: Zeroed Intensity Only (Testing feature camouflage)
echo -e "\n---> Running Config 2: Zeroed Intensity (Raw + Zero)..."
make infer SIM_VELODYNE=0 FEATURE_MODE=zero
cp -r data/predictions results_ablation/config2_zero
echo "  Evaluating original 16 classes..."
docker run --rm --user 1003:1003 \
  -v $(pwd):/workspace/project \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir results_ablation/config2_zero \
  --output_csv results_ablation/config2_zero.csv
echo "  Evaluating remapped 4 classes..."
docker run --rm --user 1003:1003 \
  -v $(pwd):/workspace/project \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir results_ablation/config2_zero \
  --output_csv results_ablation/config2_zero_remapped.csv \
  --remap

# Config 3: Thick Velodyne Rings + Zeroed Intensity
echo -e "\n---> Running Config 3: Thick Velodyne Rings (Tol: 0.15 + Zero)..."
make infer SIM_VELODYNE=1 VELODYNE_TOL=0.15 FEATURE_MODE=zero
cp -r data/predictions results_ablation/config3_velodyne_thick
echo "  Evaluating original 16 classes..."
docker run --rm --user 1003:1003 \
  -v $(pwd):/workspace/project \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir results_ablation/config3_velodyne_thick \
  --output_csv results_ablation/config3_velodyne_thick.csv
echo "  Evaluating remapped 4 classes..."
docker run --rm --user 1003:1003 \
  -v $(pwd):/workspace/project \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir results_ablation/config3_velodyne_thick \
  --output_csv results_ablation/config3_velodyne_thick_remapped.csv \
  --remap

# Config 4: Razor Thin Velodyne Rings + Zeroed Intensity
echo -e "\n---> Running Config 4: Razor Thin Velodyne Rings (Tol: 0.03 + Zero)..."
make infer SIM_VELODYNE=1 VELODYNE_TOL=0.03 FEATURE_MODE=zero
cp -r data/predictions results_ablation/config4_velodyne_thin
echo "  Evaluating original 16 classes..."
docker run --rm --user 1003:1003 \
  -v $(pwd):/workspace/project \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir results_ablation/config4_velodyne_thin \
  --output_csv results_ablation/config4_velodyne_thin.csv
echo "  Evaluating remapped 4 classes..."
docker run --rm --user 1003:1003 \
  -v $(pwd):/workspace/project \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir results_ablation/config4_velodyne_thin \
  --output_csv results_ablation/config4_velodyne_thin_remapped.csv \
  --remap

echo -e "\n====================================="
echo "  ALL EXPERIMENTS COMPLETED!           "
echo "====================================="
echo -e "\nResults saved to results_ablation/"
echo ""
echo "=== 16-Class Original Evaluation ==="
for csv in results_ablation/config[1-4]*.csv; do
    if [[ -f "$csv" ]] && [[ ! "$csv" =~ "remapped" ]]; then
        miou=$(tail -1 "$csv" | cut -d',' -f2)
        echo "  $(basename $csv .csv): mIoU = $miou%"
    fi
done
echo ""
echo "=== 4-Class Remapped Evaluation ==="
for csv in results_ablation/config*_remapped.csv; do
    if [ -f "$csv" ]; then
        miou=$(tail -1 "$csv" | cut -d',' -f2)
        echo "  $(basename $csv .csv): mIoU = $miou%"
    fi
done