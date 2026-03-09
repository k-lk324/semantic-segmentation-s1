#!/bin/bash

echo "====================================="
echo "  STARTING ZERO-SHOT ABLATION STUDY  "
echo "====================================="

# Config 1: Baseline (Raw dense points, standard FJD intensity)
echo -e "\n---> Running Config 1: Baseline..."
make infer SIM_VELODYNE=0 FEATURE_MODE=intensity
python scripts/calculate_miou.py --gt_dir data/ground_truth_labeled_tiles --pred_dir data/predictions

# Config 2: Zeroed Intensity Only (Testing feature camouflage)
echo -e "\n---> Running Config 2: Zeroed Intensity..."
make infer SIM_VELODYNE=0 FEATURE_MODE=zero
python scripts/calculate_miou.py --gt_dir data/ground_truth_labeled_tiles --pred_dir data/predictions

# Config 3: Thick Velodyne Rings + Zeroed Intensity
echo -e "\n---> Running Config 3: Thick Velodyne Rings (Tol: 0.15)..."
make infer SIM_VELODYNE=1 VELODYNE_TOL=0.15 FEATURE_MODE=zero
python scripts/calculate_miou.py --gt_dir data/ground_truth_labeled_tiles --pred_dir data/predictions

# Config 4: Razor Thin Velodyne Rings + Zeroed Intensity
echo -e "\n---> Running Config 4: Razor Thin Velodyne Rings (Tol: 0.03)..."
make infer SIM_VELODYNE=1 VELODYNE_TOL=0.03 FEATURE_MODE=zero
python scripts/calculate_miou.py --gt_dir data/ground_truth_labeled_tiles --pred_dir data/predictions

echo -e "\n====================================="
echo "  ALL EXPERIMENTS COMPLETED!           "
echo "====================================="