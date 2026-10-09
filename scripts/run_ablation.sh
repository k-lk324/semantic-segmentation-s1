#!/bin/bash

set -euo pipefail

# Parse command line flags
OVERWRITE=0
for arg in "$@"; do
    case "$arg" in
        --force|-f)
            OVERWRITE=1
            ;;
        --help|-h)
            echo "Usage: $0 [--force]"
            echo "  --force, -f    Allow overwriting existing experiment directories"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--force]" >&2
            exit 1
            ;;
    esac
done

RUNS_BASE="data/ablation_runs"
CONFIGS=(
    "config1_baseline"
    "config2_zero"
    "config3_velodyne_thick"
    "config4_velodyne_thin"
)

# Prevent accidental overwrites: refuse if any target directory already exists and is non-empty
for conf in "${CONFIGS[@]}"; do
    target_dir="$RUNS_BASE/$conf"
    if [ -d "$target_dir" ] && [ -n "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        if [ "$OVERWRITE" -ne 1 ]; then
            echo "[Error] Experiment directory '$target_dir' already exists and is not empty." >&2
            echo "Refusing to overwrite existing experiment outputs. Pass --force to overwrite." >&2
            exit 1
        else
            echo "[Warning] Overwriting existing directory '$target_dir' as requested."
            rm -rf "$target_dir"
        fi
    fi
done

echo "====================================="
echo "  STARTING ZERO-SHOT ABLATION STUDY  "
echo "====================================="

# Config 1: Baseline (Raw dense points, standard FJD intensity)
CONF1_DIR="$RUNS_BASE/config1_baseline"
CONF1_PRED="$CONF1_DIR/predictions"
mkdir -p "$CONF1_PRED"
echo -e "\n---> Running Config 1: Baseline (Raw + Intensity)..."
make infer SIM_VELODYNE=0 FEATURE_MODE=intensity OUTPUT_DIR="$CONF1_PRED"
cp "$CONF1_PRED/run_config.json" "$CONF1_DIR/run_config.json"

echo "  Evaluating original 16 classes..."
docker run --rm --user 1003:1003 \
  -v "$(pwd):/workspace/project" \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir "$CONF1_PRED" \
  --output_csv "$CONF1_DIR/eval_16class.csv"

echo "  Evaluating remapped 4 classes..."
docker run --rm --user 1003:1003 \
  -v "$(pwd):/workspace/project" \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir "$CONF1_PRED" \
  --output_csv "$CONF1_DIR/eval_4class_remapped.csv" \
  --remap

# Config 2: Zeroed Intensity Only (Testing feature camouflage)
CONF2_DIR="$RUNS_BASE/config2_zero"
CONF2_PRED="$CONF2_DIR/predictions"
mkdir -p "$CONF2_PRED"
echo -e "\n---> Running Config 2: Zeroed Intensity (Raw + Zero)..."
make infer SIM_VELODYNE=0 FEATURE_MODE=zero OUTPUT_DIR="$CONF2_PRED"
cp "$CONF2_PRED/run_config.json" "$CONF2_DIR/run_config.json"

echo "  Evaluating original 16 classes..."
docker run --rm --user 1003:1003 \
  -v "$(pwd):/workspace/project" \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir "$CONF2_PRED" \
  --output_csv "$CONF2_DIR/eval_16class.csv"

echo "  Evaluating remapped 4 classes..."
docker run --rm --user 1003:1003 \
  -v "$(pwd):/workspace/project" \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir "$CONF2_PRED" \
  --output_csv "$CONF2_DIR/eval_4class_remapped.csv" \
  --remap

# Config 3: Thick Velodyne Rings + Zeroed Intensity
CONF3_DIR="$RUNS_BASE/config3_velodyne_thick"
CONF3_PRED="$CONF3_DIR/predictions"
mkdir -p "$CONF3_PRED"
echo -e "\n---> Running Config 3: Thick Velodyne Rings (Tol: 0.15 + Zero)..."
make infer SIM_VELODYNE=1 VELODYNE_TOL=0.15 FEATURE_MODE=zero OUTPUT_DIR="$CONF3_PRED"
cp "$CONF3_PRED/run_config.json" "$CONF3_DIR/run_config.json"

echo "  Evaluating original 16 classes..."
docker run --rm --user 1003:1003 \
  -v "$(pwd):/workspace/project" \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir "$CONF3_PRED" \
  --output_csv "$CONF3_DIR/eval_16class.csv"

echo "  Evaluating remapped 4 classes..."
docker run --rm --user 1003:1003 \
  -v "$(pwd):/workspace/project" \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir "$CONF3_PRED" \
  --output_csv "$CONF3_DIR/eval_4class_remapped.csv" \
  --remap

# Config 4: Razor Thin Velodyne Rings + Zeroed Intensity
CONF4_DIR="$RUNS_BASE/config4_velodyne_thin"
CONF4_PRED="$CONF4_DIR/predictions"
mkdir -p "$CONF4_PRED"
echo -e "\n---> Running Config 4: Razor Thin Velodyne Rings (Tol: 0.03 + Zero)..."
make infer SIM_VELODYNE=1 VELODYNE_TOL=0.03 FEATURE_MODE=zero OUTPUT_DIR="$CONF4_PRED"
cp "$CONF4_PRED/run_config.json" "$CONF4_DIR/run_config.json"

echo "  Evaluating original 16 classes..."
docker run --rm --user 1003:1003 \
  -v "$(pwd):/workspace/project" \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir "$CONF4_PRED" \
  --output_csv "$CONF4_DIR/eval_16class.csv"

echo "  Evaluating remapped 4 classes..."
docker run --rm --user 1003:1003 \
  -v "$(pwd):/workspace/project" \
  s1-segmentation \
  python scripts/calculate_miou.py \
  --gt_dir data/ground_truth_labeled_tiles \
  --pred_dir "$CONF4_PRED" \
  --output_csv "$CONF4_DIR/eval_4class_remapped.csv" \
  --remap

echo -e "\n====================================="
echo "  ALL EXPERIMENTS COMPLETED!           "
echo "====================================="
echo -e "\nResults saved to $RUNS_BASE/"
echo ""
echo "=== 16-Class Original Evaluation ==="
for conf in "${CONFIGS[@]}"; do
    csv="$RUNS_BASE/$conf/eval_16class.csv"
    if [ -f "$csv" ]; then
        miou=$(tail -1 "$csv" | cut -d',' -f2)
        echo "  $conf: mIoU = $miou%"
    fi
done
echo ""
echo "=== 4-Class Remapped Evaluation ==="
for conf in "${CONFIGS[@]}"; do
    csv="$RUNS_BASE/$conf/eval_4class_remapped.csv"
    if [ -f "$csv" ]; then
        miou=$(tail -1 "$csv" | cut -d',' -f2)
        echo "  $conf: mIoU = $miou%"
    fi
done
