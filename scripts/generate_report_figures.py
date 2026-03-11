import argparse
import csv
import json
from pathlib import Path

# import laspy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import torch
from matplotlib.patches import FancyArrowPatch, Rectangle

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent

CONFIGS = [
    ("Config 1", "config1_baseline", "config1_baseline_remapped"),
    ("Config 2", "config2_zero", "config2_zero_remapped"),
    ("Config 3", "config3_velodyne_thick", "config3_velodyne_thick_remapped"),
    ("Config 4", "config4_velodyne_thin", "config4_velodyne_thin_remapped"),
]

SUPERCLASS_ORDER = ["vegetation", "object", "ground", "structure"]
SUPERCLASS_COLORS = {
    "vegetation": "#2e7d32",
    "object": "#d32f2f",
    "ground": "#8d6e63",
    "structure": "#1565c0",
}


def read_metrics(csv_path: Path):
    rows = []
    mean_iou = None
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            name = row[0].strip()
            if name.lower().startswith("mean iou"):
                mean_iou = float(row[1])
                continue
            rows.append((name, float(row[1])))
    return rows, mean_iou


def read_full_metrics(csv_path: Path):
    rows = []
    mean_iou = None
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            name = row[0].strip()
            if name.lower().startswith("mean iou"):
                mean_iou = float(row[1])
                continue
            iou = float(row[1])
            precision = float(
                row[2]) if len(row) > 2 and row[2].strip() else np.nan
            recall = float(
                row[3]) if len(row) > 3 and row[3].strip() else np.nan
            rows.append({
                "name": name,
                "iou": iou,
                "precision": precision,
                "recall": recall
            })
    return rows, mean_iou


def harmonic_mean(precision: float, recall: float) -> float:
    if np.isnan(precision) or np.isnan(recall) or precision + recall <= 0:
        return np.nan
    return 2.0 * precision * recall / (precision + recall)


def random_subsample(arrays, max_points=300_000, seed=42):
    n = arrays[0].shape[0]
    if n <= max_points:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return [arr[idx] for arr in arrays]


def subsample_indices(n: int, max_points: int, seed: int = 42):
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def make_scene_overview(src_las: Path, out_path: Path):
    las = laspy.read(src_las)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)
    x, y, z = random_subsample([x, y, z], max_points=250_000)

    xq1, xq2 = np.quantile(x, [0.35, 0.65])
    yq1, yq2 = np.quantile(y, [0.35, 0.65])
    zoom_mask = (x >= xq1) & (x <= xq2) & (y >= yq1) & (y <= yq2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=180)
    axes[0].scatter(x, y, c=z, cmap="viridis", s=0.6, alpha=0.85, linewidths=0)
    axes[0].set_title("Full scene footprint")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].axis("equal")

    axes[1].scatter(
        x[zoom_mask],
        y[zoom_mask],
        c=z[zoom_mask],
        cmap="viridis",
        s=1.0,
        alpha=0.9,
        linewidths=0,
    )
    axes[1].set_title("Zoom-in of raw point density")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].axis("equal")

    for ax in axes:
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_tiling_grid(src_las: Path, tiling_meta_path: Path, out_path: Path):
    with tiling_meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    block_size = float(meta.get("block_size", 30.0))
    stride = float(meta.get("stride", 15.0))

    las = laspy.read(src_las)
    x = np.asarray(las.x, dtype=np.float32)
    y = np.asarray(las.y, dtype=np.float32)
    z = np.asarray(las.z, dtype=np.float32)

    x, y, z = random_subsample([x, y, z], max_points=140_000, seed=7)
    min_x, max_x = float(np.min(x)), float(np.max(x))
    min_y, max_y = float(np.min(y)), float(np.max(y))

    grid_x = np.arange(min_x, max_x, stride)
    grid_y = np.arange(min_y, max_y, stride)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    ax.scatter(x, y, c=z, cmap="Greys", s=0.5, alpha=0.2, linewidths=0)

    for gx in grid_x:
        for gy in grid_y:
            rect = Rectangle(
                (gx, gy),
                block_size,
                block_size,
                fill=False,
                edgecolor="#ef6c00",
                linewidth=0.45,
                alpha=0.45,
            )
            ax.add_patch(rect)

    highlighted = [(grid_x[0], grid_y[0]),
                   (grid_x[len(grid_x) // 2], grid_y[len(grid_y) // 2])]
    labels = ["tile 0000", "example mid tile"]
    for (gx, gy), label in zip(highlighted, labels):
        rect = Rectangle((gx, gy),
                         block_size,
                         block_size,
                         fill=False,
                         edgecolor="#c62828",
                         linewidth=1.6)
        ax.add_patch(rect)
        ax.text(gx + 1.0,
                gy + block_size - 1.2,
                label,
                color="#c62828",
                fontsize=8,
                weight="bold")

    ax.set_title(
        f"Tiling grid ({block_size:.0f}m block, {stride:.0f}m stride)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.grid(True, alpha=0.18)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_class_legend(out_path: Path):
    class16 = [
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction vehicle",
        "motorcycle",
        "pedestrian",
        "traffic cone",
        "trailer",
        "truck",
        "driveable surface",
        "other flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    ]
    palette16 = plt.cm.tab20(np.linspace(0.0, 1.0, len(class16)))

    fig = plt.figure(figsize=(15.5, 8.2), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.set_title("16-class output palette",
                  fontsize=16,
                  weight="bold",
                  pad=10)
    ncols = 4
    nrows = 4
    left_margin = 0.03
    top = 0.92
    x_step = 0.24
    y_step = 0.2

    for i, (name, color) in enumerate(zip(class16, palette16)):
        row = i // ncols
        col = i % ncols
        x0 = left_margin + col * x_step
        y0 = top - row * y_step
        ax1.add_patch(
            Rectangle((x0, y0),
                      0.05,
                      0.08,
                      facecolor=color,
                      edgecolor="black",
                      linewidth=0.6))
        ax1.text(x0 + 0.065, y0 + 0.04, name, va="center", fontsize=11)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")

    ax2.set_title("Remapped 4-class palette",
                  fontsize=16,
                  weight="bold",
                  pad=10)
    for i, name in enumerate(SUPERCLASS_ORDER):
        y0 = 0.76 - i * 0.19
        ax2.add_patch(
            Rectangle(
                (0.08, y0),
                0.14,
                0.10,
                facecolor=SUPERCLASS_COLORS[name],
                edgecolor="black",
                linewidth=0.7,
            ))
        ax2.text(0.27,
                 y0 + 0.05,
                 name.capitalize(),
                 va="center",
                 fontsize=13,
                 weight="bold")

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_method_pipeline(out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 2.8), dpi=180)
    ax.axis("off")

    y = 0.5
    boxes = [
        (0.03, "Raw LAS\n(input scene)"),
        (0.28, "Tiling\n30m×30m, 50% overlap"),
        (0.53, "Inference\nPTv3 per tile"),
        (0.78, "Voting +\nReconstruction"),
    ]

    for x0, label in boxes:
        rect = Rectangle((x0, y - 0.17),
                         0.19,
                         0.34,
                         facecolor="#e3f2fd",
                         edgecolor="#1565c0",
                         linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x0 + 0.095, y, label, ha="center", va="center", fontsize=10)

    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + 0.19
        x_end = boxes[i + 1][0]
        arrow = FancyArrowPatch((x_start + 0.01, y), (x_end - 0.01, y),
                                arrowstyle="->",
                                mutation_scale=15,
                                linewidth=1.4)
        ax.add_patch(arrow)

    ax.text(
        0.50,
        0.10,
        "Zero-shot adaptation: feature masking + ring simulation + remapping",
        ha="center",
        fontsize=10)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def remap_16_to_4(pred16: np.ndarray) -> np.ndarray:
    remap = np.full(16, 1, dtype=np.int32)
    remap[15] = 0
    remap[10] = 2
    remap[11] = 2
    remap[12] = 2
    remap[13] = 2
    remap[14] = 3
    return remap[pred16]


def pick_comparison_tile_name(results_ablation_dir: Path):
    preferred = "Fh_parking_outside_2025-09-19-17-08-32_Section_Section_normal_tile_0050.pth"
    config_dirs = [
        results_ablation_dir / "config1_baseline",
        results_ablation_dir / "config2_zero",
        results_ablation_dir / "config3_velodyne_thick",
        results_ablation_dir / "config4_velodyne_thin",
    ]

    if all((d / preferred).exists() for d in config_dirs):
        return preferred

    common = None
    for d in config_dirs:
        names = {p.name for p in d.glob("*.pth")}
        common = names if common is None else (common & names)
    if not common:
        raise RuntimeError(
            "No common tile found across all ablation config folders.")
    return sorted(common)[0]


def plot_semantic(ax, x, y, labels4, title):
    color_arr = np.empty((labels4.shape[0], 3), dtype=np.float32)
    cmap = {
        0: np.array([46, 125, 50]) / 255.0,
        1: np.array([211, 47, 47]) / 255.0,
        2: np.array([141, 110, 99]) / 255.0,
        3: np.array([21, 101, 192]) / 255.0,
    }
    for cls_id, color in cmap.items():
        color_arr[labels4 == cls_id] = color
    ax.scatter(x, y, c=color_arr, s=1.2, alpha=0.9, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def make_qualitative_comparison(processed_tiles_dir: Path,
                                results_ablation_dir: Path, out_path: Path):
    tile_name = pick_comparison_tile_name(results_ablation_dir)

    tile_path = processed_tiles_dir / tile_name
    tile = torch.load(tile_path, map_location="cpu", weights_only=False)
    coord = tile["coord"]
    if torch.is_tensor(coord):
        coord = coord.cpu().numpy()

    idx = subsample_indices(coord.shape[0], max_points=180_000, seed=9)
    x = coord[idx, 0]
    y = coord[idx, 1]
    z = coord[idx, 2]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), dpi=180)
    axes[0].scatter(x, y, c=z, cmap="viridis", s=1.1, alpha=0.9, linewidths=0)
    axes[0].set_title("Raw geometry", fontsize=10)
    axes[0].set_aspect("equal")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    config_dirs = [
        ("Baseline", results_ablation_dir / "config1_baseline"),
        ("Zeroed", results_ablation_dir / "config2_zero"),
        ("Rings 0.15°", results_ablation_dir / "config3_velodyne_thick"),
        ("Rings 0.03°", results_ablation_dir / "config4_velodyne_thin"),
    ]

    for ax, (title, cdir) in zip(axes[1:], config_dirs):
        logits = torch.load(cdir / tile_name,
                            map_location="cpu",
                            weights_only=False)
        pred16 = torch.argmax(logits, dim=1).cpu().numpy()
        pred4 = remap_16_to_4(pred16)
        pred4 = pred4[idx]
        plot_semantic(ax, x, y, pred4, title)

    fig.suptitle(f"Qualitative comparison on {tile_name.replace('.pth', '')}",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_ablation_miou(results_dir: Path, out_path: Path):
    labels = []
    miou16 = []
    miou4 = []
    for label, base_name, remap_name in CONFIGS:
        labels.append(label)
        _, m16 = read_metrics(results_dir / f"{base_name}.csv")
        _, m4 = read_metrics(results_dir / f"{remap_name}.csv")
        miou16.append(m16)
        miou4.append(m4)

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    b1 = ax.bar(x - w / 2,
                miou16,
                width=w,
                label="16-class mIoU",
                color="#90caf9")
    b2 = ax.bar(x + w / 2,
                miou4,
                width=w,
                label="4-class mIoU",
                color="#ffcc80")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Ablation mIoU comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.35,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_classwise_iou(results_dir: Path, out_path: Path):
    per_config = {}
    for label, _, remap_name in CONFIGS:
        rows, _ = read_metrics(results_dir / f"{remap_name}.csv")
        metric_map = {name.strip().lower(): val for name, val in rows}
        per_config[label] = [
            metric_map.get(cls, 0.0) for cls in SUPERCLASS_ORDER
        ]

    x = np.arange(len(SUPERCLASS_ORDER))
    w = 0.19

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=180)
    palette = ["#90caf9", "#a5d6a7", "#ffcc80", "#ce93d8"]
    for i, (label, vals) in enumerate(per_config.items()):
        ax.bar(x + (i - 1.5) * w, vals, width=w, label=label, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SUPERCLASS_ORDER])
    ax.set_ylabel("IoU (%)")
    ax.set_title("Class-wise IoU in remapped 4-class space")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_delta_miou(results_dir: Path, out_path: Path):
    labels = []
    delta16 = []
    delta4 = []

    _, base16 = read_metrics(results_dir / "config1_baseline.csv")
    _, base4 = read_metrics(results_dir / "config1_baseline_remapped.csv")

    for label, base_name, remap_name in CONFIGS:
        labels.append(label)
        _, m16 = read_metrics(results_dir / f"{base_name}.csv")
        _, m4 = read_metrics(results_dir / f"{remap_name}.csv")
        delta16.append(m16 - base16)
        delta4.append(m4 - base4)

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)

    b1 = ax.bar(x - w / 2,
                delta16,
                width=w,
                label="16-class delta",
                color="#64b5f6")
    b2 = ax.bar(x + w / 2,
                delta4,
                width=w,
                label="4-class delta",
                color="#ffb74d")

    ax.axhline(0.0, color="#37474f", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta mIoU (percentage points)")
    ax.set_title("Absolute mIoU gain vs baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.2,
                    f"{h:+.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_best_config_pr_plot(results_dir: Path, out_path: Path):
    best_label = None
    best_csv = None
    best_miou = -1.0

    for label, _, remap_name in CONFIGS:
        csv_path = results_dir / f"{remap_name}.csv"
        _, miou = read_full_metrics(csv_path)
        if miou > best_miou:
            best_miou = miou
            best_label = label
            best_csv = csv_path

    rows, _ = read_full_metrics(best_csv)
    metric_map = {r["name"].strip().lower(): r for r in rows}

    classes = SUPERCLASS_ORDER
    precision = [metric_map[c]["precision"] for c in classes]
    recall = [metric_map[c]["recall"] for c in classes]
    f1 = [harmonic_mean(p, r) for p, r in zip(precision, recall)]

    x = np.arange(len(classes))
    w = 0.24

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)
    ax.bar(x - w, precision, width=w, label="Precision", color="#42a5f5")
    ax.bar(x, recall, width=w, label="Recall", color="#66bb6a")
    ax.bar(x + w, f1, width=w, label="F1", color="#ffa726")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes])
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Best config ({best_label}) per-class precision/recall/F1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_analysis_tables(results_dir: Path, out_tex_path: Path):
    summary = []
    _, base16 = read_metrics(results_dir / "config1_baseline.csv")
    _, base4 = read_metrics(results_dir / "config1_baseline_remapped.csv")

    for label, base_name, remap_name in CONFIGS:
        _, m16 = read_metrics(results_dir / f"{base_name}.csv")
        rows4, m4 = read_full_metrics(results_dir / f"{remap_name}.csv")
        summary.append({
            "label":
            label,
            "m16":
            m16,
            "m4":
            m4,
            "d16":
            m16 - base16,
            "d4":
            m4 - base4,
            "r16": (100.0 * (m16 - base16) / base16) if base16 > 0 else np.nan,
            "r4": (100.0 * (m4 - base4) / base4) if base4 > 0 else np.nan,
            "rows4":
            rows4,
        })

    best = max(summary, key=lambda d: d["m4"])
    best_map = {r["name"].strip().lower(): r for r in best["rows4"]}

    lines = []
    lines.append("% Auto-generated by generate_report_figures.py")
    lines.append("% Include with: \\\\input{figures/analysis_tables.tex}")
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Absolute and relative mIoU gains versus the baseline configuration.}"
    )
    lines.append("\\label{tab:miou-delta}")
    lines.append("\\begin{tabular}{|l|c|c|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Config} & \\textbf{$\\Delta$16 (")
    lines[
        -1] += "pp)} & \\textbf{$\\Delta$4 (pp)} & \\textbf{Rel.16 (\\%)} & \\textbf{Rel.4 (\\%)} \\\\"
    lines.append("\\hline")
    for row in summary:
        lines.append(
            f"{row['label']} & {row['d16']:+.2f} & {row['d4']:+.2f} & {row['r16']:+.1f} & {row['r4']:+.1f} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Per-superclass precision, recall, and F1 for the best 4-class configuration.}"
    )
    lines.append("\\label{tab:best-prf}")
    lines.append("\\begin{tabular}{|l|c|c|c|}")
    lines.append("\\hline")
    lines.append(
        "\\textbf{Superclass} & \\textbf{Precision (\\%)} & \\textbf{Recall (\\%)} & \\textbf{F1 (\\%)} \\\\"
    )
    lines.append("\\hline")
    for cls in SUPERCLASS_ORDER:
        r = best_map[cls]
        f1 = harmonic_mean(r["precision"], r["recall"])
        lines.append(
            f"{cls.capitalize()} & {r['precision']:.2f} & {r['recall']:.2f} & {f1:.2f} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_tex_path.parent.mkdir(parents=True, exist_ok=True)
    out_tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all report figures from project artifacts.")
    parser.add_argument("--root",
                        type=str,
                        default=str(ROOT),
                        help="Project root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # tiling_meta = root / "data" / "processed_tiles" / "tiling_metadata.json"
    # with tiling_meta.open("r", encoding="utf-8") as f:
    #     meta = json.load(f)
    # src_las = root / meta["src"]

    # make_scene_overview(src_las, figures_dir / "scene_overview.png")
    # make_tiling_grid(src_las, tiling_meta, figures_dir / "tiling_grid.png")
    make_class_legend(figures_dir / "class_legend.png")
    # make_method_pipeline(figures_dir / "method_pipeline.png")
    make_ablation_miou(root / "results_ablation",
                       figures_dir / "ablation_miou.png")
    make_classwise_iou(root / "results_ablation",
                       figures_dir / "ablation_classwise_iou.png")
    make_delta_miou(root / "results_ablation",
                    figures_dir / "ablation_miou_delta.png")
    make_best_config_pr_plot(root / "results_ablation",
                             figures_dir / "best_config_prf.png")
    write_analysis_tables(root / "results_ablation",
                          figures_dir / "analysis_tables.tex")
    # make_qualitative_comparison(
    #     root / "data" / "processed_tiles",
    #     root / "results_ablation",
    #     figures_dir / "qualitative_comparison.png",
    # )

    print("Generated figures:")
    for p in sorted(figures_dir.glob("*.png")):
        print(f" - {p.relative_to(root)}")
    analysis_tex = figures_dir / "analysis_tables.tex"
    if analysis_tex.exists():
        print(f" - {analysis_tex.relative_to(root)}")


if __name__ == "__main__":
    main()
