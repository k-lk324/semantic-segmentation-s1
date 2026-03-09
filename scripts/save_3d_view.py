import torch
import argparse
import numpy as np
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _normalize(values: np.ndarray) -> np.ndarray:
    lo = np.percentile(values, 1)
    hi = np.percentile(values, 99)
    clipped = np.clip(values, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-6)


def _pick_colors(data: dict, coord: np.ndarray) -> np.ndarray:
    if "color" in data:
        color = data["color"]
        if torch.is_tensor(color):
            color = color.cpu().numpy()
        color = np.asarray(color)
        if color.ndim == 2 and color.shape[1] >= 3 and color.shape[0] == coord.shape[0]:
            color = color[:, :3].astype(np.float32)
            if color.max() > 1.0:
                color = color / 255.0
            return np.clip(color, 0.0, 1.0)

    if "intensity" in data:
        intensity = data["intensity"]
        if torch.is_tensor(intensity):
            intensity = intensity.cpu().numpy()
        intensity = np.asarray(intensity).reshape(-1)
        if intensity.shape[0] == coord.shape[0]:
            intensity_n = _normalize(intensity)
            return plt.cm.inferno(intensity_n)[:, :3]

    z_normalized = _normalize(coord[:, 2])
    return plt.cm.viridis(z_normalized)[:, :3]


def _set_equal_axes(ax, coord: np.ndarray) -> None:
    mins = coord.min(axis=0)
    maxs = coord.max(axis=0)
    spans = maxs - mins
    max_span = spans.max()
    center = (mins + maxs) / 2.0
    half = max_span / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", required=True, help="Path to a single .pth tile")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--max_points", type=int, default=100000, help="Maximum points to render")
    parser.add_argument("--point_size", type=float, default=0.8, help="Scatter marker size")
    parser.add_argument("--bg", choices=["light", "dark"], default="light", help="Background style")
    parser.add_argument("--view", choices=["top", "oblique"], default="top", help="Camera preset")
    args = parser.parse_args()

    print(f"Loading tile: {args.tile}")
    data = torch.load(args.tile, weights_only=False)
    
    # Extract coordinates
    coord = data["coord"]
    if torch.is_tensor(coord):
        coord = coord.cpu().numpy()
    coord = np.asarray(coord)

    if coord.ndim != 2 or coord.shape[1] < 3:
        raise ValueError(f"Expected coord shape [N, 3+], got {coord.shape}")
    coord = coord[:, :3]

    if coord.shape[0] > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(coord.shape[0], size=args.max_points, replace=False)
        coord = coord[idx]
        sampled = {}
        for key in ("color", "intensity"):
            if key in data:
                values = data[key]
                if torch.is_tensor(values):
                    values = values.cpu().numpy()
                values = np.asarray(values)
                if values.shape[0] == idx.shape[0] or values.shape[0] == coord.shape[0]:
                    sampled[key] = values
                elif values.shape[0] >= idx.max() + 1:
                    sampled[key] = values[idx]
        color_source = sampled
    else:
        color_source = data

    print(f"Rendering {coord.shape[0]} points...")
    colors = _pick_colors(color_source, coord)

    if args.bg == "dark":
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        coord[:, 0],
        coord[:, 1],
        coord[:, 2],
        c=colors,
        s=args.point_size,
        marker='.',
        alpha=0.95,
        edgecolors='none'
    )

    if args.view == "top":
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=35, azim=45)

    _set_equal_axes(ax, coord)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(os.path.basename(args.tile))
    ax.grid(False)

    # --- Define Output Path and Save ---
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.tile))[0]
        output_path = f"{base_name}_3d.png"
    else:
        output_path = args.output

    print(f"Saving 3D scene to image: {output_path}...")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print("Done!")

if __name__ == "__main__":
    main()