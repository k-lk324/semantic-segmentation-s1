import laspy
import numpy as np
import pandas as pd
from typing import Tuple

VOXEL_SIZE = 0.05  # 5cm


def normalize_color(color: np.ndarray) -> np.ndarray:
    """
    Normalizes color to [-1, 1]. Detects 8-bit vs 16-bit automatically.
    """
    if color.size == 0:
        return color.astype(np.float32)

    max_val = color.max()
    if max_val > 255:
        # 16-bit case
        return (color.astype(np.float32) / 65535.0 * 2) - 1
    # 8-bit case
    return (color.astype(np.float32) / 255.0 * 2) - 1


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    """
    Normalizes intensity to [-1, 1].
    """
    if intensity.size == 0:
        return intensity
    intensity = intensity.astype(np.float32)
    max_val = np.percentile(intensity, 99)
    min_val = np.min(intensity)

    div = max_val - min_val
    if div == 0:
        div = 1.0

    norm = (intensity - min_val) / div
    norm = np.clip(norm, 0, 1)
    return (norm * 2) - 1


def extract_features(las: laspy.LasData, point_count: int) -> np.ndarray:
    """
    Extracts and normalizes features (Intensity + RGB) from the LAS object.
    Returns:
        np.ndarray: Feature matrix of shape (N, F), where:
            - F = 1 when only intensity is available -> [Intensity]
            - F = 4 when intensity and RGB are available -> [Intensity, R, G, B]
    """
    feats_list = []

    if hasattr(las, 'intensity'):
        intensity = np.array(las.intensity).reshape(-1, 1)
        feats_list.append(normalize_intensity(intensity))
    else:
        # Fallback: Use zeros if intensity is missing
        print("  [Warning] No intensity found. Using placeholders.")
        feats_list.append(np.zeros((point_count, 1), dtype=np.float32))

    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        print("  [Info] Found RGB Color. Including in features.")
        r = np.array(las.red).reshape(-1, 1)
        g = np.array(las.green).reshape(-1, 1)
        b = np.array(las.blue).reshape(-1, 1)
        rgb = np.hstack([r, g, b])
        feats_list.append(normalize_color(rgb))
    else:
        print("  [Info] No RGB Color found. Skipping color features.")

    return np.hstack(feats_list).astype(np.float32)


def voxel_grid_subsampling(
    coords: np.ndarray,
    features: np.ndarray,
    orig_indices: np.ndarray,
    voxel_size: float = VOXEL_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies a voxel grid filter.
    - Computes barycenters (mean coords + features) per voxel
    - Preserves ONE representative original index per voxel
    """

    print(
        f"  [Processing] Applying {voxel_size}m voxel filter to {len(coords)} points..."
    )

    num_feat_cols = features.shape[1]
    feat_col_names = [f'feat_{i}' for i in range(num_feat_cols)]

    df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    df_feats = pd.DataFrame(features, columns=feat_col_names)

    df['orig_idx'] = orig_indices
    df = pd.concat([df, df_feats], axis=1)

    # Voxel coordinates
    df['vx'] = np.floor(df['x'] / voxel_size).astype(np.int64)
    df['vy'] = np.floor(df['y'] / voxel_size).astype(np.int64)
    df['vz'] = np.floor(df['z'] / voxel_size).astype(np.int64)

    grouped = (df.groupby(['vx', 'vy', 'vz'], as_index=False).agg({
        'x': 'mean',
        'y': 'mean',
        'z': 'mean',
        **{
            col: 'mean'
            for col in feat_col_names
        },
        'orig_idx': 'first',
    }))

    new_coords = grouped[['x', 'y', 'z']].values.astype(np.float32)
    new_feats = grouped[feat_col_names].values.astype(np.float32)
    new_indices = grouped['orig_idx'].values.astype(np.int64)

    print(f"  [Processing] Reduced to {len(new_coords)} points "
          f"({(1 - len(new_coords) / len(coords)) * 100:.1f}% reduction).")

    return new_coords, new_feats, new_indices
