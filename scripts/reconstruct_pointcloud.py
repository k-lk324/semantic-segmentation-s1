#!/usr/bin/env python3
"""
Reconstruct full point cloud from predicted tiles.
Handles overlapping regions by voting or averaging predictions.
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import laspy
from collections import defaultdict


def load_metadata(tiles_dir: Path):
    """Load tiling metadata for reconstruction parameters."""
    metadata_path = tiles_dir / "tiling_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def reconstruct_from_tiles(predictions_dir: Path, metadata_dir: Path, 
                           output_path: Path, voting: bool = True):
    """
    Reconstruct full point cloud from tile predictions.
    
    Args:
        predictions_dir: Directory containing predicted tile .pth files
        metadata_dir: Directory containing tiling_metadata.json
        output_path: Output LAS file path
        voting: If True, use majority voting for overlapping regions
    """
    # Load metadata
    metadata = load_metadata(metadata_dir)
    coord_shift = np.array(metadata['coord_shift'])
    
    print(f"Loading predictions from: {predictions_dir}")
    print(f"Coordinate shift: {coord_shift}")
    
    # Collect all prediction files
    pred_files = sorted(predictions_dir.glob("*.pth"))
    if not pred_files:
        raise ValueError(f"No .pth files found in {predictions_dir}")
    
    print(f"Found {len(pred_files)} prediction tiles")
    
    # Dictionary to accumulate predictions for overlapping points
    # Key: (x, y, z) rounded to voxel grid, Value: list of predictions
    point_predictions = defaultdict(list)
    point_coords = {}
    point_features = {}
    
    # Load all tiles and accumulate predictions
    for pred_file in tqdm(pred_files, desc="Loading tiles"):
        data = torch.load(pred_file, weights_only=False)
        
        # Extract data
        coords = data['coord']  # Shape: (N, 3)
        predictions = data['pred']  # Shape: (N,) - predicted class per point
        
        # Optional: extract features if available
        features = data.get('features', None)
        
        # Restore original coordinates
        original_coords = coords + coord_shift
        
        # Round coordinates to handle floating point precision
        # This helps identify the same point across tiles
        for i in range(len(coords)):
            coord_key = tuple(np.round(original_coords[i], decimals=3))
            point_predictions[coord_key].append(predictions[i])
            point_coords[coord_key] = original_coords[i]
            
            if features is not None and coord_key not in point_features:
                point_features[coord_key] = features[i]
    
    print(f"\nTotal unique points: {len(point_predictions)}")
    
    # Resolve overlapping predictions
    final_coords = []
    final_predictions = []
    final_features = []
    
    for coord_key in tqdm(point_predictions.keys(), desc="Resolving overlaps"):
        preds = point_predictions[coord_key]
        
        if voting:
            # Majority voting for overlapping predictions
            final_pred = max(set(preds), key=preds.count)
        else:
            # Average (for soft predictions) or just take first
            final_pred = int(np.mean(preds))
        
        final_coords.append(point_coords[coord_key])
        final_predictions.append(final_pred)
        
        if coord_key in point_features:
            final_features.append(point_features[coord_key])
    
    final_coords = np.array(final_coords)
    final_predictions = np.array(final_predictions, dtype=np.uint8)
    
    print(f"\nWriting output to: {output_path}")
    
    # Create LAS file
    las = laspy.create()
    las.x = final_coords[:, 0]
    las.y = final_coords[:, 1]
    las.z = final_coords[:, 2]
    las.classification = final_predictions
    
    # Add features if available
    if final_features:
        final_features = np.array(final_features)
        if final_features.shape[1] >= 1:
            las.intensity = final_features[:, 0].astype(np.uint16)
        if final_features.shape[1] >= 4:  # RGB available
            las.red = final_features[:, 1].astype(np.uint16)
            las.green = final_features[:, 2].astype(np.uint16)
            las.blue = final_features[:, 3].astype(np.uint16)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(output_path))
    
    print(f"\n✓ Reconstruction complete!")
    print(f"  Output: {output_path}")
    print(f"  Points: {len(final_coords)}")
    print(f"  Classes: {np.unique(final_predictions)}")
    
    # Print class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(final_predictions, return_counts=True)
    for cls, count in zip(unique, counts):
        percentage = 100 * count / len(final_predictions)
        print(f"  Class {cls}: {count:6d} points ({percentage:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct full point cloud from predicted tiles"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Directory containing predicted tile .pth files"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Directory containing tiling_metadata.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output LAS file path"
    )
    parser.add_argument(
        "--voting",
        action="store_true",
        default=True,
        help="Use majority voting for overlapping regions (default: True)"
    )
    
    args = parser.parse_args()
    
    reconstruct_from_tiles(
        Path(args.predictions),
        Path(args.metadata),
        Path(args.output),
        voting=args.voting
    )
