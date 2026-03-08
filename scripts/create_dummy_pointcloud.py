#!/usr/bin/env python3
"""
Create a dummy LAS point cloud file for testing.
"""

import laspy
import numpy as np
from pathlib import Path
import argparse


def create_dummy_pointcloud(output_path, num_points=10000, seed=42):
    """
    Create a dummy LAS file with random point cloud data.
    
    Args:
        output_path: Path where to save the LAS file
        num_points: Number of points to generate
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Create a new LAS file with a point format that supports RGB (format 2)
    las = laspy.create(point_format=2)
    
    # Generate random points within a reasonable range (e.g., 100x100x50 meters)
    x = np.random.uniform(0, 100, num_points)
    y = np.random.uniform(0, 100, num_points)
    z = np.random.uniform(0, 50, num_points)
    
    # Set coordinates
    las.x = x
    las.y = y
    las.z = z
    
    # Add intensity (0-65535 range for 16-bit)
    las.intensity = np.random.randint(0, 65535, num_points, dtype=np.uint16)
    
    # Add RGB values if the point format supports it
    las.red = np.random.randint(0, 65535, num_points, dtype=np.uint16)
    las.green = np.random.randint(0, 65535, num_points, dtype=np.uint16)
    las.blue = np.random.randint(0, 65535, num_points, dtype=np.uint16)
    
    # Add classification (simulating some semantic classes)
    # Classes: 0=unclassified, 1=ground, 2=low vegetation, etc.
    las.classification = np.random.randint(0, 16, num_points, dtype=np.uint8)
    
    # Save the file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(output_path))
    
    print(f"Created dummy point cloud: {output_path}")
    print(f"  Points: {num_points}")
    print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Z range: [{z.min():.2f}, {z.max():.2f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dummy LAS point cloud file")
    parser.add_argument(
        "--output",
        default="data/dummy_pointcloud.las",
        help="Output path for the LAS file (default: data/dummy_pointcloud.las)"
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=10000,
        help="Number of points to generate (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    create_dummy_pointcloud(args.output, args.num_points, args.seed)
