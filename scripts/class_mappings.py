"""
Shared class definitions and mappings for semantic segmentation.

This module contains:
- nuScenes 16-class definitions
- 4-class superclass remapping
- Class name mappings
"""

import numpy as np

# nuScenes semantic segmentation classes (16 classes)
CLASS_NAMES = {
    0: "barrier",
    1: "bicycle",
    2: "bus",
    3: "car",
    4: "construction_vehicle",
    5: "motorcycle",
    6: "pedestrian",
    7: "traffic_cone",
    8: "trailer",
    9: "truck",
    10: "driveable_surface",
    11: "other_flat",
    12: "sidewalk",
    13: "terrain",
    14: "manmade",
    15: "vegetation",
}

# Superclass mapping: 16 nuScenes classes → 4 superclasses
# Index = original class ID, value = superclass ID
# Superclass IDs: 0=vegetation, 1=object, 2=ground, 3=structure
SUPERCLASS_MAPPING = np.array([
    1,  # 0: barrier → object
    1,  # 1: bicycle → object
    1,  # 2: bus → object
    1,  # 3: car → object
    1,  # 4: construction_vehicle → object
    1,  # 5: motorcycle → object
    1,  # 6: pedestrian → object
    1,  # 7: traffic_cone → object
    1,  # 8: trailer → object
    1,  # 9: truck → object
    2,  # 10: driveable_surface → ground
    2,  # 11: other_flat → ground
    2,  # 12: sidewalk → ground
    2,  # 13: terrain → ground
    3,  # 14: manmade → structure
    0,  # 15: vegetation → vegetation
])

SUPERCLASS_NAMES = {
    0: "vegetation",
    1: "object",
    2: "ground",
    3: "structure",
}
