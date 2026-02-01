_base_ = [
    "/workspace/libs/Pointcept/configs/nuscenes/semseg-pt-v3m1-0-base.py"
]

# Model Overrides
model = dict(
    backbone=dict(
        in_channels=4, ),
    # Ensure we use the standard CrossEntropy criteria
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)])

# Data Settings (Placeholder for the builder)
data = dict(
    num_classes=16,
    names=[
        "barrier", "bicycle", "bus", "car", "construction_vehicle",
        "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
        "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
        "vegetation"
    ],
    ignore_index=-1,
)

# Environment
batch_size = 1
num_worker = 2
