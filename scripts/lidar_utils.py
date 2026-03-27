import torch


def simulate_velodyne_mask(
    coords,
    tolerance=0.15,
    device="cuda",
    return_elevation=False,
):
    """Simulate a 32-beam Velodyne HDL-32E vertical sampling pattern."""
    velodyne_angles = torch.linspace(10.67, -30.67, 32, device=device)
    centered_coords = coords - coords.mean(dim=0)

    r = torch.norm(centered_coords, dim=1) + 1e-6
    z = centered_coords[:, 2]
    elevation = torch.asin(z / r) * (180.0 / torch.pi)

    angle_diffs = torch.abs(
        elevation.unsqueeze(1) - velodyne_angles.unsqueeze(0))
    min_diffs, _ = torch.min(angle_diffs, dim=1)
    mask = min_diffs < tolerance

    if return_elevation:
        return mask, elevation
    return mask
