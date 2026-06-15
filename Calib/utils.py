import numpy as np


def apply_transform(points_src, R, t):
    points_src, _ = as_xyz(points_src, "points_src")
    return (R @ points_src.T + t.reshape(-1, 1)).T


def as_xyz(points, name):
    """Return finite XYZ columns and their original row indices."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{name} must be an Nx3 or NxM array, got shape {points.shape}")

    xyz = points[:, :3]
    valid_mask = np.isfinite(xyz).all(axis=1)
    if not np.any(valid_mask):
        raise ValueError(f"{name} does not contain finite XYZ points")
    if not np.all(valid_mask):
        print(f"Warning: ignored {np.size(valid_mask) - int(np.sum(valid_mask))} invalid rows in {name}")

    return xyz[valid_mask], np.flatnonzero(valid_mask)