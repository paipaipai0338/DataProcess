from itertools import combinations, permutations
from pathlib import Path

import numpy as np

from Img2Points.utils import get_corner_coordinate, get_corner_pixel_from_img
from RadarProcess.utils import get_corner_data
from Img2Keypoint.utils import get_gt_data


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


def estimate_transform(points_src, points_dst):
    """
    通过SVD求解刚体旋转和平移矩阵。
    points_src: 源坐标系中的N个3D点 (Nx3矩阵)
    points_dst: 目标坐标系中的N个3D点 (Nx3矩阵)
    """
    points_src, _ = as_xyz(points_src, "points_src")
    points_dst, _ = as_xyz(points_dst, "points_dst")
    if points_src.shape != points_dst.shape:
        raise ValueError(f"source/destination shapes must match: {points_src.shape} vs {points_dst.shape}")
    if points_src.shape[0] < 3:
        raise ValueError("at least 3 point pairs are required to estimate a 3D rigid transform")

    centroid_src = np.mean(points_src, axis=0)
    centroid_dst = np.mean(points_dst, axis=0)

    src_centered = points_src - centroid_src
    dst_centered = points_dst - centroid_dst
    if np.linalg.matrix_rank(src_centered) < 2 or np.linalg.matrix_rank(dst_centered) < 2:
        raise ValueError("at least 3 non-collinear point pairs are required")

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_dst - R @ centroid_src

    return R, t


def apply_transform(points_src, R, t):
    points_src, _ = as_xyz(points_src, "points_src")
    return (R @ points_src.T + t.reshape(-1, 1)).T


def pairwise_distances(points_a, points_b):
    return np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)


def seed_distance_signature(points):
    distances = []
    for i, j in combinations(range(len(points)), 2):
        distances.append(np.linalg.norm(points[i] - points[j]))
    return np.sort(np.asarray(distances))


def assign_radar_points(transformed_src, radar_xyz, distance_threshold):
    """
    将每个雷达点分配给最近的重建角点；小于阈值的点才算作有效匹配。
    这样允许多个雷达点落到同一个角反射点附近。
    """
    distances = pairwise_distances(transformed_src, radar_xyz)
    nearest_src = np.argmin(distances, axis=0)
    nearest_dist = distances[nearest_src, np.arange(radar_xyz.shape[0])]

    groups = {}
    for src_idx in range(transformed_src.shape[0]):
        radar_indices = np.where((nearest_src == src_idx) & (nearest_dist <= distance_threshold))[0]
        groups[src_idx] = radar_indices.tolist()

    return groups, nearest_dist


def refine_many_to_one(points_src, radar_xyz, R, t, distance_threshold, min_matched_points, max_iters=5):
    """
    根据当前外参把多个雷达点聚到同一个角点，再用每个角点的雷达簇中心重估外参。
    """
    best_info = None
    for _ in range(max_iters):
        transformed = apply_transform(points_src, R, t)
        groups, _ = assign_radar_points(transformed, radar_xyz, distance_threshold)
        matched_src = [idx for idx, radar_indices in groups.items() if radar_indices]

        if len(matched_src) < min_matched_points:
            return None

        cluster_centers = np.asarray(
            [np.mean(radar_xyz[groups[idx]], axis=0) for idx in matched_src],
            dtype=float,
        )
        src_subset = points_src[matched_src]

        try:
            new_R, new_t = estimate_transform(src_subset, cluster_centers)
        except ValueError:
            return None

        best_info = {
            "R": new_R,
            "t": new_t,
            "groups": groups,
            "matched_src": matched_src,
            "cluster_centers": cluster_centers,
        }

        if np.linalg.norm(new_R - R) < 1e-9 and np.linalg.norm(new_t - t) < 1e-9:
            break
        R, t = new_R, new_t

    if best_info is None:
        return None

    transformed = apply_transform(points_src, best_info["R"], best_info["t"])
    groups, _ = assign_radar_points(transformed, radar_xyz, distance_threshold)
    matched_src = [idx for idx, radar_indices in groups.items() if radar_indices]
    if len(matched_src) < min_matched_points:
        return None

    cluster_centers = np.asarray(
        [np.mean(radar_xyz[groups[idx]], axis=0) for idx in matched_src],
        dtype=float,
    )
    center_errors = np.linalg.norm(transformed[matched_src] - cluster_centers, axis=1)
    inlier_errors = []
    for src_idx, radar_indices in groups.items():
        for radar_idx in radar_indices:
            inlier_errors.append(np.linalg.norm(transformed[src_idx] - radar_xyz[radar_idx]))
    inlier_errors = np.asarray(inlier_errors, dtype=float)

    return {
        "R": best_info["R"],
        "t": best_info["t"],
        "groups": groups,
        "matched_src": matched_src,
        "cluster_centers": cluster_centers,
        "center_errors": center_errors,
        "inlier_errors": inlier_errors,
        "center_rmse": float(np.sqrt(np.mean(center_errors ** 2))),
        "inlier_rmse": float(np.sqrt(np.mean(inlier_errors ** 2))) if inlier_errors.size else float("inf"),
        "inlier_count": int(inlier_errors.size),
    }


def register_unknown_correspondence(
    points_src,
    points_dst,
    distance_threshold=0.5,
    min_matched_points=3,
    max_seed_candidates=300000,
    pair_distance_tolerance=None,
):
    """
    在未知对应关系下估计刚体变换。

    支持：
    1. points_src 和 points_dst 数量不一致；
    2. 多个雷达点对应同一个角反射点；
    3. 雷达点包含额外列，例如 [x, y, z, velocity]。
    """
    points_src, src_original_indices = as_xyz(points_src, "points_src")
    points_dst, dst_original_indices = as_xyz(points_dst, "points_dst")

    if points_src.shape[0] < 3:
        raise ValueError(f"need at least 3 reconstructed corner points, got {points_src.shape[0]}")
    if points_dst.shape[0] < 3:
        raise ValueError(f"need at least 3 radar XYZ points, got {points_dst.shape[0]}")

    min_matched_points = min(min_matched_points, points_src.shape[0], points_dst.shape[0])
    if min_matched_points < 3:
        raise ValueError("min_matched_points must be at least 3")

    seed_size = 3
    if pair_distance_tolerance is None:
        pair_distance_tolerance = max(2.5 * distance_threshold, 1e-6)

    best = None
    best_key = None
    candidates_checked = 0
    candidates_used = 0

    for src_seed in combinations(range(points_src.shape[0]), seed_size):
        src_seed = np.asarray(src_seed, dtype=int)
        src_signature = seed_distance_signature(points_src[src_seed])

        for dst_seed_unordered in combinations(range(points_dst.shape[0]), seed_size):
            dst_seed_unordered = np.asarray(dst_seed_unordered, dtype=int)
            dst_signature = seed_distance_signature(points_dst[dst_seed_unordered])
            if np.any(np.abs(src_signature - dst_signature) > pair_distance_tolerance):
                continue

            for dst_seed in permutations(dst_seed_unordered.tolist()):
                candidates_checked += 1
                if candidates_checked > max_seed_candidates:
                    print(
                        f"Warning: reached max_seed_candidates={max_seed_candidates}; "
                        "increase it if the result looks unstable."
                    )
                    break

                dst_seed = np.asarray(dst_seed, dtype=int)
                try:
                    R, t = estimate_transform(points_src[src_seed], points_dst[dst_seed])
                except ValueError:
                    continue

                seed_errors = np.linalg.norm(apply_transform(points_src[src_seed], R, t) - points_dst[dst_seed], axis=1)
                if np.sqrt(np.mean(seed_errors ** 2)) > distance_threshold:
                    continue

                refined = refine_many_to_one(
                    points_src=points_src,
                    radar_xyz=points_dst,
                    R=R,
                    t=t,
                    distance_threshold=distance_threshold,
                    min_matched_points=min_matched_points,
                )
                if refined is None:
                    continue

                candidates_used += 1
                matched_count = len(refined["matched_src"])
                key = (
                    matched_count,
                    -refined["center_rmse"],
                    -refined["inlier_rmse"],
                    refined["inlier_count"],
                    -float(np.sqrt(np.mean(seed_errors ** 2))),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = refined
                    best["seed_src"] = src_seed
                    best["seed_dst"] = dst_seed
                    best["seed_errors"] = seed_errors

            if candidates_checked > max_seed_candidates:
                break
        if candidates_checked > max_seed_candidates:
            break

    if best is None:
        raise RuntimeError(
            "failed to find a valid radar/corner correspondence. "
            "Try increasing distance_threshold or check whether at least 3 radar detections "
            "really correspond to non-collinear corner reflectors."
        )

    radar_indices_by_source = {
        int(src_original_indices[src_idx]): [int(dst_original_indices[radar_idx]) for radar_idx in radar_indices]
        for src_idx, radar_indices in best["groups"].items()
        if radar_indices
    }
    cluster_centers_by_source = {
        int(src_original_indices[src_idx]): np.mean(points_dst[radar_indices], axis=0)
        for src_idx, radar_indices in best["groups"].items()
        if radar_indices
    }

    result = {
        "radar_indices_by_source": radar_indices_by_source,
        "cluster_centers_by_source": cluster_centers_by_source,
        "matched_source_count": len(best["matched_src"]),
        "inlier_radar_count": best["inlier_count"],
        "center_errors": best["center_errors"],
        "inlier_errors": best["inlier_errors"],
        "center_rmse": best["center_rmse"],
        "inlier_rmse": best["inlier_rmse"],
        "distance_threshold": distance_threshold,
        "candidates_checked": candidates_checked,
        "candidates_used": candidates_used,
        "seed_source_indices": [int(src_original_indices[idx]) for idx in best["seed_src"]],
        "seed_radar_indices": [int(dst_original_indices[idx]) for idx in best["seed_dst"]],
    }

    return best["R"], best["t"], result


def validate_transform(R, t, points_src, points_dst, distance_threshold=0.5):
    """验证多对一匹配下的变换精度。"""
    points_src, _ = as_xyz(points_src, "points_src")
    points_dst, _ = as_xyz(points_dst, "points_dst")

    transformed = apply_transform(points_src, R, t)
    groups, _ = assign_radar_points(transformed, points_dst, distance_threshold)
    matched_src = [idx for idx, radar_indices in groups.items() if radar_indices]

    if not matched_src:
        print("No radar inliers found under the validation threshold.")
        return

    cluster_centers = np.asarray([np.mean(points_dst[groups[idx]], axis=0) for idx in matched_src], dtype=float)
    center_errors = np.linalg.norm(transformed[matched_src] - cluster_centers, axis=1)
    inlier_errors = []
    for src_idx, radar_indices in groups.items():
        for radar_idx in radar_indices:
            inlier_errors.append(np.linalg.norm(transformed[src_idx] - points_dst[radar_idx]))
    inlier_errors = np.asarray(inlier_errors, dtype=float)

    print(f"匹配角点数: {len(matched_src)}/{points_src.shape[0]}")
    print(f"使用雷达点数: {inlier_errors.size}/{points_dst.shape[0]}")
    print(f"簇中心平均误差: {np.mean(center_errors):.6f} m")
    print(f"簇中心最大误差: {np.max(center_errors):.6f} m")
    print(f"簇中心RMSE: {np.sqrt(np.mean(center_errors ** 2)):.6f} m")
    print(f"雷达点到对应角点RMSE: {np.sqrt(np.mean(inlier_errors ** 2)):.6f} m")


def plot_registration(points_src, points_dst, R, t, result):
    import matplotlib.pyplot as plt

    points_src, _ = as_xyz(points_src, "points_src")
    points_dst, _ = as_xyz(points_dst, "points_dst")
    transformed_src = apply_transform(points_src, R, t)

    plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(points_dst[:, 0], points_dst[:, 1], points_dst[:, 2], c="y", s=8, label="Radar detections")
    ax.scatter(
        transformed_src[:, 0],
        transformed_src[:, 1],
        transformed_src[:, 2],
        c="b",
        s=40,
        marker="x",
        label="Reprojected camera corners",
    )

    labeled_cluster_center = False
    for src_idx, center in result["cluster_centers_by_source"].items():
        label = None if labeled_cluster_center else "Radar cluster centers"
        labeled_cluster_center = True
        ax.scatter(center[0], center[1], center[2], c="r", s=35, label=label)
        ax.text(center[0], center[1], center[2], f"corner {src_idx}")

    ax.scatter(0, 0, 0, c="k", s=35, label="Radar origin")
    ax.set_xlabel("Radar X (m)")
    ax.set_ylabel("Radar Y (m)")
    ax.set_zlabel("Radar Z (m)")
    ax.legend()
    plt.show()


def main():
    distance_threshold = 0.2  # meters; increase if radar detections are noisier
    radar_cfar_params = {
        "ref_range": 8,
        "ref_velocity": 2,
        "guard_range": 2,
        "guard_velocity": 1,
        "alpha": 2.0,
        "mode": "ca",
    }
    radar_path = Path(r'E:\20260609_164905\dpct低位机\Bin')

    img_path = Path(r'E:\20260609_164905\camera')
    pkl_save_path = Path(r'E:\20260609_164905\calib\corner_pixels.pkl')
    calib_path = Path(r'E:\20260609_164905\calib')
    R_t_save_path = Path(r'E:\20260609_164905\calib\extrinsic_img_to_radar_low.npz')

    '''人工标注像素点'''
    get_corner_pixel_from_img(img_path, pkl_save_path)

    '''根据像素点重建3D坐标'''
    pixel_coordinate, error = get_corner_coordinate(pkl_save_path, calib_path=calib_path)
    print(f"重建角点坐标:\n{pixel_coordinate}")
    print(f"重建误差: {error}")

    '''获取雷达点云'''
    radar_files = sorted([path for path in radar_path.iterdir() if path.suffix.lower() == ".bin"])
    if not radar_files:
        raise FileNotFoundError(f"No .bin radar files found in {radar_path}")
    print(f"雷达CFAR参数: {radar_cfar_params}")
    targets = get_corner_data(radar_files[-1], **radar_cfar_params)
    pc_radar = targets["cartesian coordinate"]
    print(f"雷达检测点数: {pc_radar.shape[0]}")

    mask = (
        (pc_radar[:, 0] > 0.3)
        & (pc_radar[:, 0] < 5)
        & (np.abs(pc_radar[:, 1]) < 1)
    )

    pc_radar = pc_radar[mask]
    print(f"空间ROI后雷达点数: {pc_radar.shape[0]}")

    '''对齐'''
    R_est, t_est, registration = register_unknown_correspondence(
        pixel_coordinate,
        pc_radar,
        distance_threshold=distance_threshold,
        min_matched_points=3,
    )
    print(f"\n匹配结果: {registration['radar_indices_by_source']}")
    print(f"种子角点索引: {registration['seed_source_indices']}")
    print(f"种子雷达点索引: {registration['seed_radar_indices']}")
    print(
        f"匹配到 {registration['matched_source_count']} 个角点，"
        f"使用 {registration['inlier_radar_count']} 个雷达点"
    )
    print(f"估计的旋转:\n{R_est}")
    print(f"估计的平移: {t_est}")

    validate_transform(R_est, t_est, pixel_coordinate, pc_radar, distance_threshold=distance_threshold)

    plot_registration(pixel_coordinate, pc_radar, R_est, t_est, registration)

    np.savez(R_t_save_path, R_est=R_est, t_est=t_est)

if __name__ == '__main__':
    main()
