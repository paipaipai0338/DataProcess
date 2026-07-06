from pathlib import Path
import argparse
import re
import sys
from dataclasses import dataclass
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Calib.chessboard_detection_helper import detect_chessboard_fullres, check_corners_order_minimal
from Img2Keypoint.utils import load_extrinsic_data, load_intrinsic_data

ROWS, COLS = 24, 24  # inner corners: rows, cols
SQUARE = 0.04        # meters

def parse_args():
    parser = argparse.ArgumentParser(description="检查指定数据目录的相机外参误差。")
    parser.add_argument(
        "--mode",
        type=str,
        default="check_camera_extrinsic",
        choices=["check_camera_intrinsic", "check_camera_extrinsic"],
        help="检查模式；当前脚本只执行 check_camera_extrinsic。",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\20260701\data_collection\group_005"),
        help="用于外参检查的一组数据目录；可为 group_xxx、group_xxx/camera，或包含多个 group_xxx 的父目录。",
    )
    parser.add_argument(
        "--calib_path",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\20260702\calib"),
        help="标定文件目录，包含 intrinsic_cam_*.npz 和 extrinsic_T_cam_*_to_cam_*.npy。",
    )
    parser.add_argument("--reference_cam", type=str, default="A", help="外参参考相机编号。")
    parser.add_argument(
        "--selected_groups",
        type=str,
        default=None,
        help="当 data_path 是父目录时，可指定要检查的 group，支持 5、group_005 或逗号分隔列表。",
    )
    args = parser.parse_args()

    return args

def natural_group_sort_key(group_name):
    match = re.search(r"(\d+)$", str(group_name))
    if match:
        return int(match.group(1))
    return str(group_name)


def parse_group_selection(selected_groups):
    if selected_groups is None:
        return None

    group_names = set()
    for item in str(selected_groups).split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            group_names.add(f"group_{int(item):03d}")
        else:
            group_names.add(item)
    return group_names


def resolve_group_camera_path(group_path):
    group_path = Path(group_path)
    if any(p.is_dir() and p.name.startswith("cam_") for p in group_path.iterdir()):
        return group_path
    camera_path = group_path / "camera"
    if camera_path.exists() and camera_path.is_dir():
        return camera_path
    return group_path


def has_camera_dirs(path):
    path = Path(path)
    return any(p.is_dir() and p.name.startswith("cam_") for p in path.iterdir())


def is_single_group_path(path):
    path = Path(path)
    return (
        path.name.startswith("group_")
        or has_camera_dirs(path)
        or ((path / "camera").exists() and (path / "camera").is_dir())
    )


def display_group_name(group_path):
    group_path = Path(group_path)
    if group_path.name == "camera" and group_path.parent.name:
        return group_path.parent.name
    return group_path.name


@dataclass
class Ray:
    camera_id: str
    origin: np.ndarray
    direction: np.ndarray
    frame_id: int
    pixel: np.ndarray
    pixel_id: int
    depth: float
    score: float
    person_id: int
    valid: bool


def build_ray_from_pixel(u, v, K, dist, camera_id, pixel_id, score, frame_id, person_id, valid):
    pts_obs = np.array([[[u, v]]], dtype=np.float32)
    pts_corrected = cv2.undistortPoints(pts_obs, K, dist, P=None)
    x_norm = pts_corrected[0, 0, 0]
    y_norm = pts_corrected[0, 0, 1]

    direction = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    return Ray(
        origin=np.zeros(3, dtype=np.float64),
        direction=direction,
        pixel=np.array([u, v], dtype=np.float64),
        camera_id=camera_id,
        pixel_id=int(pixel_id),
        depth=0.0,
        score=float(score),
        frame_id=int(frame_id),
        person_id=int(person_id),
        valid=bool(valid),
    )


def transform_ray_to_reference(ray, calib_path, reference_cam="A"):
    calib_path = Path(calib_path)
    if ray.camera_id == reference_cam:
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
    else:
        _, R, t = load_extrinsic_data(
            calib_path / f"extrinsic_T_cam_{ray.camera_id}_to_cam_{reference_cam}.npy"
        )

    transformed = Ray(**ray.__dict__)
    transformed.origin = R @ ray.origin + np.asarray(t, dtype=np.float64).reshape(3)
    transformed.direction = np.asarray(R, dtype=np.float64) @ ray.direction
    return transformed


def get_chessboard_ray(data_path, calib_path, image_paths_by_cam=None, rows=ROWS, cols=COLS):
    data_path = Path(data_path)
    calib_path = Path(calib_path)
    image_paths_by_cam = image_paths_by_cam or {}

    cam_dirs = [
        p for p in data_path.iterdir()
        if p.is_dir() and p.name.startswith("cam_")
    ]
    cam_ids = sorted([p.name.split("_", 1)[1] for p in cam_dirs])

    rays = []
    pixels_id = set()

    for cam_id in cam_ids:
        if cam_id in image_paths_by_cam:
            img_path = Path(image_paths_by_cam[cam_id])
        else:
            frames_path = data_path / f"cam_{cam_id}" / "frames"
            file_list = sorted([
                p for p in frames_path.iterdir()
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            ]) if frames_path.exists() else []
            if not file_list:
                continue
            img_path = file_list[0]

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        ok, corners, _ = detect_chessboard_fullres(
            gray,
            bgr=img_bgr,
            rows=rows,
            cols=cols,
            enable_color_order=True,
            class_method_flag=True,
        )
        if not ok or corners is None:
            continue

        ok_order, corners, _ = check_corners_order_minimal(
            img_bgr, corners, rows=rows, cols=cols
        )
        if not ok_order:
            continue

        K, dist = load_intrinsic_data(calib_path / f"intrinsic_cam_{cam_id}.npz")
        if K is None or dist is None:
            print(f"[WARN] cam={cam_id}: intrinsic not found or invalid")
            continue

        for idx, corner in enumerate(corners):
            ray = build_ray_from_pixel(
                corner[0, 0],
                corner[0, 1],
                K,
                dist,
                camera_id=cam_id,
                pixel_id=idx,
                score=1.0,
                frame_id=-1,
                person_id=-1,
                valid=True,
            )
            rays.append(ray)
            pixels_id.add(idx)

    return rays, sorted(pixels_id)


def calculate_chessboard_3D_coordinate(pixel_id, frame_id, rays, person_id=-1, reference_cam="A", calib_path=None):
    if calib_path is None:
        raise ValueError("calib_path is required")

    selected_rays = [r for r in rays if r.pixel_id == pixel_id]
    selected_rays = [r for r in selected_rays if r.person_id == person_id]
    selected_rays = [r for r in selected_rays if r.frame_id == frame_id]
    rays_to_cam_ref = [
        transform_ray_to_reference(r, calib_path=calib_path, reference_cam=reference_cam)
        for r in selected_rays
    ]

    I = np.eye(3, dtype=np.float64)
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)

    for ray in rays_to_cam_ref:
        origin = ray.origin
        direction = ray.direction
        P = I - np.outer(direction, direction)
        weight = ray.score if ray.score > 0.5 else 0.0
        A += weight * P
        b += weight * (P @ origin)

    coordinate, *_ = np.linalg.lstsq(A, b, rcond=None)
    for ray in rays_to_cam_ref:
        ray.depth = float(ray.direction @ (coordinate - ray.origin))
    return coordinate, rays_to_cam_ref


def check_camera_extrinsic(
    data_path,
    calib_path,
    reference_cam="A",
    selected_groups=None,
):
    """
    外参验证（group 级）：

    1. data_path 下每个 group 作为一条验证样本
    2. 每个相机在本 group 内扫描自己的图像
    3. 选择“第一张成功检测到棋盘格”的图像作为该相机观测依据
    4. 若该相机所有图像均检测失败，则认为该相机未观测到棋盘格
    5. 只在终端打印每个 group 的 3D 误差与平面残差，不保存任何文件

    打印指标：
        group_name
        used_cameras
        num_used_cameras
        num_valid_points
        mean_3d_error_m
        rmse_3d_error_m
        median_3d_error_m
        max_3d_error_m
        plane_mean_residual_m
        plane_rmse_residual_m
        plane_max_residual_m
        is_valid
        status
    """
    data_path = Path(data_path)
    calib_path = Path(calib_path)

    if not data_path.exists():
        raise FileNotFoundError(f"data_dir not found: {data_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"calib_dir not found: {calib_path}")

    selected_group_names = parse_group_selection(selected_groups)
    if is_single_group_path(data_path):
        groups = [data_path]
    else:
        groups = [
            p for p in data_path.iterdir()
            if p.is_dir() and p.name.startswith("group_")
        ]
    if selected_group_names is not None:
        groups = [
            p for p in groups
            if p.name in selected_group_names or display_group_name(p) in selected_group_names
        ]
    groups = sorted(groups, key=lambda p: natural_group_sort_key(display_group_name(p)))
    if len(groups) == 0:
        print(f"[WARN] no group data found under: {data_path}")
        return []

    # 理论棋盘 3D 点（board 坐标系）
    objp = np.zeros((ROWS * COLS, 3), np.float64)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
    objp *= float(SQUARE)

    # 预加载 reference_cam 坐标系下各相机相对位姿（保留原风格）
    cam_ref_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    forward_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    pos, direction = {}, {}
    cams = ["A", "B", "C", "D", "E", "F", "G", "H"]

    for cam_id in cams:
        if cam_id == reference_cam:
            pos[cam_id], direction[cam_id] = cam_ref_pos, forward_cam
        else:
            extr_path = calib_path / f"extrinsic_T_cam_{cam_id}_to_cam_{reference_cam}.npy"
            if extr_path.exists():
                _, R, t = load_extrinsic_data(extr_path)
                pos[cam_id] = np.asarray(t, dtype=np.float64).reshape(3)
                direction[cam_id] = np.asarray(R, dtype=np.float64) @ forward_cam

    summary_rows = []

    for group_path in groups:
        group_name = display_group_name(group_path)
        camera_path = resolve_group_camera_path(group_path)
        print(f"\n========== check extrinsic group={group_name} ==========")

        selected_image_paths = {}
        detected_cameras = []

        # 扫描本 group 下所有 cam_*
        cam_dirs = sorted([p for p in camera_path.iterdir() if p.is_dir() and p.name.startswith("cam_")])

        if len(cam_dirs) == 0:
            summary_rows.append({
                "group_name": group_name,
                "used_cameras": "",
                "num_used_cameras": 0,
                "selected_image_paths": "",
                "num_valid_points": 0,
                "mean_3d_error_m": "",
                "rmse_3d_error_m": "",
                "median_3d_error_m": "",
                "max_3d_error_m": "",
                "plane_mean_residual_m": "",
                "plane_rmse_residual_m": "",
                "plane_max_residual_m": "",
                "is_valid": 0,
                "status": "no_camera_path",
            })
            print(f"[WARN] group={group_name}: no camera path")
            continue

        # -------- Step 1: 每个相机选“第一张检测成功图像” --------
        for cam_path in cam_dirs:
            cam_name = cam_path.name.replace("cam_", "")
            frames_path = cam_path / "frames"
            if not frames_path.exists():
                continue

            img_paths = sorted([
                p for p in frames_path.iterdir()
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            ])

            selected_path = None

            for img_path in img_paths:
                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue

                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                ok, corners, detect_meta = detect_chessboard_fullres(
                    gray,
                    bgr=bgr,
                    rows=ROWS,
                    cols=COLS,
                    enable_color_order=True,
                    class_method_flag=True,
                )

                if ok and corners is not None and len(corners) == ROWS * COLS:
                    ok_order, _, _ = check_corners_order_minimal(
                        bgr, corners, rows=ROWS, cols=COLS
                    )
                    if not ok_order:
                        continue

                    selected_path = img_path
                    selected_image_paths[cam_name] = img_path
                    detected_cameras.append(cam_name)
                    break

            if selected_path is None:
                print(f"[WARN] group={group_name}, cam={cam_name}: no detectable chessboard image")

        detected_cameras = sorted(set(detected_cameras))

        if len(detected_cameras) < 2:
            summary_rows.append({
                "group_name": group_name,
                "used_cameras": "|".join(detected_cameras),
                "num_used_cameras": len(detected_cameras),
                "selected_image_paths": "|".join([f"{k}:{v}" for k, v in sorted(selected_image_paths.items())]),
                "num_valid_points": 0,
                "mean_3d_error_m": "",
                "rmse_3d_error_m": "",
                "median_3d_error_m": "",
                "max_3d_error_m": "",
                "plane_mean_residual_m": "",
                "plane_rmse_residual_m": "",
                "plane_max_residual_m": "",
                "is_valid": 0,
                "status": "too_few_detected_cameras",
            })
            print(
                f"[WARN] group={group_name}: too few detected cameras "
                f"({len(detected_cameras)})"
            )
            continue

        # -------- Step 2: 进入原有重建逻辑 --------
        # 这里默认 get_chessboard_ray(group_dir) 会基于该 group 下的相机观测进行射线构建
        # 若它内部会使用所有图片，则你后续可能还需要继续把“selected_image_paths”传入它做约束
        try:
            chessboard_ray, pixels_id = get_chessboard_ray(
                camera_path,
                calib_path=calib_path,
                image_paths_by_cam=selected_image_paths,
                rows=ROWS,
                cols=COLS,
            )

        except Exception as e:
            summary_rows.append({
                "group_name": group_name,
                "used_cameras": "|".join(detected_cameras),
                "num_used_cameras": len(detected_cameras),
                "selected_image_paths": "|".join([f"{k}:{v}" for k, v in sorted(selected_image_paths.items())]),
                "num_valid_points": 0,
                "mean_3d_error_m": "",
                "rmse_3d_error_m": "",
                "median_3d_error_m": "",
                "max_3d_error_m": "",
                "plane_mean_residual_m": "",
                "plane_rmse_residual_m": "",
                "plane_max_residual_m": "",
                "is_valid": 0,
                "status": f"get_ray_fail:{e}",
            })
            print(f"[WARN] group={group_name}: get ray failed: {e}")
            continue

        coordinates = {}
        used_camera_set = set()

        for pixel_id in pixels_id:
            try:
                coordinate, rays_to_cam_ref = calculate_chessboard_3D_coordinate(
                    pixel_id=pixel_id,
                    rays=chessboard_ray,
                    frame_id=-1,
                    person_id=-1,
                    reference_cam=reference_cam,
                    calib_path=calib_path,
                )
            except Exception:
                continue

            valid_rays = [r for r in rays_to_cam_ref if getattr(r, "valid", False) and getattr(r, "score", 0) > 0]
            if len(valid_rays) < 2 or not np.all(np.isfinite(coordinate)):
                continue

            # 尝试从 ray 中提取 cam_id / cam_name
            for r in valid_rays:
                cam_id = None
                for attr_name in ["cam_id", "camera_id", "cam_name", "camera_name"]:
                    if hasattr(r, attr_name):
                        cam_id = getattr(r, attr_name)
                        break
                if cam_id is not None:
                    used_camera_set.add(str(cam_id))

            coordinates[pixel_id] = np.asarray(coordinate, dtype=np.float64)

        valid_ids = sorted([pid for pid in coordinates.keys() if 0 <= pid < objp.shape[0]])

        if len(valid_ids) < 4:
            summary_rows.append({
                "group_name": group_name,
                "used_cameras": "|".join(sorted(used_camera_set)) if used_camera_set else "|".join(detected_cameras),
                "num_used_cameras": len(used_camera_set) if used_camera_set else len(detected_cameras),
                "selected_image_paths": "|".join([f"{k}:{v}" for k, v in sorted(selected_image_paths.items())]),
                "num_valid_points": len(valid_ids),
                "mean_3d_error_m": "",
                "rmse_3d_error_m": "",
                "median_3d_error_m": "",
                "max_3d_error_m": "",
                "plane_mean_residual_m": "",
                "plane_rmse_residual_m": "",
                "plane_max_residual_m": "",
                "is_valid": 0,
                "status": "too_few_valid_points",
            })
            print(
                f"[WARN] group={group_name}: too few valid 3D points "
                f"({len(valid_ids)})"
            )
            continue

        # reconstructed 3D points in reference_cam coordinates
        X_rec = np.stack([coordinates[pid] for pid in valid_ids], axis=0).astype(np.float64)
        X_gt = objp[valid_ids].astype(np.float64)

        # rigid alignment: X_gt -> X_rec
        mu_gt = X_gt.mean(axis=0)
        mu_rec = X_rec.mean(axis=0)

        X_gt_centered = X_gt - mu_gt
        X_rec_centered = X_rec - mu_rec

        H = X_gt_centered.T @ X_rec_centered
        U, S, Vt = np.linalg.svd(H)
        R_align = Vt.T @ U.T

        if np.linalg.det(R_align) < 0:
            Vt[-1, :] *= -1
            R_align = Vt.T @ U.T

        t_align = mu_rec - R_align @ mu_gt
        X_gt_aligned = (R_align @ X_gt.T).T + t_align

        point_errors = np.linalg.norm(X_rec - X_gt_aligned, axis=1)

        mean_err = float(np.mean(point_errors))
        rmse_err = float(np.sqrt(np.mean(point_errors ** 2)))
        median_err = float(np.median(point_errors))
        max_err = float(np.max(point_errors))

        Xc = X_rec - X_rec.mean(axis=0)
        _, _, Vt_plane = np.linalg.svd(Xc)
        normal = Vt_plane[-1]
        plane_residuals = np.abs(Xc @ normal)

        plane_mean = float(np.mean(plane_residuals))
        plane_rmse = float(np.sqrt(np.mean(plane_residuals ** 2)))
        plane_max = float(np.max(plane_residuals))

        used_cameras_final = sorted(used_camera_set) if len(used_camera_set) > 0 else detected_cameras

        summary_rows.append({
            "group_name": group_name,
            "used_cameras": "|".join(used_cameras_final),
            "num_used_cameras": len(used_cameras_final),
            "selected_image_paths": "|".join([f"{k}:{v}" for k, v in sorted(selected_image_paths.items())]),
            "num_valid_points": len(valid_ids),
            "mean_3d_error_m": mean_err,
            "rmse_3d_error_m": rmse_err,
            "median_3d_error_m": median_err,
            "max_3d_error_m": max_err,
            "plane_mean_residual_m": plane_mean,
            "plane_rmse_residual_m": plane_rmse,
            "plane_max_residual_m": plane_max,
            "is_valid": 1,
            "status": "ok",
        })

        print(f"[INFO] group={group_name}, used_cameras={used_cameras_final}, num_valid_points={len(valid_ids)}")
        print(f"[INFO] mean={mean_err:.6f} m, rmse={rmse_err:.6f} m, median={median_err:.6f} m, max={max_err:.6f} m")
        print(f"[INFO] plane_mean={plane_mean:.6f} m, plane_rmse={plane_rmse:.6f} m, plane_max={plane_max:.6f} m")

    valid_count = sum(1 for row in summary_rows if row.get("is_valid") == 1)
    print(f"\n[INFO] extrinsic check finished: valid_groups={valid_count}/{len(summary_rows)}")
    return summary_rows

if __name__ == '__main__':
    args = parse_args()
    if args.mode == "check_camera_intrinsic":
        raise NotImplementedError("check_camera_intrinsic is not implemented in this script.")
    elif args.mode == "check_camera_extrinsic":
        check_camera_extrinsic(
            data_path=args.data_path,
            calib_path=args.calib_path,
            reference_cam=args.reference_cam,
            selected_groups=args.selected_groups,
        )
