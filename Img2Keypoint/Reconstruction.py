import os
import copy
import pickle
from pathlib import Path
import open3d as o3d
import cv2
import numpy as np
from dataclasses import dataclass
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse

from Calib.chessboard_detection_helper import detect_chessboard_fullres, check_corners_order_minimal
from Img2Keypoint.utils import load_extrinsic_data, load_intrinsic_data, COCO17_SKELETON, get_matched_pairs

def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "1", "y"):
        return True
    if str(v).lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=Path, default=Path(r"C:\Users\Administrator\Desktop\20260702\data_collection\group_036"), help="dataset root")
    parser.add_argument('--calib_path', type=Path, default=Path(r'C:\Users\Administrator\Desktop\20260702\calib'), help="calibration directory, default: root_path/calib")
    parser.add_argument('--reference_cam', type=str, default="A", help="reference camera id")
    parser.add_argument('--result_2D_data_path', type=Path, default=None, help="2D result directory, default: root_path/camera results/2D")
    parser.add_argument('--result_3D_data_path', type=Path, default=None, help="3D output directory, default: root_path/camera results/3D")

    parser.add_argument('--matching_mode', type=str, default="ref_guided", choices=["exhaustive", "ref_guided"], help="multi-camera matching mode")
    parser.add_argument('--pairwise_top_k', type=int, default=1, help="pairwise candidates kept for each camera pair")
    parser.add_argument('--min_support_cams', type=int, default=3, help="minimum supporting cameras for a person candidate")
    parser.add_argument('--max_missing_cams', type=int, default=4, help="maximum missing cameras allowed while building candidates")
    parser.add_argument('--pairwise_none_error_m', type=float, default=0.10, help="ray error threshold for adding empty pairwise candidates")
    parser.add_argument('--pairwise_candidate_error_m', type=float, default=0.20, help="ray error threshold for pairwise candidates")
    parser.add_argument('--max_candidate_ray_error_m', type=float, default=0.10, help="maximum average ray error for final candidates")
    parser.add_argument('--strict_two_cam_ray_error_m', type=float, default=0.05, help="strict ray error threshold for two-camera candidates")
    parser.add_argument('--min_candidate_valid_joints', type=int, default=5, help="minimum valid reconstructed joints for final candidates")
    parser.add_argument('--support_aware_sort', type=str2bool, default=True, help="sort candidates by camera support before ray error")
    parser.add_argument('--use_2d_scores', type=str2bool, default=True, help="use 2D keypoint scores during matching and reconstruction")
    parser.add_argument('--verbose_every', type=int, default=100, help="progress print interval")

    parser.add_argument('--num_workers', type=int, default=8, help="number of worker processes")
    parser.add_argument('--parallel_chunksize', type=int, default=8, help="chunksize for process-pool map")
    parser.add_argument('--show_final_vis', type=str2bool, default=False, help="show final 3D visualization after saving")
    parser.add_argument('--skip_existing', type=str2bool, default=False, help="skip frames whose 3D pkl already exists")
    args = parser.parse_args()
    if args.root_path is None and any(
        p is None
        for p in (
            args.calib_path,
            args.result_2D_data_path,
            args.result_3D_data_path,
        )
    ):
        parser.error("Specify --root_path, or provide all input/output paths.")
    args.calib_path = args.calib_path or args.root_path / "calib"
    args.result_2D_data_path = args.result_2D_data_path or args.root_path / "camera results" / "2D"
    args.result_3D_data_path = args.result_3D_data_path or args.root_path / "camera results" / "3D"
    return args


''' ray utils '''
@dataclass
class Ray:
    """
    射线模型： 一个 cam 的一个像素点对应一个 Ray
    """
    '''Ray的时空属性'''
    # Ray 对应的camera
    camera_id: str  # 相机 ID
    # Ray所处的相机坐标系原点 世界坐标系前用np.array([0, 0, 0], dtype=np.float64)初始化
    origin: np.ndarray      # [3,]
    # Ray 的射线在相机坐标系下的朝向
    direction: np.ndarray   # [3,]
    # 帧 ID
    frame_id: int
    # Ray对应的像素坐标 [u, v]
    pixel: np.ndarray       # [2,]
    # cam 下像素点索引 chessboard 对应焦点顺序，2D keypoint 对应关键点索引
    pixel_id: int

    '''Ray的估计结果属性'''
    # 多相机联立求解深度
    depth: float
    # 2D 估计结果中的置信度
    score: float
    # 人 ID 这里由于 2D 结果具有整体性，可作为后续关键点匹配判据
    person_id: int
    # Ray 的有效性，由于会出现遮挡等情况，保证关键点的集合空间不变性
    valid: bool

def build_ray_from_pixel(u, v, K, dist, camera_id, pixel_id, score, frame_id, person_id, valid):
    """
    从像素中构建 Ray
    :param u: 像素坐标 (u,v)
    :param v: 像素坐标 (u,v)
    :param K: 相机内参
    :param dist: 相机内参
    :param camera_id: Ray 归属
    :param pixel_id: Ray 归属
    :param score: 估计结果的可靠性，来源于视觉模型
    :param frame_id: Ray 归属
    :param person_id: Ray 归属
    :param valid: Ray 的可靠性
    :return: Ray dataclass
    """
    # 像素点 OpenCV 需要 Nx1x2 格式
    pts_obs = np.array([[[u, v]]], dtype=np.float32)  # shape (1,1,2)

    # 去畸变 (Undistort Points)
    pts_corrected = cv2.undistortPoints(pts_obs, K, dist, P=None)
    x_norm = pts_corrected[0, 0, 0]
    y_norm = pts_corrected[0, 0, 1]

    # 构建相机系下的方向向量 (未单位化，Z=1)
    dir_cam = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
    # 单位化方向向量
    dir_cam_unit = dir_cam / np.linalg.norm(dir_cam)

    return Ray(
        origin=np.array([0, 0, 0], dtype=np.float64),
        direction=dir_cam_unit,
        pixel=np.array([u, v], dtype=np.float64),
        camera_id=camera_id,
        pixel_id=pixel_id,
        depth=0.0,
        score=score,
        frame_id=frame_id,
        person_id=person_id,
        valid=valid,
    )

def transform_ray_to_reference(ray, calib_path, reference_cam="A"):
    """
    将传入的单个 Ray 中的空间属性对齐到 reference_cam，即完成坐标系转换
    :param ray: dataclass
    :param reference_cam: cam id str
    :return: ray dataclass
    """
    calib_path = Path(calib_path)
    if ray.camera_id == reference_cam:
        R = np.eye(3)
        t = np.zeros(3, dtype=float)
    else:
        _, R, t = load_extrinsic_data(
            calib_path / f"extrinsic_T_cam_{ray.camera_id}_to_cam_{reference_cam}.npy"
        )
    r = copy.copy(ray)
    r.origin = R @ ray.origin + t
    r.direction = R @ ray.direction
    return r

def load_extrinsic_cache(cams, reference_cam, calib_path):
    """
    Cache camera-to-reference extrinsic for one frame/matching pass.

    Returns:
        dict[cam_id] = (R_cam_to_ref, t_cam_to_ref)
    """
    calib_path = Path(calib_path)
    extrinsic_cache = {}
    for cam_id in sorted(set(cams)):
        if cam_id == reference_cam:
            extrinsic_cache[cam_id] = (
                np.eye(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        else:
            _, R, t = load_extrinsic_data(
                calib_path / f"extrinsic_T_cam_{cam_id}_to_cam_{reference_cam}.npy"
            )
            extrinsic_cache[cam_id] = (
                np.asarray(R, dtype=np.float64),
                np.asarray(t, dtype=np.float64).reshape(3),
            )
    return extrinsic_cache

def transform_ray_to_reference_cached(ray, extrinsic_cache):
    R, t = extrinsic_cache[ray.camera_id]
    r = copy.copy(ray)
    r.origin = R @ ray.origin + t
    r.direction = R @ ray.direction
    return r

def load_intrinsic_cache(cams, calib_path):
    """
    Cache camera intrinsic once before the frame loop.

    Returns:
        dict[cam_id] = (K, dist)
    """
    calib_path = Path(calib_path)
    intrinsic_cache = {}
    for cam_id in sorted(set(cams)):
        K, dist = load_intrinsic_data(calib_path / f"intrinsic_cam_{cam_id}.npz")
        intrinsic_cache[cam_id] = (
            np.asarray(K, dtype=np.float64),
            np.asarray(dist, dtype=np.float64),
        )
    return intrinsic_cache

def build_ray_from_direction(u, v, direction, camera_id, pixel_id, score, frame_id, person_id, valid):
    """
    Fast Ray constructor used after batch undistortPoints().
    direction must already be a normalized camera-frame direction.
    """
    return Ray(
        origin=np.array([0, 0, 0], dtype=np.float64),
        direction=np.asarray(direction, dtype=np.float64).reshape(3),
        pixel=np.array([u, v], dtype=np.float64),
        camera_id=camera_id,
        pixel_id=pixel_id,
        depth=0.0,
        score=float(score),
        frame_id=frame_id,
        person_id=int(person_id),
        valid=bool(valid),
    )

def make_invalid_person_rays(camera_id, frame_id, num_joints=17):
    """
    Keep the old data-shape convention when a camera has no detected person.
    These rays are invalid and will be filtered before matching.
    """
    rays = []
    dummy_path = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for joint_idx in range(num_joints):
        rays.append(
            build_ray_from_direction(
                -1, -1, dummy_path,
                camera_id=camera_id,
                pixel_id=joint_idx,
                score=0.0,
                frame_id=frame_id,
                person_id=-1,
                valid=False,
            )
        )
    return rays

def get_keypoint_ray_fast(
    result_2D_data_path,
    group,
    frame_id,
    intrinsic_cache=None,
    calib_path=None,
    use_2d_scores=False,
):
    """
    Faster replacement for get_keypoint_ray():
    1. intrinsic are cached outside the frame loop;
    2. all keypoints of one camera are undistorted in one cv2.undistortPoints call;
    3. Ray objects are built from normalized directions directly.

    By default use_2d_scores=False to preserve the current behavior where keypoint score is forced to 1.
    """
    result_2D_data_path = Path(result_2D_data_path)
    rays = []

    for cam_id, img_path in group.items():
        if img_path is None:
            rays.extend(make_invalid_person_rays(cam_id, frame_id))
            continue

        img_name = Path(str(img_path)).stem
        result_2D_data_path_cam = result_2D_data_path / f"cam_{cam_id}" / f"{img_name}.npz"

        if not result_2D_data_path_cam.exists():
            rays.extend(make_invalid_person_rays(cam_id, frame_id))
            continue

        data = np.load(str(result_2D_data_path_cam), allow_pickle=True)
        kpts = data.get('kpts')
        scores = data.get('scores')

        if intrinsic_cache is not None and cam_id in intrinsic_cache:
            K, dist = intrinsic_cache[cam_id]
        else:
            if calib_path is None:
                raise ValueError("calib_path is required when intrinsic_cache is not provided")
            K, dist = load_intrinsic_data(Path(calib_path) / f"intrinsic_cam_{cam_id}.npz")

        if kpts is None or scores is None or kpts.shape[0] == 0 or scores.shape[0] == 0:
            rays.extend(make_invalid_person_rays(cam_id, frame_id))
            continue

        kpts = np.asarray(kpts, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        num_person, num_joints = kpts.shape[:2]

        pts = kpts.reshape(-1, 1, 2).astype(np.float32)
        undist = cv2.undistortPoints(pts, K, dist, P=None).reshape(num_person, num_joints, 2)

        dirs = np.concatenate(
            [undist, np.ones((num_person, num_joints, 1), dtype=np.float64)],
            axis=2,
        ).astype(np.float64)
        dirs /= (np.linalg.norm(dirs, axis=2, keepdims=True) + 1e-12)

        for person_id in range(num_person):
            for joint_idx in range(num_joints):
                score = float(scores[person_id, joint_idx]) if use_2d_scores else 1.0
                ray = build_ray_from_direction(
                    kpts[person_id, joint_idx, 0],
                    kpts[person_id, joint_idx, 1],
                    dirs[person_id, joint_idx],
                    camera_id=cam_id,
                    pixel_id=joint_idx,
                    score=score,
                    frame_id=frame_id,
                    person_id=person_id,
                    valid=True,
                )
                rays.append(ray)

    return rays

'''从图像数据/检测结果中获取 ray'''
def get_chessboard_ray(
    data_path,
    calib_path=None,
    image_paths_by_cam=None,
    rows=24,
    cols=24,
    frame_index=1,
):
    data_path = Path(data_path)
    if calib_path is None:
        raise ValueError("calib_path is required")
    calib_path = Path(calib_path)
    image_paths_by_cam = image_paths_by_cam or {}

    cams = [
        p for p in data_path.iterdir()
        if p.is_dir() and p.name.startswith("cam_")
    ]
    cams_id = sorted([cam.name.split("_", 1)[1] for cam in cams])
    rays = []
    pixels_id = []
    for cam_id in cams_id:
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
            img_path = file_list[min(frame_index, len(file_list) - 1)]

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gray is None:
            continue

        ok, corners, detect_meta = detect_chessboard_fullres(
            gray,
            bgr=img_bgr,
            rows=rows,
            cols=cols,
            enable_color_order=True,
        )
        if not ok:
            continue

        ok_order, corners, color_marks = check_corners_order_minimal(
            img_bgr, corners, rows=rows, cols=cols
        )
        if not ok_order:
            continue
        K, dist = load_intrinsic_data(calib_path / f"intrinsic_cam_{cam_id}.npz")
        for idx, corner in enumerate(corners):
            ray = build_ray_from_pixel(corner[0, 0], corner[0, 1], K, dist, camera_id=cam_id, pixel_id=idx, score=1, frame_id=-1, person_id=-1, valid=True)
            rays.append(ray)
            pixels_id.append(idx)
        pixels_id = list(set(pixels_id))
    return rays, pixels_id

def get_keypoint_ray(result_2D_data_path, group, frame_id, calib_path):
    """
    从保存的 2D 结果中构建 ray
    :param result_2D_data_path: 2D 结果的保存路径
    :param group: 同步后的一组数据
    :param frame_id: Ray 的时间属性
    :return: rays List[ray]
    """
    # 利用 get_matched_pairs 获取匹配的多个图像 pair ，即为时间同步
    # pair中可能存在 img_path 为 None 、或未检测出人的情况，所有失败情况传出的下列数据中 num of person 为 0
    # 所有情况均传出 masks (W,H)、kpts(num of person, 17, 2)、scores(num of person, 17)
    # 无论如何都需要构建 ray 以保证每帧的数据不丢失
    # 失败情况参考 if kpts.shape[0] == 0 or scores.shape[0] == 0
    # 多人情况依据 person_id 区分，注意这里可能出现同一个人的 kpt 在不同的 frame_id, camera_id 下不同，因此为保证正确性应当在同一 frame_id 下 对不同的 camera_id 进行假设匹配
    result_2D_data_path = Path(result_2D_data_path)
    calib_path = Path(calib_path)
    rays = []
    for cam_id, img_path in group.items():
        img_name = img_path.split("\\")[-1].split(".")[0]
        result_2D_data_path_cam = result_2D_data_path / f"cam_{cam_id}/{img_name}.npz"
        data = np.load(result_2D_data_path_cam)
        masks = data.get('masks')
        kpts = data.get('kpts')
        scores = data.get('scores')
        K, dist = load_intrinsic_data(calib_path / f"intrinsic_cam_{cam_id}.npz")
        if kpts.shape[0] == 0 or scores.shape[0] == 0:
            # 未检测到人时为保证完整性传入ray 但valid设为False
            for joint_idx in range(17):
                ray = build_ray_from_pixel(-1, -1, K, dist, camera_id=cam_id, pixel_id=joint_idx, score=0, person_id=-1, frame_id=frame_id, valid=False)
                rays.append(ray)
            continue
        # 为每个检测到的人构建射线模型
        for person_id in range(kpts.shape[0]):
            for joint_idx in range(17):
                # ray = build_ray_from_pixel(kpts[person_id, joint_idx, 0], kpts[person_id, joint_idx, 1], K, dist, camera_id=cam_id, pixel_id=joint_idx, score=scores[person_id, joint_idx], person_id=person_id, frame_id=frame_id, valid=True)
                ray = build_ray_from_pixel(kpts[person_id, joint_idx, 0], kpts[person_id, joint_idx, 1], K, dist,
                                           camera_id=cam_id, pixel_id=joint_idx, score=1,
                                           person_id=person_id, frame_id=frame_id, valid=True)
                rays.append(ray)
    return rays

'''获取相对位姿'''
def build_camera_relative_position(cams=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], reference_cam="A", calib_path=None):
    """
    基于外参矩阵求解相机的相对位姿，以 reference_cam 基准，reference_cam位置为 [0, 0, 0], 朝向为 [0, 0, 1]
    :param cams: List[str] 相机名称
    :param reference_cam: str 参考相机
    :return:
        pos 相对位置: dict[cam_id: str] = np.array (3,)
        dir 相对朝向: dict[cam_id: str] = np.array (3,)
    """

    # 根据外参矩阵与预设 reference_cam 位置、朝向计算相对位置关系
    if calib_path is None:
        raise ValueError("calib_path is required")
    calib_path = Path(calib_path)

    cam_ref_pos = np.array([0, 0, 0], dtype=np.float64)         # reference_cam 位置 (3,)
    forward_cam = np.array([0, 0, 1], dtype=np.float64)         # reference_cam 朝向 (3,)

    pos, dir = {}, {}
    for cam_id in cams:
        if cam_id == reference_cam:
            pos[cam_id], dir[cam_id] = cam_ref_pos, forward_cam
        else:
            _, R, t = load_extrinsic_data(calib_path / f"extrinsic_T_cam_{cam_id}_to_cam_{reference_cam}.npy")
            pos[cam_id] = t
            dir[cam_id] = R @ forward_cam
    return pos, dir

'''可视化 chessboard'''
def build_sence(pos, dir, box_size=(0.20, 0.12, 0.08), axis_size=0.5, rays=None, ray_color=(1.0, 0.5, 0.2), pixels_id=None, reference_cam="A", calib_path=None):
    """
    可视化 棋盘格数据 相机位置与焦点位置
    """
    if calib_path is None:
        raise ValueError("calib_path is required")
    calib_path = Path(calib_path)

    # +Z    forward
    # +X    right
    # +Y    down
    # --- 向量归一化：避免零向量除零 ---
    def norm(v, eps=1e-12):
        v = np.asarray(v, dtype=np.float64).reshape(3)
        n = np.linalg.norm(v)
        return v * 0.0 if n < eps else v / n

    # --- 相机盒子尺寸（沿 forward/right/up 的长宽高），以及半尺寸 ---
    L, W, H = box_size
    hx, hy, hz = L / 2, W / 2, H / 2

    # --- 在相机局部坐标系中定义盒子 8 个顶点 (x=right, y=down, z=forward) ---
    corners_local = np.array([
        [-hx, -hy, -hz],
        [-hx, -hy, +hz],
        [-hx, +hy, -hz],
        [-hx, +hy, +hz],
        [+hx, -hy, -hz],
        [+hx, -hy, +hz],
        [+hx, +hy, -hz],
        [+hx, +hy, +hz],
    ], dtype=np.float64)

    # --- 盒子 12 条边（顶点索引对） ---
    edges = np.array([
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    # --- 场景几何体列表：先添加世界坐标系轴 ---
    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axis_size), origin=[0, 0, 0])]
    cams_id = sorted(pos.keys())
    # --- 逐相机绘制：相机盒子 ---
    for cam_id in cams_id:
        c = np.asarray(pos[cam_id], dtype=np.float64).reshape(3)
        if cam_id == reference_cam:
            R = np.eye(3)
        else:
            _, R, _ = load_extrinsic_data(calib_path / f"extrinsic_T_cam_{cam_id}_to_cam_{reference_cam}.npy")
        corners_world  = (R @ corners_local.T).T + c

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners_world)
        ls.lines = o3d.utility.Vector2iVector(edges)

        col = np.array([0.2, 1.0, 0.4] if cam_id == reference_cam else [0.2, 0.6, 1.0], dtype=np.float64)
        ls.colors = o3d.utility.Vector3dVector(np.tile(col, (len(edges), 1)))
        geoms.append(ls)

    if rays is not None:
        coordinates = {}
        for pixel_id in pixels_id:
            coordinate, rays_to_cam_ref = calculate_chessboard_3D_coordinate(
                pixel_id=pixel_id,
                rays=rays,
                frame_id=-1,
                person_id=-1,
                reference_cam=reference_cam,
                calib_path=calib_path,
            )

            # 没有足够观测，跳过
            valid_rays = [r for r in rays_to_cam_ref if r.valid and r.score > 0]
            if len(valid_rays) < 2 or not np.all(np.isfinite(coordinate)):
                continue

            coordinates[pixel_id] = coordinate

            # 只取真正参与这个点求解/可视化的相机中心
            ray_centers = np.stack([pos[r.camera_id] for r in valid_rays], axis=0)
            d_world = np.stack([norm(r.direction) for r in valid_rays], axis=0)
            depth = np.array([r.depth for r in valid_rays], dtype=np.float64)

            # 深度非法也跳过
            valid_mask = np.isfinite(depth) & (depth > 0)
            if np.sum(valid_mask) < 2:
                continue

            ray_centers = ray_centers[valid_mask]
            d_world = d_world[valid_mask]
            depth = depth[valid_mask]

            ends = ray_centers + d_world * depth[:, None]
            pts = np.vstack([ray_centers, ends])
            n = len(ray_centers)
            lines = np.array([[i, i + n] for i in range(n)], dtype=np.int32)

            ray_ls = o3d.geometry.LineSet()
            ray_ls.points = o3d.utility.Vector3dVector(pts)
            ray_ls.lines = o3d.utility.Vector2iVector(lines)
            ray_ls.colors = o3d.utility.Vector3dVector(
                np.tile(np.asarray(ray_color, np.float64), (len(lines), 1))
            )
            geoms.append(ray_ls)

            ray_ls = o3d.geometry.LineSet()
            ray_ls.points = o3d.utility.Vector3dVector(pts)
            ray_ls.lines = o3d.utility.Vector2iVector(lines)
            ray_ls.colors = o3d.utility.Vector3dVector(
                np.tile(np.asarray(ray_color, np.float64), (len(lines), 1))
            )
            geoms.append(ray_ls)

        point = np.array([coordinates[k] for k in sorted(coordinates.keys())])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        color = [0, 0, 0]  # 红色
        pcd.colors = o3d.utility.Vector3dVector([color] * len(point))

        geoms.append(pcd)
    # --- Open3D 可视化窗口 ---
    o3d.visualization.draw_geometries(geoms)

'''根据ray 计算 3D 点'''
def calculate_chessboard_3D_coordinate(pixel_id, frame_id, rays, person_id=-1, reference_cam="A", calib_path=None):
    """
    求解棋盘格数据的 3D 坐标，确保相机的像素 id 在多个 cam 下是对应关系，否则求解失败
    :param pixel_id: 相机的像素id
    :param frame_id: 对应帧数
    :param person_id: 棋盘格数据无此信息，但为保证集合不变性传入-1
    :param rays: 所有 ray 集合 List[Ray]
    :param reference_cam: cam id str
    :return:
        coordinate: 求解结果
        rays_to_cam_ref: 将所有 Ray 的空间信息转换至 reference_cam坐标系下
    """
    rays_seleted = [r for r in rays if r.pixel_id == pixel_id]
    rays_seleted = [r for r in rays_seleted if r.person_id == person_id]
    rays_seleted = [r for r in rays_seleted if r.frame_id == frame_id]
    if calib_path is None:
        raise ValueError("calib_path is required")
    rays_to_cam_ref = [
        transform_ray_to_reference(r, calib_path=calib_path, reference_cam=reference_cam)
        for r in rays_seleted
    ]

    I = np.eye(3)
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)

    for p in rays_to_cam_ref:
        o = p.origin
        d = p.direction
        P = I - np.outer(d, d)
        wi = p.score if p.score > 0.5 else 0
        A += wi * P
        b += wi * (P @ o)

    coordinate, *_ = np.linalg.lstsq(A, b, rcond=None)
    for p in rays_to_cam_ref:
        p.depth = float(p.direction @ (coordinate - p.origin))
    return coordinate, rays_to_cam_ref

def compute_chessboard_reprojection_debug(
    chessboard_img_data_path,
    rays,
    pixels_id,
    reference_cam="A",
    calib_path=None,
    cams=("A", "B", "C", "D", "E", "F", "G", "H"),
    rows=24,
    cols=24,
):
    """
    计算棋盘格重投影误差

    Args:
        chessboard_img_data_path: 棋盘格图像目录
        rays: get_chessboard_ray 返回的 rays
        pixels_id: get_chessboard_ray 返回的像素角点索引
        reference_cam: 参考相机
        cams: 相机顺序
        rows, cols: 棋盘格角点行列数

    Returns:
        reproj_debug: dict
    """
    if calib_path is None:
        raise ValueError("calib_path is required")
    calib_path = Path(calib_path)

    chessboard_img_data_path = Path(chessboard_img_data_path)

    # 先重新读取每个相机实际检测到的角点，作为 observed 2D
    obs_by_cam = {}
    for cam in cams:
        cam_path = chessboard_img_data_path / f"cam_{cam}" / "frames"
        if not cam_path.exists():
            continue

        file_list = os.listdir(cam_path)
        file_list.sort()
        if len(file_list) == 0:
            continue

        img_path = cam_path / file_list[-1]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        ok, corners, _ = detect_chessboard_fullres(gray, bgr=img_bgr, rows=rows, cols=cols, enable_color_order=True)
        if not ok:
            continue

        ok_order, corners, _ = check_corners_order_minimal(
            img_bgr, corners, rows=rows, cols=cols
        )
        if not ok_order:
            continue

        obs_by_cam[cam] = {
            "img_path": img_path,
            "corners": corners.reshape(-1, 2).astype(np.float64),   # (N,2)
        }

    reproj_debug = {
        "reference_cam": reference_cam,
        "pixels_id": list(sorted(pixels_id)),
        "per_cam": {},
        "coordinates_3d": {},   # pixel_id -> (3,)
    }

    # 先求每个角点的3D
    for pixel_id in sorted(pixels_id):
        coordinate_3d, _ = calculate_chessboard_3D_coordinate(
            pixel_id=pixel_id,
            rays=rays,
            frame_id=-1,
            person_id=-1,
            reference_cam=reference_cam,
            calib_path=calib_path,
        )
        reproj_debug["coordinates_3d"][pixel_id] = coordinate_3d

    # 每个相机计算重投影误差
    all_errors_global = []

    for cam in cams:
        if cam not in obs_by_cam:
            continue

        observed_corners = obs_by_cam[cam]["corners"]
        joint_errors = np.full((len(pixels_id),), np.nan, dtype=float)
        projected_kpts = np.full((len(pixels_id), 2), np.nan, dtype=float)
        observed_kpts = np.full((len(pixels_id), 2), np.nan, dtype=float)
        point_status = ["missing_obs"] * len(pixels_id)

        # 取 reference_cam -> 当前 cam 的变换
        if cam == reference_cam:
            R_ref_to_cam = np.eye(3, dtype=float)
            t_ref_to_cam = np.zeros(3, dtype=float)
        else:
            _, R_cam_to_ref, t_cam_to_ref = load_extrinsic_data(
                calib_path / f"extrinsic_T_cam_{cam}_to_cam_{reference_cam}.npy"
            )
            # X_ref = R_cam_to_ref @ X_cam + t_cam_to_ref
            # => X_cam = R_cam_to_ref.T @ (X_ref - t_cam_to_ref)
            R_ref_to_cam = R_cam_to_ref.T
            t_ref_to_cam = -R_cam_to_ref.T @ t_cam_to_ref

        K, dist = load_intrinsic_data(calib_path / f"intrinsic_cam_{cam}.npz")

        valid_errors_this_cam = []

        for local_idx, pixel_id in enumerate(sorted(pixels_id)):
            if pixel_id >= observed_corners.shape[0]:
                point_status[local_idx] = "missing_obs"
                continue

            pt3d_ref = reproj_debug["coordinates_3d"][pixel_id]
            observed_uv = observed_corners[pixel_id]

            observed_kpts[local_idx] = observed_uv

            if not np.all(np.isfinite(pt3d_ref)):
                point_status[local_idx] = "missing_3d"
                continue

            pt3d_cam = R_ref_to_cam @ pt3d_ref + t_ref_to_cam

            if pt3d_cam[2] <= 1e-8:
                point_status[local_idx] = "behind_camera"
                continue

            obj_pts = np.array(pt3d_cam, dtype=np.float64).reshape(1, 1, 3)
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)

            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            uv = proj.reshape(2)

            if not np.all(np.isfinite(uv)):
                point_status[local_idx] = "invalid_projection"
                continue

            projected_kpts[local_idx] = uv
            err = float(np.linalg.norm(uv - observed_uv))
            joint_errors[local_idx] = err
            point_status[local_idx] = "valid"

            valid_errors_this_cam.append(err)
            all_errors_global.append(err)

        reproj_debug["per_cam"][cam] = {
            "img_path": obs_by_cam[cam]["img_path"],
            "mean_error": float(np.mean(valid_errors_this_cam)) if len(valid_errors_this_cam) > 0 else np.nan,
            "valid_point_count": len(valid_errors_this_cam),
            "point_errors": joint_errors,
            "point_status": point_status,
            "projected_kpts": projected_kpts,
            "observed_kpts": observed_kpts,
        }

    reproj_debug["global_mean_error"] = float(np.mean(all_errors_global)) if len(all_errors_global) > 0 else np.nan
    reproj_debug["global_valid_point_count"] = len(all_errors_global)

    return reproj_debug

def visualize_chessboard_reprojection_debug(
    reproj_debug,
    cams=("A", "B", "C", "D", "E", "F", "G", "H"),
    error_vis_threshold=5.0,
    draw_all_points=False,
):
    """
    按4x4布局显示棋盘格重投影debug结果
    左图：角点检测结果
    右图：重投影结果 + 误差
    """
    plt.figure(figsize=(18, 18))

    for i, cam in enumerate(cams):
        if cam not in reproj_debug["per_cam"]:
            continue

        cam_debug = reproj_debug["per_cam"][cam]
        img_path = cam_debug["img_path"]
        img = plt.imread(img_path)

        observed_kpts = cam_debug["observed_kpts"]
        projected_kpts = cam_debug["projected_kpts"]
        point_errors = cam_debug["point_errors"]
        point_status = cam_debug["point_status"]
        mean_error = cam_debug["mean_error"]
        valid_point_count = cam_debug["valid_point_count"]

        # ---------- 重投影结果 + 误差 ----------
        ax = plt.subplot(4, 2, i + 1)
        ax.imshow(img)

        valid_obs = []
        for point_idx in range(len(point_status)):
            if point_status[point_idx] != "valid":
                continue

            x_obs, y_obs = observed_kpts[point_idx]
            x_proj, y_proj = projected_kpts[point_idx]
            err = point_errors[point_idx]

            # 右图可选择全部画，或者只画误差较大的点
            if (not draw_all_points) and np.isfinite(err) and err < error_vis_threshold:
                continue

            # 观测角点：绿色圆点
            ax.scatter(x_obs, y_obs, c='lime', s=10, marker='o')
            # 重投影角点：红色叉号
            ax.scatter(x_proj, y_proj, c='red', s=16, marker='x')
            # 误差连线：黄色
            ax.plot([x_obs, x_proj], [y_obs, y_proj], color='yellow', linewidth=0.8, alpha=0.8)

            if np.isfinite(err) and err > error_vis_threshold:
                ax.text(
                    x_proj + 2, y_proj + 2,
                    f"{err:.1f}",
                    color='red',
                    fontsize=7
                )

            valid_obs.append([x_obs, y_obs])

        summary_lines = [
            f"Camera {cam} Reprojection",
            f"err={mean_error:.2f}px n={valid_point_count}" if np.isfinite(mean_error) else "err=nan n=0"
        ]

        if len(valid_obs) > 0:
            valid_obs = np.array(valid_obs)
            center_x, center_y = np.mean(valid_obs, axis=0)
            err_text = f"{mean_error:.2f}px" if np.isfinite(mean_error) else "nan"
            ax.text(
                center_x, center_y,
                f"Chessboard\n{err_text}",
                color='yellow',
                fontsize=10,
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=2)
            )

        ax.set_title("\n".join(summary_lines), fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def calculate_human_3D_coordinate(selected_rays_to_ref, num_joints=17, score_threshold=0.2):
    """
    求解人体的 3D 坐标
    :param selected_rays_to_ref: 通过算法筛选出的 rays List[Ray], 确保是对应关系
    :param num_joints:
    :param score_threshold: 2D估计置信度最小可用阈值
    """
    # 初始化结果
    kpts3d = np.full((num_joints, 3), np.nan, dtype=float)
    joint_errors = np.full((num_joints,), np.inf, dtype=float)
    used_rays_by_joint = {j: [] for j in range(num_joints)}
    per_cam_joint_errors = {}
    # 求解人体坐标时 pixel_id 表示关键点索引
    for pixel_id in range(num_joints):
        # 根据关键点结果以及ray的有效性选取有效 ray
        valid_rays = [
            r for r in selected_rays_to_ref
            if r.pixel_id == pixel_id and r.valid and r.score > score_threshold
        ]
        used_rays_by_joint[pixel_id] = list(valid_rays)

        # 只有单个 ray 无法求解
        if len(valid_rays) < 2:
            continue

        I = np.eye(3)
        # ---------- 第一次：用全部 ray 求初值 ----------
        A = np.zeros((3, 3), dtype=float)
        b = np.zeros(3, dtype=float)

        for p in valid_rays:
            o = p.origin
            d = p.direction / (np.linalg.norm(p.direction) + 1e-12)
            P = I - np.outer(d, d)
            w = max(float(p.score), 1e-6)
            A += w * P
            b += w * (P @ o)

        if np.linalg.matrix_rank(A) < 2:
            continue

        coordinate_pixel, *_ = np.linalg.lstsq(A, b, rcond=None)

        # ---------- 计算每条 ray 对初值的残差 ----------
        ray_dists = []
        for p in valid_rays:
            o = p.origin
            d = p.direction / (np.linalg.norm(p.direction) + 1e-12)
            dist = float(np.linalg.norm(np.cross(coordinate_pixel - o, d)))
            ray_dists.append(dist)

        # ---------- 鲁棒筛选：如果最大残差明显偏大，就剔掉 1 条 ----------
        filtered_rays = list(valid_rays)
        if len(valid_rays) >= 3:
            dists_np = np.asarray(ray_dists, dtype=float)
            worst_idx = int(np.argmax(dists_np))
            worst_dist = float(dists_np[worst_idx])

            other_dists = np.delete(dists_np, worst_idx)
            other_mean = float(np.mean(other_dists)) if len(other_dists) > 0 else 0.0

            # 条件：最大残差绝对值够大，且明显大于其它 ray
            if worst_dist > 0.05 and worst_dist > other_mean * 2.5:
                filtered_rays.pop(worst_idx)

        # 把最终参与求解的 ray 记录下来，UI stats 会直接显示这里
        used_rays_by_joint[pixel_id] = list(filtered_rays)

        if len(filtered_rays) < 2:
            continue

        # ---------- 第二次：用筛选后的 ray 重新求解 ----------
        A = np.zeros((3, 3), dtype=float)
        b = np.zeros(3, dtype=float)

        for p in filtered_rays:
            o = p.origin
            d = p.direction / (np.linalg.norm(p.direction) + 1e-12)
            P = I - np.outer(d, d)
            w = max(float(p.score), 1e-6)
            A += w * P
            b += w * (P @ o)

        if np.linalg.matrix_rank(A) < 2:
            continue

        coordinate_pixel, *_ = np.linalg.lstsq(A, b, rcond=None)

        errors_pixel = []
        valid_depth_count = 0
        for p in filtered_rays:
            o = p.origin
            d = p.direction / (np.linalg.norm(p.direction) + 1e-12)
            depth = float(d @ (coordinate_pixel - o))
            dist = float(np.linalg.norm(np.cross(coordinate_pixel - o, d)))
            if depth > 0:
                p.depth = depth
                errors_pixel.append(dist)

                if p.camera_id not in per_cam_joint_errors:
                    per_cam_joint_errors[p.camera_id] = {}
                if pixel_id not in per_cam_joint_errors[p.camera_id]:
                    per_cam_joint_errors[p.camera_id][pixel_id] = []
                per_cam_joint_errors[p.camera_id][pixel_id].append(dist)

                valid_depth_count += 1

        if valid_depth_count < 2:
            continue

        kpts3d[pixel_id] = coordinate_pixel
        joint_errors[pixel_id] = float(np.mean(errors_pixel)) if errors_pixel else np.inf

    valid_joint_mask = np.isfinite(joint_errors)
    total_error = float(np.mean(joint_errors[valid_joint_mask])) if np.any(valid_joint_mask) else np.inf

    per_cam_error = {}
    for cam_id, joint_dict in per_cam_joint_errors.items():
        vals = []
        for _, dist_list in joint_dict.items():
            vals.extend(dist_list)
        per_cam_error[cam_id] = float(np.mean(vals)) if len(vals) > 0 else np.inf

    return kpts3d, total_error, joint_errors, per_cam_error


'''人体匹配'''
def get_3D_keypoints_from_rays(rays, frame_id, reference_cam="A", calib_path=None):
    # 输出 coordinates coordinates[person_id] = 3D_keypoints (17, 3)
    # 考虑到多人匹配的问题，先统计该 frame_id 下，所有有效数据中 camera 出现的 person_id 个数，再取至少 3/4 相机都达到的人数上界作为场景候选人数（保守估计）
    # 利用笛卡尔积穷举所有可能匹配  all_combinations (person_i_in_camera_0, person_j_in_camera_1,...)
    # 计算所有可能匹配对应的结果 append 对应位置，得到 error, candidate_coordinates, candidate_combinations
    # 直接选取误差最小的前 num_person_estimated 存在问题：cam0,1,3 检测到了3个人而 cam2 只检测到了1个那么可能会出现, 第1小误差组合 (0, 1, 0, 2) 第2小误差组合 (0, 1, None, 2) 是一组数据
    # 解决方案为：从 ID 层面加入唯一性约束，避免“同一个人被重复使用” 最终按照“误差优先 + ID唯一性约束”的方式，依次选出不重复的最优组合

    coordinates = {}

    # 按照有效性以及帧id筛选rays
    selected_rays = [r for r in rays if r.frame_id == frame_id and r.valid]
    if len(selected_rays) == 0:
        coordinates[frame_id] = np.empty((0, 17, 3), dtype=float)
        return coordinates, [], []
    # 获取当前cam id
    cams = sorted(set(r.camera_id for r in selected_rays))
    # 统计cam下对应的人数
    person = {camera_id: len(set([r.person_id for r in selected_rays if r.camera_id == camera_id])) for camera_id in cams}
    num_person_estimated = 0
    if len(list(person.values())) > 0:
        # 保守估计场景人数
        threshold_ratio = 1 / 4

        for num in range(max(list(person.values())), 0, -1):
            count = 0
            for camera_id in cams:
                if person[camera_id] >= num:
                    count += 1
            if count >= threshold_ratio * len(cams):
                num_person_estimated = num
                break
    #     # 激进估计
    #     num_person_estimated = max(list(person.values()))
    #     num_person_estimated = 4
    if num_person_estimated == 0:
        coordinates[frame_id] = np.empty((0, 17, 3), dtype=float)
        return coordinates, [], []
    # 枚举所有情况（裁剪版）
    candidate_person_id_across_cams = []
    for camera_id in cams:
        if person[camera_id] < num_person_estimated:
            # 当相机出现的人数少于估计人数则加入None
            candidate_person_id_across_cams.append([i for i in range(person[camera_id])] + [None])
        else:
            # 当相机出现人数大于等于则认为不会出现丢失的情况
            candidate_person_id_across_cams.append([i for i in range(person[camera_id])])
    all_combinations = product(*candidate_person_id_across_cams)
    error = []
    candidate_coordinates = []
    candidate_combinations = []
    candidate_per_cam_errors = []
    for combination in all_combinations:
        candidate_combinations.append(combination)
        candidate_rays = []
        for idx, camera_id in enumerate(cams):
            person_id = combination[idx]
            candidate_rays += [r for r in selected_rays if r.person_id == person_id and r.camera_id == camera_id]
        candidate_rays_to_cam_ref = [
            transform_ray_to_reference(r, calib_path=calib_path, reference_cam=reference_cam)
            for r in candidate_rays
        ]
        kpts3D, total_error, joint_errors, per_cam_error = calculate_human_3D_coordinate(candidate_rays_to_cam_ref)

        candidate_coordinates.append(kpts3D)
        error.append(total_error)
        candidate_per_cam_errors.append(per_cam_error)
    # 选择候选策略
    used_pairs = set()  # (camera_id, person_id)
    sorted_idx = np.argsort(np.array(error))
    selected_kpts3D = []
    selected_combinations = []
    selected_per_cam_errors = []
    for idx in sorted_idx:
        kpts3D = candidate_coordinates[idx]
        combination = candidate_combinations[idx]

        # 1. 检查是否有重复使用 person
        valid = True
        temp_pairs = []
        for cam_idx, person_id in enumerate(combination):
            camera_id = cams[cam_idx]

            if person_id is None:
                continue

            pair = (camera_id, person_id)

            if pair in used_pairs:
                valid = False
                break

            temp_pairs.append(pair)

        if not valid:
            continue

        # 2. 检查是否有效3D
        valid_mask = ~np.isnan(kpts3D).any(axis=1)
        if not np.any(valid_mask):
            continue

        # ✅ 通过：加入结果
        selected_kpts3D.append(kpts3D)
        selected_combinations.append(combination)
        selected_per_cam_errors.append(candidate_per_cam_errors[idx])

        # 记录使用情况
        for pair in temp_pairs:
            used_pairs.add(pair)

        if len(selected_kpts3D) >= num_person_estimated:
            break

    if len(selected_kpts3D) > 0:
        coordinates[frame_id] = np.stack(selected_kpts3D)
    else:
        coordinates[frame_id] = np.empty((0, 17, 3), dtype=float)
    return coordinates, selected_per_cam_errors, selected_combinations

def build_ray_index(rays_to_ref):
    """
    Build a fast lookup:
        ray_index[camera_id][person_id][joint_id] = Ray
    If duplicate rays exist for the same joint, keep the one with higher score.
    """
    ray_index = {}
    for ray in rays_to_ref:
        if ray.person_id is None or ray.person_id < 0:
            continue
        cam_index = ray_index.setdefault(ray.camera_id, {})
        person_index = cam_index.setdefault(ray.person_id, {})
        old_ray = person_index.get(ray.pixel_id)
        if old_ray is None or ray.score > old_ray.score:
            person_index[ray.pixel_id] = ray
    return ray_index

def get_person_ids_by_camera(ray_index, cams):
    return {
        cam_id: sorted(ray_index.get(cam_id, {}).keys())
        for cam_id in cams
    }

def collect_person_rays(ray_index, camera_id, person_id):
    if person_id is None:
        return []
    person_rays = ray_index.get(camera_id, {}).get(person_id, {})
    return [person_rays[joint_id] for joint_id in sorted(person_rays.keys())]

def collect_combination_rays(ray_index, cams, combination):
    candidate_rays = []
    for camera_id, person_id in zip(cams, combination):
        if person_id is None:
            continue
        candidate_rays.extend(collect_person_rays(ray_index, camera_id, person_id))
    return candidate_rays

def build_numpy_ray_table(ray_index):
    """
    Compact per-person ray arrays for pairwise matching.
    Keeps Ray objects for final triangulation, but avoids Python object loops in pairwise scoring.
    """
    table = {}
    for cam_id, persons in ray_index.items():
        table[cam_id] = {}
        for person_id, joints in persons.items():
            origins = np.full((17, 3), np.nan, dtype=np.float64)
            directions = np.full((17, 3), np.nan, dtype=np.float64)
            scores = np.zeros((17,), dtype=np.float64)
            valid = np.zeros((17,), dtype=bool)

            for joint_id, ray in joints.items():
                if 0 <= int(joint_id) < 17:
                    origins[joint_id] = ray.origin
                    directions[joint_id] = ray.direction
                    scores[joint_id] = float(ray.score)
                    valid[joint_id] = bool(ray.valid)

            table[cam_id][person_id] = {
                "origins": origins,
                "directions": directions,
                "scores": scores,
                "valid": valid,
            }
    return table

def compute_pairwise_ray_distance_error_np(
    ray_table,
    ref_cam,
    ref_person_id,
    target_cam,
    target_person_id,
    score_threshold=0.2,
    min_valid_joints=3,
    robust="median",
):
    ref_person = ray_table.get(ref_cam, {}).get(ref_person_id)
    target_person = ray_table.get(target_cam, {}).get(target_person_id)
    if ref_person is None or target_person is None:
        return np.inf

    valid = (
        ref_person["valid"]
        & target_person["valid"]
        & (ref_person["scores"] > score_threshold)
        & (target_person["scores"] > score_threshold)
    )
    if int(np.sum(valid)) < int(min_valid_joints):
        return np.inf

    o1 = ref_person["origins"][valid]
    d1 = ref_person["directions"][valid]
    o2 = target_person["origins"][valid]
    d2 = target_person["directions"][valid]

    d1 = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-12)
    d2 = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-12)
    n = np.cross(d1, d2)
    n_norm = np.linalg.norm(n, axis=1)
    w = o2 - o1

    dists = np.empty((o1.shape[0],), dtype=np.float64)
    parallel = n_norm < 1e-12
    if np.any(parallel):
        dists[parallel] = np.linalg.norm(np.cross(w[parallel], d1[parallel]), axis=1)
    if np.any(~parallel):
        dists[~parallel] = np.abs(np.sum(w[~parallel] * n[~parallel], axis=1)) / n_norm[~parallel]

    dists = dists[np.isfinite(dists)]
    if len(dists) < int(min_valid_joints):
        return np.inf

    if robust == "mean":
        return float(np.mean(dists))
    return float(np.median(dists))

def iter_pruned_combinations(candidate_ids_by_cam, output_cams, active_person_cams, min_support_cams, max_missing_cams):
    """
    DFS generator with support/missing pruning. This avoids materializing the full product.
    """
    output_cams = list(output_cams)
    active_person_cams = set(active_person_cams)
    min_support_cams = int(min_support_cams)
    max_missing_cams = int(max_missing_cams)
    n = len(output_cams)
    can_support_suffix = np.zeros((n + 1,), dtype=np.int32)
    for i in range(n - 1, -1, -1):
        cam_id = output_cams[i]
        can_support = any(person_id is not None for person_id in candidate_ids_by_cam[cam_id])
        can_support_suffix[i] = can_support_suffix[i + 1] + (1 if can_support else 0)

    def dfs(idx, current, support_count, missing_count):
        if missing_count > max_missing_cams:
            return
        if support_count + int(can_support_suffix[idx]) < min_support_cams:
            return
        if idx >= n:
            yield tuple(current)
            return

        cam_id = output_cams[idx]
        for person_id in candidate_ids_by_cam[cam_id]:
            next_support = support_count + (0 if person_id is None else 1)
            next_missing = missing_count + (1 if cam_id in active_person_cams and person_id is None else 0)
            current.append(person_id)
            yield from dfs(idx + 1, current, next_support, next_missing)
            current.pop()

    yield from dfs(0, [], 0, 0)

def estimate_num_person_from_counts(person_counts, cams):
    valid_counts = [person_counts.get(cam_id, 0) for cam_id in cams]
    if len(valid_counts) == 0 or max(valid_counts) <= 0:
        return 0

    threshold_ratio = 1 / 4
    for num in range(max(valid_counts), 0, -1):
        count = sum(1 for cam_id in cams if person_counts.get(cam_id, 0) >= num)
        if count >= threshold_ratio * len(cams):
            return num
    return 0

def estimate_exhaustive_combination_count(person_counts, cams, num_person_estimated):
    if len(cams) == 0:
        return 0
    count = 1
    for cam_id in cams:
        num_person = person_counts.get(cam_id, 0)
        if num_person_estimated > 0 and num_person < num_person_estimated:
            count *= (num_person + 1)
        else:
            count *= max(num_person, 1)
    return count

def ray_ray_distance(o1, d1, o2, d2, eps=1e-12):
    """
    Distance between two 3D lines defined by o + t d.
    Used only as a lightweight pairwise matching cost.
    """
    o1 = np.asarray(o1, dtype=np.float64).reshape(3)
    d1 = np.asarray(d1, dtype=np.float64).reshape(3)
    o2 = np.asarray(o2, dtype=np.float64).reshape(3)
    d2 = np.asarray(d2, dtype=np.float64).reshape(3)

    d1 = d1 / (np.linalg.norm(d1) + eps)
    d2 = d2 / (np.linalg.norm(d2) + eps)

    n = np.cross(d1, d2)
    n_norm = np.linalg.norm(n)
    w = o2 - o1

    if n_norm < eps:
        # Nearly parallel lines: distance from o2 to line 1.
        return float(np.linalg.norm(np.cross(w, d1)))

    return float(abs(np.dot(w, n)) / n_norm)

def compute_pairwise_ray_distance_error(
    ray_index,
    ref_cam,
    ref_person_id,
    target_cam,
    target_person_id,
    score_threshold=0.2,
    min_valid_joints=3,
    robust="median",
):
    """
    Lightweight pairwise cost for top-K candidate generation.
    This function intentionally does NOT call calculate_human_3D_coordinate().
    """
    ref_joints = ray_index.get(ref_cam, {}).get(ref_person_id, {})
    target_joints = ray_index.get(target_cam, {}).get(target_person_id, {})
    if len(ref_joints) == 0 or len(target_joints) == 0:
        return np.inf

    common_joints = sorted(set(ref_joints.keys()) & set(target_joints.keys()))
    dists = []
    for joint_id in common_joints:
        r1 = ref_joints[joint_id]
        r2 = target_joints[joint_id]
        if (
            (not r1.valid) or (not r2.valid)
            or r1.score <= score_threshold
            or r2.score <= score_threshold
        ):
            continue

        dist = ray_ray_distance(r1.origin, r1.direction, r2.origin, r2.direction)
        if np.isfinite(dist):
            dists.append(dist)

    if len(dists) < int(min_valid_joints):
        return np.inf

    dists = np.asarray(dists, dtype=np.float64)
    if robust == "mean":
        return float(np.mean(dists))
    return float(np.median(dists))

def compute_pairwise_person_error(
    ray_index,
    ref_cam,
    ref_person_id,
    target_cam,
    target_person_id,
    score_threshold=0.2
):
    """
    Backward-compatible wrapper.
    The old implementation performed full 17-joint triangulation here;
    this optimized version uses ray distance only.
    """
    return compute_pairwise_ray_distance_error(
        ray_index=ray_index,
        ref_cam=ref_cam,
        ref_person_id=ref_person_id,
        target_cam=target_cam,
        target_person_id=target_person_id,
        score_threshold=score_threshold,
        min_valid_joints=3,
    )

def get_3D_keypoints_from_rays_ref_guided(
    rays,
    frame_id,
    reference_cam="A",
    top_k_pairwise=1,
    min_support_cams=3,
    score_threshold=0.2,
    max_missing_cams=1,
    pairwise_none_error_m=0.10,
    pairwise_candidate_error_m=0.20,
    extrinsic_cache=None,
    calib_path=None,
    output_cams=("A", "B", "C", "D", "E", "F", "G", "H"),
    verbose_every=0,
    max_candidate_ray_error_m=0.10,
    strict_two_cam_ray_error_m=0.05,
    min_candidate_valid_joints=5,
    support_aware_sort=True,
):
    """
    Ref-guided top-K + None closure + full multiview validation.

    Speed-oriented changes:
    1. pairwise stage uses ray-ray distance, not full triangulation;
    2. None is added only when a camera has fewer detections than the expected
       person count, or when the best pairwise match is too poor;
    3. pairwise candidate scoring uses compact numpy arrays;
    4. combinations are generated with DFS pruning instead of full product materialization;
    5. final candidates can be filtered by mean ray-to-point distance and valid joint count.
    """
    coordinates = {}
    selected_rays = [r for r in rays if r.frame_id == frame_id and r.valid and r.person_id >= 0]
    if len(selected_rays) == 0:
        coordinates[frame_id] = np.empty((0, 17, 3), dtype=float)
        return coordinates, [], []

    active_cams = sorted(set(r.camera_id for r in selected_rays))
    output_cams = list(output_cams)
    output_cams.extend([cam_id for cam_id in active_cams if cam_id not in output_cams])

    if extrinsic_cache is None:
        if calib_path is None:
            raise ValueError("calib_path is required when extrinsic_cache is not provided")
        extrinsic_cache_local = load_extrinsic_cache(active_cams, reference_cam, calib_path)
    else:
        extrinsic_cache_local = extrinsic_cache

    selected_rays_to_ref = [
        transform_ray_to_reference_cached(ray, extrinsic_cache_local)
        for ray in selected_rays
    ]
    ray_index = build_ray_index(selected_rays_to_ref)
    ray_table = build_numpy_ray_table(ray_index)

    person_ids_by_cam = get_person_ids_by_camera(ray_index, output_cams)
    person_counts = {
        cam_id: len(person_ids_by_cam.get(cam_id, []))
        for cam_id in output_cams
    }
    active_person_cams = [cam_id for cam_id in active_cams if person_counts.get(cam_id, 0) > 0]
    if len(active_person_cams) == 0:
        coordinates[frame_id] = np.empty((0, 17, 3), dtype=float)
        return coordinates, [], []

    max_person_count = max(person_counts.get(cam_id, 0) for cam_id in active_person_cams)
    if person_counts.get(reference_cam, 0) > 0 and person_counts.get(reference_cam, 0) == max_person_count:
        ref_cam_for_matching = reference_cam
    else:
        ref_cam_for_matching = sorted(
            active_person_cams,
            key=lambda cam_id: (-person_counts.get(cam_id, 0), cam_id)
        )[0]

    ref_person_ids = person_ids_by_cam[ref_cam_for_matching]
    num_person_estimated = estimate_num_person_from_counts(person_counts, active_cams)
    max_selected_persons = len(ref_person_ids)
    exhaustive_theory_count = estimate_exhaustive_combination_count(
        person_counts,
        active_cams,
        num_person_estimated,
    )

    candidate_coordinates = []
    candidate_errors = []
    candidate_combinations = []
    candidate_per_cam_errors = []
    candidate_support_counts = []
    candidate_missing_counts = []
    candidate_valid_joint_counts = []

    generated_combination_count = 0
    kept_after_missing_limit = 0
    final_valid_candidate_count = 0
    rejected_by_valid_joints = 0
    rejected_by_ray_error = 0
    pairwise_cost_count = 0

    top_k_pairwise = max(int(top_k_pairwise), 1)
    min_support_cams = max(int(min_support_cams), 1)
    max_missing_cams = max(int(max_missing_cams), 0)
    pairwise_none_error_m = float(pairwise_none_error_m)
    pairwise_candidate_error_m = float(pairwise_candidate_error_m)
    max_candidate_ray_error_m = float(max_candidate_ray_error_m)
    strict_two_cam_ray_error_m = float(strict_two_cam_ray_error_m)
    min_candidate_valid_joints = max(int(min_candidate_valid_joints), 1)
    support_aware_sort = bool(support_aware_sort)

    for ref_person_id in ref_person_ids:
        candidate_ids_by_cam = {}

        for target_cam in output_cams:
            if target_cam == ref_cam_for_matching:
                candidate_ids_by_cam[target_cam] = [ref_person_id]
                continue

            target_person_ids = person_ids_by_cam.get(target_cam, [])
            target_count = len(target_person_ids)
            candidates = []
            best_pairwise_error = np.inf

            if target_count > 0:
                pairwise_errors = []
                for target_person_id in target_person_ids:
                    pair_error = compute_pairwise_ray_distance_error_np(
                        ray_table,
                        ref_cam_for_matching,
                        ref_person_id,
                        target_cam,
                        target_person_id,
                        score_threshold=score_threshold,
                        min_valid_joints=3,
                    )
                    pairwise_cost_count += 1
                    if np.isfinite(pair_error):
                        pairwise_errors.append((pair_error, target_person_id))

                pairwise_errors.sort(key=lambda item: (item[0], item[1]))
                if len(pairwise_errors) > 0:
                    best_pairwise_error = float(pairwise_errors[0][0])
                if pairwise_candidate_error_m > 0:
                    pairwise_errors = [
                        item for item in pairwise_errors
                        if item[0] <= pairwise_candidate_error_m
                    ]
                candidates = [person_id for _, person_id in pairwise_errors[:top_k_pairwise]]

            should_add_none = target_count < num_person_estimated
            if pairwise_none_error_m > 0:
                should_add_none = should_add_none or (not np.isfinite(best_pairwise_error))
                should_add_none = should_add_none or (best_pairwise_error > pairwise_none_error_m)

            if should_add_none and None not in candidates:
                candidates.append(None)
            if len(candidates) == 0:
                candidates = [None]

            candidate_ids_by_cam[target_cam] = candidates

        for combination in iter_pruned_combinations(
            candidate_ids_by_cam,
            output_cams,
            active_person_cams,
            min_support_cams,
            max_missing_cams,
        ):
            generated_combination_count += 1

            support_cam_count = sum(1 for person_id in combination if person_id is not None)
            missing_count = sum(
                1
                for camera_id, person_id in zip(output_cams, combination)
                if camera_id in active_person_cams and person_id is None
            )
            kept_after_missing_limit += 1

            candidate_rays = collect_combination_rays(ray_index, output_cams, combination)
            if len(candidate_rays) < 2:
                continue

            kpts3D, total_error, joint_errors, per_cam_error = calculate_human_3D_coordinate(
                candidate_rays,
                score_threshold=score_threshold,
            )
            valid_mask = np.all(np.isfinite(kpts3D), axis=1)
            valid_joint_count = int(np.sum(valid_mask))

            # Gate 1: a candidate must reconstruct enough valid joints.
            if valid_joint_count < min_candidate_valid_joints:
                rejected_by_valid_joints += 1
                continue

            # Gate 2: total_error is mean ray-to-3D-point perpendicular distance
            # over valid joints. With meter-scale extrinsic, 0.10 means 10 cm.
            if not np.isfinite(total_error):
                rejected_by_ray_error += 1
                continue

            ray_error_thresh = max_candidate_ray_error_m
            if support_cam_count <= 2 and strict_two_cam_ray_error_m > 0:
                ray_error_thresh = min(ray_error_thresh, strict_two_cam_ray_error_m)

            if ray_error_thresh > 0 and float(total_error) > ray_error_thresh:
                rejected_by_ray_error += 1
                continue

            final_valid_candidate_count += 1
            candidate_coordinates.append(kpts3D)
            candidate_errors.append(float(total_error))
            candidate_combinations.append(tuple(combination))
            candidate_per_cam_errors.append(per_cam_error)
            candidate_support_counts.append(support_cam_count)
            candidate_missing_counts.append(missing_count)
            candidate_valid_joint_counts.append(valid_joint_count)

    used_pairs = set()
    if support_aware_sort:
        # Prefer candidates supported by more cameras. This prevents weak two-camera
        # candidates from winning only because they are easier to self-fit.
        sorted_idx = sorted(
            range(len(candidate_errors)),
            key=lambda i: (
                -candidate_support_counts[i],
                candidate_errors[i],
                candidate_missing_counts[i],
                -candidate_valid_joint_counts[i],
            )
        )
    else:
        sorted_idx = np.argsort(np.array(candidate_errors, dtype=float))
    selected_kpts3D = []
    selected_combinations = []
    selected_per_cam_errors = []

    for idx in sorted_idx:
        combination = candidate_combinations[idx]
        kpts3D = candidate_coordinates[idx]

        valid = True
        temp_pairs = []
        for camera_id, person_id in zip(output_cams, combination):
            if person_id is None:
                continue
            pair = (camera_id, person_id)
            if pair in used_pairs:
                valid = False
                break
            temp_pairs.append(pair)

        if not valid:
            continue

        valid_mask = np.all(np.isfinite(kpts3D), axis=1)
        if not np.any(valid_mask):
            continue

        selected_kpts3D.append(kpts3D)
        selected_combinations.append(combination)
        selected_per_cam_errors.append(candidate_per_cam_errors[idx])

        for pair in temp_pairs:
            used_pairs.add(pair)

        if len(selected_kpts3D) >= max_selected_persons:
            break

    if len(selected_kpts3D) > 0:
        coordinates[frame_id] = np.stack(selected_kpts3D)
    else:
        coordinates[frame_id] = np.empty((0, 17, 3), dtype=float)

    return coordinates, selected_per_cam_errors, selected_combinations

def align_combinations_to_cams(selected_combinations, source_cams, target_cams):
    """
    Convert person-id combinations from their source camera order to target camera order.
    Missing cameras are filled with None so reprojection cannot compare against the wrong 2D person.
    """
    source_cams = list(source_cams)
    target_cams = list(target_cams)
    aligned = []

    for combination in selected_combinations:
        by_cam = {
            cam: combination[idx]
            for idx, cam in enumerate(source_cams)
            if idx < len(combination)
        }
        aligned.append(tuple(by_cam.get(cam, None) for cam in target_cams))

    return aligned

def _legacy_compute_frame_reprojection_debug(
    coordinates_3d,
    selected_combinations,
    group,
    result_2D_data_path,
    reference_cam="A",
    calib_path=None,
    score_threshold=0.2,
    cams=("A", "B", "C", "D", "E", "F", "G", "H"),
):
    """
    计算单帧3D结果在各相机下的重投影误差

    Args:
        coordinates_3d: np.ndarray, (num_person, 17, 3), 位于reference_cam坐标系
        selected_combinations: List[Tuple]
            get_3D_keypoints_from_rays() 选出的多人匹配结果
            第i个元素对应第i个3D person在各cam下匹配到的2D person_id
        group: dict[cam_id] = img_path
        result_2D_data_path: 2D结果目录
        reference_cam: 参考相机
        score_threshold: 2D关键点有效阈值
        cams: 相机顺序

    Returns:
        reproj_debug: dict
    """
    if calib_path is None:
        raise ValueError("calib_path is required")
    calib_path = Path(calib_path)
    result_2D_data_path = Path(result_2D_data_path)

    reproj_debug = {
        "reference_cam": reference_cam,
        "persons": []
    }

    # 先缓存这一帧所有相机的2D检测结果
    frame_obs = {}
    for cam in cams:
        if cam not in group:
            continue

        filename = Path(group[cam]).stem
        npz_path = result_2D_data_path / f"cam_{cam}" / f"{filename}.npz"

        if not npz_path.exists():
            frame_obs[cam] = {
                "kpts": np.empty((0, 17, 2), dtype=float),
                "scores": np.empty((0, 17), dtype=float),
                "masks": None,
            }
            continue

        data = np.load(str(npz_path), allow_pickle=True)
        frame_obs[cam] = {
            "kpts": data.get("kpts"),
            "scores": data.get("scores"),
            "masks": data.get("masks"),
        }

    num_person = coordinates_3d.shape[0] if coordinates_3d is not None else 0

    for person_3d_id in range(num_person):
        kpts3d_ref = coordinates_3d[person_3d_id]
        combination = selected_combinations[person_3d_id] if person_3d_id < len(selected_combinations) else tuple([None] * len(cams))

        person_debug = {
            "person_3d_id": person_3d_id,
            "match": {},
            "per_cam": {},
            "global_mean_error": np.nan,
            "global_valid_joint_count": 0,
        }

        all_errors = []

        for cam_idx, cam in enumerate(cams):
            if cam not in group:
                continue

            person_2d_id = combination[cam_idx] if cam_idx < len(combination) else None
            person_debug["match"][cam] = person_2d_id

            joint_errors = np.full((17,), np.nan, dtype=float)
            projected_kpts = np.full((17, 2), np.nan, dtype=float)
            observed_kpts = np.full((17, 2), np.nan, dtype=float)
            scores = np.zeros((17,), dtype=float)
            joint_status = ["missing_match"] * 17

            # 没匹配到人，直接记状态
            if person_2d_id is None:
                person_debug["per_cam"][cam] = {
                    "person_2d_id": None,
                    "mean_error": np.nan,
                    "valid_joint_count": 0,
                    "joint_errors": joint_errors,
                    "joint_status": joint_status,
                    "projected_kpts": projected_kpts,
                    "observed_kpts": observed_kpts,
                    "scores": scores,
                }
                continue

            obs = frame_obs[cam]
            kpts2d_all = obs["kpts"]
            scores_all = obs["scores"]

            if kpts2d_all is None or scores_all is None or len(kpts2d_all) == 0:
                joint_status = ["missing_2d"] * 17
                person_debug["per_cam"][cam] = {
                    "person_2d_id": person_2d_id,
                    "mean_error": np.nan,
                    "valid_joint_count": 0,
                    "joint_errors": joint_errors,
                    "joint_status": joint_status,
                    "projected_kpts": projected_kpts,
                    "observed_kpts": observed_kpts,
                    "scores": scores,
                }
                continue

            if person_2d_id >= kpts2d_all.shape[0]:
                joint_status = ["missing_2d"] * 17
                person_debug["per_cam"][cam] = {
                    "person_2d_id": person_2d_id,
                    "mean_error": np.nan,
                    "valid_joint_count": 0,
                    "joint_errors": joint_errors,
                    "joint_status": joint_status,
                    "projected_kpts": projected_kpts,
                    "observed_kpts": observed_kpts,
                    "scores": scores,
                }
                continue

            kpts2d = kpts2d_all[person_2d_id]
            scores2d = scores_all[person_2d_id]

            observed_kpts[:] = kpts2d
            scores[:] = scores2d

            # 取 reference_cam -> 当前cam 的变换
            if cam == reference_cam:
                R_ref_to_cam = np.eye(3, dtype=float)
                t_ref_to_cam = np.zeros(3, dtype=float)
            else:
                _, R_cam_to_ref, t_cam_to_ref = load_extrinsic_data(
                    calib_path / f"extrinsic_T_cam_{cam}_to_cam_{reference_cam}.npy"
                )
                # X_ref = R_cam_to_ref @ X_cam + t_cam_to_ref
                # => X_cam = R_cam_to_ref.T @ (X_ref - t_cam_to_ref)
                R_ref_to_cam = R_cam_to_ref.T
                t_ref_to_cam = -R_cam_to_ref.T @ t_cam_to_ref

            K, dist = load_intrinsic_data(calib_path / f"intrinsic_cam_{cam}.npz")

            valid_errors_this_cam = []

            for joint_idx in range(17):
                pt3d_ref = kpts3d_ref[joint_idx]

                if not np.all(np.isfinite(pt3d_ref)):
                    joint_status[joint_idx] = "missing_3d"
                    continue

                if scores2d[joint_idx] <= score_threshold:
                    joint_status[joint_idx] = "missing_2d"
                    continue

                pt3d_cam = R_ref_to_cam @ pt3d_ref + t_ref_to_cam

                if pt3d_cam[2] <= 1e-8:
                    joint_status[joint_idx] = "behind_camera"
                    continue

                obj_pts = np.array(pt3d_cam, dtype=np.float64).reshape(1, 1, 3)
                rvec = np.zeros((3, 1), dtype=np.float64)
                tvec = np.zeros((3, 1), dtype=np.float64)

                proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
                uv = proj.reshape(2)

                if not np.all(np.isfinite(uv)):
                    joint_status[joint_idx] = "invalid_projection"
                    continue

                projected_kpts[joint_idx] = uv
                err = float(np.linalg.norm(uv - kpts2d[joint_idx]))
                joint_errors[joint_idx] = err
                joint_status[joint_idx] = "valid"
                valid_errors_this_cam.append(err)
                all_errors.append(err)

            mean_error = float(np.mean(valid_errors_this_cam)) if len(valid_errors_this_cam) > 0 else np.nan
            valid_joint_count = len(valid_errors_this_cam)

            person_debug["per_cam"][cam] = {
                "person_2d_id": person_2d_id,
                "mean_error": mean_error,
                "valid_joint_count": valid_joint_count,
                "joint_errors": joint_errors,
                "joint_status": joint_status,
                "projected_kpts": projected_kpts,
                "observed_kpts": observed_kpts,
                "scores": scores,
            }

        person_debug["global_mean_error"] = float(np.mean(all_errors)) if len(all_errors) > 0 else np.nan
        person_debug["global_valid_joint_count"] = len(all_errors)

        reproj_debug["persons"].append(person_debug)

    return reproj_debug

def _safe_npz_get(data, keys, default=None):
    for key in keys:
        if key in data.files:
            return data[key]
    return default


def _load_2d_npz_for_frame(result_2D_data_path, cam, img_path):
    result_2D_data_path = Path(result_2D_data_path)
    filename = Path(img_path).stem
    npz_path = result_2D_data_path / f"cam_{cam}" / f"{filename}.npz"
    empty = {
        "npz_path": npz_path,
        "kpts": np.empty((0, 17, 2), dtype=np.float32),
        "scores": np.empty((0, 17), dtype=np.float32),
        "masks": None,
        "bboxes": np.empty((0, 4), dtype=np.float32),
        "bboxes_scores": np.empty((0,), dtype=np.float32),
    }
    if not npz_path.exists():
        return empty

    data = np.load(str(npz_path), allow_pickle=True)
    kpts = _safe_npz_get(data, ["kpts"], empty["kpts"])
    scores = _safe_npz_get(data, ["scores"], empty["scores"])
    masks = _safe_npz_get(data, ["masks"], None)
    bboxes = _safe_npz_get(
        data,
        ["bboxes", "boxes", "bbox", "det_boxes", "person_bboxes"],
        empty["bboxes"],
    )
    bboxes_scores = _safe_npz_get(
        data,
        ["bboxes_scores", "bbox_scores", "box_scores", "det_scores"],
        empty["bboxes_scores"],
    )

    kpts = empty["kpts"] if kpts is None else np.asarray(kpts, dtype=np.float32)
    scores = empty["scores"] if scores is None else np.asarray(scores, dtype=np.float32)
    bboxes = empty["bboxes"] if bboxes is None else np.asarray(bboxes, dtype=np.float32)
    bboxes_scores = (
        empty["bboxes_scores"]
        if bboxes_scores is None
        else np.asarray(bboxes_scores, dtype=np.float32).reshape(-1)
    )
    if bboxes.ndim == 1 and bboxes.shape[0] >= 4:
        bboxes = bboxes.reshape(1, -1)
    if bboxes.ndim != 2 or bboxes.shape[1] < 4:
        bboxes = empty["bboxes"]

    return {
        "npz_path": npz_path,
        "kpts": kpts,
        "scores": scores,
        "masks": masks,
        "bboxes": bboxes,
        "bboxes_scores": bboxes_scores,
    }


def compute_frame_reprojection_debug(
    coordinates_3d,
    selected_combinations,
    group,
    result_2D_data_path,
    reference_cam="A",
    calib_path=None,
    score_threshold=0.5,
    cams=("A", "B", "C", "D", "E", "F", "G", "H"),
):
    if calib_path is None:
        raise ValueError("calib_path is required")
    calib_path = Path(calib_path)
    result_2D_data_path = Path(result_2D_data_path)
    cams = list(cams)
    reproj_debug = {
        "reference_cam": reference_cam,
        "score_threshold": score_threshold,
        "cams": cams,
        "persons": [],
    }

    frame_obs = {}
    for cam in cams:
        if cam not in group:
            continue
        frame_obs[cam] = _load_2d_npz_for_frame(
            result_2D_data_path=result_2D_data_path,
            cam=cam,
            img_path=group[cam],
        )

    if coordinates_3d is None:
        return reproj_debug
    coordinates_3d = np.asarray(coordinates_3d, dtype=np.float64)
    if coordinates_3d.ndim != 3 or coordinates_3d.shape[1:] != (17, 3):
        return reproj_debug

    for person_3d_id, kpts3d_ref in enumerate(coordinates_3d):
        if person_3d_id < len(selected_combinations):
            combination = selected_combinations[person_3d_id]
        else:
            combination = tuple([None] * len(cams))

        person_debug = {
            "person_3d_id": person_3d_id,
            "match": {},
            "per_cam": {},
            "global_mean_error": np.nan,
            "global_valid_joint_count": 0,
        }
        all_errors = []

        for cam_idx, cam in enumerate(cams):
            if cam not in group:
                continue

            person_2d_id = combination[cam_idx] if cam_idx < len(combination) else None
            person_debug["match"][cam] = person_2d_id

            joint_errors = np.full((17,), np.nan, dtype=np.float64)
            projected_kpts = np.full((17, 2), np.nan, dtype=np.float64)
            observed_kpts = np.full((17, 2), np.nan, dtype=np.float64)
            scores = np.zeros((17,), dtype=np.float64)
            joint_status = ["missing_match"] * 17

            def store_empty(status=None):
                person_debug["per_cam"][cam] = {
                    "person_2d_id": person_2d_id,
                    "mean_error": np.nan,
                    "valid_joint_count": 0,
                    "joint_errors": joint_errors,
                    "joint_status": status or joint_status,
                    "projected_kpts": projected_kpts,
                    "observed_kpts": observed_kpts,
                    "scores": scores,
                }

            if person_2d_id is None:
                store_empty()
                continue

            obs = frame_obs.get(cam)
            if obs is None:
                store_empty(["missing_2d_file"] * 17)
                continue

            kpts2d_all = obs["kpts"]
            scores_all = obs["scores"]
            if kpts2d_all is None or scores_all is None or len(kpts2d_all) == 0:
                store_empty(["missing_2d"] * 17)
                continue
            if person_2d_id < 0 or person_2d_id >= kpts2d_all.shape[0]:
                store_empty(["person_id_out_of_range"] * 17)
                continue

            kpts2d = np.asarray(kpts2d_all[person_2d_id], dtype=np.float64)
            scores2d = np.asarray(scores_all[person_2d_id], dtype=np.float64)
            observed_kpts[:] = kpts2d
            scores[:] = scores2d

            if cam == reference_cam:
                R_ref_to_cam = np.eye(3, dtype=np.float64)
                t_ref_to_cam = np.zeros(3, dtype=np.float64)
            else:
                _, R_cam_to_ref, t_cam_to_ref = load_extrinsic_data(
                    calib_path / f"extrinsic_T_cam_{cam}_to_cam_{reference_cam}.npy"
                )
                R_cam_to_ref = np.asarray(R_cam_to_ref, dtype=np.float64)
                t_cam_to_ref = np.asarray(t_cam_to_ref, dtype=np.float64).reshape(3)
                R_ref_to_cam = R_cam_to_ref.T
                t_ref_to_cam = -R_cam_to_ref.T @ t_cam_to_ref

            K, dist = load_intrinsic_data(calib_path / f"intrinsic_cam_{cam}.npz")
            K = np.asarray(K, dtype=np.float64)
            dist = np.asarray(dist, dtype=np.float64)
            valid_errors_this_cam = []

            for joint_idx in range(17):
                pt3d_ref = kpts3d_ref[joint_idx]
                if not np.all(np.isfinite(pt3d_ref)):
                    joint_status[joint_idx] = "missing_3d"
                    continue
                if joint_idx >= len(scores2d) or scores2d[joint_idx] <= score_threshold:
                    joint_status[joint_idx] = "low_score_2d"
                    continue

                pt3d_cam = R_ref_to_cam @ pt3d_ref + t_ref_to_cam
                if not np.all(np.isfinite(pt3d_cam)):
                    joint_status[joint_idx] = "invalid_3d_cam"
                    continue
                if pt3d_cam[2] <= 1e-8:
                    joint_status[joint_idx] = "behind_camera"
                    continue

                obj_pts = pt3d_cam.reshape(1, 1, 3).astype(np.float64)
                proj, _ = cv2.projectPoints(
                    obj_pts,
                    np.zeros((3, 1), dtype=np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    K,
                    dist,
                )
                uv = proj.reshape(2)
                if not np.all(np.isfinite(uv)):
                    joint_status[joint_idx] = "invalid_projection"
                    continue
                if not np.all(np.isfinite(kpts2d[joint_idx])):
                    joint_status[joint_idx] = "invalid_2d"
                    continue

                projected_kpts[joint_idx] = uv
                err = float(np.linalg.norm(uv - kpts2d[joint_idx]))
                joint_errors[joint_idx] = err
                joint_status[joint_idx] = "valid"
                valid_errors_this_cam.append(err)
                all_errors.append(err)

            person_debug["per_cam"][cam] = {
                "person_2d_id": person_2d_id,
                "mean_error": float(np.mean(valid_errors_this_cam)) if valid_errors_this_cam else np.nan,
                "valid_joint_count": len(valid_errors_this_cam),
                "joint_errors": joint_errors,
                "joint_status": joint_status,
                "projected_kpts": projected_kpts,
                "observed_kpts": observed_kpts,
                "scores": scores,
            }

        person_debug["global_mean_error"] = float(np.mean(all_errors)) if all_errors else np.nan
        person_debug["global_valid_joint_count"] = len(all_errors)
        reproj_debug["persons"].append(person_debug)

    return reproj_debug


def _draw_bbox_on_axis(ax, bboxes, bboxes_scores=None, color="lime", linewidth=1.8):
    if bboxes is None:
        return
    bboxes = np.asarray(bboxes)
    if bboxes.ndim == 1 and bboxes.shape[0] >= 4:
        bboxes = bboxes.reshape(1, -1)
    if bboxes.ndim != 2 or bboxes.shape[1] < 4:
        return
    if bboxes_scores is not None:
        bboxes_scores = np.asarray(bboxes_scores).reshape(-1)

    for pid, box in enumerate(bboxes):
        x1, y1, x2, y2 = box[:4]
        if not np.all(np.isfinite([x1, y1, x2, y2])):
            continue
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
        ax.add_patch(rect)
        score_text = ""
        if bboxes_scores is not None and pid < len(bboxes_scores):
            score_text = f" {float(bboxes_scores[pid]):.2f}"
        elif box.shape[0] >= 5:
            score_text = f" {float(box[4]):.2f}"
        ax.text(
            x1,
            max(y1 - 4, 0),
            f"2D{pid}{score_text}",
            color=color,
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1),
        )


def _draw_pose_on_axis(
    ax,
    kpts,
    scores,
    person_ids=None,
    kpt_thr=0.3,
    keypoint_color="red",
    skeleton_color="red",
    linewidth=1.2,
    keypoint_size=10,
):
    if kpts is None or scores is None:
        return
    kpts = np.asarray(kpts)
    scores = np.asarray(scores)
    if kpts.ndim != 3 or scores.ndim != 2 or len(kpts) == 0:
        return

    person_ids = range(kpts.shape[0]) if person_ids is None else list(person_ids)
    for pid in person_ids:
        if pid is None or pid < 0 or pid >= kpts.shape[0]:
            continue
        pts = kpts[pid]
        scs = scores[pid]
        for joint_idx in range(min(17, pts.shape[0])):
            if joint_idx < len(scs) and scs[joint_idx] > kpt_thr:
                x, y = pts[joint_idx]
                if np.all(np.isfinite([x, y])):
                    ax.scatter(x, y, c=keypoint_color, s=keypoint_size, marker="o")
        for j1, j2 in COCO17_SKELETON:
            if j1 >= pts.shape[0] or j2 >= pts.shape[0] or j1 >= len(scs) or j2 >= len(scs):
                continue
            if scs[j1] > kpt_thr and scs[j2] > kpt_thr:
                x1, y1 = pts[j1]
                x2, y2 = pts[j2]
                if np.all(np.isfinite([x1, y1, x2, y2])):
                    ax.plot([x1, x2], [y1, y2], color=skeleton_color, linewidth=linewidth)


def visualize_frame_reprojection_debug(
    group,
    reproj_debug,
    result_2D_data_path,
    cams=("A", "B", "C", "D", "E", "F", "G", "H"),
    error_vis_threshold=10.0,
    draw_all_2d=True,
    kpt_thr_2d=0.3,
    max_summary_lines=4,
):
    result_2D_data_path = Path(result_2D_data_path)
    cams = list(cams)
    n_cam = len([cam for cam in cams if cam in group])
    n_rows = max(1, int(np.ceil(n_cam / 2)))
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5.2 * n_rows), squeeze=False)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, wspace=0.03, hspace=0.16)
    for ax in axes.ravel():
        ax.axis("off")

    cam_plot_idx = 0
    for cam in cams:
        if cam not in group:
            continue
        img = plt.imread(group[cam])
        h, w = img.shape[:2]
        obs = _load_2d_npz_for_frame(result_2D_data_path, cam, group[cam])
        kpts = obs["kpts"]
        scores = obs["scores"]
        bboxes = obs["bboxes"]
        bboxes_scores = obs["bboxes_scores"]

        row = cam_plot_idx // 2
        col_base = (cam_plot_idx % 2) * 2
        ax1 = axes[row, col_base]
        ax2 = axes[row, col_base + 1]

        ax1.imshow(img, aspect="auto")
        _draw_bbox_on_axis(ax1, bboxes, bboxes_scores, color="lime", linewidth=1.8)
        if draw_all_2d:
            _draw_pose_on_axis(ax1, kpts, scores, None, kpt_thr_2d, "red", "red", 1.1, 10)

        bbox_info = f"bbox:{bboxes.shape[0]}" if bboxes is not None and len(bboxes) > 0 else "bbox:none"
        pose_info = f"pose:{kpts.shape[0]}" if kpts is not None and len(kpts) > 0 else "pose:none"
        ax1.set_title(f"Camera {cam} 2D Detection\n{pose_info}, {bbox_info}", fontsize=12, pad=5)
        ax1.set_xlim([0, w])
        ax1.set_ylim([h, 0])
        ax1.set_aspect("auto")
        ax1.axis("off")

        ax2.imshow(img, aspect="auto")
        summary_lines = []
        used_2d_ids = set()
        for person_debug in reproj_debug.get("persons", []):
            person_3d_id = person_debug["person_3d_id"]
            if cam not in person_debug["per_cam"]:
                continue
            cam_debug = person_debug["per_cam"][cam]
            person_2d_id = cam_debug["person_2d_id"]
            mean_error = cam_debug["mean_error"]
            valid_joint_count = cam_debug["valid_joint_count"]
            projected_kpts = cam_debug["projected_kpts"]
            observed_kpts = cam_debug["observed_kpts"]
            joint_errors = cam_debug["joint_errors"]
            joint_status = cam_debug["joint_status"]

            if person_2d_id is None:
                summary_lines.append(f"P3D{person_3d_id}->None")
                continue
            used_2d_ids.add(person_2d_id)
            err_text = f"{mean_error:.2f}px" if np.isfinite(mean_error) else "nan"
            summary_lines.append(f"P3D{person_3d_id}->2D{person_2d_id} err={err_text} n={valid_joint_count}")
            _draw_pose_on_axis(ax2, kpts, scores, [person_2d_id], kpt_thr_2d, "orange", "orange", 0.8, 8)

            valid_obs = []
            for joint_idx in range(17):
                if joint_status[joint_idx] != "valid":
                    continue
                x_obs, y_obs = observed_kpts[joint_idx]
                x_proj, y_proj = projected_kpts[joint_idx]
                if not np.all(np.isfinite([x_obs, y_obs, x_proj, y_proj])):
                    continue
                ax2.scatter(x_obs, y_obs, c="red", s=14, marker="o")
                ax2.scatter(x_proj, y_proj, c="blue", s=18, marker="x")
                ax2.plot([x_obs, x_proj], [y_obs, y_proj], color="yellow", linewidth=0.9, alpha=0.85)
                if np.isfinite(joint_errors[joint_idx]) and joint_errors[joint_idx] > error_vis_threshold:
                    ax2.text(
                        x_proj + 2,
                        y_proj + 2,
                        f"{joint_errors[joint_idx]:.1f}",
                        color="yellow",
                        fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
                    )
                valid_obs.append([x_obs, y_obs])

            if valid_obs:
                cx, cy = np.nanmean(np.asarray(valid_obs, dtype=float), axis=0)
                ax2.text(
                    cx,
                    cy,
                    f"P3D{person_3d_id}->2D{person_2d_id}\n{err_text}",
                    color="white",
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
                )

        if bboxes is not None and len(bboxes) > 0 and used_2d_ids:
            for pid in used_2d_ids:
                if pid is None or pid < 0 or pid >= len(bboxes):
                    continue
                _draw_bbox_on_axis(
                    ax2,
                    np.asarray(bboxes[pid]).reshape(1, -1),
                    np.asarray([bboxes_scores[pid]]) if bboxes_scores is not None and pid < len(bboxes_scores) else None,
                    color="cyan",
                    linewidth=2.2,
                )

        title_text = f"Camera {cam} Reprojection"
        if summary_lines:
            title_text += "\n" + "\n".join(summary_lines[:max_summary_lines])
        ax2.set_title(title_text, fontsize=11, pad=5)
        ax2.set_xlim([0, w])
        ax2.set_ylim([h, 0])
        ax2.set_aspect("auto")
        ax2.axis("off")
        cam_plot_idx += 1

    plt.show()


def vis_results_3D(path, pos_world, cams=["A", "B", "C", "D", "E", "F", "G", "H"],):
    person_color = ['red', 'blue', 'green', 'orange', 'purple']
    path = Path(path)
    files = os.listdir(path)
    files.sort()
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    plt.ioff()
    for f in files:
        with open(path / f, "rb") as file:
            data = pickle.load(file)
        plt.cla()
        for cam in cams:
            ax.scatter(pos_world[cam][0], pos_world[cam][1], pos_world[cam][2], c='r', s=5)
            ax.text(pos_world[cam][0], pos_world[cam][1], pos_world[cam][2], s=cam, fontsize=10)
        for person_id, person in enumerate(data):
            ax.scatter(person[:, 0], person[:, 1], person[:, 2], c='b', s=5)
            for edge in COCO17_SKELETON:
                ax.plot(
                    [person[edge[0], 0], person[edge[1], 0]],
                    [person[edge[0], 1], person[edge[1], 1]],
                    [person[edge[0], 2], person[edge[1], 2]],
                    color='b',
                )
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([0, 2.2])
        plt.pause(0.01)
    plt.show()

_FRAME_WORKER_CFG = None

def init_frame_worker(cfg):
    global _FRAME_WORKER_CFG
    _FRAME_WORKER_CFG = cfg

def reconstruct_and_save_frame_no_debug_global(frame_item):
    if _FRAME_WORKER_CFG is None:
        raise RuntimeError("Frame worker config has not been initialized.")
    return reconstruct_and_save_frame_no_debug(frame_item, _FRAME_WORKER_CFG)

def frame_result_3d_filename(group):
    valid_paths = [img_path for img_path in group.values() if img_path is not None]
    if len(valid_paths) == 0:
        return None

    avg_t_ns = int(round(np.mean([
        int(Path(img_path).stem.split("_")[0]) * 1_000_000_000
        + int(Path(img_path).stem.split("_")[1])
        for img_path in valid_paths
    ])))

    sec = avg_t_ns // 1_000_000_000
    nsec = avg_t_ns % 1_000_000_000
    return f"{sec}_{nsec:09d}"

def has_camera_result_dirs(path):
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("cam_") for p in path.iterdir())

def infer_group_jobs(human_img_data_path, result_2D_data_path, result_3D_data_path):
    """
    Infer group folders from result_2D_data_path.

    Supports either:
    - result_2D_data_path = .../results/2D
    - result_2D_data_path = .../results/2D/4
    """
    human_img_data_path = Path(human_img_data_path)
    result_2D_data_path = Path(result_2D_data_path)
    result_3D_data_path = Path(result_3D_data_path)

    if has_camera_result_dirs(human_img_data_path) and has_camera_result_dirs(result_2D_data_path):
        return [(
            result_2D_data_path.name,
            human_img_data_path,
            result_2D_data_path,
            result_3D_data_path,
        )]

    if has_camera_result_dirs(result_2D_data_path):
        result_2d_base = result_2D_data_path.parent
        default_group_name = result_2D_data_path.name
    else:
        result_2d_base = result_2D_data_path
        default_group_name = None

    if human_img_data_path.name.isdigit():
        human_base = human_img_data_path.parent
    elif any((human_img_data_path / name).is_dir() for name in ("0", "1", "2")):
        human_base = human_img_data_path
    else:
        human_base = human_img_data_path.parent

    if result_3D_data_path.name.isdigit():
        result_3d_base = result_3D_data_path.parent
    else:
        result_3d_base = result_3D_data_path

    group_dirs = [
        p for p in result_2d_base.iterdir()
        if p.is_dir() and has_camera_result_dirs(p)
    ]
    group_dirs.sort(key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name))

    jobs = []
    for result_2d_group_path in group_dirs:
        group_name = result_2d_group_path.name
        human_group_path = human_base / group_name
        result_3d_group_path = result_3d_base / group_name
        if not human_group_path.exists():
            print(f"[all_groups] skip group={group_name}: missing human dir {human_group_path}")
            continue
        jobs.append((group_name, human_group_path, result_2d_group_path, result_3d_group_path))

    return jobs

def collect_missing_frame_items(synchronized_groups, result_3D_data_path, skip_existing=True):
    result_3D_data_path = Path(result_3D_data_path)
    frame_items = []
    skipped = 0
    for frame_id, group in synchronized_groups.items():
        result_3D_filename = frame_result_3d_filename(group)
        if result_3D_filename is None:
            skipped += 1
            continue
        out_path = result_3D_data_path / f"{result_3D_filename}.pkl"
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        frame_items.append((frame_id, group))
    return frame_items, skipped

def reconstruct_and_save_frame_no_debug(frame_item, cfg):
    """
    Non-visual per-frame worker used by the parallel path.
    Visualization and reprojection debug stay in the serial debug path.
    """
    frame_id, group = frame_item
    all_cams = cfg["all_cams"]
    result_3D_filename = frame_result_3d_filename(group)
    if result_3D_filename is None:
        return {
            "frame_id": frame_id,
            "filename": None,
            "num_person": 0,
            "skipped": True,
        }

    out_path = Path(cfg["result_3D_data_path"]) / f"{result_3D_filename}.pkl"
    if cfg.get("skip_existing", True) and out_path.exists():
        return {
            "frame_id": frame_id,
            "filename": result_3D_filename,
            "num_person": 0,
            "skipped": True,
        }

    rays = get_keypoint_ray_fast(
        result_2D_data_path=cfg["result_2D_data_path"],
        group=group,
        frame_id=frame_id,
        intrinsic_cache=cfg["intrinsic_cache"],
        calib_path=cfg["calib_path"],
        use_2d_scores=cfg["use_2d_scores"],
    )

    if cfg["matching_mode"] == "exhaustive":
        coordinates_dict, _, selected_combinations = get_3D_keypoints_from_rays(
            rays,
            frame_id=frame_id,
            reference_cam=cfg["reference_cam"],
            calib_path=cfg["calib_path"],
        )
        combination_cams = sorted(set(r.camera_id for r in rays if r.frame_id == frame_id and r.valid))
    else:
        coordinates_dict, _, selected_combinations = get_3D_keypoints_from_rays_ref_guided(
            rays,
            frame_id=frame_id,
            reference_cam=cfg["reference_cam"],
            top_k_pairwise=cfg["pairwise_top_k"],
            min_support_cams=cfg["min_support_cams"],
            score_threshold=0.5,
            max_missing_cams=cfg["max_missing_cams"],
            pairwise_none_error_m=cfg["pairwise_none_error_m"],
            pairwise_candidate_error_m=cfg["pairwise_candidate_error_m"],
            extrinsic_cache=cfg["extrinsic_cache"],
            calib_path=cfg["calib_path"],
            output_cams=tuple(all_cams),
            verbose_every=cfg["verbose_every"],
            max_candidate_ray_error_m=cfg["max_candidate_ray_error_m"],
            strict_two_cam_ray_error_m=cfg["strict_two_cam_ray_error_m"],
            min_candidate_valid_joints=cfg["min_candidate_valid_joints"],
            support_aware_sort=cfg["support_aware_sort"],
        )
        combination_cams = list(all_cams)

    coordinates = coordinates_dict[frame_id]
    coordinates_world = coordinates.copy()
    R_ref_to_world = cfg["R_ref_to_world"]
    world_origin_ref = cfg["world_origin_ref"]
    for person_id in range(coordinates.shape[0]):
        valid_mask = np.all(np.isfinite(coordinates[person_id]), axis=1)
        if np.any(valid_mask):
            coordinates_world[person_id, valid_mask] = (
                R_ref_to_world @ (coordinates[person_id, valid_mask] - world_origin_ref).T
            ).T

    with open(out_path, "wb") as f:
        pickle.dump(coordinates_world, f)

    return {
        "frame_id": frame_id,
        "filename": result_3D_filename,
        "num_person": int(coordinates_world.shape[0]),
        "skipped": False,
    }

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.result_3D_data_path, exist_ok=True)
    person_color = ['red', 'blue', 'green', 'orange', 'purple']
    human_img_data_path = args.root_path / "camera"
    result_2D_data_path = Path(args.result_2D_data_path)
    result_3D_data_path = Path(args.result_3D_data_path)
    all_cams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    intrinsic_cache = load_intrinsic_cache(all_cams, args.calib_path)
    extrinsic_cache = load_extrinsic_cache(all_cams, args.reference_cam, args.calib_path)

    pos, dir = build_camera_relative_position(cams=all_cams, reference_cam=args.reference_cam, calib_path=args.calib_path)
    R_ref_to_world = np.eye(3, dtype=np.float64)
    world_origin_ref = np.zeros(3, dtype=np.float64)
    pos_world = {cam: R_ref_to_world @ (pos[cam] - world_origin_ref) for cam in pos.keys()}

    group_jobs = infer_group_jobs(human_img_data_path, result_2D_data_path, result_3D_data_path)
    print(f"[all_groups] found {len(group_jobs)} groups from result_2D_data_dir={result_2D_data_path}")

    total_done = 0
    total_skipped = 0
    workers = max(1, int(args.num_workers))
    chunksize = max(1, int(args.parallel_chunksize))

    for group_name, human_group_path, result_2d_group_path, result_3d_group_path in group_jobs:
        os.makedirs(result_3d_group_path, exist_ok=True)
        synchronized_groups = get_matched_pairs(human_group_path)
        frame_items, skipped = collect_missing_frame_items(
            synchronized_groups,
            result_3d_group_path,
            skip_existing=args.skip_existing,
        )
        total_skipped += skipped
        print(
            f"[all_groups] group={group_name} "
            f"total={len(synchronized_groups)} todo={len(frame_items)} skipped={skipped}"
        )
        if len(frame_items) == 0:
            continue

        worker_cfg = {
            "all_cams": all_cams,
            "result_2D_data_path": result_2d_group_path,
            "result_3D_data_path": str(result_3d_group_path),
            "intrinsic_cache": intrinsic_cache,
            "extrinsic_cache": extrinsic_cache,
            "reference_cam": args.reference_cam,
            "calib_path": args.calib_path,
            "matching_mode": args.matching_mode,
            "pairwise_top_k": args.pairwise_top_k,
            "min_support_cams": args.min_support_cams,
            "max_missing_cams": args.max_missing_cams,
            "pairwise_none_error_m": args.pairwise_none_error_m,
            "pairwise_candidate_error_m": args.pairwise_candidate_error_m,
            "max_candidate_ray_error_m": args.max_candidate_ray_error_m,
            "strict_two_cam_ray_error_m": args.strict_two_cam_ray_error_m,
            "min_candidate_valid_joints": args.min_candidate_valid_joints,
            "support_aware_sort": args.support_aware_sort,
            "use_2d_scores": args.use_2d_scores,
            "verbose_every": args.verbose_every,
            "R_ref_to_world": R_ref_to_world,
            "world_origin_ref": world_origin_ref,
            "skip_existing": args.skip_existing,
        }

        if workers > 1:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_frame_worker,
                initargs=(worker_cfg,),
            ) as executor:
                results = executor.map(
                    reconstruct_and_save_frame_no_debug_global,
                    frame_items,
                    chunksize=chunksize,
                )
                for result in tqdm(results, total=len(frame_items), desc=f"Group {group_name} x{workers}"):
                    if not result.get("skipped", False):
                        total_done += 1
        else:
            for frame_item in tqdm(frame_items, total=len(frame_items), desc=f"Group {group_name}"):
                result = reconstruct_and_save_frame_no_debug(frame_item, worker_cfg)
                if not result.get("skipped", False):
                    total_done += 1

    print(f"[all_groups] done={total_done} skipped={total_skipped}")
    if args.show_final_vis and len(group_jobs) > 0:
        vis_results_3D(group_jobs[-1][3], pos_world)
