import math
import pickle
from pathlib import Path
import scipy.io as sio
import numpy as np
import cv2

from TimeProcess.utils import timestamp_to_ms

ROWS, COLS = 24, 24  # 棋盘内角点数: rows x cols
CRIT = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    50,
    1e-6
)

COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
COCO17_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

def load_extrinsic_data(file_full_path):
    T = np.load(file_full_path)
    R = T[:3, :3]
    t = T[:3, 3]
    return T, R, t

def load_intrinsic_data(file_full_path):
    file_type = Path(file_full_path).suffix.lower()
    if file_type == ".mat":
        data = sio.loadmat(file_full_path)
        K = data["K"]
        dist = data["dist"]
    elif file_type == ".npz":
        data = np.load(file_full_path, allow_pickle=True)
        K = data["K"]
        dist = data["dist"]
    else:
        K = None
        dist = None
    return K, dist

def get_matched_pairs(data_path, max_error_ms=25):
    data_path = Path(data_path)
    cams = sorted(p for p in data_path.iterdir() if p.is_dir())
    cam_data = {}

    for cam_path in cams:
        frames_path = cam_path / "frames"
        files = sorted(p for p in frames_path.iterdir() if p.is_file())

        cam_id = cam_path.name.split("_")[-1]
        cam_data[cam_id] = []

        for file_path in files:
            if file_path.suffix != ".jpg":
                continue
            name = file_path.stem
            t = timestamp_to_ms(name)
            cam_data[cam_id].append({"t": t, "file": file_path, "used": False})

        print(f"cam{cam_id} has {len(files)} files")

    cam_ids = sorted(cam_data.keys())
    min_required = math.ceil(0.75 * len(cam_ids))
    # min_required = math.ceil(len(cam_ids))

    synchronized_groups = {}
    gid = 0

    base_cam = cam_ids[0]
    for frame in cam_data[base_cam]:

        if frame["used"]:
            continue

        base_t = frame["t"]
        group = {cid: None for cid in cam_ids}
        candidates = []

        for cid in cam_ids:
            best = None
            best_err = max_error_ms + 1

            for f in cam_data[cid]:
                if f["used"]:
                    continue

                err = abs(f["t"] - base_t)
                if err <= max_error_ms and err < best_err:
                    best = f
                    best_err = err

            if best:
                candidates.append((cid, best))
                group[cid] = best["file"]

        if len(candidates) >= min_required:
            for cid, f in candidates:
                f["used"] = True

            synchronized_groups[gid] = group
            gid += 1

    print(f"synchronized groups: {len(synchronized_groups)}")
    return synchronized_groups

def _to_uint8_image(img):
    if img is None:
        return None

    arr = np.asarray(img)
    if arr.size == 0:
        return None

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    if not np.issubdtype(arr.dtype, np.number):
        return None

    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return None

    min_val = float(arr[finite].min())
    max_val = float(arr[finite].max())
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = np.nan_to_num(arr, nan=min_val, posinf=max_val, neginf=min_val)
    arr = (arr - min_val) * (255.0 / (max_val - min_val))
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))

def _normalize_gray(gray):
    gray_u8 = _to_uint8_image(gray)
    if gray_u8 is None:
        return None

    if gray_u8.ndim == 2:
        return gray_u8
    if gray_u8.ndim == 3 and gray_u8.shape[2] == 1:
        return np.ascontiguousarray(gray_u8[:, :, 0])
    if gray_u8.ndim == 3 and gray_u8.shape[2] == 3:
        return cv2.cvtColor(gray_u8, cv2.COLOR_BGR2GRAY)
    if gray_u8.ndim == 3 and gray_u8.shape[2] == 4:
        return cv2.cvtColor(gray_u8, cv2.COLOR_BGRA2GRAY)
    return None


def _normalize_bgr(bgr):
    bgr_u8 = _to_uint8_image(bgr)
    if bgr_u8 is None:
        return None

    if bgr_u8.ndim == 2:
        return cv2.cvtColor(bgr_u8, cv2.COLOR_GRAY2BGR)
    if bgr_u8.ndim == 3 and bgr_u8.shape[2] == 1:
        return cv2.cvtColor(bgr_u8[:, :, 0], cv2.COLOR_GRAY2BGR)
    if bgr_u8.ndim == 3 and bgr_u8.shape[2] == 3:
        return np.ascontiguousarray(bgr_u8)
    if bgr_u8.ndim == 3 and bgr_u8.shape[2] == 4:
        return cv2.cvtColor(bgr_u8, cv2.COLOR_BGRA2BGR)
    return None


def preprocess_gray_variants(gray):
    """
    生成若干灰度预处理版本，提高棋盘检测成功率。
    返回:
        [(processed_img, name), ...]
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return [
        (gray, "gray"),
        (clahe, "clahe"),
        (cv2.GaussianBlur(gray, (3, 3), 0), "gauss3"),
        (cv2.GaussianBlur(clahe, (3, 3), 0), "clahe_gauss3"),
    ]


def _refine_corners(gray, corners, method):
    """
    对检测到的角点做亚像素优化。
    不同检测方法使用不同窗口大小。
    """
    win = (3, 3) if method == "SB" else (3, 3)
    corners = corners.astype(np.float32)
    refined = cv2.cornerSubPix(gray, corners, win, (-1, -1), CRIT)
    return refined.astype(np.float32)


def detect_chessboard_fullres(gray, bgr=None, rows=ROWS, cols=COLS, enable_color_order=True, class_method_flag=True):
    """
    检测棋盘内角点，并可选地根据颜色标签统一角点顺序。

    Args:
        gray: 灰度图
        bgr: 彩色图，仅在 enable_color_order=True 时使用
        rows, cols: 棋盘内角点数
        enable_color_order: 是否启用颜色检测并重排角点顺序

    Returns:
        成功: (True, corners, info)
        失败: (False, None, None)
    """
    pattern = (cols, rows)
    expected_n = rows * cols
    gray = _normalize_gray(gray)
    bgr = _normalize_bgr(bgr) if bgr is not None else None
    if gray is None:
        return False, None, None

    sb_flag_candidates = [
        cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        0,
    ]

    classic_flag_candidates = [
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        0,
    ]

    for img_in, prep_name in preprocess_gray_variants(gray):
        # 1) 优先使用更稳的 SB 方法
        for flags in sb_flag_candidates:
            try:
                ok, corners = cv2.findChessboardCornersSB(img_in, pattern, flags=flags)
            except cv2.error:
                continue
            if ok and corners is not None and corners.shape[0] == expected_n:
                refined = _refine_corners(gray, corners, method="SB")

                if enable_color_order and bgr is not None:
                    ok_order, corners_ordered, _ = check_corners_order_minimal(
                        bgr, refined, rows, cols
                    )
                    if ok_order:
                        refined = corners_ordered

                return True, refined, {
                    "prep": prep_name,
                    "flags": int(flags),
                    "method": "SB",
                }

        # # 2) SB 失败后，回退到 classic 方法
        if class_method_flag:
            for flags in classic_flag_candidates:
                try:
                    ok, corners = cv2.findChessboardCorners(img_in, pattern, flags=flags)
                except cv2.error:
                    continue
                if ok and corners is not None and corners.shape[0] == expected_n:
                    refined = _refine_corners(gray, corners, method="classic")

                    if enable_color_order and bgr is not None:
                        ok_order, corners_ordered, _ = check_corners_order_minimal(
                            bgr, refined, rows, cols
                        )
                        if ok_order:
                            refined = corners_ordered

                    return True, refined, {
                        "prep": prep_name,
                        "flags": int(flags),
                        "method": "classic",
                    }

    return False, None, None


def make_rotations_and_flips(grid):
    """
    枚举角点网格的候选朝向。
    当前只保留 4 种旋转，不做翻转。
    """
    return [np.rot90(grid, k=k, axes=(0, 1)).copy() for k in range(4)]


def _find_color_centers_near_board_corners(bgr, grid, rows, cols):
    """
    在棋盘四角附近搜索 green / yellow / red 三种颜色中心。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    # 当前角点网格四个角
    board_corners = {
        "TL": grid[0, 0],
        "TR": grid[0, -1],
        "BR": grid[-1, -1],
        "BL": grid[-1, 0],
    }

    # 用边界相邻角点距离估计单格大小，从而决定颜色搜索半径
    d_list = []
    for j in range(cols - 1):
        d_list.append(np.linalg.norm(grid[0, j + 1] - grid[0, j]))
        d_list.append(np.linalg.norm(grid[-1, j + 1] - grid[-1, j]))
    for i in range(rows - 1):
        d_list.append(np.linalg.norm(grid[i + 1, 0] - grid[i, 0]))
        d_list.append(np.linalg.norm(grid[i + 1, -1] - grid[i, -1]))

    cell = float(np.median(d_list)) if d_list else 20.0
    roi_r = int(np.clip(cell * 6.0, 40, 220))

    color_ranges = {
        "red": [((0, 90, 50), (10, 255, 255)), ((170, 90, 50), (179, 255, 255))],
        "yellow": [((16, 70, 70), (42, 255, 255))],
        "green": [((35, 20, 120), (95, 180, 255))],
        "blue": [((90, 70, 50), (130, 255, 255))],
    }

    candidates = {"green": [], "yellow": [], "red": [], "blue": []}

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 在四个棋盘角附近分别搜索颜色块
    for pt in board_corners.values():
        cx0, cy0 = int(pt[0]), int(pt[1])

        x0 = max(0, cx0 - roi_r)
        x1 = min(w, cx0 + roi_r)
        y0 = max(0, cy0 - roi_r)
        y1 = min(h, cy0 + roi_r)

        if x1 - x0 < 10 or y1 - y0 < 10:
            continue

        hsv_roi = hsv[y0:y1, x0:x1]
        roi_center = np.array([cx0, cy0], dtype=np.float32)

        for color, hs_ranges in color_ranges.items():
            mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            for lo, hi in hs_ranges:
                mask |= cv2.inRange(
                    hsv_roi,
                    np.array(lo, np.uint8),
                    np.array(hi, np.uint8)
                )

            # 简单去噪
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = float(cv2.contourArea(c))
                if area < 8:
                    continue

                m = cv2.moments(c)
                if m["m00"] == 0:
                    continue

                cx = (m["m10"] / m["m00"]) + x0
                cy = (m["m01"] / m["m00"]) + y0
                cand = np.array([cx, cy], dtype=np.float32)

                # 分数越小越好：越靠近角点越好，面积越大越好
                score = np.linalg.norm(cand - roi_center) - 0.1 * area
                candidates[color].append((score, cand))

    centers = {}
    for color in ("green", "yellow", "red", "blue"):
        if not candidates[color]:
            centers[color] = None
        else:
            candidates[color].sort(key=lambda x: x[0])
            centers[color] = candidates[color][0][1]

    return centers


def check_corners_order_minimal(bgr, corners, rows, cols):
    """
    用颜色标签统一 corners 顺序：
        green  -> TL
        yellow -> TR
        red    -> BR
        blue   -> BL

    Returns:
        ok_order, corners_ordered, color_centers
    """
    if corners is None or len(corners) != rows * cols:
        return False, corners, {}

    grid = np.asarray(corners, dtype=np.float32).reshape(rows, cols, 1, 2)[:, :, 0, :]
    color_centers = _find_color_centers_near_board_corners(bgr, grid, rows, cols)

    required_colors = ("green", "yellow", "red", "blue")
    missing = [c for c in required_colors if color_centers[c] is None]
    if len(missing) > 1:
        return False, corners, color_centers

    best_grid = None
    best_score = float("inf")

    for g in make_rotations_and_flips(grid):
        tl = g[0, 0]
        tr = g[0, -1]
        bl = g[-1, 0]
        br = g[-1, -1]


        expected = {
            "green": tl,
            "yellow": tr,
            "red": br,
            "blue": bl,
        }

        score = 0.0
        for color, corner_pt in expected.items():
            if color_centers[color] is not None:
                score += np.linalg.norm(color_centers[color] - corner_pt)

        if score < best_score:
            best_score = score
            best_grid = g

    if best_grid is None:
        return False, corners, color_centers

    corners_ordered = best_grid.reshape(rows * cols, 1, 2).astype(np.float32)
    return True, corners_ordered, color_centers

def get_gt_data(path: Path|str) -> np.ndarray:
    with open(path, 'rb') as ff:
        gt = pickle.load(ff)
    return gt