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


def get_gt_data(path: Path|str) -> np.ndarray:
    with open(path, 'rb') as ff:
        gt = pickle.load(ff)
    has_nan = np.isnan(gt).any(axis=(1, 2))  # 形状 (a,)
    gt = gt[~has_nan]  # 形状 (a-1, b, c)
    return gt