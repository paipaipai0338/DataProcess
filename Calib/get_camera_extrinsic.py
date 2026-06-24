import random
from pathlib import Path
import csv
from collections import defaultdict
import argparse
import sys
import re
import numpy as np
import cv2
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Img2Keypoint.utils import load_intrinsic_data
from Calib.chessboard_detection_helper import (
    detect_chessboard_fullres,
    check_corners_order_minimal,
)

ROWS, COLS = 24, 24
SQUARE = 0.04

DRAW_COLORS = {
    "green": (144, 238, 144),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
}


def natural_group_sort_key(group_name):
    match = re.search(r"(\d+)$", str(group_name))
    if match:
        return int(match.group(1))
    return str(group_name)

def parse_args():
    parser = argparse.ArgumentParser(description="根据多相机棋盘格图像估计相机外参。")
    parser.add_argument('--data_path', type=Path, default=Path(r'G:\20260615\data_collection'), help="外参标定训练数据目录，默认使用 root_path/chessboard/extrinsic/train。")
    parser.add_argument('--save_path', type=Path, default=Path(r'G:\20260615\calib'), help="标定结果保存目录，同时用于读取已有内参文件，默认使用 root_path/calib。")
    parser.add_argument('--reference_cam', type=str, default="A", help="外参统一到的参考相机编号。")
    parser.add_argument('--selected_groups', type=int, default=4, help="只使用前 N 个 group 进行外参估计；不填则使用全部 group。")
    parser.add_argument('--target_valid_frames_per_group_cam', type=int, default=3, help="每个 group、每个相机最多使用的有效棋盘格帧数。")
    parser.add_argument('--reproj_rmse_threshold', type=float, default=1.2, help="单帧棋盘格 PnP 重投影 RMSE 阈值，超过则剔除。")
    args = parser.parse_args()
    return args

# ============================================================
# Geometry helpers
# ============================================================
def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def T_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def T_to_rtvec(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)
    return rvec.reshape(3), tvec.reshape(3)


def rtvec_to_T(rt6):
    rvec = np.asarray(rt6[:3], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(rt6[3:6], dtype=np.float64).reshape(3, 1)
    return rvec_tvec_to_T(rvec, tvec)


def rot_err_deg(Ra, Rb):
    R = Ra @ Rb.T
    cosang = (np.trace(R) - 1.0) / 2.0
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def trans_err_m(Ta, Tb):
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))


def reproj_rmse_single_cam(objp_, imgpts_, rvec_, tvec_, K_, dist_):
    proj, _ = cv2.projectPoints(objp_, rvec_, tvec_, K_, dist_)
    proj = proj.reshape(-1, 2)
    obs = np.asarray(imgpts_, dtype=np.float64).reshape(-1, 2)
    e = proj - obs
    return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))


def project_rmse_from_T(objp_, imgpts_, T_cb, K_, dist_):
    R = T_cb[:3, :3]
    t = T_cb[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(objp_, rvec, t, K_, dist_)
    proj = proj.reshape(-1, 2)
    obs = np.asarray(imgpts_, dtype=np.float64).reshape(-1, 2)
    e = proj - obs
    rmse = float(np.sqrt(np.mean(np.sum(e * e, axis=1))))
    return rmse, proj


# ============================================================
# Fusion helpers (仅用于初始化)
# ============================================================
def fuse_translations(ts):
    ts = np.asarray(ts, dtype=np.float64)
    return np.median(ts, axis=0)


def fuse_rotations_so3(Rs, max_iter=50, eps=1e-12):
    if len(Rs) == 1:
        return Rs[0].copy()

    R_mean = Rs[0].copy()
    for _ in range(max_iter):
        Rt = R_mean.T
        deltas = []
        for Rk in Rs:
            rv, _ = cv2.Rodrigues(Rt @ Rk)
            deltas.append(rv.reshape(3))
        delta = np.mean(np.stack(deltas, axis=0), axis=0).reshape(3, 1)
        if float(np.linalg.norm(delta)) < eps:
            break
        dR, _ = cv2.Rodrigues(delta)
        R_mean = R_mean @ dR
    return R_mean


def fuse_Ts_so3(T_list):
    if len(T_list) == 0:
        return None
    if len(T_list) == 1:
        return T_list[0].copy()

    translations = np.stack([T[:3, 3] for T in T_list], axis=0)
    t_fused = fuse_translations(translations)
    Rs = [T[:3, :3] for T in T_list]
    R_fused = fuse_rotations_so3(Rs)

    T_fused = np.eye(4, dtype=np.float64)
    T_fused[:3, :3] = R_fused
    T_fused[:3, 3] = t_fused
    return T_fused


# ============================================================
# Detection
# ============================================================
def detect_and_estimate_pose(
    img_bgr,
    gray,
    K,
    dist,
    objp,
    rows=ROWS,
    cols=COLS,
):
    ok, corners, detect_meta = detect_chessboard_fullres(
        gray,
        bgr=img_bgr,
        rows=rows,
        cols=cols,
        enable_color_order=True,
        class_method_flag=True,
    )
    if not ok or corners is None:
        return {"status": "detect_fail", "detect_meta": detect_meta}

    ok_order, ordered, color_marks = check_corners_order_minimal(
        img_bgr, corners, rows=rows, cols=cols
    )
    if not ok_order:
        return {
            "status": "color_reject",
            "corners": None,
            "color_marks": color_marks,
            "detect_meta": detect_meta,
        }

    ok2, rvec, tvec = cv2.solvePnP(objp, ordered, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok2:
        return {
            "status": "pnp_fail",
            "corners": ordered,
            "color_marks": color_marks,
            "detect_meta": detect_meta,
        }

    rmse = reproj_rmse_single_cam(objp, ordered, rvec, tvec, K, dist)
    return {
        "status": "ok",
        "corners": ordered,
        "color_marks": color_marks,
        "detect_meta": detect_meta,
        "rvec": rvec,
        "tvec": tvec,
        "T_cb": rvec_tvec_to_T(rvec, tvec),   # board -> cam
        "rmse": rmse,
    }


# ============================================================
# Visualization
# ============================================================
def draw_corners_overlay_bgr(img_bgr, corners, title_text="", color_marks=None):
    vis = img_bgr.copy()

    if corners is not None:
        pts = np.asarray(corners).reshape(-1, 2).astype(int)
        for p in pts:
            cv2.circle(vis, tuple(p), 2, (0, 255, 0), -1)

        if pts.shape[0] > 0:
            p0 = tuple(pts[0])
            plast = tuple(pts[-1])
            cv2.circle(vis, p0, 4, (0, 0, 255), -1)
            cv2.circle(vis, plast, 4, (255, 0, 0), -1)
            cv2.line(vis, p0, plast, (0, 255, 255), 1)

    if color_marks is not None:
        for cname, cpt in color_marks.items():
            if cpt is None or cname not in DRAW_COLORS:
                continue
            x, y = int(cpt[0]), int(cpt[1])
            color = DRAW_COLORS[cname]
            cv2.circle(vis, (x, y), 16, color, 2)
            cv2.circle(vis, (x, y), 4, color, -1)
            cv2.putText(vis, cname, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if title_text:
        cv2.putText(vis, title_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    return vis


# ============================================================
# Joint optimization helpers
# ============================================================
def build_joint_initial_guess(cams, groups, group_ref_board_init, cam_to_ref_init):
    """
    参数顺序：
      [cam1(6), cam2(6), ..., group1(6), group2(6), ...]
    """
    x0 = []
    cam_index = {}
    group_index = {}

    offset = 0
    for cam_name in cams:
        T = cam_to_ref_init[cam_name]
        rv, tv = T_to_rtvec(T)
        x0.extend(rv.tolist() + tv.tolist())
        cam_index[cam_name] = offset
        offset += 6

    for group_name in groups:
        T = group_ref_board_init[group_name]   # board -> ref
        rv, tv = T_to_rtvec(T)
        x0.extend(rv.tolist() + tv.tolist())
        group_index[group_name] = offset
        offset += 6

    return np.asarray(x0, dtype=np.float64), cam_index, group_index


def unpack_joint_params(x, cams, groups, cam_index, group_index):
    cam_to_ref = {}
    board_to_ref = {}

    for cam_name in cams:
        off = cam_index[cam_name]
        cam_to_ref[cam_name] = rtvec_to_T(x[off:off + 6])

    for group_name in groups:
        off = group_index[group_name]
        board_to_ref[group_name] = rtvec_to_T(x[off:off + 6])

    return cam_to_ref, board_to_ref


def joint_residuals(
    x,
    cams,
    groups,
    cam_index,
    group_index,
    frame_records,
    intrinsic_cache,
    objp,
    reference_cam,
):
    cam_to_ref, board_to_ref = unpack_joint_params(
        x, cams, groups, cam_index, group_index
    )

    residuals = []

    # 遍历所有有效观测
    for (group_name, cam_name), recs in frame_records.items():
        if group_name not in groups:
            continue

        K, dist = intrinsic_cache[cam_name]

        if cam_name == reference_cam:
            T_board_cam = board_to_ref[group_name]   # board -> ref
        else:
            if cam_name not in cam_to_ref:
                continue
            T_board_cam = T_inv(cam_to_ref[cam_name]) @ board_to_ref[group_name]

        R = T_board_cam[:3, :3]
        t = T_board_cam[:3, 3].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)

        for rec in recs:
            obs = np.asarray(rec["corners"], dtype=np.float64).reshape(-1, 2)
            proj, _ = cv2.projectPoints(objp, rvec, t, K, dist)
            proj = proj.reshape(-1, 2)
            e = (proj - obs).reshape(-1)
            residuals.extend(e.tolist())

    return np.asarray(residuals, dtype=np.float64)


def compute_joint_frame_rmse(
    group_name,
    cam_name,
    rec,
    cam_to_ref_opt,
    board_to_ref_opt,
    intrinsic_cache,
    objp,
    reference_cam,
):
    K, dist = intrinsic_cache[cam_name]

    if cam_name == reference_cam:
        T_board_cam = board_to_ref_opt[group_name]
    else:
        T_board_cam = T_inv(cam_to_ref_opt[cam_name]) @ board_to_ref_opt[group_name]

    rmse, _ = project_rmse_from_T(objp, rec["corners"], T_board_cam, K, dist)
    return rmse, T_board_cam


# ============================================================
# Main pipeline
# ============================================================
def get_camera_extrinsic(
    data_path,
    save_path,
    selected_groups=None,
    reference_cam="A",
    target_valid_frames_per_group_cam=5,
    reproj_rmse_threshold=1.2,
):
    data_path = Path(data_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
    objp *= SQUARE

    group_dirs = sorted(
        [
            p for p in data_path.iterdir()
            if p.is_dir()
            and p.name.startswith("group_")
            and (p / "camera").is_dir()
        ],
        key=lambda x: natural_group_sort_key(x.name)
    )
    group_dirs = group_dirs[:selected_groups] if selected_groups else group_dirs[:5]
    if not group_dirs:
        print(f"[ERROR] No group directories found under: {data_path}")
        return


    intrinsic_cache = {}

    def get_intrinsic(cam_name):
        if cam_name not in intrinsic_cache:
            intrinsic_cache[cam_name] = load_intrinsic_data(
                save_path / f"intrinsic_cam_{cam_name}.npz"
            )
        return intrinsic_cache[cam_name]

    # (group, cam) -> list[frame_record]
    frame_records = defaultdict(list)

    # ------------------------------------------------------------
    # Step 1. 单帧检测 + 单帧筛选
    # ------------------------------------------------------------
    for gpath in group_dirs:
        group_name = gpath.name
        gpath = Path(gpath) / 'camera'
        cam_dirs = sorted([p for p in gpath.iterdir() if p.is_dir() and p.name.startswith("cam_")])
        print(f"\n=== Processing group {group_name} ===")

        for cpath in cam_dirs:

            cam_name = cpath.name.split("_", 1)[1]
            print('Cam:{}'.format(cam_name))
            frames_path = cpath / "frames"
            imgs = sorted([
                p for p in frames_path.iterdir()
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            ]) if frames_path.exists() else []
            if not imgs:
                continue
            random.shuffle(imgs)

            K, dist = get_intrinsic(cam_name)
            ok_count = 0
            status_counts = defaultdict(int)

            for idx, img_path in enumerate(imgs):
                if ok_count >= target_valid_frames_per_group_cam:
                    break

                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img_bgr is None or gray is None:
                    status_counts["read_fail"] += 1
                    print("idx:{} / valid:{} / status:read_fail / file:{}".format(idx, ok_count, img_path.name))
                    continue

                result = detect_and_estimate_pose(
                    img_bgr=img_bgr,
                    gray=gray,
                    K=K,
                    dist=dist,
                    objp=objp,
                    rows=ROWS,
                    cols=COLS,
                )
                if result["status"] != "ok":
                    status_counts[result["status"]] += 1
                    print("idx:{} / valid:{} / status:{} / file:{}".format(
                        idx, ok_count, result["status"], img_path.name
                    ))
                    continue

                if result["rmse"] > reproj_rmse_threshold:
                    status_counts["rmse_reject"] += 1
                    print("idx:{} / valid:{} / status:rmse_reject / rmse:{:.3f} / file:{}".format(
                        idx, ok_count, float(result["rmse"]), img_path.name
                    ))
                    continue

                frame_name = Path(img_path).stem
                rec = {
                    "group": group_name,
                    "cam": cam_name,
                    "imgpath": img_path,
                    "frame_name": frame_name,
                    "T_cb": result["T_cb"],                       # board -> cam
                    "corners": result["corners"].reshape(-1, 2), # 颜色矫正后的角点
                    "color_marks": result["color_marks"],
                    "single_reproj_rmse_px": float(result["rmse"]),
                }
                frame_records[(group_name, cam_name)].append(rec)
                ok_count += 1
                status_counts["ok"] += 1
                meta = result.get("detect_meta") or {}
                print("idx:{} / valid:{} / status:ok / rmse:{:.3f} / method:{} / prep:{} / scale:{} / file:{}".format(
                    idx,
                    ok_count,
                    float(result["rmse"]),
                    meta.get("method", ""),
                    meta.get("prep", ""),
                    meta.get("scale", ""),
                    img_path.name,
                ))

            if status_counts:
                summary = ", ".join(
                    "{}:{}".format(k, status_counts[k])
                    for k in sorted(status_counts.keys())
                )
                print("[Detect summary] group {} cam {} -> {}".format(group_name, cam_name, summary))

    all_groups = sorted({g for (g, _) in frame_records.keys()}, key=natural_group_sort_key)
    all_cam_names = sorted({cam for (_, cam) in frame_records.keys()})
    non_ref_cams = [c for c in all_cam_names if c != reference_cam]

    if not non_ref_cams:
        print(f"[ERROR] No non-reference cameras besides {reference_cam}")
        return

    # 只保留 reference cam 存在的 group
    valid_groups = []
    for g in all_groups:
        if len(frame_records.get((g, reference_cam), [])) > 0:
            valid_groups.append(g)
        else:
            print(f"[WARN] group {g}: reference cam {reference_cam} missing")

    if len(valid_groups) == 0:
        print("[ERROR] No valid groups with reference camera.")
        return

    # ------------------------------------------------------------
    # Step 2. 组内融合，生成联合优化初值
    # ------------------------------------------------------------
    group_ref_board_init = {}            # group -> fused board->ref
    group_cam_board_init = {}            # (group, cam) -> fused board->cam
    per_group_cam_to_ref = defaultdict(list)

    for group_name in valid_groups:
        ref_records = frame_records[(group_name, reference_cam)]
        T_ref_board_group = fuse_Ts_so3([r["T_cb"] for r in ref_records])
        if T_ref_board_group is None:
            continue
        group_ref_board_init[group_name] = T_ref_board_group

        for cam_name in non_ref_cams:
            cam_records = frame_records.get((group_name, cam_name), [])
            if len(cam_records) == 0:
                continue

            T_cam_board_group = fuse_Ts_so3([r["T_cb"] for r in cam_records])
            if T_cam_board_group is None:
                continue

            group_cam_board_init[(group_name, cam_name)] = T_cam_board_group

            T_cam_to_ref_group = T_ref_board_group @ T_inv(T_cam_board_group)
            per_group_cam_to_ref[cam_name].append(T_cam_to_ref_group)

            print(f"[Group {group_name}] cam {cam_name} -> {reference_cam}:")
            print(T_cam_to_ref_group)

    # 每台非参考相机的初值：跨 group 粗融合
    cam_to_ref_init = {}
    usable_cams = []
    for cam_name in non_ref_cams:
        Ts = per_group_cam_to_ref.get(cam_name, [])
        if len(Ts) == 0:
            print(f"[WARN] cam {cam_name}: no initial relative pose")
            continue
        cam_to_ref_init[cam_name] = fuse_Ts_so3(Ts)
        usable_cams.append(cam_name)

    if len(usable_cams) == 0:
        print("[ERROR] No usable non-reference cameras.")
        return

    valid_groups = [g for g in valid_groups if g in group_ref_board_init]
    if len(valid_groups) == 0:
        print("[ERROR] No valid groups with initial board pose.")
        return

    # ------------------------------------------------------------
    # Step 3. 联合重投影优化
    # ------------------------------------------------------------
    x0, cam_index, group_index = build_joint_initial_guess(
        cams=usable_cams,
        groups=valid_groups,
        group_ref_board_init=group_ref_board_init,
        cam_to_ref_init=cam_to_ref_init,
    )

    print("\n[INFO] Start joint bundle-like reprojection optimization ...")
    opt = least_squares(
        fun=joint_residuals,
        x0=x0,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        verbose=2,
        max_nfev=200,
        args=(
            usable_cams,
            valid_groups,
            cam_index,
            group_index,
            frame_records,
            intrinsic_cache,
            objp,
            reference_cam,
        ),
    )

    print(f"[INFO] Optimization success: {opt.success}, status={opt.status}")
    print(f"[INFO] Optimization message: {opt.message}")

    cam_to_ref_opt, board_to_ref_opt = unpack_joint_params(
        opt.x, usable_cams, valid_groups, cam_index, group_index
    )

    # ------------------------------------------------------------
    # Step 4. 保存最终外参
    # ------------------------------------------------------------
    for cam_name in usable_cams:
        T_opt = cam_to_ref_opt[cam_name]
        np.save(
            save_path / f"extrinsic_T_cam_{cam_name}_to_cam_{reference_cam}.npy",
            T_opt,
        )

        for group_name in valid_groups:
            T_ref_board_group_init = group_ref_board_init[group_name]
            T_ref_board_group_opt = board_to_ref_opt[group_name]

            for cam_name in usable_cams:
                recs = frame_records.get((group_name, cam_name), [])
                if len(recs) == 0:
                    continue
                if (group_name, cam_name) not in group_cam_board_init:
                    continue

                T_group = T_ref_board_group_init @ T_inv(group_cam_board_init[(group_name, cam_name)])
                T_joint = cam_to_ref_opt[cam_name]

                K, dist = intrinsic_cache[cam_name]

                # 当前组初始化重投影
                T_cam_board_pred_group = T_inv(T_group) @ T_ref_board_group_init

                for rec in recs:
                    T_single = T_ref_board_group_init @ T_inv(rec["T_cb"])

                    rot_err_group = rot_err_deg(T_group[:3, :3], T_single[:3, :3])
                    trans_err_group = trans_err_m(T_group, T_single)

                    rot_err_joint = rot_err_deg(T_joint[:3, :3], T_single[:3, :3])
                    trans_err_joint = trans_err_m(T_joint, T_single)

                    group_reproj_rmse_px, _ = project_rmse_from_T(
                        objp, rec["corners"], T_cam_board_pred_group, K, dist
                    )

                    joint_reproj_rmse_px, _ = compute_joint_frame_rmse(
                        group_name=group_name,
                        cam_name=cam_name,
                        rec=rec,
                        cam_to_ref_opt=cam_to_ref_opt,
                        board_to_ref_opt=board_to_ref_opt,
                        intrinsic_cache=intrinsic_cache,
                        objp=objp,
                        reference_cam=reference_cam,
                    )

    print(f"[INFO] extrinsic npy saved to: {save_path}")


def load_extrinsic_data(file_full_path):
    """读取保存的 4x4 外参矩阵。"""
    T = np.load(file_full_path)
    R = T[:3, :3]
    t = T[:3, 3]
    return T, R, t


if __name__ == "__main__":
    args = parse_args()
    get_camera_extrinsic(
        data_path=args.data_path,
        save_path=args.save_path,
        selected_groups=args.selected_groups,
        reference_cam=args.reference_cam,
        target_valid_frames_per_group_cam=args.target_valid_frames_per_group_cam,
        reproj_rmse_threshold=args.reproj_rmse_threshold,
    )
