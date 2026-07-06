import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy import signal



COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

DIRECTED_BONES = [
    (11, 12),
    (11, 5),
    (12, 6),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 0),
    (0, 1),
    (1, 3),
    (0, 2),
    (2, 4),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=Path, default=Path(r"C:\Users\Administrator\Desktop\20260702\data_collection\group_036"), help="本次采集或标定的数据根目录。其余路径默认由该目录自动派生。")
    parser.add_argument("--input_path", type=Path, default=None, help="原始 3D 关键点结果输入目录，默认使用 root_path/results/3D/group。")
    parser.add_argument("--output_path", type=Path, default=None, help="时序平滑后 3D 关键点结果的输出目录，默认使用 root_path/results/smoothed 3D/group。")
    parser.add_argument("--smooth_alpha", type=float, default=0.35, help="指数平滑系数，用于控制新观测与历史轨迹的融合比例。")
    parser.add_argument("--max_match_distance", type=float, default=0.50, help="跨帧轨迹匹配允许的最大中心点距离。")
    parser.add_argument("--max_missing", type=int, default=8, help="轨迹在被删除前允许连续缺失的最大帧数。")
    parser.add_argument("--min_valid_joints", type=int, default=5, help="判定单个人体姿态有效所需的最少有效关节点数量。")
    parser.add_argument("--max_tracks", type=int, default=0, help="最多保留的轨迹数量，0 表示不限制。")
    parser.add_argument("--max_interp_gap", type=int, default=8, help="允许进行缺失插值的最大连续帧间隔。")
    parser.add_argument("--medfilt_kernel", type=int, default=7, help="中值滤波窗口大小，用于抑制关键点时序抖动。")
    parser.add_argument("--velocity_threshold", type=float, default=0.35, help="速度异常检测阈值，用于识别并修正突变点。")
    parser.add_argument("--bone_length_weight", type=float, default=0.65, help="骨长约束修正的权重，用于保持人体骨架长度稳定。")
    parser.add_argument("--bone_iters", type=int, default=2, help="骨长约束修正的迭代次数。")
    parser.add_argument("--show", action="store_true", default=True, help="是否显示平滑前后对比可视化窗口。")
    parser.add_argument("--save_plot", action="store_true", help="是否保存平滑前后对比图。")
    parser.add_argument("--pause", type=float, default=0.01, help="可视化播放时每帧暂停时间，单位为秒。")
    parser.add_argument("--process_all_groups", type=lambda v: str(v).lower() in ("1", "true", "yes", "y"), default=False, help="是否处理输入目录下推断出的所有分组。")
    parser.add_argument("--skip_existing", type=lambda v: str(v).lower() in ("1", "true", "yes", "y"), default=False, help="输出文件已存在时是否跳过对应帧。")
    args = parser.parse_args()
    if args.root_path is None and any(p is None for p in (args.input_path, args.output_path)):
        parser.error("请指定 --root_path，或同时指定 --input_path 和 --output_path。")
    args.input_path = args.input_path or args.root_path / "camera results" / "3D"
    args.output_path = args.output_path or args.root_path / "camera results" / "smoothed 3D"
    return args


@dataclass
class TrackState:
    track_id: int
    pose: np.ndarray
    center: np.ndarray
    velocity: np.ndarray
    missing_count: int = 0
    age: int = 0


def normalize_frame_array(data):
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 17, 3), dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (17, 3):
        return arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (17, 3):
        raise ValueError(f"Unexpected frame shape: {arr.shape}")
    return arr


def load_sequence(input_path):
    input_path = Path(input_path)
    files = sorted([p for p in input_path.iterdir() if p.suffix.lower() == ".pkl"])
    frames = []
    for path in files:
        with open(path, "rb") as f:
            frames.append(normalize_frame_array(pickle.load(f)))
    return files, frames

def infer_group_jobs(input_path, output_path):
    """
    Infer group folders from input_path.

    Supports either:
    - input_path = .../results/3D
    - input_path = .../results/3D/4
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.parent / "smoothed"

    if any(input_path.glob("*.pkl")):
        input_base = input_path.parent
    else:
        input_base = input_path

    if output_path.name.isdigit():
        output_base = output_path.parent
    else:
        output_base = output_path

    group_dirs = [
        p for p in input_base.iterdir()
        if p.is_dir() and any(p.glob("*.pkl"))
    ]
    group_dirs.sort(key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name))

    return [
        (p.name, p, output_base / p.name)
        for p in group_dirs
    ]

def output_files_complete(file_paths, output_path):
    output_path = Path(output_path)
    if len(file_paths) == 0:
        return False
    return all((output_path / src_path.name).exists() for src_path in file_paths)


def save_sequence(output_path, file_paths, frames, skip_existing=False):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    for src_path, frame in zip(file_paths, frames):
        out_path = output_path / src_path.name
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        with open(out_path, "wb") as f:
            pickle.dump(frame, f)
        saved += 1
    return saved, skipped


def valid_joint_mask(pose):
    return np.all(np.isfinite(pose), axis=1)


def pose_center(pose):
    mask = valid_joint_mask(pose)
    if not np.any(mask):
        return None
    return np.median(pose[mask], axis=0)


def count_valid_joints(pose):
    return int(np.sum(valid_joint_mask(pose)))


def pose_distance(a, b):
    mask = valid_joint_mask(a) & valid_joint_mask(b)
    if np.sum(mask) == 0:
        ca = pose_center(a)
        cb = pose_center(b)
        if ca is None or cb is None:
            return np.inf
        return float(np.linalg.norm(ca - cb))
    diffs = a[mask] - b[mask]
    return float(np.median(np.linalg.norm(diffs, axis=1)))


def greedy_assignment(cost_matrix, max_cost):
    assignments = []
    if cost_matrix.size == 0:
        return assignments

    used_rows = set()
    used_cols = set()
    for flat_idx in np.argsort(cost_matrix, axis=None):
        row, col = np.unravel_index(flat_idx, cost_matrix.shape)
        cost = cost_matrix[row, col]
        if not np.isfinite(cost) or cost > max_cost:
            break
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        assignments.append((row, col))
    return assignments


def assign_detections(tracks, detections, max_match_distance):
    if len(tracks) == 0 or len(detections) == 0:
        return []

    cost_matrix = np.full((len(tracks), len(detections)), np.inf, dtype=np.float64)
    for i, track in enumerate(tracks):
        predicted_pose = track.pose.copy()
        pred_mask = valid_joint_mask(predicted_pose)
        if np.any(pred_mask):
            predicted_pose[pred_mask] = predicted_pose[pred_mask] + track.velocity

        for j, det in enumerate(detections):
            center = pose_center(det)
            if center is None:
                continue
            center_cost = float(np.linalg.norm((track.center + track.velocity) - center))
            pose_cost = pose_distance(predicted_pose, det)
            cost_matrix[i, j] = 0.6 * center_cost + 0.4 * pose_cost

    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost_matrix)
        assignments = []
        for row, col in zip(rows, cols):
            cost = cost_matrix[row, col]
            if np.isfinite(cost) and cost <= max_match_distance:
                assignments.append((row, col))
        return assignments

    return greedy_assignment(cost_matrix, max_match_distance)


def create_track(track_id, pose):
    center = pose_center(pose)
    return TrackState(
        track_id=track_id,
        pose=pose.copy(),
        center=center.copy(),
        velocity=np.zeros(3, dtype=np.float64),
        missing_count=0,
        age=1,
    )


def get_next_track_id(active_tracks, max_tracks):
    used_ids = {track.track_id for track in active_tracks}
    for track_id in range(max_tracks):
        if track_id not in used_ids:
            return track_id
    return None


def build_tracked_tensor(frames, max_match_distance, max_missing, min_valid_joints, max_tracks):
    if max_tracks <= 0:
        max_tracks = max((frame.shape[0] for frame in frames), default=0)

    tracked = np.full((len(frames), max_tracks, 17, 3), np.nan, dtype=np.float64)
    active_tracks = []

    for frame_idx, frame in enumerate(frames):
        detections = [pose for pose in frame if count_valid_joints(pose) >= min_valid_joints]
        assignments = assign_detections(active_tracks, detections, max_match_distance)
        assigned_rows = {row for row, _ in assignments}
        assigned_cols = {col for _, col in assignments}
        new_active = []

        for row, track in enumerate(active_tracks):
            if row not in assigned_rows:
                track.missing_count += 1
                if track.missing_count <= max_missing:
                    new_active.append(track)
                continue

            det = detections[next(col for r, col in assignments if r == row)]
            prev_center = track.center.copy()
            center = pose_center(det)
            track.velocity = center - prev_center if center is not None else np.zeros(3, dtype=np.float64)
            track.center = center if center is not None else prev_center
            track.pose = det.copy()
            track.missing_count = 0
            track.age += 1
            new_active.append(track)

        active_tracks = new_active

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_cols:
                continue
            track_id = get_next_track_id(active_tracks, max_tracks)
            if track_id is None:
                break
            active_tracks.append(create_track(track_id, det))

        for track in active_tracks:
            if track.missing_count == 0:
                tracked[frame_idx, track.track_id] = track.pose

    return tracked


def nan_helper(values):
    return np.isnan(values), lambda z: z.nonzero()[0]


def interpolate_1d(values, max_gap):
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    nans, idx = nan_helper(out)
    if np.all(nans):
        return out

    valid_idx = idx(~nans)
    out[nans] = np.interp(idx(nans), valid_idx, out[~nans])

    if max_gap >= 0:
        gaps = []
        start = None
        for i, is_nan in enumerate(nans):
            if is_nan and start is None:
                start = i
            if not is_nan and start is not None:
                gaps.append((start, i - 1))
                start = None
        if start is not None:
            gaps.append((start, len(nans) - 1))

        for gap_start, gap_end in gaps:
            if (gap_end - gap_start + 1) > max_gap:
                out[gap_start:gap_end + 1] = np.nan

    return out


def median_filter_1d(values, kernel_size):
    if kernel_size <= 1:
        return values.copy()

    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    if signal is not None:
        pad = kernel_size + 4
        padded = np.pad(values, (pad, pad), mode="reflect")
        filtered = signal.medfilt(padded, kernel_size=kernel_size)
        return filtered[pad:-pad]

    radius = kernel_size // 2
    out = np.empty_like(values)
    for i in range(len(values)):
        left = max(0, i - radius)
        right = min(len(values), i + radius + 1)
        out[i] = np.median(values[left:right])
    return out


def exponential_smooth_1d(values, alpha):
    out = values.copy()
    if len(out) == 0:
        return out
    for i in range(1, len(out)):
        out[i] = (1.0 - alpha) * out[i - 1] + alpha * out[i]
    return out


def suppress_velocity_outliers(sequence, threshold):
    if threshold <= 0:
        return sequence.copy()

    out = sequence.copy()
    diffs = np.diff(out, axis=0)
    speed = np.linalg.norm(diffs, axis=2)
    bad = speed > threshold
    for t in range(1, out.shape[0]):
        for j in range(out.shape[1]):
            if bad[t - 1, j]:
                out[t, j] = np.nan
    return out


def measure_bone_lengths(pose):
    lengths = np.full(len(DIRECTED_BONES), np.nan, dtype=np.float64)
    for idx, (parent, child) in enumerate(DIRECTED_BONES):
        if np.all(np.isfinite(pose[parent])) and np.all(np.isfinite(pose[child])):
            lengths[idx] = float(np.linalg.norm(pose[child] - pose[parent]))
    return lengths


def compute_bone_template(sequence):
    lengths = np.stack([measure_bone_lengths(pose) for pose in sequence], axis=0)
    return np.nanmedian(lengths, axis=0)


def enforce_bone_lengths(pose, template, weight, iterations):
    adjusted = pose.copy()
    if weight <= 0:
        return adjusted

    for _ in range(max(1, iterations)):
        for idx, (parent, child) in enumerate(DIRECTED_BONES):
            target = template[idx]
            if not np.isfinite(target):
                continue
            if not (np.all(np.isfinite(adjusted[parent])) and np.all(np.isfinite(adjusted[child]))):
                continue

            vec = adjusted[child] - adjusted[parent]
            length = float(np.linalg.norm(vec))
            if length <= 1e-8:
                continue

            desired = adjusted[parent] + vec / length * target
            adjusted[child] = (1.0 - weight) * adjusted[child] + weight * desired

    return adjusted


def process_single_track(
    track_sequence,
    smooth_alpha,
    max_interp_gap,
    medfilt_kernel,
    velocity_threshold,
    bone_length_weight,
    bone_iters,
):
    seq = track_sequence.copy()
    seq = suppress_velocity_outliers(seq, velocity_threshold)

    valid_mask = np.all(np.isfinite(seq), axis=2)
    for joint_idx in range(seq.shape[1]):
        joint_valid = valid_mask[:, joint_idx]
        if np.sum(joint_valid) < 2:
            continue

        for dim in range(3):
            values = seq[:, joint_idx, dim]
            values = interpolate_1d(values, max_gap=max_interp_gap)
            finite = np.isfinite(values)
            if np.sum(finite) < 2:
                seq[:, joint_idx, dim] = values
                continue

            filled = values.copy()
            missing = ~finite
            if np.any(missing):
                valid_idx = np.where(finite)[0]
                filled[missing] = np.interp(np.where(missing)[0], valid_idx, filled[finite])

            filtered = median_filter_1d(filled, medfilt_kernel)
            filtered = exponential_smooth_1d(filtered, smooth_alpha)
            filtered[~np.isfinite(values)] = np.nan
            seq[:, joint_idx, dim] = filtered

    template = compute_bone_template(seq)
    for frame_idx in range(seq.shape[0]):
        seq[frame_idx] = enforce_bone_lengths(seq[frame_idx], template, bone_length_weight, bone_iters)
    return seq, template


def process_tracked_tensor(
    tracked,
    smooth_alpha,
    max_interp_gap,
    medfilt_kernel,
    velocity_threshold,
    bone_length_weight,
    bone_iters,
):
    output = tracked.copy()
    templates = []
    for track_id in range(tracked.shape[1]):
        processed, template = process_single_track(
            tracked[:, track_id],
            smooth_alpha=smooth_alpha,
            max_interp_gap=max_interp_gap,
            medfilt_kernel=medfilt_kernel,
            velocity_threshold=velocity_threshold,
            bone_length_weight=bone_length_weight,
            bone_iters=bone_iters,
        )
        output[:, track_id] = processed
        templates.append(template)
    return output, np.asarray(templates)


def frame_list_from_tensor(tensor):
    return [tensor[i] for i in range(tensor.shape[0])]


def pad_frames_to_max_tracks(frames, max_tracks):
    padded = []
    for frame in frames:
        arr = normalize_frame_array(frame)
        out = np.full((max_tracks, 17, 3), np.nan, dtype=np.float64)
        count = min(arr.shape[0], max_tracks)
        out[:count] = arr[:count]
        padded.append(out)
    return padded


def collect_axis_limits(raw_frames, processed_frames):
    all_points = []
    for frame in raw_frames + processed_frames:
        arr = normalize_frame_array(frame)
        points = arr[np.all(np.isfinite(arr), axis=-1)]
        if points.size > 0:
            all_points.append(points)

    if len(all_points) == 0:
        return (-3, 3), (-3, 3), (0, 2.2)

    pts = np.concatenate(all_points, axis=0)
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) * 0.6, 1.0)
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (max(center[2] - radius, -0.2), center[2] + radius),
    )


def draw_frame(ax, frame, title, colors):
    ax.cla()
    ax.set_title(title)
    for person_idx, person in enumerate(frame):
        mask = valid_joint_mask(person)
        if not np.any(mask):
            continue

        color = colors[person_idx % len(colors)]
        pts = person[mask]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=10)
        for j1, j2 in COCO17_SKELETON:
            if np.all(np.isfinite(person[j1])) and np.all(np.isfinite(person[j2])):
                ax.plot(
                    [person[j1, 0], person[j2, 0]],
                    [person[j1, 1], person[j2, 1]],
                    [person[j1, 2], person[j2, 2]],
                    color=color,
                    linewidth=2,
                )


def compute_jitter_metric(frames):
    data = np.stack(frames, axis=0)
    diffs = np.diff(data, axis=0)
    speed = np.linalg.norm(diffs, axis=3)
    return np.nanmedian(speed, axis=(1, 2))


def compute_bone_error_metric(frames, templates):
    data = np.stack(frames, axis=0)
    errors = np.full((data.shape[0], data.shape[1], len(DIRECTED_BONES)), np.nan, dtype=np.float64)
    for t in range(data.shape[0]):
        for track_id in range(data.shape[1]):
            current = measure_bone_lengths(data[t, track_id])
            template = templates[track_id]
            valid = np.isfinite(current) & np.isfinite(template)
            errors[t, track_id, valid] = np.abs(current[valid] - template[valid])
    return np.nanmedian(errors, axis=(1, 2))


def create_debug_plot(raw_frames, processed_frames, templates, output_path=None):
    raw_jitter = compute_jitter_metric(raw_frames)
    processed_jitter = compute_jitter_metric(processed_frames)
    raw_bone_error = compute_bone_error_metric(raw_frames, templates)
    processed_bone_error = compute_bone_error_metric(processed_frames, templates)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(raw_jitter, label="before", color="tab:red", alpha=0.85)
    axes[0].plot(processed_jitter, label="after", color="tab:blue", alpha=0.85)
    axes[0].set_ylabel("median frame-to-frame motion (m)")
    axes[0].set_title("Temporal jitter")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(raw_bone_error, label="before", color="tab:red", alpha=0.85)
    axes[1].plot(processed_bone_error, label="after", color="tab:blue", alpha=0.85)
    axes[1].set_ylabel("median bone length deviation (m)")
    axes[1].set_xlabel("frame")
    axes[1].set_title("Bone consistency")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(Path(output_path) / "temporal_postprocess_metrics.png", dpi=200)
    return fig


def visualize_comparison(raw_frames, processed_frames, file_paths, pause):
    colors = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta", "olive", "black"]
    xlim, ylim, zlim = collect_axis_limits(raw_frames, processed_frames)

    # plt.ion()
    fig = plt.figure(figsize=(14, 6))
    ax_raw = fig.add_subplot(121, projection="3d")
    ax_processed = fig.add_subplot(122, projection="3d")
    for idx, (raw_frame, processed_frame, src_path) in enumerate(zip(raw_frames, processed_frames, file_paths)):
        draw_frame(ax_raw, raw_frame, f"Before: {src_path.stem}", colors)
        draw_frame(ax_processed, processed_frame, f"After: {src_path.stem}", colors)
        for ax in (ax_raw, ax_processed):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        fig.suptitle(f"Frame {idx + 1}/{len(file_paths)}")
        plt.pause(pause)
        if not plt.fignum_exists(fig.number):
            break
    # plt.ioff()
    plt.show()


def process_one_sequence(args, input_path, output_path, group_name=None):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path / "smoothed"
    file_paths, raw_frames = load_sequence(input_path)
    if len(raw_frames) == 0:
        raise FileNotFoundError(f"No pkl files found in: {input_path}")

    if args.skip_existing and output_files_complete(file_paths, output_path):
        label = f" group={group_name}" if group_name is not None else ""
        print(f"[postprocess]{label} skip: all {len(file_paths)} output files already exist")
        return {
            "input": input_path,
            "output": output_path,
            "frames": len(file_paths),
            "saved": 0,
            "skipped": len(file_paths),
        }

    tracked = build_tracked_tensor(
        frames=raw_frames,
        max_match_distance=args.max_match_distance,
        max_missing=args.max_missing,
        min_valid_joints=args.min_valid_joints,
        max_tracks=args.max_tracks,
    )
    processed, templates = process_tracked_tensor(
        tracked,
        smooth_alpha=args.smooth_alpha,
        max_interp_gap=args.max_interp_gap,
        medfilt_kernel=args.medfilt_kernel,
        velocity_threshold=args.velocity_threshold,
        bone_length_weight=args.bone_length_weight,
        bone_iters=args.bone_iters,
    )

    raw_frames_padded = pad_frames_to_max_tracks(raw_frames, processed.shape[1])
    processed_frames = frame_list_from_tensor(processed)
    saved, skipped = save_sequence(
        output_path,
        file_paths,
        processed_frames,
        skip_existing=args.skip_existing,
    )
    label = f" group={group_name}" if group_name is not None else ""
    print(f"[postprocess]{label} input frames: {len(raw_frames)}")
    print(f"[postprocess]{label} output dir: {output_path}")
    print(f"[postprocess]{label} saved={saved} skipped_existing={skipped}")

    create_debug_plot(raw_frames_padded, processed_frames, templates, output_path if args.save_plot else None)
    if args.show:
        visualize_comparison(raw_frames_padded, processed_frames, file_paths, args.pause)

    return {
        "input": input_path,
        "output": output_path,
        "frames": len(file_paths),
        "saved": saved,
        "skipped": skipped,
    }


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path / "smoothed"

    if args.process_all_groups:
        group_jobs = infer_group_jobs(input_path, output_path)
        print(f"[postprocess] found {len(group_jobs)} groups from input_dir={input_path}")
        if args.show:
            print("[postprocess] process_all_groups enabled: disabling interactive show for batch processing")
            args.show = False

        total_saved = 0
        total_skipped = 0
        for group_name, group_input_path, group_output_path in group_jobs:
            result = process_one_sequence(
                args,
                input_path=group_input_path,
                output_path=group_output_path,
                group_name=group_name,
            )
            total_saved += result["saved"]
            total_skipped += result["skipped"]

        print(f"[postprocess] all groups done saved={total_saved} skipped={total_skipped}")
        return

    process_one_sequence(args, input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
