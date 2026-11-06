"""Debug saved raw/smoothed 3D poses against eight-camera detections.

No poses are filtered or overwritten. Correspondences are re-estimated because
the saved pickle arrays do not contain the original 2D detection IDs.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
import math
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Img2Keypoint.utils import (  # noqa: E402
    COCO17_SKELETON,
    load_extrinsic_data,
    load_intrinsic_data,
)
from TimeProcess.utils import timestamp_to_ms  # noqa: E402

CAMS = tuple("ABCDEFGH")
CAM_FIELDS = (
    "image_frame", "error_px", "person_2d_id", "valid_joint_count", "status",
    "nearest_person_2d_id", "nearest_error_px", "nearest_valid_joint_count",
    "detection_count", "projected_joint_count",
)
CSV_FIELDS = (
    "group", "frame", "source", "person_3d_id", "mean_error_px",
    "max_error_px", "valid_camera_count", "nearest_mean_error_px",
    "nearest_valid_camera_count", "score_threshold", "min_valid_joints", "sync_error_ms",
) + tuple(f"cam_{cam}_{field}" for cam in CAMS for field in CAM_FIELDS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("debug",), default="debug")
    parser.add_argument("--root_path", "--data_path", dest="root_path", type=Path,
                        default=Path(r"F:\20261028\data_collection"))
    parser.add_argument("--groups", nargs="*", help="Group folder names; default: all groups.")
    parser.add_argument("--calib_path", type=Path)
    parser.add_argument("--reference_cam", choices=CAMS, default="A")
    parser.add_argument("--sources", nargs="+", choices=("raw", "smoothed"),
                        default=["raw", "smoothed"])
    parser.add_argument("--result_2D_data_path", "--result_2d_data_path",
                        dest="result_2d_path", type=Path)
    parser.add_argument("--result_3D_data_path", "--result_3d_data_path",
                        dest="raw_path", type=Path)
    parser.add_argument("--smooth_3D_data_path", "--smooth_3d_data_path",
                        dest="smoothed_path", type=Path)
    parser.add_argument("--result_2d_subdir", default="camera results/2D")
    parser.add_argument("--result_3d_subdir", default="camera results/3D")
    parser.add_argument("--smooth_3d_subdir", default="camera results/smoothed 3D")
    parser.add_argument("--output_dir", type=Path,
                        default=PROJECT_ROOT / "check_results_debug")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--min_valid_joints", type=int, default=5)
    parser.add_argument("--sync_error_ms", type=int, default=25)
    parser.add_argument("--frames", nargs="+", help="Saved 3D filename stems to inspect.")
    parser.add_argument("--max_frames", type=int, help="Limit frames per group, for a quick check.")
    parser.add_argument("--no_view", action="store_true", help="Export CSV without opening a viewer.")
    args = parser.parse_args(argv)
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score_threshold must be between 0 and 1")
    if not 1 <= args.min_valid_joints <= 17:
        parser.error("--min_valid_joints must be between 1 and 17")
    if args.sync_error_ms < 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("--sync_error_ms must be nonnegative and --max_frames must be positive")
    args.sources = list(dict.fromkeys(args.sources))
    return args


def frame_name(group):
    """Match Reconstruction.frame_result_3d_filename, including its rounding."""
    paths = [path for path in group.values() if path is not None]
    if not paths:
        return None
    avg_ns = int(round(np.mean([
        int(Path(path).stem.split("_")[0]) * 1_000_000_000
        + int(Path(path).stem.split("_")[1]) for path in paths
    ])))
    return f"{avg_ns // 1_000_000_000}_{avg_ns % 1_000_000_000:09d}"


def timestamp_ns(stem):
    sec, ns = stem.split("_")
    return int(sec) * 1_000_000_000 + int(ns)


def synchronized_images(camera_path, max_error_ms):
    """Indexed equivalent of utils.get_matched_pairs, including tie/used rules.

    Keep original file order for ties and base-frame traversal. Search only the
    timestamp window rather than every image, without changing reconstruction.
    """
    cameras = {}
    for folder in sorted(p for p in camera_path.iterdir() if p.is_dir()):
        files = sorted(p for p in (folder / "frames").iterdir() if p.suffix == ".jpg")
        times = [timestamp_to_ms(p.stem) for p in files]
        order = sorted(range(len(files)), key=lambda i: (times[i], i))
        cameras[folder.name.split("_")[-1]] = dict(
            files=files, times=times, order=order, sorted_times=[times[i] for i in order],
            used=set())
    if not cameras:
        raise FileNotFoundError(f"No camera directories: {camera_path}")
    ids = sorted(cameras)
    required = math.ceil(0.75 * len(ids))
    base = cameras[ids[0]]
    for base_index, base_time in enumerate(base["times"]):
        if base_index in base["used"]:
            continue
        matches = {}
        for cam in ids:
            data = cameras[cam]
            start = bisect_left(data["sorted_times"], base_time - max_error_ms)
            end = bisect_right(data["sorted_times"], base_time + max_error_ms)
            candidates = (i for i in data["order"][start:end] if i not in data["used"])
            best = min(candidates, key=lambda i: (abs(data["times"][i] - base_time), i), default=None)
            if best is not None:
                matches[cam] = best
        if len(matches) >= required:
            group = {cam: None for cam in ids}
            for cam, index in matches.items():
                cameras[cam]["used"].add(index)
                group[cam] = cameras[cam]["files"][index]
            yield group


def load_calibration(path, reference_cam):
    calibration = {}
    for cam in CAMS:
        K, dist = load_intrinsic_data(path / f"intrinsic_cam_{cam}.npz")
        R, t = np.eye(3), np.zeros(3)
        if cam != reference_cam:
            _, R_to_ref, t_to_ref = load_extrinsic_data(
                path / f"extrinsic_T_cam_{cam}_to_cam_{reference_cam}.npy")
            R = np.asarray(R_to_ref, dtype=float).T
            t = -R @ np.asarray(t_to_ref, dtype=float).reshape(3)
        K, dist = np.asarray(K, dtype=float), np.asarray(dist, dtype=float)
        if K.shape != (3, 3) or not all(np.isfinite(x).all() for x in (K, dist, R, t)):
            raise ValueError(f"Invalid calibration for camera {cam}: {path}")
        calibration[cam] = (K, dist, R, t)
    return calibration


def collect_samples(args):
    root = args.root_path
    if args.groups:
        groups = [root if root.name == name and (root / "camera").is_dir()
                  else root / name for name in args.groups]
    elif (root / "camera").is_dir():
        groups = [root]
    else:
        groups = sorted(p for p in root.iterdir() if p.is_dir() and (p / "camera").is_dir())
    if not groups:
        raise FileNotFoundError(f"No groups with camera/ found under {root}")
    samples, calibration_cache = [], {}
    for group_root in groups:
        def result_path(explicit, subdir):
            if explicit is None:
                return group_root / subdir
            return explicit / group_root.name if len(groups) > 1 else explicit

        candidates = [group_root / "calib", root / "calib", group_root.parent / "calib",
                      root.parent / "calib", group_root.parent.parent / "calib"]
        calib_path = args.calib_path or next((p for p in candidates if p.is_dir()), None)
        if calib_path is None:
            raise FileNotFoundError(f"No calibration directory for {group_root}")
        cache_key = calib_path.resolve()
        if cache_key not in calibration_cache:
            calibration_cache[cache_key] = load_calibration(calib_path, args.reference_cam)
        result_2d = result_path(args.result_2d_path, args.result_2d_subdir)
        if not result_2d.is_dir():
            raise FileNotFoundError(f"Missing 2D result directory: {result_2d}")
        paths_by_frame = {}
        for source in args.sources:
            folder = result_path(args.raw_path, args.result_3d_subdir) if source == "raw" else (
                result_path(args.smoothed_path, args.smooth_3d_subdir))
            files = list(folder.glob("*.pkl"))
            if not files:
                raise FileNotFoundError(f"No {source} 3D results: {folder}")
            for path in files:
                if args.frames is None or path.stem in args.frames:
                    paths_by_frame.setdefault(path.stem, {})[source] = path

        images_by_frame = {}
        for images in synchronized_images(group_root / "camera", args.sync_error_ms):
            name = frame_name(images)
            if name in images_by_frame:
                raise ValueError(f"Ambiguous synchronized timestamp: {group_root.name}/{name}")
            images_by_frame[name] = images
        names = sorted(paths_by_frame, key=timestamp_ns)
        if args.max_frames is not None:
            names = names[:args.max_frames]
        missing_sync = 0
        for name in names:
            images = images_by_frame.get(name)
            missing_sync += images is None
            samples.append(dict(group=group_root.name, frame=name, paths=paths_by_frame[name],
                                images=images, result_2d=result_2d,
                                calibration=calibration_cache[cache_key]))
        print(f"[debug] {group_root.name}: {len(names)} frames, {missing_sync} missing synchronization")
    if not samples:
        raise FileNotFoundError("No 3D frames matched the requested selection")
    return samples


def load_poses(path):
    with path.open("rb") as stream:
        poses = np.asarray(pickle.load(stream), dtype=float)
    if poses.size == 0:
        return np.empty((0, 17, 3), dtype=float)
    if poses.shape == (17, 3):
        poses = poses[None]
    if poses.ndim != 3 or poses.shape[1:] != (17, 3):
        raise ValueError(f"Expected [people, 17, 3], got {poses.shape}: {path}")
    return poses


def load_observations(result_2d, image):
    empty = (np.empty((0, 17, 2)), np.empty((0, 17)))
    if image is None:
        return *empty, "missing_image"
    path = result_2d / image.parent.parent.name / f"{image.stem}.npz"
    if not path.exists():
        return *empty, "missing_2d_file"
    with np.load(path) as data:
        kpts = np.asarray(data["kpts"], dtype=float)
        scores = np.asarray(data["scores"], dtype=float)
    if kpts.ndim != 3 or kpts.shape[1:] != (17, 2) or scores.shape != kpts.shape[:2]:
        raise ValueError(f"Invalid kpts/scores shapes {kpts.shape}/{scores.shape}: {path}")
    return kpts, scores, "ok" if len(kpts) else "no_detections"


def project_poses(poses, calibration):
    K, dist, R, t = calibration
    projected = np.full((*poses.shape[:2], 2), np.nan)
    camera_points = poses @ R.T + t
    valid = np.isfinite(camera_points).all(axis=-1) & (camera_points[..., 2] > 1e-8)
    if np.any(valid):
        pixels, _ = cv2.projectPoints(camera_points[valid].reshape(-1, 1, 3),
                                      np.zeros(3), np.zeros(3), K, dist)
        projected[valid] = pixels.reshape(-1, 2)
    return projected


def match_camera(projected, kpts, scores, score_threshold, min_valid_joints, status="ok"):
    """Keep all residuals; only joint availability gates one-to-one matching.

    No error cutoff is used in debug: even arbitrarily large matched residuals
    remain visible. Dummy columns allow people with no eligible/free detection
    to stay unmatched. Nearest candidates are recorded independently.
    """
    valid = (np.isfinite(projected).all(axis=-1)[:, None, :]
             & np.isfinite(kpts).all(axis=-1)[None, :, :]
             & (np.isfinite(scores) & (scores > score_threshold))[None, :, :])
    distances = np.linalg.norm(projected[:, None] - kpts[None], axis=-1)
    errors = np.where(valid, distances, np.nan)
    counts = valid.sum(axis=-1)
    means = np.divide(np.where(valid, distances, 0).sum(axis=-1), counts,
                      out=np.full(counts.shape, np.nan), where=counts > 0)
    eligible = (counts >= min_valid_joints) & np.isfinite(means)
    matches = {}
    if eligible.any():
        # Maximize supported match count first, then minimize total pixel error.
        unmatched_cost = (float(means[eligible].max()) + 1) * (len(projected) + 1)
        costs = np.where(eligible, means, unmatched_cost * (len(projected) + 2))
        costs = np.column_stack([costs, np.full((len(projected), len(projected)), unmatched_cost)])
        rows, cols = linear_sum_assignment(costs)
        matches = {int(row): int(col) for row, col in zip(rows, cols)
                   if col < len(kpts) and eligible[row, col]}
    people = []
    for person_id, projection in enumerate(projected):
        supported = np.flatnonzero(eligible[person_id])
        candidates = supported if len(supported) else np.flatnonzero(counts[person_id] > 0)
        nearest = int(candidates[np.argmin(means[person_id, candidates])]) if len(candidates) else None
        matched = matches.get(person_id)
        projected_count = int(np.isfinite(projection).all(axis=-1).sum())
        reason = status
        if status == "ok":
            reason = ("matched" if matched is not None else "no_valid_projection" if projected_count == 0
                      else "conflict" if len(supported) else "insufficient_joints")
        selected = matched if matched is not None else nearest
        people.append(dict(
            error_px=float(means[person_id, matched]) if matched is not None else np.nan,
            person_2d_id=matched,
            valid_joint_count=int(counts[person_id, matched]) if matched is not None else 0,
            status=reason,
            nearest_person_2d_id=nearest,
            nearest_error_px=float(means[person_id, nearest]) if nearest is not None else np.nan,
            nearest_valid_joint_count=int(counts[person_id, nearest]) if nearest is not None else 0,
            detection_count=len(kpts), projected_joint_count=projected_count,
            projected=projection,
            observed=kpts[selected] if selected is not None else np.full((17, 2), np.nan),
            joint_errors=errors[person_id, selected] if selected is not None else np.full(17, np.nan),
        ))
    return people


def analyze_frame(sample, source, args):
    poses = load_poses(sample["paths"][source])
    rows = [dict(group=sample["group"], frame=sample["frame"], source=source,
                 person_3d_id=person_id, score_threshold=args.score_threshold,
                 min_valid_joints=args.min_valid_joints, sync_error_ms=args.sync_error_ms)
            for person_id in range(len(poses))]
    debug = {}
    for cam in CAMS:
        image = sample["images"].get(cam) if sample["images"] is not None else None
        kpts, scores, status = load_observations(sample["result_2d"], image)
        if sample["images"] is None:
            status = "missing_sync"
        projected = project_poses(poses, sample["calibration"][cam])
        people = match_camera(projected, kpts, scores, args.score_threshold, args.min_valid_joints, status)
        for row, person in zip(rows, people):
            person["image_frame"] = image.stem if image is not None else ""
            row.update({f"cam_{cam}_{field}": person[field] for field in CAM_FIELDS})
        debug[cam] = dict(image=image, kpts=kpts, scores=scores, people=people, status=status)
    for row in rows:
        means = [row[f"cam_{cam}_error_px"] for cam in CAMS
                 if np.isfinite(row[f"cam_{cam}_error_px"])]
        nearest = [row[f"cam_{cam}_nearest_error_px"] for cam in CAMS
                   if row[f"cam_{cam}_nearest_valid_joint_count"] >= args.min_valid_joints
                   and np.isfinite(row[f"cam_{cam}_nearest_error_px"])]
        row.update(mean_error_px=float(np.mean(means)) if means else np.nan,
                   max_error_px=max(means) if means else np.nan, valid_camera_count=len(means),
                   nearest_mean_error_px=float(np.mean(nearest)) if nearest else np.nan,
                   nearest_valid_camera_count=len(nearest))
    return rows, debug


def export_csv(samples, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "reprojection_errors.csv"
    # Avoid truncating an earlier completed report if a corrupt frame aborts this run.
    temporary = output.with_suffix(".csv.tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, sample in enumerate(samples):
            for source in args.sources:
                if source not in sample["paths"]:
                    print(f"[debug] missing {source}: {sample['group']}/{sample['frame']}")
                    continue
                rows, _ = analyze_frame(sample, source, args)
                writer.writerows(rows)
                count += len(rows)
            if (index + 1) % 100 == 0 or index + 1 == len(samples):
                print(f"[debug] exported {index + 1}/{len(samples)} frames, {count} person rows")
    temporary.replace(output)
    print(f"[debug] CSV: {output}")
    return output


def draw_pose(ax, points, color, marker="o", alpha=1.0):
    valid = np.isfinite(points).all(axis=-1)
    for a, b in COCO17_SKELETON:
        if valid[a] and valid[b]:
            ax.plot(points[[a, b], 0], points[[a, b], 1], color=color, alpha=alpha, linewidth=1)
    ax.scatter(points[valid, 0], points[valid, 1], color=color, marker=marker, alpha=alpha, s=14)


class ReprojectionViewer:
    """One frame per page; raw/smoothed share navigation, never person IDs."""

    def __init__(self, samples, args):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.samples, self.args, self.index = samples, args, 0
        self.source = "smoothed" if "smoothed" in args.sources else args.sources[0]
        self.fig, axes = plt.subplots(2, 4, figsize=(20, 11))
        self.axes = axes.ravel()
        self.fig.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.12,
                                 wspace=0.05, hspace=0.23)
        self.buttons = []
        for x, label, callback in (
            (0.20, "Previous", lambda event: self.move(-1)),
            (0.35, "Next", lambda event: self.move(1)),
            (0.50, "Raw / Smoothed", lambda event: self.toggle_source()),
            (0.65, "Save PNG", lambda event: self.save()),
        ):
            button = Button(self.fig.add_axes([x, 0.025, 0.14, 0.04]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig._reprojection_viewer = self
        self.draw()

    def draw(self):
        import matplotlib.pyplot as plt

        sample = self.samples[self.index]
        available = [s for s in self.args.sources if s in sample["paths"]]
        if self.source not in available:
            self.source = available[0]
        rows, debug = analyze_frame(sample, self.source, self.args)
        palette = plt.get_cmap("tab10")
        for cam, ax in zip(CAMS, self.axes):
            ax.clear()
            data = debug[cam]
            image = None
            if data["image"] is not None:
                # imdecode supports Windows paths containing non-ASCII characters.
                image = cv2.imdecode(np.fromfile(data["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                ax.text(0.5, 0.5, "Missing image" if sample["images"] is not None else "Missing synchronization",
                        transform=ax.transAxes, ha="center")
                ax.set_title(f"Camera {cam} | {data['status']}", fontsize=9)
                ax.axis("off")
                continue
            for detection_id, (pose, scores) in enumerate(zip(data["kpts"], data["scores"])):
                visible = pose.copy()
                visible[~np.isfinite(scores) | (scores <= self.args.score_threshold)] = np.nan
                draw_pose(ax, visible, "gray", alpha=0.45)
                valid = np.isfinite(visible).all(axis=-1)
                if valid.any():
                    x, y = np.mean(visible[valid], axis=0)
                    ax.text(x, y, f"D{detection_id}", fontsize=8, color="white", clip_on=True,
                            bbox=dict(facecolor="black", alpha=0.5, pad=1))
            lines = []
            for person_id, person in enumerate(data["people"]):
                color = palette(person_id % 10)
                draw_pose(ax, person["projected"], color, marker="x")
                valid = np.isfinite(person["projected"]).all(axis=-1)
                if valid.any():
                    x, y = np.mean(person["projected"][valid], axis=0)
                    ax.text(x, y - 20, f"P{person_id}", fontsize=8, color=color, clip_on=True,
                            bbox=dict(facecolor="white", alpha=0.8, pad=1))
                matched = person["person_2d_id"]
                candidate = person["nearest_person_2d_id"]
                if matched is not None or candidate is not None:
                    observed = person["observed"].copy()
                    observed[~np.isfinite(person["joint_errors"])] = np.nan
                    draw_pose(ax, observed, color, alpha=0.9 if matched is not None else 0.4)
                    for joint in np.flatnonzero(np.isfinite(person["joint_errors"])):
                        points = np.array([observed[joint], person["projected"][joint]])
                        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=0.7,
                                linestyle="-" if matched is not None else ":", alpha=0.6)
                if matched is not None:
                    text = f"P{person_id}->D{matched}: {person['error_px']:.1f}px n={person['valid_joint_count']}"
                else:
                    text = f"P{person_id}: {person['status']}"
                    if candidate is not None:
                        text += f" (nearest D{candidate}: {person['nearest_error_px']:.1f}px)"
                lines.append(text)
            ax.set_title(f"Camera {cam} | detections={len(data['kpts'])}\n" +
                         ("\n".join(lines) if lines else data["status"]), fontsize=9)
            if image is not None:
                ax.set_xlim(0, image.shape[1])
                ax.set_ylim(image.shape[0], 0)
            ax.axis("off")
        summary = " | ".join(
            f"P{row['person_3d_id']}: mean={row['mean_error_px']:.1f}px cams={row['valid_camera_count']}/8"
            for row in rows) or "No 3D people in this frame"
        self.fig.suptitle(
            f"{sample['group']} / {sample['frame']} | {self.source} | {self.index + 1}/{len(self.samples)}\n"
            f"{summary}\nGray: all 2D detections; colored circles: observed; crosses: projected; "
            "dotted: unmatched nearest candidate\nLeft/Right: frame | R: raw/smoothed | S: save | Q: close",
            fontsize=11)
        self.fig.canvas.draw_idle()

    def move(self, delta):
        self.index = min(max(self.index + delta, 0), len(self.samples) - 1)
        self.draw()

    def toggle_source(self):
        available = [s for s in self.args.sources if s in self.samples[self.index]["paths"]]
        self.source = available[(available.index(self.source) + 1) % len(available)]
        self.draw()

    def save(self):
        sample = self.samples[self.index]
        folder = self.args.output_dir / "figures" / sample["group"]
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"{sample['frame']}_{self.source}.png"
        self.fig.savefig(output, dpi=150)
        print(f"[debug] PNG: {output}")
        return output

    def on_key(self, event):
        if event.key in ("right", "left"):
            self.move(1 if event.key == "right" else -1)
        elif event.key == "r":
            self.toggle_source()
        elif event.key == "s":
            self.save()
        elif event.key == "q":
            import matplotlib.pyplot as plt
            plt.close(self.fig)


def main(argv=None):
    args = parse_args(argv)
    samples = collect_samples(args)
    export_csv(samples, args)
    if not args.no_view:
        import matplotlib.pyplot as plt
        ReprojectionViewer(samples, args)
        plt.show()


if __name__ == "__main__":
    main()
