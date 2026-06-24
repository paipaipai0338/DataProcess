from __future__ import annotations

import argparse
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reconstruction.py forces Qt5Agg at import time for its visual tools. Keep this
# batch entry on a non-interactive backend while still reusing its compute code.
matplotlib.use("Agg")
_MATPLOTLIB_USE = matplotlib.use


def _batch_matplotlib_use(backend, *args, **kwargs):
    if str(backend).lower() == "qt5agg":
        return None
    return _MATPLOTLIB_USE(backend, *args, **kwargs)


matplotlib.use = _batch_matplotlib_use

try:
    import open3d  # noqa: F401
except ModuleNotFoundError:
    sys.modules["open3d"] = types.ModuleType("open3d")

from Img2Keypoint.Reconstruction import (  # noqa: E402
    collect_missing_frame_items,
    init_frame_worker,
    load_extrinsic_cache,
    load_intrinsic_cache,
    reconstruct_and_save_frame_no_debug,
    reconstruct_and_save_frame_no_debug_global,
)
from Img2Keypoint.temporal_postprocess_compare import (  # noqa: E402
    build_tracked_tensor,
    frame_list_from_tensor,
    load_sequence,
    output_files_complete,
    process_tracked_tensor,
    save_sequence,
)
from Img2Keypoint.utils import get_matched_pairs  # noqa: E402


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("yes", "true", "t", "1", "y"):
        return True
    if lowered in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


@dataclass(frozen=True)
class GroupJob:
    name: str
    root_path: Path
    camera_path: Path
    result_2d_path: Path
    result_3d_path: Path
    smooth_3d_path: Path
    calib_path: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch 3D reconstruction and temporal postprocess for one group "
            "or a collection of groups. No visualization is performed."
        )
    )
    parser.add_argument(
        "--root_path",
        "--data_path",
        dest="root_path",
        type=Path,
        default=Path(r'G:\20260615\data_collection'),
        help="One group root containing camera/, or a collection root containing group folders.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Group folder names to process. If omitted, all groups are processed.",
    )
    parser.add_argument(
        "--calib_path",
        type=Path,
        default=None,
        help="Calibration directory. If omitted, it is inferred from root/group parents.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "reconstruct", "postprocess"),
        default="all",
        help="Processing stage to run.",
    )

    parser.add_argument(
        "--result_2D_data_path",
        "--result_2d_data_path",
        dest="result_2d_data_path",
        type=Path,
        default=None,
        help="2D result path. For multiple groups, this is treated as a base path with group subfolders.",
    )
    parser.add_argument(
        "--result_3D_data_path",
        "--result_3d_data_path",
        "--input_path",
        dest="result_3d_data_path",
        type=Path,
        default=None,
        help="Raw 3D output/input path. For multiple groups, this is treated as a base path with group subfolders.",
    )
    parser.add_argument(
        "--smooth_3D_data_path",
        "--smooth_3d_data_path",
        "--output_path",
        dest="smooth_3d_data_path",
        type=Path,
        default=None,
        help="Smoothed 3D output path. For multiple groups, this is treated as a base path with group subfolders.",
    )
    parser.add_argument("--result_2d_subdir", default="camera results/2D")
    parser.add_argument("--result_3d_subdir", default="camera results/3D")
    parser.add_argument("--smooth_3d_subdir", default="camera results/smoothed 3D")

    parser.add_argument("--reference_cam", type=str, default="A")
    parser.add_argument(
        "--matching_mode",
        type=str,
        default="ref_guided",
        choices=("exhaustive", "ref_guided"),
    )
    parser.add_argument("--pairwise_top_k", type=int, default=1)
    parser.add_argument("--min_support_cams", type=int, default=3)
    parser.add_argument("--max_missing_cams", type=int, default=4)
    parser.add_argument("--pairwise_none_error_m", type=float, default=0.10)
    parser.add_argument("--pairwise_candidate_error_m", type=float, default=0.20)
    parser.add_argument("--max_candidate_ray_error_m", type=float, default=0.10)
    parser.add_argument("--strict_two_cam_ray_error_m", type=float, default=0.05)
    parser.add_argument("--min_candidate_valid_joints", type=int, default=5)
    parser.add_argument("--support_aware_sort", type=str2bool, default=True)
    parser.add_argument("--use_2d_scores", type=str2bool, default=True)
    parser.add_argument("--verbose_every", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--parallel_chunksize", type=int, default=8)

    parser.add_argument("--smooth_alpha", type=float, default=0.35)
    parser.add_argument("--max_match_distance", type=float, default=0.50)
    parser.add_argument("--max_missing", type=int, default=8)
    parser.add_argument("--min_valid_joints", type=int, default=5)
    parser.add_argument("--max_tracks", type=int, default=0)
    parser.add_argument("--max_interp_gap", type=int, default=8)
    parser.add_argument("--medfilt_kernel", type=int, default=7)
    parser.add_argument("--velocity_threshold", type=float, default=0.35)
    parser.add_argument("--bone_length_weight", type=float, default=0.65)
    parser.add_argument("--bone_iters", type=int, default=2)

    parser.add_argument(
        "--skip_existing",
        type=str2bool,
        default=False,
        help="Skip existing raw/smoothed 3D frame files.",
    )
    parser.add_argument(
        "--continue_on_error",
        type=str2bool,
        default=False,
        help="Continue with later groups when one group fails.",
    )
    return parser.parse_args()


def group_sort_key(item: Tuple[str, Path]):
    name = item[0]
    return (not name.isdigit(), int(name) if name.isdigit() else name)


def iter_group_roots(root_path: Path, groups: Optional[Sequence[str]]) -> List[Tuple[str, Path]]:
    root_path = Path(root_path)
    if groups:
        group_roots = []
        for group_name in groups:
            if root_path.name == group_name and (root_path / "camera").is_dir():
                group_root = root_path
            else:
                group_root = root_path / group_name
            group_roots.append((group_name, group_root))
        return group_roots

    if (root_path / "camera").is_dir():
        return [(root_path.name, root_path)]

    group_roots = [
        (p.name, p)
        for p in root_path.iterdir()
        if p.is_dir() and (p / "camera").is_dir()
    ]
    return sorted(group_roots, key=group_sort_key)


def resolve_group_path(
    explicit_path: Optional[Path],
    group_name: str,
    group_root: Path,
    default_subdir: str,
    multi_group: bool,
) -> Path:
    if explicit_path is None:
        return group_root / default_subdir

    explicit_path = Path(explicit_path)
    if multi_group:
        return explicit_path / group_name
    return explicit_path


def infer_calib_path(root_path: Path, group_root: Path, explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)

    candidates = [
        group_root / "calib",
        root_path / "calib",
        group_root.parent / "calib",
        root_path.parent / "calib",
        group_root.parent.parent / "calib",
        root_path.parent.parent / "calib",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot infer calibration directory. Tried: {tried}")


def build_group_jobs(args) -> List[GroupJob]:
    root_path = Path(args.root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")

    group_roots = iter_group_roots(root_path, args.groups)
    if not group_roots:
        raise FileNotFoundError(f"No group folders with camera/ found under: {root_path}")

    multi_group = len(group_roots) > 1
    jobs = []
    for group_name, group_root in group_roots:
        camera_path = group_root / "camera"
        if not camera_path.is_dir():
            raise FileNotFoundError(f"Missing camera directory for group {group_name}: {camera_path}")

        result_2d_path = resolve_group_path(
            args.result_2d_data_path,
            group_name,
            group_root,
            args.result_2d_subdir,
            multi_group,
        )
        result_3d_path = resolve_group_path(
            args.result_3d_data_path,
            group_name,
            group_root,
            args.result_3d_subdir,
            multi_group,
        )
        smooth_3d_path = resolve_group_path(
            args.smooth_3d_data_path,
            group_name,
            group_root,
            args.smooth_3d_subdir,
            multi_group,
        )
        calib_path = infer_calib_path(root_path, group_root, args.calib_path)

        jobs.append(
            GroupJob(
                name=group_name,
                root_path=group_root,
                camera_path=camera_path,
                result_2d_path=result_2d_path,
                result_3d_path=result_3d_path,
                smooth_3d_path=smooth_3d_path,
                calib_path=calib_path,
            )
        )
    return jobs


def extract_camera_id(camera_dir: Path) -> str:
    return camera_dir.name.split("_")[-1]


def infer_camera_ids(camera_path: Path) -> List[str]:
    camera_ids = [
        extract_camera_id(path)
        for path in camera_path.iterdir()
        if path.is_dir()
    ]
    if not camera_ids:
        raise FileNotFoundError(f"No camera folders found in: {camera_path}")
    return sorted(camera_ids)


def reconstruction_cfg(
    args,
    job: GroupJob,
    camera_ids: Sequence[str],
    intrinsic_cache,
    extrinsic_cache,
):
    return {
        "all_cams": list(camera_ids),
        "result_2D_data_path": job.result_2d_path,
        "result_3D_data_path": str(job.result_3d_path),
        "intrinsic_cache": intrinsic_cache,
        "extrinsic_cache": extrinsic_cache,
        "reference_cam": args.reference_cam,
        "calib_path": job.calib_path,
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
        "R_ref_to_world": np.eye(3, dtype=np.float64),
        "world_origin_ref": np.zeros(3, dtype=np.float64),
        "skip_existing": args.skip_existing,
    }


def run_reconstruction(args, job: GroupJob, cache_store: Dict[Tuple[str, Tuple[str, ...]], tuple]):
    if not job.result_2d_path.is_dir():
        raise FileNotFoundError(f"Missing 2D result directory for group {job.name}: {job.result_2d_path}")

    job.result_3d_path.mkdir(parents=True, exist_ok=True)
    camera_ids = infer_camera_ids(job.camera_path)
    cache_key = (str(job.calib_path.resolve()), tuple(camera_ids))
    if cache_key not in cache_store:
        intrinsic_cache = load_intrinsic_cache(camera_ids, job.calib_path)
        extrinsic_cache = load_extrinsic_cache(camera_ids, args.reference_cam, job.calib_path)
        cache_store[cache_key] = (intrinsic_cache, extrinsic_cache)
    else:
        intrinsic_cache, extrinsic_cache = cache_store[cache_key]

    synchronized_groups = get_matched_pairs(job.camera_path)
    frame_items, skipped = collect_missing_frame_items(
        synchronized_groups,
        job.result_3d_path,
        skip_existing=args.skip_existing,
    )

    print(
        f"[reconstruct] group={job.name} total={len(synchronized_groups)} "
        f"todo={len(frame_items)} skipped={skipped}"
    )
    if not frame_items:
        return {"done": 0, "skipped": skipped}

    worker_cfg = reconstruction_cfg(
        args,
        job,
        camera_ids,
        intrinsic_cache,
        extrinsic_cache,
    )
    workers = max(1, int(args.num_workers))
    chunksize = max(1, int(args.parallel_chunksize))
    done = 0

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
            for result in tqdm(results, total=len(frame_items), desc=f"Reconstruct {job.name} x{workers}"):
                if not result.get("skipped", False):
                    done += 1
    else:
        for frame_item in tqdm(frame_items, total=len(frame_items), desc=f"Reconstruct {job.name}"):
            result = reconstruct_and_save_frame_no_debug(frame_item, worker_cfg)
            if not result.get("skipped", False):
                done += 1

    return {"done": done, "skipped": skipped}


def run_postprocess(args, job: GroupJob):
    if not job.result_3d_path.is_dir():
        raise FileNotFoundError(f"Missing raw 3D directory for group {job.name}: {job.result_3d_path}")

    file_paths, raw_frames = load_sequence(job.result_3d_path)
    if not raw_frames:
        print(f"[postprocess] group={job.name} skip: no pkl files in {job.result_3d_path}")
        return {"saved": 0, "skipped": 0}

    if args.skip_existing and output_files_complete(file_paths, job.smooth_3d_path):
        print(f"[postprocess] group={job.name} skip: all {len(file_paths)} output files already exist")
        return {"saved": 0, "skipped": len(file_paths)}

    tracked = build_tracked_tensor(
        frames=raw_frames,
        max_match_distance=args.max_match_distance,
        max_missing=args.max_missing,
        min_valid_joints=args.min_valid_joints,
        max_tracks=args.max_tracks,
    )
    processed, _ = process_tracked_tensor(
        tracked,
        smooth_alpha=args.smooth_alpha,
        max_interp_gap=args.max_interp_gap,
        medfilt_kernel=args.medfilt_kernel,
        velocity_threshold=args.velocity_threshold,
        bone_length_weight=args.bone_length_weight,
        bone_iters=args.bone_iters,
    )
    processed_frames = frame_list_from_tensor(processed)
    saved, skipped = save_sequence(
        job.smooth_3d_path,
        file_paths,
        processed_frames,
        skip_existing=args.skip_existing,
    )
    print(
        f"[postprocess] group={job.name} frames={len(file_paths)} "
        f"saved={saved} skipped_existing={skipped} output={job.smooth_3d_path}"
    )
    return {"saved": saved, "skipped": skipped}


def run_group(args, job: GroupJob, cache_store):
    summary = {
        "reconstruct_done": 0,
        "reconstruct_skipped": 0,
        "postprocess_saved": 0,
        "postprocess_skipped": 0,
    }
    print(f"[group] {job.name}")
    print(f"[group] camera={job.camera_path}")
    print(f"[group] 2d={job.result_2d_path}")
    print(f"[group] raw3d={job.result_3d_path}")
    print(f"[group] smooth3d={job.smooth_3d_path}")
    print(f"[group] calib={job.calib_path}")

    if args.stage in ("all", "reconstruct"):
        result = run_reconstruction(args, job, cache_store)
        summary["reconstruct_done"] = result["done"]
        summary["reconstruct_skipped"] = result["skipped"]

    if args.stage in ("all", "postprocess"):
        result = run_postprocess(args, job)
        summary["postprocess_saved"] = result["saved"]
        summary["postprocess_skipped"] = result["skipped"]

    return summary


def main():
    args = parse_args()
    jobs = build_group_jobs(args)
    print(f"[batch] found {len(jobs)} group(s)")

    cache_store = {}
    totals = {
        "reconstruct_done": 0,
        "reconstruct_skipped": 0,
        "postprocess_saved": 0,
        "postprocess_skipped": 0,
        "failed": 0,
    }

    for job in jobs:
        try:
            summary = run_group(args, job, cache_store)
        except Exception as exc:
            totals["failed"] += 1
            print(f"[batch] group={job.name} failed: {exc}")
            if not args.continue_on_error:
                raise
            continue

        for key, value in summary.items():
            totals[key] += value

    print(
        "[batch] done "
        f"reconstructed={totals['reconstruct_done']} "
        f"reconstruct_skipped={totals['reconstruct_skipped']} "
        f"smoothed={totals['postprocess_saved']} "
        f"smooth_skipped={totals['postprocess_skipped']} "
        f"failed={totals['failed']}"
    )


if __name__ == "__main__":
    main()
