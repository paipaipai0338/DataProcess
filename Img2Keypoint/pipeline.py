from pathlib import Path
import argparse
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import cv2
import numpy as np
import torch
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_DIR = SCRIPT_DIR / "2D_KeyPoints_helper"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(r"F:\20260703\data_collection"),
        help="Dataset root path, or one group root that contains camera/.",
    )
    parser.add_argument("--groups", nargs="*", default=None, help="Group folders to process. If omitted, all group folders are processed.")
    parser.add_argument("--det_score_thr", type=float, default=0.5, help="Person detection score threshold.")
    parser.add_argument("--det_model", choices=("large", "tiny"), default="large", help="Detector size. tiny is faster and less accurate.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing npz files.")
    return parser.parse_args()


def model_for_pose_init(device):
    init_default_scope("mmpose")
    pose_config = HELPER_DIR / "td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192.py"
    pose_ckpt = HELPER_DIR / "td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192-ffd48c05_20230314.pth"
    pose_model = init_model(str(pose_config), str(pose_ckpt), device=device)
    pose_model.eval()
    return pose_model


def model_for_seg_init(device, model_size="large"):
    init_default_scope("mmdet")
    if model_size == "tiny":
        seg_config = HELPER_DIR / "rtmdet-ins_tiny_8xb32-300e_coco.py"
        seg_ckpt = HELPER_DIR / "rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
    else:
        seg_config = HELPER_DIR / "rtmdet-ins_l_8xb32-300e_coco.py"
        seg_ckpt = HELPER_DIR / "rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth"

    seg_model = init_detector(str(seg_config), str(seg_ckpt), device=device)
    seg_model.eval()
    return seg_model


def expand_xyxy(box_xyxy, img_w, img_h, scale=1.2):
    x1, y1, x2, y2 = box_xyxy
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    nx1 = max(0.0, cx - bw * 0.5)
    ny1 = max(0.0, cy - bh * 0.5)
    nx2 = min(float(img_w - 1), cx + bw * 0.5)
    ny2 = min(float(img_h - 1), cy + bh * 0.5)
    return [nx1, ny1, nx2, ny2]


def empty_result():
    return (
        None,
        np.empty((0, 17, 2), dtype=np.float32),
        np.empty((0, 17), dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )


def run_inference(img_path, det_model, pose_model, det_score_thr=0.5, return_masks=False):
    img_path = Path(img_path)
    masks_np, kpts, kpts_scores, bboxes_np, bboxes_scores_np = empty_result()

    img = cv2.imread(str(img_path))
    if img is None:
        return masks_np, kpts, kpts_scores, bboxes_np, bboxes_scores_np
    img_h, img_w = img.shape[:2]

    with torch.inference_mode():
        init_default_scope("mmdet")
        # Avoid a second disk read by passing the already decoded image.
        result = inference_detector(det_model, img)
        pred = result.pred_instances

        labels = pred.labels
        bboxes = pred.bboxes
        det_scores = pred.scores
        mask_valid = (det_scores > det_score_thr) & (labels == 0)

        bboxes_np = bboxes[mask_valid].detach().cpu().numpy().astype(np.float32)
        bboxes_scores_np = det_scores[mask_valid].detach().cpu().numpy().astype(np.float32)
        if bboxes_np.shape[0] == 0:
            return masks_np, kpts, kpts_scores, bboxes_np, bboxes_scores_np

        if return_masks:
            masks = getattr(pred, "masks", None)
            if masks is not None:
                masks_np = masks[mask_valid].detach().cpu().numpy()

        bboxes_for_pose = np.array(
            [expand_xyxy(b, img_w, img_h, scale=1.2) for b in bboxes_np],
            dtype=np.float32,
        )

        init_default_scope("mmpose")
        pose_results = inference_topdown(
            pose_model,
            img,
            bboxes=bboxes_for_pose,
            bbox_format="xyxy",
        )

    if len(pose_results) == 0:
        return masks_np, kpts, kpts_scores, bboxes_np, bboxes_scores_np

    kpts = np.stack([person.pred_instances.keypoints[0] for person in pose_results]).astype(np.float32)
    kpts_scores = np.stack([person.pred_instances.keypoint_scores[0] for person in pose_results]).astype(np.float32)
    return masks_np, kpts, kpts_scores, bboxes_np, bboxes_scores_np


def iter_group_roots(data_path, groups):
    data_path = Path(data_path)
    if groups is None:
        if (data_path / "camera").is_dir():
            return [(data_path.name, data_path)]
        return sorted(
            (p.name, p)
            for p in data_path.iterdir()
            if p.is_dir() and (p / "camera").is_dir()
        )

    group_roots = []
    for group in groups:
        group_root = data_path / group
        if not (group_root / "camera").is_dir() and data_path.name == group and (data_path / "camera").is_dir():
            group_root = data_path
        group_roots.append((group, group_root))
    return group_roots


def iter_frames(cam_path):
    frames_dir = cam_path / "frames"
    if not frames_dir.exists():
        return []
    return sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    group_roots = iter_group_roots(args.data_path, args.groups)
    print("Groups: ", [group for group, _ in group_roots])

    det_model = model_for_seg_init(device=device, model_size=args.det_model)
    pose_model = model_for_pose_init(device=device)

    for group, group_root in group_roots:
        data_path = group_root / "camera"
        save_path = group_root / "camera results" / "2D"
        save_path.mkdir(parents=True, exist_ok=True)

        cams = sorted(p for p in data_path.iterdir() if p.is_dir())
        for cam_path in cams:
            cam_id = cam_path.name
            save_cam_path = save_path / cam_id
            save_cam_path.mkdir(parents=True, exist_ok=True)

            frame_paths = iter_frames(cam_path)
            for img_path in tqdm(frame_paths, desc=f"Group: {group}, Cam: {cam_id}", ncols=100, total=len(frame_paths)):
                output_path = save_cam_path / f"{img_path.stem}.npz"
                if output_path.exists() and not args.overwrite:
                    continue

                masks_np, kpts, scores, bboxes, bboxes_scores = run_inference(
                    img_path,
                    det_model,
                    pose_model,
                    det_score_thr=args.det_score_thr,
                )
                np.savez(
                    output_path,
                    # masks=masks_np,
                    kpts=kpts,
                    scores=scores,
                    bboxes=bboxes,
                    bboxes_scores=bboxes_scores,
                )


if __name__ == "__main__":
    main()
