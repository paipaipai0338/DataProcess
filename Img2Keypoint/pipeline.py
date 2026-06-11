from pathlib import Path
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown

import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0", help="模型推理使用的计算设备，例如 cuda:0 或 cpu。")
    parser.add_argument('--root_path', type=Path, default=Path(r"E:\20260609_164720\camera"), help="完整数据集的根目录，用于读取输入数据并保存处理结果。")
    parser.add_argument('--data_path', type=Path, default=None, help="兼容旧参数；不指定时使用 root_path。")
    args = parser.parse_args()
    if args.root_path is None and args.data_path is None:
        parser.error("请指定 --root_path，或使用兼容旧参数 --data_path。")
    args.data_path = args.data_path or args.root_path
    return args


def model_for_pose_init(device):
    init_default_scope("mmpose")
    pose_config = './2D_KeyPoints_helper/td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192.py'
    pose_ckpt   = './2D_KeyPoints_helper/td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192-ffd48c05_20230314.pth'
    pose_model = init_model(pose_config, pose_ckpt, device=device)
    return pose_model

def model_for_seg_init(device):
    init_default_scope("mmdet")
    # seg_config = './2D_KeyPoints_helper/rtmdet-ins_tiny_8xb32-300e_coco.py'
    # seg_ckpt   = './2D_KeyPoints_helper/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'
    seg_config = './2D_KeyPoints_helper/rtmdet-ins_l_8xb32-300e_coco.py'
    seg_ckpt = './2D_KeyPoints_helper/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth'

    seg_model = init_detector(seg_config, seg_ckpt, device=device)
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

def run_inference(img_path, det_model, pose_model, det_score_thr=0.5):
    img_path = Path(img_path)
    # 初始化结果
    masks_np, kpts, kpts_scores, bboxes_np, bboxes_scores_np  = np.zeros((0, 1080, 1920)), np.zeros((0, 17, 2)), np.zeros((0, 17,)), np.zeros((0, 4)), np.zeros((0,))
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        return masks_np, kpts, kpts_scores, bboxes_np, bboxes_scores_np
    H, W = img.shape[:2]

    # 目标检测
    init_default_scope("mmdet")
    result = inference_detector(det_model, str(img_path))
    pred = result.pred_instances
    # 获取 Person 类的 bbox
    labels = pred.labels
    bboxes = pred.bboxes
    det_scores = pred.scores
    masks = pred.masks
    mask_valid = (det_scores > det_score_thr) & (labels == 0)
    bboxes_np = bboxes[mask_valid].detach().cpu().numpy().astype(np.float32)
    bboxes_scores_np = det_scores[mask_valid].detach().cpu().numpy().astype(np.float32)

    if int(bboxes_np.shape[0]) == 0:
        # 未检测到人
        return masks_np, kpts, kpts_scores, bboxes_np, bboxes_scores_np
    else:
        # 检测到人
        masks_np = masks[mask_valid].detach().cpu().numpy()
        # 框适当膨胀以防止 pose 检测失败
        bboxes_for_pose = np.array([expand_xyxy(b, W, H, scale=1.2) for b in bboxes_np], dtype=np.float32)
        # 获取 pose 结果
        pose_results = inference_topdown(
            pose_model,
            img,
            bboxes=bboxes_for_pose,
            bbox_format="xyxy"
        )
        kpts = np.stack([person.pred_instances.keypoints[0] for person in pose_results])
        kpts_scores = np.stack([person.pred_instances.keypoint_scores[0] for person in pose_results])

    return masks_np, kpts, kpts_scores, bboxes_np, bboxes_scores_np



if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    data_path = args.data_path / "human"
    save_path = args.data_path / "results" / "2D"
    save_path.mkdir(parents=True, exist_ok=True)
    det_model = model_for_seg_init(device=device)
    pose_model = model_for_pose_init(device=device)

    groups = sorted(p for p in data_path.iterdir() if p.is_dir())
    for group_path in groups:
        group = group_path.name
        cams = sorted(p for p in group_path.iterdir() if p.is_dir())
        for cam_path in cams:
            cam_id = cam_path.name
            save_cam_path = save_path / group / cam_id
            save_cam_path.mkdir(parents=True, exist_ok=True)
            frame_paths = sorted((cam_path / "frames").iterdir())
            for idx, img_path in tqdm(enumerate(frame_paths),desc=f"Group: {group}, Cam: {cam_id}", ncols=100):
                img_name = img_path.stem
                output_path = save_cam_path / f"{img_name}.npz"
                masks_np, kpts, scores, bboxes, bboxes_scores = run_inference(img_path, det_model, pose_model)
                np.savez(
                        output_path,
                        # masks=masks_np,
                        kpts=kpts,
                        scores=scores,
                        bboxes=bboxes,
                        bboxes_scores=bboxes_scores,
                        )
