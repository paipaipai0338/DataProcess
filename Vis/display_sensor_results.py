from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

from RadarProcess import lky as lky_module
from RadarProcess.bin_to_pc import get_bin_pc

sys.modules.setdefault("lky", lky_module)
from RadarProcess.lky_pointcloud_utils import generate_point_cloud_array_from_bin
from TimeProcess.utils import timestamp_to_ms, align_multi_sensor_files
from LidarProcess.utils import read_pcd
from RealSenseProcess.utils import get_realsense_data
from RadarProcess.utils import get_corner_data, get_pc_data
from Img2Keypoint.utils import COCO17_SKELETON
from Calib.utils import apply_transform


if TYPE_CHECKING:
    import torch


def get_gt_data(path: Path|str) -> np.ndarray:
    with open(path, 'rb') as ff:
        gt = pickle.load(ff)
    has_nan = np.isnan(gt).any(axis=(1, 2))  # 形状 (a,)
    gt = gt[~has_nan]  # 形状 (a-1, b, c)
    return gt

def get_gt_boxes(gt: torch.Tensor, gt_mask: torch.Tensor, threshold: float=0.1):
    """
    gt: [B, T, K, J, 3]
    gt_mask: [B, T, K] - 布尔值，True表示有效

    Returns: bbox_3d [B, T, K, 6]
    """
    import torch

    mask_expanded = gt_mask.unsqueeze(-1).unsqueeze(-1).expand_as(gt)

    # 使用 +/-inf 忽略无效人，避免依赖 torch.nanmin/nanmax。
    min_xyz = gt.masked_fill(~mask_expanded, torch.inf).amin(dim=-2) - threshold
    max_xyz = gt.masked_fill(~mask_expanded, -torch.inf).amax(dim=-2) + threshold

    bbox_3d = torch.cat([min_xyz, max_xyz], dim=-1)  # [B, T, K, 6]
    return bbox_3d

def get_gt_boxes_list(
    gt_list: Sequence[np.ndarray],
    threshold: float = 0.3,
) -> List[np.ndarray]:
    """根据逐帧、变长人数的 GT 关节点计算 3D 包围盒。

    Args:
        gt_list:
            长度为 T 的序列。每帧 GT 的形状为 ``[P, J, 3]``，其中
            P 可以随帧变化，也可以为 0。
        threshold:
            包围盒在 x、y、z 三个方向上向外扩展的距离。

    Returns:
        长度为 T 的 list。每帧元素形状为 ``[P, 6]``，最后一维依次为
        ``[xmin, ymin, zmin, xmax, ymax, zmax]``。对于没有人的帧，返回
        形状为 ``[0, 6]`` 的数组。
    """
    if threshold < 0:
        raise ValueError(f"threshold 必须大于等于 0，当前为 {threshold}")

    bbox_list = []

    for frame_idx, frame_gt in enumerate(gt_list):
        frame_gt = np.asarray(frame_gt)

        if frame_gt.ndim != 3 or frame_gt.shape[-1] != 3:
            raise ValueError(
                "每帧 GT 必须为 [P, J, 3]，"
                f"frame_idx={frame_idx}, actual_shape={frame_gt.shape}"
            )

        num_people, num_joints, _ = frame_gt.shape

        if num_joints == 0:
            raise ValueError(
                f"每个人必须至少包含一个关节点，frame_idx={frame_idx}"
            )

        if num_people == 0:
            bbox_list.append(
                np.empty((0, 6), dtype=frame_gt.dtype)
            )
            continue

        if not np.isfinite(frame_gt).all():
            raise ValueError(
                f"GT 中包含 NaN 或 Inf，frame_idx={frame_idx}"
            )

        min_xyz = frame_gt.min(axis=1) - threshold
        max_xyz = frame_gt.max(axis=1) + threshold
        bbox_list.append(
            np.concatenate([min_xyz, max_xyz], axis=-1)
        )

    return bbox_list

def get_gt_detection_targets(
    gt: torch.Tensor,
    gt_mask: torch.Tensor,
    heatmap_shape: tuple = (64, 64),
    xy_limits: tuple = ((0.0, 5.0), (-2.0, 2.0)),
    box_margin: float = 0.1,
    sigma: float = 1.0,
    clip_boxes: bool = True,
    require_full_box_inside: bool = False,
):
    """
    Returns:
        center_heatmap: [B, T, 1, H, W]
            GT heatmap for Center Heatmap Head.
        center_indices: [B, T, K, 2]
            Integer [row, col] index of GT center on heatmap.
        center_offsets: [B, T, K, 2]
            Fractional offset [d_col, d_row] for Center Offset Head.
        box_sizes: [B, T, K, 4]
            GT [left, top, right, bottom] distances in heatmap coordinates.
        inside: [B, T, K]
            Valid mask for offset and size losses.
    """
    import torch

    B, T, K, J, _ = gt.shape
    H, W = heatmap_shape
    device = gt.device
    dtype = gt.dtype

    valid = gt_mask.to(device=device, dtype=torch.bool)

    # ----------------------------
    # 1. Build GT xy boxes from joints
    # ----------------------------
    xy = gt[..., :2]  # [B, T, K, J, 2]
    joint_mask = valid[..., None, None].expand_as(xy)

    min_xy = xy.masked_fill(~joint_mask, torch.inf).amin(dim=-2)   # [B, T, K, 2]
    max_xy = xy.masked_fill(~joint_mask, -torch.inf).amax(dim=-2)  # [B, T, K, 2]

    min_xy = min_xy - box_margin
    max_xy = max_xy + box_margin

    x1 = min_xy[..., 0]
    y1 = min_xy[..., 1]
    x2 = max_xy[..., 0]
    y2 = max_xy[..., 1]

    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    (x_min, x_max), (y_min, y_max) = xy_limits

    # ----------------------------
    # 2. Validity mask
    # ----------------------------
    center_inside = (
        valid
        & (center_x >= x_min) & (center_x <= x_max)
        & (center_y >= y_min) & (center_y <= y_max)
    )

    full_box_inside = (
        valid
        & (x1 >= x_min) & (x2 <= x_max)
        & (y1 >= y_min) & (y2 <= y_max)
    )

    inside = full_box_inside if require_full_box_inside else center_inside

    # ----------------------------
    # 3. World xy -> heatmap continuous coordinates
    #    Standard image/ROIAlign convention:
    #       row corresponds to y
    #       col corresponds to x
    # ----------------------------
    x1_col = (x1 - x_min) / (x_max - x_min) * (W - 1)
    x2_col = (x2 - x_min) / (x_max - x_min) * (W - 1)
    y1_row = (y1 - y_min) / (y_max - y_min) * (H - 1)
    y2_row = (y2 - y_min) / (y_max - y_min) * (H - 1)

    center_col = (center_x - x_min) / (x_max - x_min) * (W - 1)
    center_row = (center_y - y_min) / (y_max - y_min) * (H - 1)

    if clip_boxes:
        x1_col = x1_col.clamp(0, W - 1)
        x2_col = x2_col.clamp(0, W - 1)
        y1_row = y1_row.clamp(0, H - 1)
        y2_row = y2_row.clamp(0, H - 1)

    # ----------------------------
    # 4. Center index and offset
    # ----------------------------
    center_col_int = center_col.floor().clamp(0, W - 1)
    center_row_int = center_row.floor().clamp(0, H - 1)

    offset_col = center_col - center_col_int
    offset_row = center_row - center_row_int

    center_indices = torch.stack(
        [center_row_int.long(), center_col_int.long()],
        dim=-1,
    )  # [B, T, K, 2], [row, col]

    center_offsets = torch.stack(
        [offset_col, offset_row],
        dim=-1,
    )  # [B, T, K, 2], [dx/col, dy/row]

    # ----------------------------
    # 5. Box size target: [left, top, right, bottom]
    #    All in heatmap coordinates.
    # ----------------------------
    left = center_col - x1_col
    top = center_row - y1_row
    right = x2_col - center_col
    bottom = y2_row - center_row

    box_sizes = torch.stack(
        [left, top, right, bottom],
        dim=-1,
    )  # [B, T, K, 4]

    # Avoid invalid targets polluting later operations.
    center_offsets = center_offsets.masked_fill(~inside[..., None], 0.0)
    box_sizes = box_sizes.masked_fill(~inside[..., None], 0.0)

    # ----------------------------
    # 6. Center heatmap target
    # ----------------------------
    row_for_hm = center_row_int.masked_fill(~inside, 0).reshape(B, T, K, 1, 1)
    col_for_hm = center_col_int.masked_fill(~inside, 0).reshape(B, T, K, 1, 1)

    rows = torch.arange(H, device=device, dtype=dtype).reshape(1, 1, 1, H, 1)
    cols = torch.arange(W, device=device, dtype=dtype).reshape(1, 1, 1, 1, W)

    dist_sq = (rows - row_for_hm).square() + (cols - col_for_hm).square()

    heatmaps_per_person = torch.exp(-dist_sq / (sigma ** 2))
    heatmaps_per_person = heatmaps_per_person * inside[..., None, None].to(dtype)

    center_heatmap = heatmaps_per_person.sum(dim=2, keepdim=True)

    return center_heatmap, center_indices, center_offsets, box_sizes, inside


def _filter_points_by_gt_boxes(points, boxes_high, R_to_high=None, t_to_high=None, return_mask=False):
    """Keep complete point rows inside any GT box in native high coordinates."""
    if points is None:
        return None
    points = np.asarray(points)
    if points.ndim == 1 and points.size == 0:
        points = points.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Point cloud must be [N, >=3], got {points.shape}")
    if len(points) == 0 or len(boxes_high) == 0:
        return np.zeros(len(points), dtype=bool) if return_mask else points[:0]
    xyz_high = points[:, :3]
    if R_to_high is not None:
        xyz_high = xyz_high @ R_to_high.T + np.asarray(t_to_high).reshape(3)
    # Only allocate per-point masks, avoiding a [people, points, 3] array.
    mask = np.zeros(len(points), dtype=bool)
    for box in boxes_high:
        mask |= ((xyz_high >= box[:3]) & (xyz_high <= box[3:])).all(axis=1)
    mask &= np.isfinite(xyz_high).all(axis=1)
    return mask if return_mask else points[mask]


def _box_corners(boxes):
    """Eight corners of each native high box; transform corners, never re-box."""
    bits = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    return boxes[:, None, :3] + bits[None, :, :] * (boxes[:, None, 3:] - boxes[:, None, :3])


def _plot_boxes_matplotlib(ax, corners, color):
    edges = [(i, i ^ step) for i in range(8) for step in (1, 2, 4) if i < (i ^ step)]
    for person_idx, vertices in enumerate(corners):
        for edge_idx, (start, end) in enumerate(edges):
            segment = vertices[[start, end]]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=1.2,
                    label="GT boxes" if person_idx == 0 and edge_idx == 0 else None)


def _plot_coordinate_frame(ax, size=1.0):
    origin = np.zeros(3)
    axes = (
        ("X", np.array([size, 0, 0]), "r"),
        ("Y", np.array([0, size, 0]), "g"),
        ("Z", np.array([0, 0, size]), "b"),
    )
    for label, direction, color in axes:
        ax.plot(
            [origin[0], direction[0]],
            [origin[1], direction[1]],
            [origin[2], direction[2]],
            color=color,
            linewidth=1.5,
        )
        ax.text(direction[0], direction[1], direction[2], label, color=color)


def _plot_gt_matplotlib(ax, gt, color, label):
    gt = np.asarray(gt)
    if gt.size == 0:
        return

    if gt.ndim == 2:
        max_index = max(max(pair) for pair in COCO17_SKELETON) + 1
        if gt.shape[0] % max_index != 0:
            points = gt[:, :3]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=12, color=[color], label=label)
            return
        gt = gt.reshape(gt.shape[0] // max_index, max_index, 3)

    points = gt.reshape(-1, gt.shape[-1])[:, :3]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=12, color=[color], label=label)

    for person_points in gt:
        person_points = person_points[:, :3]
        for start, end in COCO17_SKELETON:
            if start < len(person_points) and end < len(person_points):
                segment = person_points[[start, end]]
                ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=1.0)

def random_uniform_sample(points, target_points=5000):
    """随机采样固定数量点，天然均匀"""
    if len(points) <= target_points:
        return points
    indices = np.random.choice(len(points), target_points, replace=False)
    return points[indices]
def _plot_multi_sensor_frame_matplotlib(
        fig,
        ax,
        data,
        colors,
        frame_idx,
):
    ax.cla()
    _plot_coordinate_frame(ax, size=1.0)

    has_label = False
    has_outside_label = False
    for data_key, data_value in data.items():
        if data_value is None:
            continue

        data_value = np.asarray(data_value)
        if data_value.size == 0:
            continue

        color = colors.get(data_key, [0.7, 0.7, 0.7])
        if data_key == "gt_boxes":
            _plot_boxes_matplotlib(ax, data_value, color)
            has_label = True
            continue
        if data_key == "gt":
            _plot_gt_matplotlib(ax, data_value, color, data_key)
            has_label = True
            continue

        points = data_value.reshape(-1, data_value.shape[-1])[:, :3]
        outside = data_key.endswith("_outside")
        label = ("Outside GT boxes" if not has_outside_label else None) if outside else data_key
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2 if outside else 4,
                   color=[color], alpha=0.3 if outside else 1.0, label=label)
        has_outside_label |= outside
        has_label = True

    ax.set_title(f"Frame {frame_idx:04d}")
    ax.set_xlim(0, 5)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-2.2, 2.2)
    ax.set_xlabel("X/m")
    ax.set_ylabel("Y/m")
    ax.set_zlabel("Z/m")
    if has_label:
        ax.legend(loc="upper right")
    fig.canvas.draw_idle()


class _SensorPlayback:
    """Play processed frames using the WebAgg event loop."""

    def __init__(self, frames, colors, play_interval_sec):
        self.frames = frames
        self.colors = colors
        self.index = 0
        self.playing = len(frames) > 1
        self.fig = plt.figure("Multi-Sensor Playback")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(bottom=0.25)
        self.previous = Button(self.fig.add_axes([0.20, 0.13, 0.18, 0.06]), "Previous")
        self.pause = Button(self.fig.add_axes([0.41, 0.13, 0.18, 0.06]), "Pause")
        self.next = Button(self.fig.add_axes([0.62, 0.13, 0.18, 0.06]), "Next")
        self.progress = Slider(
            self.fig.add_axes([0.20, 0.06, 0.60, 0.03]),
            "Frame", 0, max(1, len(frames) - 1), valinit=0, valstep=1,
            valfmt="%d",
        )
        self.progress.set_active(len(frames) > 1)
        self.previous.on_clicked(lambda event: self.step(-1))
        self.next.on_clicked(lambda event: self.step(1))
        self.pause.on_clicked(self.toggle)
        self.progress.on_changed(self.seek)
        self.timer = self.fig.canvas.new_timer(interval=max(100, int(play_interval_sec * 1000)))
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect("close_event", lambda event: self.timer.stop())
        # Keep widgets and their callbacks alive throughout plt.show().
        self.fig._sensor_playback = self
        self.draw()
        self.set_playing(self.playing)

    def set_playing(self, playing):
        self.playing = playing and len(self.frames) > 1
        self.pause.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def draw(self):
        frame_idx, data = self.frames[self.index]
        _plot_multi_sensor_frame_matplotlib(self.fig, self.ax, data, self.colors, frame_idx)
        self.ax.set_title(f"Frame {frame_idx:04d} ({self.index + 1}/{len(self.frames)})")
        self.progress.eventson = False
        try:
            self.progress.set_val(self.index)
        finally:
            self.progress.eventson = True
        self.fig.canvas.draw_idle()

    def step(self, offset):
        self.set_playing(False)
        self.index = max(0, min(len(self.frames) - 1, self.index + offset))
        self.draw()

    def seek(self, value):
        self.set_playing(False)
        self.index = max(0, min(len(self.frames) - 1, int(value)))
        self.draw()

    def toggle(self, event):
        if not self.playing and self.index == len(self.frames) - 1:
            self.index = 0
            self.draw()
        self.set_playing(not self.playing)

    def tick(self):
        if not self.playing:
            return
        if self.index < len(self.frames) - 1:
            self.index += 1
            self.draw()
        if self.index == len(self.frames) - 1:
            self.set_playing(False)


def main():
    root_path = Path(r'F:\20261010\data_collection\group_011')
    calib_path = root_path.parent.parent / 'calib'
    calib_flag = True
    select_flag = True
    select_threshold = 0.3  # high 原生坐标系中包围盒向外扩展的距离（米）
    continuous_play_flag = True
    play_interval_sec = 0.1
    vis_scene_center = [0, 0, 0]
    vis_scene_size = 40
    # vis_data = ['radar_low_bin', 'radar_high_bin', 'radar_low_pc', 'radar_high_pc', 'realsense', 'lidar', 'gt']
    # vis_data = ['radar_low_bin', 'radar_high_bin', 'radar_low_pc', 'radar_high_pc', 'gt']
    # vis_data = ['radar_low_bin', 'radar_high_bin', 'realsense', 'lidar']
    # vis_data = ['radar_high_pc', 'gt', 'radar_high_bin']
    vis_data = ['radar_high_pc', 'gt']


    if calib_flag or select_flag:
        extrinsic_img_to_radar_low = np.load(calib_path / 'extrinsic_img_to_radar_low.npz')
        extrinsic_img_to_radar_high = np.load(calib_path / 'extrinsic_img_to_radar_high.npz')
        extrinsic_realsense_to_radar = np.load(calib_path / 'extrinsic_realsense_to_radar.npz')
        if calib_flag:
            R_high_to_low = extrinsic_img_to_radar_low['R_est'] @ extrinsic_img_to_radar_high['R_est'].T
            t_high_to_low = (
                extrinsic_img_to_radar_low['t_est'].reshape(3)
                - R_high_to_low @ extrinsic_img_to_radar_high['t_est'].reshape(3)
            )
        if select_flag:
            # Low -> camera -> high. LiDAR already uses the low/display frame.
            R_low_to_high = extrinsic_img_to_radar_high['R_est'] @ extrinsic_img_to_radar_low['R_est'].T
            t_low_to_high = (
                extrinsic_img_to_radar_high['t_est'].reshape(3)
                - R_low_to_high @ extrinsic_img_to_radar_low['t_est'].reshape(3)
            )



    radar_low_cfar_params = {
        "ref_range": 8,
        "ref_velocity": 8,
        "guard_range": 4,
        "guard_velocity": 4,
        "alpha": 10.0,
        "mode": "ca",

    }
    radar_high_cfar_params = {
        "ref_range": 9,
        "ref_velocity": 8,
        "guard_range": 8,
        "guard_velocity": 4,
        "alpha": 10.0,
        "mode": "ca",
    }

    radar_low_path = root_path / 'dpct低位机'
    radar_low_bin_path = radar_low_path / 'Bin'
    radar_low_pc_path = radar_low_path / 'PC'

    radar_high_path = root_path / 'dpct高位机'
    radar_high_bin_path = radar_high_path / 'Bin'
    radar_high_pc_path = radar_high_path / 'PC'

    lidar_path = root_path / 'robosense'

    realsense_path = root_path / 'realsense' / 'aligned_depth'

    gt_path = root_path / 'camera results' / 'smoothed 3D'

    sensors = {
        'lidar': lidar_path if 'lidar' in vis_data else None,
        'radar_low_bin': radar_low_bin_path if 'radar_low_bin' in vis_data else None,
        'radar_high_bin': radar_high_bin_path if 'radar_high_bin' in vis_data else None,
        'radar_low_pc': radar_low_pc_path if 'radar_low_pc' in vis_data else None,
        'radar_high_pc': radar_high_pc_path if 'radar_high_pc' in vis_data else None,
        'gt': gt_path if 'gt' in vis_data or select_flag else None,
        'realsense': realsense_path if 'realsense' in vis_data else None,
    }

    # 自定义后缀（如果需要）
    suffix_map = {
        'lidar': '.pcd',
        'radar_low_bin': '.bin',
        'radar_high_bin': '.bin',
        'radar_low_pc': '.npy',
        'radar_high_pc': '.npy',
        'gt': '.pkl',
        'realsense': '.bin',
    }

    # 执行对齐
    result = align_multi_sensor_files(
        sources=sensors,
        max_delta_sec=0.05,
        one_to_one=True,
        base_source='radar_high_pc',  # 可选，不指定则自动选择
        suffix_map=suffix_map,
        # time_offsets_sec={
        #     "gt": -0.2
        # }
    )

    base_files_matched = result.get('radar_high_pc') or next((files for files in result.values() if files), [])
    frames = len(base_files_matched)

    def get_matched_files(name):
        return result.get(name) or [None] * frames

    lidar_files_matched = get_matched_files('lidar')
    radar_low_bin_files_matched = get_matched_files('radar_low_bin')
    radar_low_pc_files_matched = get_matched_files('radar_low_pc')
    radar_high_bin_files_matched = get_matched_files('radar_high_bin')
    radar_high_pc_files_matched = get_matched_files('radar_high_pc')
    gt_files_matched = get_matched_files('gt')
    realsense_files_matched = get_matched_files('realsense')

    colors = {
        'lidar_pcd': [0.5, 0.5, 0.5],  # 中灰色 (RGB: 128, 128, 128)

        'pc_from_bin_radar_low': [1.0, 0.5, 0.0],  # 纯橙色 (RGB: 255, 128, 0)
        'pc_radar_low': [1.0, 0.75, 0.25],  # 浅橙色/金橙色 (RGB: 255, 191, 64)

        'pc_from_bin_radar_high': [0.0, 0.0, 1.0],  # 纯蓝色 (RGB: 0, 0, 255)
        'pc_radar_high': [0.25, 0.55, 1.0],  # 天蓝色/亮蓝色 (RGB: 64, 140, 255)

        'gt_boxes': [0.1, 0.8, 0.2],
        'gt': [1.0, 0.0, 0.0],  # 纯红色 (RGB: 255, 0, 0)

        'realsense_depth': [0.0, 1.0, 0.0],  # 纯绿色 (RGB: 0, 255, 0)
    }

    processed_frames = []

    for frame_idx in range(0, frames, 1):
        print('*' * 50, f'frame: {frame_idx}', '*' * 50)
        (
            lidar_pcd,
            pc_from_bin_radar_low,
            pc_from_bin_radar_high,
            pc_radar_low,
            pc_radar_high,
            gt,
            realsense_depth,
        ) = [None] * 7
        # ========== LiDAR ==========
        if lidar_files_matched[frame_idx]:
            lidar_pcd = read_pcd(lidar_files_matched[frame_idx])
            lidar_time_samp = Path(lidar_files_matched[frame_idx]).stem
            lidar_time_ms = timestamp_to_ms(lidar_time_samp)
            print(f"LiDAR: {lidar_time_ms}, Shape: {lidar_pcd.shape}")
        # ========== 低位雷达 ==========
        if radar_low_bin_files_matched[frame_idx] and 'radar_low_bin' in vis_data:
            '''信号处理方式1'''
            targets_low = get_corner_data(Path(radar_low_bin_files_matched[frame_idx]), **radar_low_cfar_params)
            pc_from_bin_radar_low = targets_low["cartesian coordinate"]
            '''信号处理方式2'''
            pc_from_bin_radar_low = generate_point_cloud_array_from_bin(
                str(radar_low_bin_files_matched[frame_idx]),
            )
            '''信号处理方式3'''
            pc_from_bin_radar_low = get_bin_pc(
                str(radar_low_bin_files_matched[frame_idx]),
            )

            radar_low_time_samp = Path(radar_low_bin_files_matched[frame_idx]).stem
            radar_low_time_ms = timestamp_to_ms(radar_low_time_samp)
            print(f"Radar Bin Low: {radar_low_time_ms}, Shape: {pc_from_bin_radar_low.shape}")
        else:
            print("Radar Bin Low: None")
        if radar_low_pc_files_matched[frame_idx] and 'radar_low_pc' in vis_data:
            pc_radar_low = get_pc_data(Path(radar_low_pc_files_matched[frame_idx]))
            radar_low_time_samp = Path(radar_low_pc_files_matched[frame_idx]).stem
            radar_low_time_ms = timestamp_to_ms(radar_low_time_samp)
            print(f"Radar PC Low: {radar_low_time_ms}, Shape: {pc_radar_low.shape}")
        else:
            print("Radar PC Low: None")
        # ========== 高位雷达 ==========
        if radar_high_bin_files_matched[frame_idx] and 'radar_high_bin' in vis_data:
            '''信号处理方式1'''
            targets_high = get_corner_data(Path(radar_high_bin_files_matched[frame_idx]), **radar_high_cfar_params)
            pc_from_bin_radar_high = targets_high["cartesian coordinate"]
            '''信号处理方式2'''
            pc_from_bin_radar_high = generate_point_cloud_array_from_bin(
                str(radar_high_bin_files_matched[frame_idx]),
            )
            '''信号处理方式3'''
            pc_from_bin_radar_high = get_bin_pc(
                str(radar_high_bin_files_matched[frame_idx]),
            )
            radar_high_time_samp = Path(radar_high_bin_files_matched[frame_idx]).stem
            radar_high_time_ms = timestamp_to_ms(radar_high_time_samp)
            print(f"Radar Bin High: {radar_high_time_ms}, Shape: {pc_from_bin_radar_high.shape}")
        else:
            print("Radar Bin High: None")

        if radar_high_pc_files_matched[frame_idx] and 'radar_high_pc' in vis_data:
            pc_radar_high = get_pc_data(Path(radar_high_pc_files_matched[frame_idx]))
            radar_high_time_samp = Path(radar_high_pc_files_matched[frame_idx]).stem
            radar_high_time_ms = timestamp_to_ms(radar_high_time_samp)
            print(f"Radar PC High: {radar_high_time_ms}, Shape: {pc_radar_high.shape}")
        else:
            print("Radar PC High: None")
        # ========== GT ==========
        gt_high = None
        if gt_files_matched[frame_idx] is not None and ('gt' in vis_data or select_flag):
            gt = get_gt_data(gt_files_matched[frame_idx])
            gt_time_samp = Path(gt_files_matched[frame_idx]).stem
            gt_time_ms = timestamp_to_ms(gt_time_samp)
            print(f"GT: {gt_time_ms}, Shape: {gt.shape}")
            num_person, num_joint, _ = gt.shape
            if select_flag:
                gt_high = apply_transform(
                    gt.reshape(-1, 3),
                    R=extrinsic_img_to_radar_high['R_est'],
                    t=extrinsic_img_to_radar_high['t_est'],
                ).reshape(num_person, num_joint, 3)
            '坐标转换'
            if calib_flag and num_person > 0:
                gt = apply_transform(gt.reshape(-1, 3), R=extrinsic_img_to_radar_low['R_est'], t=extrinsic_img_to_radar_low['t_est']).reshape(num_person, num_joint, 3)
        else:
            print("GT: None")
        # ========== RealSense ==========
        if realsense_files_matched[frame_idx] is not None and 'realsense' in vis_data:
            realsense_depth = get_realsense_data(Path(realsense_files_matched[frame_idx]))
            realsense_time_samp = Path(realsense_files_matched[frame_idx]).stem
            realsense_time_ms = timestamp_to_ms(realsense_time_samp)
            if realsense_depth is not None and len(realsense_depth) > 0:
                # 过滤无效点（深度为 0 或无穷）
                valid_mask = (realsense_depth[:, 2] > 0) & np.isfinite(realsense_depth[:, 2])
                realsense_depth = realsense_depth[valid_mask]
                print(f"Realsense: {realsense_time_ms}, Shape: {realsense_depth.shape}")
            '坐标转换'
            if calib_flag:
                realsense_depth = apply_transform(realsense_depth, R=extrinsic_realsense_to_radar['R_est'], t=extrinsic_realsense_to_radar['t_est'])
        else:
            print("Realsense: None")

        data_for_vis = {
            'lidar_pcd': lidar_pcd if 'lidar' in vis_data else None,
            'pc_from_bin_radar_low': pc_from_bin_radar_low if 'radar_low_bin' in vis_data else None,
            'pc_from_bin_radar_high': pc_from_bin_radar_high if 'radar_high_bin' in vis_data else None,
            'pc_radar_low': pc_radar_low if 'radar_low_pc' in vis_data else None,
            'pc_radar_high': pc_radar_high if 'radar_high_pc' in vis_data else None,
            'gt': gt if 'gt' in vis_data else None,
            'realsense_depth' : realsense_depth if 'realsense' in vis_data else None,
        }

        if select_flag:
            boxes_high = (
                get_gt_boxes_list([gt_high], threshold=select_threshold)[0]
                if gt_high is not None else np.empty((0, 6))
            )
            for key, points in list(data_for_vis.items()):
                if key == 'gt' or points is None:
                    continue
                points = np.asarray(points)
                if points.ndim == 1 and points.size == 0:
                    points = points.reshape(0, 3)
                if key in {'pc_radar_high', 'pc_from_bin_radar_high'}:
                    mask = _filter_points_by_gt_boxes(points, boxes_high, return_mask=True)
                else:
                    R_to_high, t_to_high = R_low_to_high, t_low_to_high
                    if key == 'realsense_depth' and not calib_flag:
                        # Uncalibrated RealSense still needs depth -> low -> high for selection.
                        R_depth_to_low = extrinsic_realsense_to_radar['R_est']
                        t_depth_to_low = extrinsic_realsense_to_radar['t_est'].reshape(3)
                        R_to_high = R_low_to_high @ R_depth_to_low
                        t_to_high = R_low_to_high @ t_depth_to_low + t_low_to_high
                    mask = _filter_points_by_gt_boxes(points, boxes_high, R_to_high, t_to_high, return_mask=True)
                data_for_vis[key] = points[mask]
                outside_key = key + '_outside'
                data_for_vis[outside_key] = points[~mask & np.isfinite(points[:, :3]).all(axis=1)]
                colors[outside_key] = [0.65, 0.65, 0.65]
                print(f"{key} selected in high coordinates: {len(points)} -> {int(mask.sum())}")
            corners = _box_corners(boxes_high)
            data_for_vis['gt_boxes'] = (
                corners @ R_high_to_low.T + t_high_to_low if calib_flag else corners
            )

        # High point clouds stay native until selection is complete.
        if calib_flag:
            for key in ('pc_from_bin_radar_high', 'pc_radar_high',
                        'pc_from_bin_radar_high_outside', 'pc_radar_high_outside'):
                points = data_for_vis.get(key)
                if points is not None and len(points):
                    xyz_low = points[:, :3] @ R_high_to_low.T + t_high_to_low
                    data_for_vis[key] = (
                        np.column_stack((xyz_low, points[:, 3:])) if points.shape[1] > 3 else xyz_low
                    )

        if continuous_play_flag:
            # Sample once so seeking back to a frame shows the same points.
            for key in ("lidar_pcd", "realsense_depth", "lidar_pcd_outside", "realsense_depth_outside"):
                if data_for_vis.get(key) is not None:
                    data_for_vis[key] = random_uniform_sample(data_for_vis[key], target_points=500)
            processed_frames.append((frame_idx, data_for_vis))
        else:
            from Vis.utils import visualize_multi_sensor

            visualize_multi_sensor(
                data_for_vis,
                colors,
                window_name=f"Frame {frame_idx:04d}",
                scene_center=vis_scene_center,
                scene_size=vis_scene_size,
            )

    if continuous_play_flag:
        if not processed_frames:
            print("No frames available for playback.")
            return
        print(f"Processed {len(processed_frames)} frames. Starting WebAgg playback.")
        plt.switch_backend("WebAgg")
        plt.ioff()
        _SensorPlayback(processed_frames, colors, play_interval_sec)
        plt.show()


if __name__ == '__main__':
    main()
