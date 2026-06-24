import os
import pickle
import sys
from pathlib import Path

import numpy as np

from RadarProcess import lky as lky_module
from RadarProcess.bin_to_pc import get_bin_pc

sys.modules.setdefault("lky", lky_module)
from RadarProcess.lky_pointcloud_utils import generate_point_cloud_array_from_bin
from Vis.utils import (
    close_multi_sensor_player,
    create_multi_sensor_player,
    play_multi_sensor_frame,
    visualize_multi_sensor,
)
from TimeProcess.utils import timestamp_to_ms, align_multi_sensor_files
from LidarProcess.utils import read_pcd
from RealSenseProcess.utils import get_realsense_data
from RadarProcess.utils import get_corner_data, get_pc_data
from Img2Keypoint.utils import get_gt_data
from Calib.utils import apply_transform

def main():
    root_path = Path(r'G:\20260615\data_collection\group_025')
    calib_path = Path(r'G:\20260615\calib')
    calib_flag = True
    select_flag = False
    continuous_play_flag = False
    play_interval_sec = 0.1
    vis_scene_center = [0, 0, 0]
    vis_scene_size = 40
    vis_data = ['radar_low_bin', 'radar_high_bin', 'radar_low_pc', 'radar_high_pc', 'realsense', 'lidar', 'gt']
    # vis_data = ['radar_low_bin', 'radar_high_bin', 'realsense', 'lidar']
    # vis_data = ['radar_high_pc', 'gt']


    if calib_flag:
        extrinsic_img_to_radar_low = np.load(calib_path / 'extrinsic_img_to_radar_low.npz')
        extrinsic_img_to_radar_high = np.load(calib_path / 'extrinsic_img_to_radar_high.npz')
        extrinsic_realsense_to_radar = np.load(calib_path / 'extrinsic_realsense_to_radar.npz')


    radar_low_cfar_params = {
        "ref_range": 8,
        "ref_velocity": 8,
        "guard_range": 4,
        "guard_velocity": 4,
        "alpha": 10.0,
        "mode": "ca",
    }
    radar_high_cfar_params = {
        "ref_range": 10,
        "ref_velocity": 10,
        "guard_range": 4,
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

    realsense_path = root_path / 'realsense' / 'undistorted_depth'

    gt_path = root_path / 'camera results' / 'smoothed 3D'

    sensors = {
        'lidar': lidar_path if 'lidar' in vis_data else None,
        'radar_low_bin': radar_low_bin_path if 'radar_low_bin' in vis_data else None,
        'radar_high_bin': radar_high_bin_path if 'radar_high_bin' in vis_data else None,
        'radar_low_pc': radar_low_pc_path if 'radar_low_pc' in vis_data else None,
        'radar_high_pc': radar_high_pc_path if 'radar_high_pc' in vis_data else None,
        'gt': gt_path if 'gt' in vis_data else None,
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
        base_source='radar_low_pc',  # 可选，不指定则自动选择
        suffix_map=suffix_map,
        time_offsets_sec={
            "gt": -0.2
        }
    )

    base_files_matched = result.get('radar_low_pc') or next((files for files in result.values() if files), [])
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

        'gt': [1.0, 0.0, 0.0],  # 纯红色 (RGB: 255, 0, 0)

        'realsense_depth': [0.0, 1.0, 0.0],  # 纯绿色 (RGB: 0, 255, 0)
    }

    player = None
    if continuous_play_flag:
        player = create_multi_sensor_player(
            colors,
            window_name="Multi-Sensor Playback",
            frame_interval_sec=play_interval_sec,
            scene_center=vis_scene_center,
            scene_size=vis_scene_size,
        )

    for frame_idx in range(frames):
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
            '坐标转换'
            if calib_flag:
                R_inv = extrinsic_img_to_radar_high['R_est'].T
                t_inv = -extrinsic_img_to_radar_high['R_est'].T @ extrinsic_img_to_radar_high['t_est'].T
                pc_from_bin_radar_high = apply_transform(pc_from_bin_radar_high, R_inv, t_inv)
                pc_from_bin_radar_high = apply_transform(pc_from_bin_radar_high, extrinsic_img_to_radar_low['R_est'], extrinsic_img_to_radar_low['t_est'])
        else:
            print("Radar Bin High: None")

        if radar_high_pc_files_matched[frame_idx] and 'radar_high_pc' in vis_data:
            pc_radar_high = get_pc_data(Path(radar_high_pc_files_matched[frame_idx]))
            radar_high_time_samp = Path(radar_high_pc_files_matched[frame_idx]).stem
            radar_high_time_ms = timestamp_to_ms(radar_high_time_samp)
            print(f"Radar PC High: {radar_high_time_ms}, Shape: {pc_radar_high.shape}")
            '坐标转换'
            if calib_flag:
                R_inv = extrinsic_img_to_radar_high['R_est'].T
                t_inv = -extrinsic_img_to_radar_high['R_est'].T @ extrinsic_img_to_radar_high['t_est'].T
                pc_radar_high = apply_transform(pc_radar_high, R_inv, t_inv)
                pc_radar_high = apply_transform(pc_radar_high, extrinsic_img_to_radar_low['R_est'], extrinsic_img_to_radar_low['t_est'])
        else:
            print("Radar PC High: None")
        # ========== GT ==========
        if gt_files_matched[frame_idx] is not None and 'gt' in vis_data:
            gt = get_gt_data(gt_files_matched[frame_idx])
            gt_time_samp = Path(gt_files_matched[frame_idx]).stem
            gt_time_ms = timestamp_to_ms(gt_time_samp)
            print(f"GT: {gt_time_ms}, Shape: {gt.shape}")
            num_person, num_joint, _ = gt.shape
            if num_person == 0:
                continue
            '坐标转换'
            if calib_flag:
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

        bbox_3d = None
        if gt is not None:
            threshold  = 0.2
            min_xyz = gt.min(axis=1) - threshold   # shape: (num_person, 3)
            max_xyz = gt.max(axis=1) + threshold   # shape: (num_person, 3)

            # 组合成 [x_min, y_min, z_min, x_max, y_max, z_max] 格式
            bbox_3d = np.concatenate([min_xyz, max_xyz], axis=1)  # shape: (num_person, 6)
        if select_flag and gt is not None and bbox_3d is not None:
            if lidar_pcd is not None:
                xyz = lidar_pcd[:, :3]
                diff_min = xyz[None, :, :] - bbox_3d[:, None, :3]
                diff_max = bbox_3d[:, None, 3:] - xyz[None, :, :]
                in_bounds = (diff_min >= 0) & (diff_max >= 0)
                valid_in_bbox = np.all(in_bounds, axis=-1)
                point_mask = np.any(valid_in_bbox, axis=0)
                lidar_pcd = lidar_pcd[point_mask]
                print("lidar_pcd selected: ", lidar_pcd.shape)
            if pc_from_bin_radar_low is not None:
                xyz = pc_from_bin_radar_low[:, :3]
                diff_min = xyz[None, :, :] - bbox_3d[:, None, :3]
                diff_max = bbox_3d[:, None, 3:] - xyz[None, :, :]
                in_bounds = (diff_min >= 0) & (diff_max >= 0)
                valid_in_bbox = np.all(in_bounds, axis=-1)
                point_mask = np.any(valid_in_bbox, axis=0)
                pc_from_bin_radar_low = pc_from_bin_radar_low[point_mask]
                print("pc_from_bin_radar_low selected: ", pc_from_bin_radar_low.shape)
            if pc_from_bin_radar_high is not None:
                xyz = pc_from_bin_radar_high[:, :3]
                diff_min = xyz[None, :, :] - bbox_3d[:, None, :3]
                diff_max = bbox_3d[:, None, 3:] - xyz[None, :, :]
                in_bounds = (diff_min >= 0) & (diff_max >= 0)
                valid_in_bbox = np.all(in_bounds, axis=-1)
                point_mask = np.any(valid_in_bbox, axis=0)
                pc_from_bin_radar_high = pc_from_bin_radar_high[point_mask]
                print("pc_from_bin_radar_high selected: ", pc_from_bin_radar_high.shape)
            if pc_radar_low is not None:
                xyz = pc_radar_low[:, :3]
                diff_min = xyz[None, :, :] - bbox_3d[:, None, :3]
                diff_max = bbox_3d[:, None, 3:] - xyz[None, :, :]
                in_bounds = (diff_min >= 0) & (diff_max >= 0)
                valid_in_bbox = np.all(in_bounds, axis=-1)
                point_mask = np.any(valid_in_bbox, axis=0)
                pc_radar_low = pc_radar_low[point_mask]
                print("pc_radar_low selected: ", pc_radar_low.shape)
            if pc_radar_high is not None:
                xyz = pc_radar_high[:, :3]
                diff_min = xyz[None, :, :] - bbox_3d[:, None, :3]
                diff_max = bbox_3d[:, None, 3:] - xyz[None, :, :]
                in_bounds = (diff_min >= 0) & (diff_max >= 0)
                valid_in_bbox = np.all(in_bounds, axis=-1)
                point_mask = np.any(valid_in_bbox, axis=0)
                pc_radar_high = pc_radar_high[point_mask]
                print("pc_radar_high selected: ", pc_radar_high.shape)


        data_for_vis = {
            'lidar_pcd': lidar_pcd if 'lidar' in vis_data else None,
            'pc_from_bin_radar_low': pc_from_bin_radar_low if 'radar_low_bin' in vis_data else None,
            'pc_from_bin_radar_high': pc_from_bin_radar_high if 'radar_high_bin' in vis_data else None,
            'pc_radar_low': pc_radar_low if 'radar_low_pc' in vis_data else None,
            'pc_radar_high': pc_radar_high if 'radar_high_pc' in vis_data else None,
            'gt': gt if 'gt' in vis_data else None,
            'realsense_depth' : realsense_depth if 'realsense' in vis_data else None,
        }

        if continuous_play_flag:
            if not play_multi_sensor_frame(player, data_for_vis):
                break
        else:
            visualize_multi_sensor(
                data_for_vis,
                colors,
                window_name=f"Frame {frame_idx:04d}",
                scene_center=vis_scene_center,
                scene_size=vis_scene_size,
            )

    if player is not None:
        close_multi_sensor_player(player)


if __name__ == '__main__':
    main()
