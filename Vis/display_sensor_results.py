import os
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d

from TimeProcess.utils import timestamp_to_ms, align_multi_sensor_files
from LidarProcess.utils import read_pcd
from RealSenseProcess.utils import get_realsense_data
from RadarProcess.utils import get_corner_data, get_pc_data
from Img2Keypoint.utils import get_gt_data

def visualize_multi_sensor(data: dict, colors: dict, window_name="Multi-Sensor Visualization", window_size=(1280, 720), show_coord_frame=True, coord_frame_size=1.0):
    """
    可视化多个传感器点云数据

    Parameters:
        data: dict, 键为传感器名称，值为 numpy数组 (N,3) 或 None
        colors: dict, 键为传感器名称，值为 RGB 颜色列表 [R,G,B], 范围 0~1
        window_name: str, 窗口标题
        window_size: tuple, 窗口大小 (width, height)
        show_coord_frame: bool, 是否显示坐标轴
        coord_frame_size: float, 坐标轴大小
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=window_size[0], height=window_size[1])

    # 添加坐标轴（可选）
    if show_coord_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size)
        vis.add_geometry(coord_frame)

    # 遍历数据，为每个非空点云创建几何体并添加到窗口
    for name, points in data.items():
        if points is None or len(points) == 0:
            continue
        # radar x,y,z,v,snr,type
        if points.shape[-1] != 3:
            points = points[..., :3]
        # gt num person, num joint, 3
        if name == 'gt':
            points = points.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # 设置颜色（如果 colors 中没有定义该名称，则使用默认灰色）
        color = colors.get(name, [0.7, 0.7, 0.7])
        pcd.paint_uniform_color(color)
        vis.add_geometry(pcd)

    # 添加点云后获取视图控制
    view_control = vis.get_view_control()

    # 设置初始视图
    view_control.set_front([-1, 0, 0])  # 相机指向的方向（从相机指向原点）
    view_control.set_lookat([1, 0, 0])  # 相机注视的点（目标位置）
    view_control.set_up([0, 0, 1])  # 向上的方向
    view_control.set_zoom(0.5)  # 缩放倍数（1.0为标准距离）
    # 运行可视化（阻塞直到窗口关闭）
    vis.run()
    vis.destroy_window()

def main():
    radar_low_cfar_params = {
        "ref_range": 10,
        "ref_velocity": 8,
        "guard_range": 2,
        "guard_velocity": 2,
        "alpha": 2.0,
        "mode": "ca",
    }
    radar_high_cfar_params = {
        "ref_range": 8,
        "ref_velocity": 8,
        "guard_range": 2,
        "guard_velocity": 2,
        "alpha": 1.5,
        "mode": "ca",
    }
    root_path = Path(r'E:\20260609_165004')
    calib_path = root_path / 'calib'

    img_to_radar_low_extrinsic = np.load(calib_path / 'extrinsic_img_to_radar_low.npz')
    img_to_radar_high_extrinsic = np.load(calib_path / 'extrinsic_img_to_radar_high.npz')


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
        'lidar': lidar_path,
        'radar_low_bin': radar_low_bin_path,
        'radar_high_bin': radar_high_bin_path,
        'radar_low_pc': radar_low_pc_path,
        'radar_high_pc': radar_high_pc_path,
        'gt': gt_path,
        'realsense': realsense_path,
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
        base_source='lidar',  # 可选，不指定则自动选择
        suffix_map=suffix_map
    )

    lidar_files_matched = result.get('lidar')
    radar_low_bin_files_matched = result.get('radar_low_bin')
    radar_low_pc_files_matched = result.get('radar_low_pc')
    radar_high_bin_files_matched = result.get('radar_high_bin')
    radar_high_pc_files_matched = result.get('radar_high_pc')
    gt_files_matched = result.get('gt')
    realsense_files_matched = result.get('realsense')

    frames = len(lidar_files_matched)

    for frame_idx in range(frames):
        lidar_pcd, pc_from_bin_radar_low, pc_from_bin_radar_high, pc_radar_low, pc_radar_high, gt, realsense_depth = [None] * 7
        # ========== LiDAR ==========
        if lidar_files_matched[frame_idx]:
            lidar_pcd = read_pcd(lidar_files_matched[frame_idx])
            lidar_time_samp = Path(lidar_files_matched[frame_idx]).stem
            lidar_time_ms = timestamp_to_ms(lidar_time_samp)
            print(f"LiDAR: {lidar_time_ms}, Shape: {lidar_pcd.shape}")
        # ========== 低位雷达 ==========
        if radar_low_bin_files_matched[frame_idx]:
            targets_low = get_corner_data(Path(radar_low_bin_files_matched[frame_idx]), **radar_low_cfar_params)
            pc_from_bin_radar_low = targets_low["cartesian coordinate"]
            radar_low_time_samp = Path(radar_low_bin_files_matched[frame_idx]).stem
            radar_low_time_ms = timestamp_to_ms(radar_low_time_samp)
            print(f"Radar Bin Low: {radar_low_time_ms}, Shape: {pc_from_bin_radar_low.shape}")
        if radar_low_pc_files_matched[frame_idx]:
            pc_radar_low = get_pc_data(Path(radar_low_pc_files_matched[frame_idx]))
            radar_low_time_samp = Path(radar_low_pc_files_matched[frame_idx]).stem
            radar_low_time_ms = timestamp_to_ms(radar_low_time_samp)
            print(f"Radar PC Low: {radar_low_time_ms}, Shape: {pc_radar_low.shape}")
        # ========== 高位雷达 ==========
        if radar_high_bin_files_matched[frame_idx]:
            targets_high = get_corner_data(Path(radar_high_bin_files_matched[frame_idx]), **radar_high_cfar_params)
            pc_from_bin_radar_high = targets_high["cartesian coordinate"]
            radar_high_time_samp = Path(radar_high_bin_files_matched[frame_idx]).stem
            radar_high_time_ms = timestamp_to_ms(radar_high_time_samp)
            print(f"Radar High: {radar_high_time_ms}, Shape: {pc_from_bin_radar_high.shape}")
            # TODO：坐标转换
        if radar_high_pc_files_matched[frame_idx]:
            pc_radar_high = get_pc_data(Path(radar_high_pc_files_matched[frame_idx]))
            radar_high_time_samp = Path(radar_high_pc_files_matched[frame_idx]).stem
            radar_high_time_ms = timestamp_to_ms(radar_high_time_samp)
            print(f"Radar PC High: {radar_high_time_ms}, Shape: {pc_radar_high.shape}")
            # TODO：坐标转换
        # ========== GT ==========
        if gt_files_matched[frame_idx] is not None:
            gt = get_gt_data(gt_files_matched[frame_idx])
            gt_time_samp = Path(gt_files_matched[frame_idx]).stem
            gt_time_ms = timestamp_to_ms(gt_time_samp)
            print(f"GT: {gt_time_ms}, Shape: {gt.shape}")
            num_person, num_joint, _ = gt.shape
            # TODO：坐标转换
            # gt =(img_to_radar_low_extrinsic['R_est'] @ gt.reshape(-1, 3).T + img_to_radar_low_extrinsic['t_est'].reshape(-1, 1)).T
            # gt = gt.reshape(num_person, num_joint, 3)
        # ========== RealSense ==========
        if realsense_files_matched[frame_idx] is not None:
            realsense_depth = get_realsense_data(Path(realsense_files_matched[frame_idx]))
            realsense_time_samp = Path(realsense_files_matched[frame_idx]).stem
            realsense_time_ms = timestamp_to_ms(realsense_time_samp)
            if realsense_depth is not None and len(realsense_depth) > 0:
                # 过滤无效点（深度为 0 或无穷）
                valid_mask = (realsense_depth[:, 2] > 0) & np.isfinite(realsense_depth[:, 2])
                realsense_depth = realsense_depth[valid_mask]
                print(f"Realsense: {realsense_time_ms}, Shape: {realsense_depth.shape}")
            # TODO：坐标转换

        data_for_vis = {
            'lidar_pcd': lidar_pcd,
            # 'pc_from_bin_radar_low': pc_from_bin_radar_low,
            # 'pc_from_bin_radar_high': pc_from_bin_radar_high,
            'pc_radar_low': pc_radar_low,
            'pc_radar_high': pc_radar_high,
            'gt': gt,
            'realsense_depth' : realsense_depth,
        }

        # 添加颜色和大小定义
        colors = {
            'lidar_pcd': [0.5, 0.5, 0.5],  # 灰色
            'pc_from_bin_radar_low': [1.0, 0.5, 0.0],  # 橙色
            'pc_radar_low': [1.0, 0.75, 0.25],  # 浅橙色

            'pc_from_bin_radar_high': [0.0, 0.0, 1.0],  # 蓝色
            'pc_radar_high': [0.25, 0.55, 1.0],  # 浅蓝色

            'gt': [1.0, 0.0, 0.0],  # 红色
            'realsense_depth': [0.0, 1.0, 0.0],  # 绿色
        }
        visualize_multi_sensor(data_for_vis, colors, window_name=f"Frame {frame_idx:04d}")


if __name__ == '__main__':
    main()
