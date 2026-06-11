import os
from pathlib import Path
import json
import numpy as np
import open3d as o3d

from sensors_utils import *
from lky_pointcloud_utils import generate_point_cloud_array_from_bin




def read_pcd(file_path: Path) -> np.ndarray:
    """
    获取激光雷达 EM4 数据
    输入:
      file_path: str 或 path-like，PCD 点云文件路径，shape 为标量路径。
    输出:
      pointcloud: np.ndarray，dtype=float64，shape=(N, 3)。
    """
    pcd = o3d.io.read_point_cloud(file_path)
    pointcloud = np.asarray(pcd.points)
    return pointcloud


def visualize_pointclouds(lidar_points: np.ndarray, radar_points: np.ndarray):
    """
    可视化激光雷达和雷达点云数据

    参数:
        lidar_points: 激光雷达点云，shape=(N, 3)
        radar_points: 雷达点云，shape=(M, 3)
    """
    # 创建Open3D点云对象
    lidar_pcd = o3d.geometry.PointCloud()
    radar_pcd = o3d.geometry.PointCloud()

    # 设置点云数据
    lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points)
    radar_pcd.points = o3d.utility.Vector3dVector(radar_points)

    # 为激光雷达点云设置颜色（蓝色）
    lidar_colors = np.tile([0, 0, 1], (len(lidar_points), 1))  # RGB蓝色
    lidar_pcd.colors = o3d.utility.Vector3dVector(lidar_colors)

    # 为雷达点云设置颜色（红色）
    radar_colors = np.tile([1, 0, 0], (len(radar_points), 1))  # RGB红色
    radar_pcd.colors = o3d.utility.Vector3dVector(radar_colors)

    # 可选：为雷达点云设置更大的点大小（如果雷达点较少）
    # 注意：点大小需要在可视化时设置，不能直接设置在点云对象上

    # 可视化
    print(f"激光雷达点云点数: {len(lidar_points)}")
    print(f"雷达点云点数: {len(radar_points)}")

    # 方式1：同时显示两个点云
    o3d.visualization.draw_geometries(
        [lidar_pcd, radar_pcd],
        window_name="LiDAR (Blue) and Radar (Red) Point Cloud",
        width=1024,
        height=768,
        left=50,
        top=50,
        point_show_normal=False
    )

    # 方式2：如果需要分别显示，可以取消下面的注释
    # o3d.visualization.draw_geometries(
    #     [lidar_pcd],
    #     window_name="LiDAR Point Cloud",
    #     width=1024,
    #     height=768
    # )
    # o3d.visualization.draw_geometries(
    #     [radar_pcd],
    #     window_name="Radar Point Cloud",
    #     width=1024,
    #     height=768
    # )


def visualize_with_customization(lidar_points: np.ndarray, radar_points: np.ndarray):
    """
    带更多自定义选项的可视化函数
    """
    # 创建点云对象
    lidar_pcd = o3d.geometry.PointCloud()
    radar_pcd = o3d.geometry.PointCloud()

    lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points)
    radar_pcd.points = o3d.utility.Vector3dVector(radar_points)

    # 为点云着色
    lidar_colors = np.tile([0, 0.5, 1], (len(lidar_points), 1))  # 浅蓝色
    radar_colors = np.tile([1, 0.2, 0.2], (len(radar_points), 1))  # 亮红色

    lidar_pcd.colors = o3d.utility.Vector3dVector(lidar_colors)
    radar_pcd.colors = o3d.utility.Vector3dVector(radar_colors)

    # 添加坐标系（可选）
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5.0, origin=[0, 0, 0]
    )

    # 设置可视化参数
    vis_params = {
        "window_name": "Point Cloud Visualization - LiDAR (Cyan) & Radar (Red)",
        "width": 1280,
        "height": 720,
        "left": 50,
        "top": 50,
        "point_show_normal": False,
    }

    # 显示点云
    o3d.visualization.draw_geometries(
        [lidar_pcd, radar_pcd, coordinate_frame],
        **vis_params
    )



def main():
    # 读取数据
    radar_low_path = Path(r"E:\20260609_164905\dpct低位机\Bin")
    radar_high_path = Path(r"E:\20260609_164905\dpct高位机\Bin")
    lidar_path = Path(r"E:\20260609_164905\robosense")

    radar_low_bin_files = os.listdir(radar_low_path)
    radar_high_bin_files = os.listdir(radar_high_path)
    lidar_bin_files = os.listdir(lidar_path)

    # 读取点云数据
    pc_lidar = read_pcd(lidar_path / lidar_bin_files[-10])
    targets = get_corner_data(radar_low_path / radar_low_bin_files[-10])
    pc_radar_low = targets["cartesian coordinate"]
    targets = get_corner_data(radar_high_path / radar_high_bin_files[-10])
    pc_radar_high = targets["cartesian coordinate"]

    # pc_radar = generate_point_cloud_array_from_bin(
    #     str(radar_path / radar_bin_files[10]),
    #     branch="dbf_first",
    #     output_format="xyz",
    #     mti="none",
    #     cfar_mul_fac=20.0,
    # )


    print(f"激光雷达点云形状: {pc_lidar.shape}")
    print(f"雷达点云形状: {pc_radar_low.shape, pc_radar_high.shape}")

    # 基本可视化
    visualize_pointclouds(pc_lidar, pc_radar_low[:, :3])
    visualize_pointclouds(pc_lidar, pc_radar_high[:, :3])



if __name__ == "__main__":
    main()