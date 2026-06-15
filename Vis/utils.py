import open3d as o3d
from Img2Keypoint.utils import COCO17_SKELETON
import open3d as o3d
import numpy as np

def visualize_multi_sensor(data: dict, colors: dict, window_name="Multi-Sensor Visualization",
                           window_size=(1280, 720), show_coord_frame=True,
                           coord_frame_size=1.0, point_size=5.0):
    """
    可视化多个传感器点云数据，支持对 'gt' 绘制骨架线段

    Parameters:
        data: dict, 键为传感器名称，值为 numpy数组。
              - 普通传感器：形状 (N, 3) 或 (N, D) 且 D>=3，取前3维作为坐标。
              - gt 传感器：形状 (num_person, num_joint, 3)，例如 (1, 17, 3)。
        colors: dict, 键为传感器名称，值为 RGB 颜色列表 [R,G,B], 范围 0~1。
        window_name: str, 窗口标题。
        window_size: tuple, 窗口大小 (width, height)。
        show_coord_frame: bool, 是否显示坐标轴。
        coord_frame_size: float, 坐标轴大小。
        point_size: float, 点的大小（对所有点云有效）。
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=window_size[0], height=window_size[1])

    if show_coord_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size)
        vis.add_geometry(coord_frame)

    # 遍历数据
    for name, points in data.items():
        if points is None or len(points) == 0:
            continue

        # 通用处理：保证坐标维度为3
        if points.shape[-1] != 3:
            points = points[..., :3]

        # ========== 特殊处理 gt：绘制骨架线段 ==========
        if name == 'gt':
            # 期望形状: (num_person, num_joint, 3)
            # 如果传入的是 (num_person*num_joint, 3)，尝试恢复人数（但无法知道人数，建议用户按格式传入）
            if points.ndim == 2:
                # 假设 COCO17 有 17 个关键点，尝试自动 reshape
                num_joint = len(COCO17_SKELETON) + 1  # 17 个点，但为了通用，计算最大索引值
                max_index = max(max(pair) for pair in COCO17_SKELETON) + 1
                if points.shape[0] % max_index == 0:
                    num_person = points.shape[0] // max_index
                    points = points.reshape(num_person, max_index, 3)
                    print(f"Auto reshape gt to {num_person} persons, {max_index} joints each")
                else:
                    print("Warning: gt points shape cannot be automatically reshaped. Skipping skeleton drawing.")
                    # 退化为普通点云
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.paint_uniform_color(colors.get(name, [0.7, 0.7, 0.7]))
                    vis.add_geometry(pcd)
                    continue

            # 获取颜色
            color = colors.get(name, [1.0, 0.0, 0.0])  # 默认红色

            # 1) 添加所有关节点的点云（可选，便于观察关键点位置）
            all_points = points.reshape(-1, 3)
            pcd_all = o3d.geometry.PointCloud()
            pcd_all.points = o3d.utility.Vector3dVector(all_points)
            pcd_all.paint_uniform_color(color)
            vis.add_geometry(pcd_all)

            # 2) 为每个人添加骨架线段
            num_person = points.shape[0]
            for i in range(num_person):
                person_points = points[i]  # (num_joint, 3)
                # 创建 LineSet
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(person_points)
                line_set.lines = o3d.utility.Vector2iVector(COCO17_SKELETON)
                line_set.paint_uniform_color(color)
                vis.add_geometry(line_set)

            # gt 处理完毕，跳过后续普通点云添加步骤
            continue

        # ========== 普通传感器：仅作为点云添加 ==========
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        color = colors.get(name, [0.7, 0.7, 0.7])
        pcd.paint_uniform_color(color)
        vis.add_geometry(pcd)

    # 设置渲染选项（点大小）
    opt = vis.get_render_option()
    opt.point_size = point_size

    # 视图控制
    view_control = vis.get_view_control()
    view_control.set_front([-1, 0, 0])
    view_control.set_lookat([1, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(0.5)

    vis.run()
    vis.destroy_window()