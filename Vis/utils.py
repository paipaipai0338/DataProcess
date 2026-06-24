import time

import numpy as np
import open3d as o3d

from Img2Keypoint.utils import COCO17_SKELETON


def _add_geometry(vis, geometry, reset_bounding_box=True):
    try:
        vis.add_geometry(geometry, reset_bounding_box=reset_bounding_box)
    except TypeError:
        vis.add_geometry(geometry)


def _set_render_options(vis, point_size):
    opt = vis.get_render_option()
    opt.point_size = point_size


def _scene_size_to_zoom(scene_size, reference_scene_size=8.0, reference_zoom=0.5):
    if scene_size is None:
        return reference_zoom
    if isinstance(scene_size, (list, tuple, np.ndarray)):
        scene_size = max(float(value) for value in scene_size)
    scene_size = max(float(scene_size), 1e-6)
    return float(np.clip(reference_zoom * reference_scene_size / scene_size, 0.02, 2.0))


def _set_default_view(vis, scene_center=(1, 0, 0), scene_size=None):
    view_control = vis.get_view_control()
    view_control.set_front([-1, 0, 0])
    view_control.set_lookat(scene_center)
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(_scene_size_to_zoom(scene_size))


def _add_multi_sensor_geometries(vis, data, colors, reset_bounding_box=True):
    for name, points in data.items():
        if points is None or len(points) == 0:
            continue

        points = np.asarray(points)
        if points.shape[-1] != 3:
            points = points[..., :3]

        if name == "gt":
            if points.ndim == 2:
                max_index = max(max(pair) for pair in COCO17_SKELETON) + 1
                if points.shape[0] % max_index == 0:
                    num_person = points.shape[0] // max_index
                    points = points.reshape(num_person, max_index, 3)
                    print(f"Auto reshape gt to {num_person} persons, {max_index} joints each")
                else:
                    print("Warning: gt points shape cannot be automatically reshaped. Skipping skeleton drawing.")
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.paint_uniform_color(colors.get(name, [0.7, 0.7, 0.7]))
                    _add_geometry(vis, pcd, reset_bounding_box=reset_bounding_box)
                    continue

            color = colors.get(name, [1.0, 0.0, 0.0])

            all_points = points.reshape(-1, 3)
            pcd_all = o3d.geometry.PointCloud()
            pcd_all.points = o3d.utility.Vector3dVector(all_points)
            pcd_all.paint_uniform_color(color)
            _add_geometry(vis, pcd_all, reset_bounding_box=reset_bounding_box)

            for person_points in points:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(person_points)
                line_set.lines = o3d.utility.Vector2iVector(COCO17_SKELETON)
                line_set.paint_uniform_color(color)
                _add_geometry(vis, line_set, reset_bounding_box=reset_bounding_box)

            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(colors.get(name, [0.7, 0.7, 0.7]))
        _add_geometry(vis, pcd, reset_bounding_box=reset_bounding_box)


def create_multi_sensor_player(colors: dict, window_name="Multi-Sensor Playback",
                               window_size=(1280, 720), show_coord_frame=True,
                               coord_frame_size=1.0, point_size=5.0,
                               frame_interval_sec=0.1, scene_center=(1, 0, 0),
                               scene_size=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=window_size[0], height=window_size[1])
    _set_render_options(vis, point_size)
    return {
        "vis": vis,
        "colors": colors,
        "show_coord_frame": show_coord_frame,
        "coord_frame_size": coord_frame_size,
        "point_size": point_size,
        "frame_interval_sec": frame_interval_sec,
        "scene_center": scene_center,
        "scene_size": scene_size,
        "first_frame": True,
    }


def play_multi_sensor_frame(player: dict, data: dict):
    vis = player["vis"]
    reset_bounding_box = bool(player["first_frame"])

    vis.clear_geometries()
    if player["show_coord_frame"]:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=player["coord_frame_size"]
        )
        _add_geometry(vis, coord_frame, reset_bounding_box=reset_bounding_box)

    _add_multi_sensor_geometries(
        vis,
        data,
        player["colors"],
        reset_bounding_box=reset_bounding_box,
    )
    _set_render_options(vis, player["point_size"])

    if player["first_frame"]:
        _set_default_view(
            vis,
            scene_center=player["scene_center"],
            scene_size=player["scene_size"],
        )
        player["first_frame"] = False

    window_alive = vis.poll_events()
    vis.update_renderer()
    if player["frame_interval_sec"] > 0:
        time.sleep(player["frame_interval_sec"])
    return window_alive


def close_multi_sensor_player(player: dict):
    player["vis"].destroy_window()


def play_multi_sensor_sequence(data_sequence, colors: dict, window_name="Multi-Sensor Playback",
                               window_size=(1280, 720), show_coord_frame=True,
                               coord_frame_size=1.0, point_size=5.0,
                               frame_interval_sec=0.1, scene_center=(1, 0, 0),
                               scene_size=None):
    player = create_multi_sensor_player(
        colors,
        window_name=window_name,
        window_size=window_size,
        show_coord_frame=show_coord_frame,
        coord_frame_size=coord_frame_size,
        point_size=point_size,
        frame_interval_sec=frame_interval_sec,
        scene_center=scene_center,
        scene_size=scene_size,
    )
    try:
        for data in data_sequence:
            if not play_multi_sensor_frame(player, data):
                break
    finally:
        close_multi_sensor_player(player)


def visualize_multi_sensor(data: dict, colors: dict, window_name="Multi-Sensor Visualization",
                           window_size=(1280, 720), show_coord_frame=True,
                           coord_frame_size=1.0, point_size=5.0,
                           scene_center=(1, 0, 0), scene_size=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=window_size[0], height=window_size[1])

    if show_coord_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_frame_size)
        _add_geometry(vis, coord_frame)

    _add_multi_sensor_geometries(vis, data, colors)
    _set_render_options(vis, point_size)
    _set_default_view(vis, scene_center=scene_center, scene_size=scene_size)

    vis.run()
    vis.destroy_window()
