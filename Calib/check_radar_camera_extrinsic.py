import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Calib.utils import apply_transform
from Img2Points.utils import get_corner_coordinate
from RadarProcess.utils import get_corner_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='叠加角反射器与高低位雷达点云，检查相机雷达外参。',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_path', type=Path, default=Path('C:\\Users\\Administrator\\Desktop\\20260702\\data_collection\\group_007'), help='包含相机与高低位雷达数据的采集组目录。')
    parser.add_argument('--calib_path', type=Path, default=Path('C:\\Users\\Administrator\\Desktop\\20260702\\calib'), help='已有相机内外参、雷达外参和角反射器标注的目录。')
    parser.add_argument('--frame_index', type=int, default=-1, help='检查的雷达帧索引，负数表示从末尾计数。')
    parser.add_argument('--reference_cam', default='A', help='重建角反射器使用的参考相机编号。')
    parser.add_argument('--ref_range', type=int, default=9, help='距离维参考单元数。')
    parser.add_argument('--ref_velocity', type=int, default=8, help='速度维参考单元数。')
    parser.add_argument('--guard_range', type=int, default=8, help='距离维保护单元数。')
    parser.add_argument('--guard_velocity', type=int, default=4, help='速度维保护单元数。')
    parser.add_argument('--alpha', type=float, default=15.0, help='CFAR 检测门限倍率。')
    parser.add_argument('--mode', type=str, choices=('ca', 'so', 'go'), default='ca', help='CFAR 检测模式。')
    args = parser.parse_args(argv)
    return args


def main():
    args = parse_args()
    data_path = args.data_path
    calib_path = args.calib_path
    radar_path_low = data_path / 'dpct低位机/Bin'
    radar_path_high = data_path / 'dpct高位机/Bin'
    radar_cfar_params = {key: getattr(args, key) for key in (
        "ref_range", "ref_velocity", "guard_range", "guard_velocity", "alpha", "mode"
    )}
    extrinsic_img_to_radar_high = np.load(calib_path / 'extrinsic_img_to_radar_high.npz')
    extrinsic_img_to_radar_low = np.load(calib_path / 'extrinsic_img_to_radar_low.npz')
    pkl_save_path = args.calib_path / f"corner_pixels_{data_path.stem.split('_')[-1]}.pkl"
    pixel_coordinate, error = get_corner_coordinate(pkl_save_path, calib_path=calib_path, reference_cam=args.reference_cam)
    radar_files_low = sorted([path for path in radar_path_low.iterdir() if path.suffix.lower() == ".bin"])
    radar_files_high = sorted([path for path in radar_path_high.iterdir() if path.suffix.lower() == ".bin"])

    for files in (radar_files_low, radar_files_high):
        if not -len(files) <= args.frame_index < len(files):
            raise ValueError(f"雷达帧索引 {args.frame_index} 超出范围，共 {len(files)} 帧")
    radar_pc_low = get_corner_data(radar_files_low[args.frame_index], **radar_cfar_params)
    radar_pc_low = radar_pc_low["cartesian coordinate"][:, :3]

    radar_pc_high = get_corner_data(radar_files_high[args.frame_index], **radar_cfar_params)
    radar_pc_high = radar_pc_high["cartesian coordinate"][:, :3]




    R_inv = extrinsic_img_to_radar_high['R_est'].T
    t_inv = -extrinsic_img_to_radar_high['R_est'].T @ extrinsic_img_to_radar_high['t_est'].T
    radar_pc_high = apply_transform(radar_pc_high, R_inv, t_inv)
    radar_pc_high = apply_transform(radar_pc_high, extrinsic_img_to_radar_low['R_est'], extrinsic_img_to_radar_low['t_est'])
    pixel_coordinate = apply_transform(pixel_coordinate, extrinsic_img_to_radar_low['R_est'], extrinsic_img_to_radar_low['t_est'])


    print(pixel_coordinate.shape, radar_pc_low.shape, radar_pc_high.shape)

    plt.figure()
    ax = plt.subplot(projection='3d')

    ax.scatter(pixel_coordinate[:, 0], pixel_coordinate[:, 1], pixel_coordinate[:, 2], c='r', label='pixel')
    ax.scatter(radar_pc_high[:, 0], radar_pc_high[:, 1], radar_pc_high[:, 2], c='b', label='radar_pc_high')
    ax.scatter(radar_pc_low[:, 0], radar_pc_low[:, 1], radar_pc_low[:, 2], c='g', label='radar_pc_low')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
