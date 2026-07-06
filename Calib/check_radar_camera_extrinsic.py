import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Calib.utils import apply_transform
from Img2Points.utils import get_corner_coordinate
from RadarProcess.utils import get_corner_data


def parse_args():
    parser = argparse.ArgumentParser(description="根据多相机棋盘格图像估计相机外参。")
    parser.add_argument('--data_path', type=Path, default=Path(r'C:\Users\Administrator\Desktop\20260702\data_collection\group_007'), help="外参标定训练数据目录，默认使用 data_path/chessboard/extrinsic/train。")
    parser.add_argument('--calib_path', type=Path, default=Path(r'C:\Users\Administrator\Desktop\20260702\calib'), help="标定结果保存目录，同时用于读取已有内参文件，默认使用 data_path/calib。")
    args = parser.parse_args()
    return args

args = parse_args()
data_path = args.data_path
calib_path = args.calib_path
radar_path_low = data_path / 'dpct低位机/Bin'
radar_path_high = data_path / 'dpct高位机/Bin'
radar_cfar_params = {
    "ref_range": 9,
    "ref_velocity": 8,
    "guard_range": 8,
    "guard_velocity": 4,
    "alpha": 15.0,
    "mode": "ca",
}
extrinsic_img_to_radar_high = np.load(calib_path / 'extrinsic_img_to_radar_high.npz')
extrinsic_img_to_radar_low = np.load(calib_path / 'extrinsic_img_to_radar_low.npz')
pkl_save_path = args.calib_path / f"corner_pixels_{data_path.stem.split('_')[-1]}.pkl"
pixel_coordinate, error = get_corner_coordinate(pkl_save_path, calib_path=calib_path)
radar_files_low = sorted([path for path in radar_path_low.iterdir()])
radar_files_high = sorted([path for path in radar_path_high.iterdir()])

radar_pc_low = get_corner_data(radar_files_low[-1], **radar_cfar_params)
radar_pc_low = radar_pc_low["cartesian coordinate"][:, :3]

radar_pc_high = get_corner_data(radar_files_high[-1], **radar_cfar_params)
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
