import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from RadarProcess.utils import bin_to_cube_range_fft, doppler_fft, Radar_Config, get_radar_res

if __name__ == '__main__':
    root_path = Path(r'E:\20260611\group_000')

    radar_path = root_path / 'dpct'
    radar_bin_path = radar_path / 'Bin'

    bin_files = os.listdir(radar_bin_path)
    bin_files.sort()

    radar_config = Radar_Config()

    plt.figure()
    plt.ion()

    for idx, file in enumerate(bin_files):
        plt.cla()
        bin_file_path = radar_bin_path / file
        range_fft_data = bin_to_cube_range_fft(bin_file_path, radar_config)

        range_res, _, _, _ = get_radar_res(radar_config)
        r_axis = np.arange(range_fft_data.shape[0], dtype=float) * range_res

        # 固件风格 TDM Doppler FFT
        rd_cube, v_axis = doppler_fft(
            range_fft_data,
            radar_config,
            window=True,
            doppler_mode="firmware_tdm",
        )
        rd_map_abs_sum = np.sum(np.abs(rd_cube), axis=-1)
        rd_map_abs_sum_log = np.log10(rd_map_abs_sum / np.max(rd_map_abs_sum, axis=(0, 1)))
        plt.imshow(rd_map_abs_sum.T, extent=(r_axis[0], r_axis[-1], v_axis[0], v_axis[-1]), aspect='auto')
        plt.title('{}/{}'.format(idx+1, len(bin_files)))
        plt.draw()
        plt.xlabel('range(m)')
        plt.ylabel('velocity(m/s)')
        plt.pause(0.01)
    plt.ioff()
    plt.show()




