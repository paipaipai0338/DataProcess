
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from RadarProcess.utils import bin_to_cube_range_fft, doppler_fft, Radar_Config, get_radar_res


if __name__ == '__main__':
    root_path = Path(r'C:\Users\Administrator\Desktop\frames')
    radar_bin_path = root_path
    # radar_path = root_path / 'dpct低位机'
    # radar_bin_path = radar_path / 'Bin'

    bin_files = os.listdir(radar_bin_path)
    bin_files.sort()

    if len(bin_files) == 0:
        raise RuntimeError(f'目录中不存在 bin 文件：{radar_bin_path}')

    radar_config = Radar_Config()

    # 1. 相位差分曲线
    # 2. 截取前 RD
    # 3. 截取前 RD clean
    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 18),
        gridspec_kw={'height_ratios': [0.5, 1.5, 1.5]},
    )
    plt.ion()
    plt.show()

    first_file = radar_bin_path / bin_files[0]
    range_fft_data_first = bin_to_cube_range_fft(first_file, radar_config)

    if range_fft_data_first.ndim != 3:
        raise ValueError(f'期望 [range, chirp, antenna]，实际为 {range_fft_data_first.shape}')

    range_res, _, _, _ = get_radar_res(radar_config)
    r_axis = np.arange(range_fft_data_first.shape[0], dtype=np.float32) * float(range_res)

    for idx, file in enumerate(bin_files):
        bin_file_path = radar_bin_path / file
        range_fft_data = bin_to_cube_range_fft(bin_file_path, radar_config)
        range_fft_data = np.asarray(range_fft_data)

        if range_fft_data.ndim != 3:
            print(f'{file} shape 异常：{range_fft_data.shape}，跳过')
            continue

        if not np.isfinite(range_fft_data).all():
            print(f'{file} 存在 NaN 或 Inf，跳过')
            continue

        num_range, num_chirp, num_ant = range_fft_data.shape

        if num_chirp < 4:
            print(f'{file} chirp 数过少：{num_chirp}，跳过')
            continue

        previous = range_fft_data[:, :-1, :]
        current = range_fft_data[:, 1:, :]

        # [range, chirp-1, antenna]
        cross = current * np.conj(previous)
        phase_step = np.angle(cross)

        # 排除低功率 range bin
        range_power = np.mean(np.abs(range_fft_data) ** 2, axis=(1, 2))
        finite_range = np.isfinite(range_power)

        if not np.any(finite_range):
            print(f'{file} range power 全部无效，跳过')
            continue

        power_threshold = np.percentile(range_power[finite_range], 60)
        valid_range = finite_range & (range_power > power_threshold)

        if not np.any(valid_range):
            print(f'{file} 不存在有效 range bin，跳过')
            continue

        # 每个 range、每根天线的典型相位增量
        reference_phase = np.angle(
            np.sum(np.exp(1j * phase_step), axis=1, keepdims=True)
        )

        # 圆形相位残差
        phase_residual = np.abs(
            np.angle(np.exp(1j * (phase_step - reference_phase)))
        )

        phase_residual[~valid_range, :, :] = np.nan

        # [chirp-1, antenna]
        phase_score_per_ant = np.nanmedian(phase_residual, axis=0)

        # 只用于寻找全局 suspect boundary
        global_phase_score = np.nanmedian(phase_score_per_ant, axis=-1)

        if not np.isfinite(global_phase_score).any():
            print(f'{file} 无法计算有效相位异常分数，跳过')
            continue

        suspect_boundary = int(np.nanargmax(global_phase_score)) + 1
        selected_range_fft = range_fft_data[:, suspect_boundary:, :]

        if selected_range_fft.shape[1] < 4:
            print(
                f'{file} suspect boundary={suspect_boundary}，'
                f'截取后仅剩 {selected_range_fft.shape[1]} 个 chirp，跳过'
            )
            continue

        # =========================================================
        # 截取前完整数据的 RD map
        # =========================================================
        rd_cube_full, rd_cube_clean_full, v_axis_full = doppler_fft(
            range_fft_data,
            radar_config,
            window=True,
            doppler_mode='firmware_tdm',
        )

        rd_power_full = np.mean(np.abs(rd_cube_full) ** 2, axis=-1)
        rd_power_clean_full = np.mean(np.abs(rd_cube_clean_full) ** 2, axis=-1)

        rd_map_db_full = 10.0 * np.log10(
            rd_power_full / (np.max(rd_power_full) + 1e-12) + 1e-12
        )

        rd_map_clean_db_full = 10.0 * np.log10(
            rd_power_clean_full / (np.max(rd_power_clean_full) + 1e-12) + 1e-12
        )

        # =========================================================
        # suspect boundary 截取后的 RD map
        # =========================================================
        rd_cube_selected, rd_cube_clean_selected, v_axis_selected = doppler_fft(
            selected_range_fft,
            radar_config,
            window=True,
            doppler_mode='firmware_tdm',
        )

        rd_power_selected = np.mean(np.abs(rd_cube_selected) ** 2, axis=-1)
        rd_power_clean_selected = np.mean(np.abs(rd_cube_clean_selected) ** 2, axis=-1)

        rd_map_db_selected = 10.0 * np.log10(
            rd_power_selected / (np.max(rd_power_selected) + 1e-12) + 1e-12
        )

        rd_map_clean_db_selected = 10.0 * np.log10(
            rd_power_clean_selected / (np.max(rd_power_clean_selected) + 1e-12) + 1e-12
        )

        for ax in axes:
            ax.clear()

        # =========================================================
        # 第一张：每根天线的相位差分异常曲线
        # =========================================================
        chirp_boundaries = np.arange(1, num_chirp)
        antenna_colors = plt.cm.hsv(np.linspace(0, 1, num_ant, endpoint=False))

        for ant_idx in range(num_ant):
            axes[0].plot(
                chirp_boundaries,
                phase_score_per_ant[:, ant_idx],
                color=antenna_colors[ant_idx],
                linewidth=1.0,
                alpha=0.9,
                label=f'Ant {ant_idx}',
            )

        axes[0].plot(
            chirp_boundaries,
            global_phase_score,
            color='black',
            linewidth=2.0,
            linestyle='--',
            label='Median across antennas',
        )

        axes[0].axvline(
            suspect_boundary,
            color='red',
            linewidth=2.0,
            linestyle='--',
            label=f'Suspect boundary {suspect_boundary}',
        )

        axes[0].set_title('Phase-step residual for each antenna')
        axes[0].set_xlabel('Chirp boundary')
        axes[0].set_ylabel('Median phase residual (rad)')
        axes[0].set_xlim(1, num_chirp - 1)
        # axes[0].set_ylim(0, np.pi)
        axes[0].grid(True)
        axes[0].legend(fontsize=7, ncol=4, loc='upper right')

        # =========================================================
        # 第二张：截取前原始 RD map
        # =========================================================
        axes[1].imshow(
            rd_map_db_full.T,
            origin='lower',
            aspect='auto',
            extent=[
                float(r_axis[0]),
                float(r_axis[-1]),
                float(v_axis_full[0]),
                float(v_axis_full[-1]),
            ],
            cmap='viridis',
            vmin=-40,
            vmax=0,
        )

        axes[1].set_title(f'Full RD map ({num_chirp} chirps)')
        axes[1].set_xlabel('Range (m)')
        axes[1].set_ylabel('Velocity (m/s)')

        # =========================================================
        # 第三张：截取前均值对消 RD map
        # =========================================================
        axes[2].imshow(
            rd_map_clean_db_full.T,
            origin='lower',
            aspect='auto',
            extent=[
                float(r_axis[0]),
                float(r_axis[-1]),
                float(v_axis_full[0]),
                float(v_axis_full[-1]),
            ],
            cmap='viridis',
            vmin=-40,
            vmax=0,
        )

        axes[2].set_title(f'Full RD map clean ({num_chirp} chirps)')
        axes[2].set_xlabel('Range (m)')
        axes[2].set_ylabel('Velocity (m/s)')

        print(
            f'{idx + 1}/{len(bin_files)} {file}: '
            f'suspect boundary={suspect_boundary}, '
            f'score={global_phase_score[suspect_boundary - 1]:.4f} rad, '
            f'full chirps={num_chirp}, '
            f'remaining chirps={selected_range_fft.shape[1]}'
        )
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    plt.ioff()
    plt.show()
