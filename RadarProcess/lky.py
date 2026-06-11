#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent radar point-cloud generator extracted from the two uploaded scripts:
  1) pointcloud_generate.py: USB FFT frame reader, pseudo-float complex decode,
     TDM Doppler FFT, CFAR, DBF angle estimation, and point-cloud construction.
  2) steeringVec_calu.py: ideal steering-vector generation and single-value
     pseudo-float-to-complex verification helper.

Only numpy and Python standard library are required. GUI / PyQtGraph / Matplotlib
code has been removed.

Main public entry:
    generate_point_cloud_from_bin(bin_file, branch="dbf_first")

Returned point cloud format:
    A dict[str, np.ndarray] with aligned 1-D columns, including:
      Frame, RangeBin, RangeM, DopplerBin, Amplitude, CfarThreshold,
      AzimuthDeg, ElevationDeg, AzimuthPeakIndex, X, Y, Z

Recommended use:
    from radar_pointcloud_generator_refactored import generate_point_cloud_from_bin
    pc = generate_point_cloud_from_bin("usb_fft_xxx.bin")
    xyz = point_table_to_xyz(pc)

Function responsibilities / IO summary:

1. pseudo_float_complex_to_complex(pf_u32)
   - Purpose: vectorized conversion of packed 32-bit pseudo-float values to
     complex values.
   - Input: np.ndarray uint32/int-compatible, arbitrary shape.
   - Output: np.ndarray complex, same shape.

2. convert_single_pseudo_float_to_complex(pf_val)
   - Purpose: scalar version copied from steeringVec_calu.py for checking C
     steering-vector constants or a single pseudo-float value.
   - Input: one Python int / np.uint32.
   - Output: Python complex.

3. generate_ideal_steering_vectors(num_angle, num_rx)
   - Purpose: generate ideal DFT-style steering vectors in MATLAB column-major
     flatten order, matching the role of generate_matlab_vectors in
     steeringVec_calu.py.
   - Input: num_angle, num_rx.
   - Output: complex ndarray of shape [num_angle * num_rx].

4. read_usb_fft_frame(file_path, common_cfg)
   - Purpose: read one USB FFT bin file, reorder bytes, decode pseudo-float,
     and reshape to [range, chirp, virtual_ant].
   - Input: bin file path and CommonConfig.
   - Output: fft_cube, complex ndarray [num_range_bin, num_chirp, num_ant].

5. firmware_tdm_doppler_fft(doppler_input, proc_cfg)
   - Purpose: reproduce firmware-style TDM Doppler FFT using the exact C
     velocity window table.
   - Input: doppler_input [range, chirp, ant].
   - Output: rd_cube [range, doppler, ant].

6. firmware_velocity_cfar(amp_map, cfar_cfg)
   - Purpose: velocity-dimension CFAR over each range bin.
   - Input: amplitude map [range, doppler].
   - Output: cfar_mask, threshold_map, score_map, base_map.

7. build_array_geometry(array_cfg, proc_cfg)
   - Purpose: build physical x/y coordinates of the virtual antenna layout and
     elevation-compensation lookup.
   - Input: ArrayConfig and ProcessConfig.
   - Output: updated ArrayConfig.

8. build_target_table(...)
   - Purpose: original branch: noncoherent RD accumulation -> CFAR -> DBF -> XYZ.
   - Input: RD cube, RD amplitude map, CFAR threshold/mask, configs.
   - Output: point table dict.

9. build_spatial_dbf_first_rd_map(...) + build_spatial_dbf_first_target_table(...)
   - Purpose: DBF-first branch: azimuth DBF enhanced RD map -> CFAR -> elevation
     recovery -> XYZ.
   - Input: RD cube / CFAR maps / configs.
   - Output: point table dict.

10. process_usb_fft_frame(...)
    - Purpose: run both original and DBF-first branches from one decoded FFT cube.
    - Input: fft_cube and configs.
    - Output: radar_data dict and original-branch target_table.

11. generate_point_cloud_from_bin(bin_file, branch="dbf_first", ...)
    - Purpose: total reusable function requested by the user.
    - Input: one USB FFT bin file path; branch can be "dbf_first", "original",
      or "both".
    - Output: point table dict for one branch, or a dict containing both branches.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# ============================================================================
# Steering-vector helpers from steeringVec_calu.py
# ============================================================================


def generate_ideal_steering_vectors(num_angle: int, num_rx: int) -> np.ndarray:
    """Generate ideal steering vectors and flatten by MATLAB column-major order.

    Args:
        num_angle: Number of spatial angle FFT/grid bins, e.g. 128.
        num_rx: Number of RX channels/columns, e.g. 4.

    Returns:
        Complex ndarray with shape [num_angle * num_rx].  Index order is
        column-major: all angle bins of rx=1, then all angle bins of rx=2, etc.
    """
    steering = np.zeros((num_angle, num_rx), dtype=np.complex128)
    angle_indices = np.arange(num_angle)
    for rx_now in range(1, num_rx + 1):
        steering[:, rx_now - 1] = np.exp(
            -1j * 2.0 * np.pi / float(num_angle) * angle_indices * (rx_now - 1)
        )
    return steering.flatten(order="F")


def convert_single_pseudo_float_to_complex(pf_val: int) -> complex:
    """Convert one packed 32-bit pseudo-float integer to a Python complex value.

    Args:
        pf_val: One 32-bit integer whose high 4 bits store exponent, low 14 bits
            store signed real, and middle 14 bits store signed imaginary.

    Returns:
        complex(real, imag), scaled by 2 ** (exp - 13).
    """
    pf_val = int(pf_val) & 0xFFFFFFFF
    exp = (pf_val >> 28) & 0xF
    imag_raw = (pf_val >> 14) & 0x3FFF
    real_raw = pf_val & 0x3FFF
    int_real = (real_raw - 0x4000) if (real_raw & 0x2000) else real_raw
    int_imag = (imag_raw - 0x4000) if (imag_raw & 0x2000) else imag_raw
    factor = 2.0 ** (exp - 13)
    return complex(int_real * factor, int_imag * factor)



# ============================================================================
# Configuration copied from pointcloud_generate.py
# ============================================================================

@dataclass
class CommonConfig:
    num_range_bin: int = 256
    num_chirp: int = 64
    num_ant: int = 16


@dataclass
class CfarConfig:
    mode: str = "SO"
    search_size: int = 10
    guard_size: int = 16
    mul_fac: float = 10.0
    thres_div: int = 8
    peak_det_enable: bool = False


@dataclass
class ProcessConfig:
    usb_dir: str = "usb"
    file_pattern: str = "usb_fft_*.bin"
    pcd_dir: str = "pcd"
    pcd_file_pattern: str = "pcd_*.bin"
    max_frames: Optional[int] = None

    num_doppler_fft: int = 64
    num_tx: int = 4
    num_rx: int = 4
    bandwidth_mode: int = 1
    freq_start: float = 56.4e9
    bandwidth: float = 6.4453e9
    chirp_gap: float = 80e-6
    c: float = 299_792_458.0

    angle_grid_deg: np.ndarray = field(default_factory=lambda: np.arange(-80.0, 80.0 + 0.25, 0.25))
    expert_az_grid_deg: np.ndarray = field(default_factory=lambda: np.arange(-80.0, 80.0 + 1.0, 1.0))
    expert_el_grid_deg: np.ndarray = field(default_factory=lambda: np.arange(-80.0, 80.0 + 1.0, 1.0))
    num_angle: int = 128

    # MATLAB channels are 1-based. Python stores zero-based indices.
    az_channels: np.ndarray = field(default_factory=lambda: np.arange(16, 8, -1) - 1)
    el_channels: np.ndarray = field(default_factory=lambda: np.array([1, 5, 9]) - 1)
    el_coherent_ants: np.ndarray = field(
        default_factory=lambda: np.array(
            [[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]], dtype=int
        )
        - 1
    )
    el_ref_column: int = 3  # MATLAB elRefColumn=4 -> zero-based 3

    max_targets: int = 3000
    max_targets_to_dbf: float = math.inf
    max_azimuth_peaks_per_rd: int = 1
    min_azimuth_peak_ratio: float = 0.60
    min_azimuth_peak_separation_deg: float = 10.0
    expert_dbf_mode: str = "separable_current_array"
    angle_estimator_mode: str = "pcd_aligned_dbf"
    min_valid_range_bin: int = 8

    rd_display_low_prctile: float = 0.0
    rd_display_high_prctile: float = 99.9
    rd_display_gamma: float = 1.0
    dbf_az_snr_gate_enable: bool = True
    dbf_az_snr_threshold: float = 4.0

    pc_marker_size: float = 7.0
    pc_x_lim: Tuple[float, float] = (0.0, 4.0)
    pc_y_lim: Tuple[float, float] = (-1.5, 1.5)
    pc_z_lim: Tuple[float, float] = (-1.2, 1.2)
    # Display/export-only vertical offset.  Use this to compensate radar height
    # without changing the underlying algorithm output.
    pc_z_height_comp_m: float = 1.0
    pc_sync_view: bool = False
    pc_view_elev: float = 30.0
    pc_view_azim: float = -60.0
    pc_view_roll: float = 0.0
    play_pause_sec: float = 0.15
    play_pc_stride: int = 1

    @property
    def firmware_doppler_fft_size(self) -> int:
        return self.num_doppler_fft * self.num_tx

    @property
    def fc(self) -> float:
        return self.freq_start + self.bandwidth / 2.0

    @property
    def lambda_(self) -> float:
        return self.c / self.fc

    @property
    def range_resolution(self) -> float:
        return self.c / (2.0 * self.bandwidth)

    @property
    def pcd_range_resolution(self) -> float:
        return self.range_resolution

    @property
    def pcd_velocity_resolution(self) -> float:
        return self.lambda_ / (2.0 * self.num_doppler_fft * self.num_tx * self.chirp_gap)

    @property
    def rx_spacing_x(self) -> float:
        return self.lambda_ / 2.0

    @property
    def tx_spacing_y(self) -> float:
        return self.lambda_ / 2.0

    @property
    def tx_spacing_x(self) -> float:
        return 4.0 * self.rx_spacing_x



# ============================================================================
# Reusable point-cloud processing functions copied/refactored from pointcloud_generate.py
# ============================================================================

@dataclass
class ArrayConfig:
    layout: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0, 0, 0, 0, 4, 3, 2, 1], [0, 0, 0, 0, 8, 7, 6, 5], [16, 15, 14, 13, 12, 11, 10, 9]],
            dtype=int,
        )
    )
    row: np.ndarray = field(default_factory=lambda: np.empty(0))
    col: np.ndarray = field(default_factory=lambda: np.empty(0))
    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    el_coherent_ants: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    el_dbf_coord: np.ndarray = field(default_factory=lambda: np.empty(0))
    el_column_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    el_az_comp_lut: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.complex128))


# ============================================================================
# Small point-table helpers.  Dicts avoid pandas dependency and export logic.
# ============================================================================


def empty_point_table() -> Dict[str, np.ndarray]:
    names = [
        "Frame",
        "RangeBin",
        "RangeM",
        "DopplerBin",
        "Amplitude",
        "CfarThreshold",
        "AzimuthDeg",
        "ElevationDeg",
        "AzimuthPeakIndex",
        "X",
        "Y",
        "Z",
        "Power",
        "DopplerIndex",
    ]
    return {name: np.empty(0, dtype=float) for name in names}


def table_len(table: Dict[str, np.ndarray]) -> int:
    if not table:
        return 0
    lengths = [len(value) for value in table.values() if hasattr(value, "__len__")]
    return int(max(lengths)) if lengths else 0


def make_table(**cols: np.ndarray) -> Dict[str, np.ndarray]:
    table = empty_point_table()
    for key, value in cols.items():
        table[key] = np.asarray(value)
    return table


def point_table_from_target(target_table: Dict[str, np.ndarray], frame_idx: int) -> Dict[str, np.ndarray]:
    n = table_len(target_table)
    return make_table(
        Frame=np.full(n, frame_idx, dtype=int),
        RangeBin=target_table.get("RangeBin", np.empty(0)),
        RangeM=target_table.get("RangeM", np.empty(0)),
        DopplerBin=target_table.get("DopplerBin", np.empty(0)),
        Amplitude=target_table.get("Amplitude", np.empty(0)),
        CfarThreshold=target_table.get("CfarThreshold", np.empty(0)),
        AzimuthDeg=target_table.get("AzimuthDeg", np.empty(0)),
        ElevationDeg=target_table.get("ElevationDeg", np.empty(0)),
        AzimuthPeakIndex=target_table.get("AzimuthPeakIndex", np.empty(0)),
        X=target_table.get("X", np.empty(0)),
        Y=target_table.get("Y", np.empty(0)),
        Z=target_table.get("Z", np.empty(0)),
    )


# ============================================================================
# File discovery and readers
# ============================================================================


def discover_files(folder: str, pattern: str, max_frames: Optional[int] = None) -> List[str]:
    listing = sorted(glob.glob(os.path.join(folder, pattern)))
    if not listing and folder != ".":
        listing = sorted(glob.glob(pattern))
    if max_frames is not None and math.isfinite(max_frames):
        listing = listing[: int(max_frames)]
    return listing


def filename_timestamp_seconds(file_path: str) -> float:
    name = os.path.splitext(os.path.basename(file_path))[0]
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{6})", name)
    if not match:
        return float("nan")
    y, mo, d, h, mi, s, us = [int(x) for x in match.groups()]
    dt = datetime(y, mo, d, h, mi, s, us, tzinfo=timezone.utc)
    return dt.timestamp()


def match_nearest_timestamp_file(reference_file: str, candidate_files: List[str]) -> str:
    if not candidate_files:
        return ""
    reference_time = filename_timestamp_seconds(reference_file)
    candidate_times = np.array([filename_timestamp_seconds(path) for path in candidate_files], dtype=float)
    if not np.isfinite(reference_time) or not np.any(np.isfinite(candidate_times)):
        return candidate_files[0]
    best_idx = int(np.nanargmin(np.abs(candidate_times - reference_time)))
    return candidate_files[best_idx]


def pseudo_float_complex_to_complex(pf_u32: np.ndarray) -> np.ndarray:
    pf_u32 = pf_u32.astype(np.uint32, copy=False)
    idx = (pf_u32 >> np.uint32(28)).astype(np.float64)

    real_part = (pf_u32 & np.uint32(2**14 - 1)).astype(np.float64)
    real_part[real_part >= 2**13] -= 2**14

    imag_part = ((pf_u32 >> np.uint32(14)) & np.uint32(2**14 - 1)).astype(np.float64)
    imag_part[imag_part >= 2**13] -= 2**14

    scale = np.power(2.0, idx - 13.0)
    return (real_part + 1j * imag_part) * scale


def read_usb_fft_frame(file_path: str, common_cfg: CommonConfig) -> np.ndarray:
    raw_bytes = np.fromfile(file_path, dtype=np.uint8)
    num_whole_chunks = raw_bytes.size // 8
    raw_bytes = raw_bytes[: num_whole_chunks * 8]
    byte_rows = raw_bytes.reshape(-1, 8)[:, ::-1]
    reordered_bytes = np.ascontiguousarray(byte_rows.reshape(-1))
    pf_u32 = reordered_bytes.view("<u4")

    need = common_cfg.num_range_bin * common_cfg.num_ant * common_cfg.num_chirp
    if pf_u32.size < need:
        raise ValueError(f"USB FFT file is too short: need {need}, got {pf_u32.size}: {file_path}")

    c = pseudo_float_complex_to_complex(pf_u32[:need])
    # MATLAB reshape: [range, ant, chirp], then permute to [range, chirp, ant].
    mcu_timing = np.reshape(
        c,
        (common_cfg.num_range_bin, common_cfg.num_ant, common_cfg.num_chirp),
        order="F",
    )
    return np.transpose(mcu_timing, (0, 2, 1))


def pcd_signed_angle_index(idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=float)
    return np.where(idx > 63, idx - 128, idx)


def read_dynamic_pcd_frame(file_path: str, proc_cfg: ProcessConfig) -> Dict[str, np.ndarray]:
    if not file_path or not os.path.isfile(file_path):
        return empty_point_table()

    data = np.fromfile(file_path, dtype=np.uint8)
    if data.size < 9 or not np.array_equal(data[:2], np.array([85, 170], dtype=np.uint8)):
        return empty_point_table()

    frame_length = int(np.frombuffer(data[2:6].tobytes(), dtype="<u4")[0])
    if data.size != 6 + frame_length:
        return empty_point_table()

    pos = 6
    pos += 2  # time interval
    if pos >= data.size:
        return empty_point_table()
    num_tlvs = int(data[pos])
    pos += 1

    range_bins: List[int] = []
    ranges: List[float] = []
    doppler_indices: List[int] = []
    doppler_bins: List[float] = []
    velocities: List[float] = []
    angle_h: List[float] = []
    angle_v: List[float] = []
    powers: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []

    for _ in range(min(7, num_tlvs)):
        if pos + 2 > data.size:
            break
        type_id = int(data[pos])
        pos += 1
        target_num = int(np.frombuffer(data[pos : pos + 2].tobytes(), dtype="<u2")[0])
        pos += 2
        tlv_length = target_num * 6
        if pos + tlv_length > data.size:
            break

        if type_id == 1:
            for point_idx in range(target_num):
                point_start = pos + point_idx * 6
                idx1 = int(np.frombuffer(data[point_start : point_start + 2].tobytes(), dtype="<u2")[0])
                idx2 = int(data[point_start + 2])
                idx3 = int(data[point_start + 3])
                idx4 = int(data[point_start + 4])
                pow_abs = float(data[point_start + 5])

                doppler_bin = idx2 - proc_cfg.num_doppler_fft / 2.0
                range_val = idx1 * proc_cfg.pcd_range_resolution
                velocity = doppler_bin * proc_cfg.pcd_velocity_resolution
                angle_horizontal = np.degrees(np.arcsin(pcd_signed_angle_index(idx3) / 64.0))
                angle_vertical = np.degrees(np.arcsin(pcd_signed_angle_index(idx4) / 64.0))

                cos_v = np.cos(np.deg2rad(angle_vertical))
                x = range_val * np.cos(np.deg2rad(angle_horizontal)) * cos_v
                y = range_val * np.sin(np.deg2rad(angle_horizontal)) * cos_v
                z = range_val * np.sin(np.deg2rad(angle_vertical))

                range_bins.append(idx1)
                ranges.append(range_val)
                doppler_indices.append(idx2)
                doppler_bins.append(doppler_bin)
                velocities.append(velocity)
                angle_h.append(float(angle_horizontal))
                angle_v.append(float(angle_vertical))
                powers.append(pow_abs)
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))

        pos += tlv_length

    if not ranges:
        return empty_point_table()

    return make_table(
        RangeBin=np.array(range_bins, dtype=float),
        RangeM=np.array(ranges, dtype=float),
        DopplerIndex=np.array(doppler_indices, dtype=float),
        DopplerBin=np.array(doppler_bins, dtype=float),
        Amplitude=np.array(powers, dtype=float),
        Power=np.array(powers, dtype=float),
        AzimuthDeg=np.array(angle_h, dtype=float),
        ElevationDeg=np.array(angle_v, dtype=float),
        X=np.array(xs, dtype=float),
        Y=np.array(ys, dtype=float),
        Z=np.array(zs, dtype=float),
    )


# ============================================================================
# Radar processing
# ============================================================================


def build_array_geometry(array_cfg: ArrayConfig, proc_cfg: ProcessConfig) -> ArrayConfig:
    layout = array_cfg.layout
    max_channel = int(np.max(layout))
    row = np.full(max_channel, np.nan)
    col = np.full(max_channel, np.nan)

    for r in range(layout.shape[0]):
        for c in range(layout.shape[1]):
            ch = int(layout[r, c])
            if ch > 0:
                row[ch - 1] = r + 1
                col[ch - 1] = c + 1

    x_center_col = np.nanmean(col)
    y_center_row = np.nanmean(row)
    x = (col - x_center_col) * proc_cfg.rx_spacing_x
    y = (row - y_center_row) * proc_cfg.tx_spacing_y

    array_cfg.row = row
    array_cfg.col = col
    array_cfg.x = x
    array_cfg.y = y
    array_cfg.el_coherent_ants = proc_cfg.el_coherent_ants.copy()
    array_cfg.el_dbf_coord = y[array_cfg.el_coherent_ants[:, proc_cfg.el_ref_column]].reshape(-1)

    el_column_x = np.zeros(proc_cfg.el_coherent_ants.shape[1], dtype=float)
    for col_idx in range(proc_cfg.el_coherent_ants.shape[1]):
        el_column_x[col_idx] = np.mean(x[proc_cfg.el_coherent_ants[:, col_idx]])
    ref_x = el_column_x[proc_cfg.el_ref_column]
    array_cfg.el_column_x = el_column_x
    array_cfg.el_az_comp_lut = np.exp(
        -1j
        * 2.0
        * np.pi
        / proc_cfg.lambda_
        * (el_column_x.reshape(1, -1) - ref_x)
        * np.sin(np.deg2rad(proc_cfg.angle_grid_deg.reshape(-1, 1)))
    )
    return array_cfg


# Exact C firmware velocity window table from Radar_Config.c winVelPre.
# C writes winVel1[i]=winVelPre[4*i+0], winVel2[i]=winVelPre[4*i+1], etc.
C_WINVEL_PRE = np.array([1,5,11,20,31,44,60,78,99,122,147,175,205,238,272,309,349,390,434,480,528,578,631,685,742,800,860,923,987,1053,1121,1191,1262,1335,1410,1487,1565,1644,1725,1807,1891,1976,2062,2150,2239,2328,2419,2511,2604,2698,2792,2888,2984,3080,3178,3275,3374,3473,3572,3671,3771,3871,3971,4071,4171,4271,4371,4471,4571,4670,4769,4867,4966,5063,5160,5257,5352,5447,5541,5635,5727,5818,5909,5998,6086,6173,6259,6343,6426,6508,6588,6667,6744,6819,6893,6966,7036,7105,7172,7237,7301,7362,7421,7479,7534,7588,7639,7688,7735,7780,7823,7863,7901,7937,7971,8002,8031,8058,8082,8104,8123,8140,8155,8167,8177,8184,8189,8191,8191,8189,8184,8177,8167,8155,8140,8123,8104,8082,8058,8031,8002,7971,7937,7901,7863,7823,7780,7735,7688,7639,7588,7534,7479,7421,7362,7301,7237,7172,7105,7036,6966,6893,6819,6744,6667,6588,6508,6426,6343,6259,6173,6086,5998,5909,5818,5727,5635,5541,5447,5352,5257,5160,5063,4966,4867,4769,4670,4571,4471,4371,4271,4171,4071,3971,3871,3771,3671,3572,3473,3374,3275,3178,3080,2984,2888,2792,2698,2604,2511,2419,2328,2239,2150,2062,1976,1891,1807,1725,1644,1565,1487,1410,1335,1262,1191,1121,1053,987,923,860,800,742,685,631,578,528,480,434,390,349,309,272,238,205,175,147,122,99,78,60,44,31,20,11,5,1], dtype=np.float64)

def firmware_velocity_window_exact_c(proc_cfg: ProcessConfig) -> np.ndarray:
    n = proc_cfg.num_doppler_fft
    tx = proc_cfg.num_tx
    need = n * tx
    if C_WINVEL_PRE.size < need:
        raise ValueError("C_WINVEL_PRE is shorter than num_doppler_fft*num_tx")
    out = np.zeros((tx, n), dtype=np.float64)
    for tx_idx in range(tx):
        out[tx_idx, :] = C_WINVEL_PRE[tx_idx:need:tx]
    scale = np.max(C_WINVEL_PRE[:need])
    if scale <= 0:
        scale = 1.0
    return out / scale

def firmware_tdm_doppler_fft(doppler_input: np.ndarray, proc_cfg: ProcessConfig) -> np.ndarray:
    """Replicate BB_AlgProc.c::fft2dCalc() for exported 1D FFT frames.

    C behavior reproduced here:
      - numChirp=64 per virtual antenna, numTX=4, FFTPT=256.
      - each TX stream is placed at TDM slots txIdx + chirpIdx*numTX.
      - velocity window comes from Radar_Config.c winVelPre, split as winVel1..winVel4.
      - FFT_MODE_FORWARD.
      - unload order follows fft2d_useA=224..255, then 0..31.
    """
    num_range, num_chirp, num_ant = doppler_input.shape
    if num_chirp != proc_cfg.num_doppler_fft:
        raise ValueError(f"Expected {proc_cfg.num_doppler_fft} chirps, got {num_chirp}.")
    if num_ant != proc_cfg.num_tx * proc_cfg.num_rx:
        raise ValueError(f"Expected {proc_cfg.num_tx * proc_cfg.num_rx} virtual antennas, got {num_ant}.")

    fft_size = proc_cfg.firmware_doppler_fft_size
    use_a = (proc_cfg.num_tx - 1) * proc_cfg.num_doppler_fft + proc_cfg.num_doppler_fft // 2
    use_b = proc_cfg.num_doppler_fft // 2 - 1
    unload_bins = np.r_[use_a:fft_size, 0:use_b + 1]
    doppler_windows = firmware_velocity_window_exact_c(proc_cfg)
    rd_cube = np.zeros((num_range, proc_cfg.num_doppler_fft, num_ant), dtype=np.complex128)

    for ant_idx in range(num_ant):
        tx_idx = ant_idx // proc_cfg.num_rx
        slot_idx = tx_idx + np.arange(num_chirp) * proc_cfg.num_tx
        tdm_input = np.zeros((num_range, fft_size), dtype=np.complex128)
        tdm_input[:, slot_idx] = doppler_input[:, :, ant_idx] * doppler_windows[tx_idx][None, :]
        fft_out = np.fft.fft(tdm_input, n=fft_size, axis=1)
        rd_cube[:, :, ant_idx] = fft_out[:, unload_bins]

    return rd_cube

def wrap_index(idx: np.ndarray, length: int) -> np.ndarray:
    return np.mod(idx, length)


def velocity_cfar_base_map(amp_map: np.ndarray, cfar_cfg: CfarConfig) -> np.ndarray:
    num_range, num_doppler = amp_map.shape
    base_map = np.zeros((num_range, num_doppler), dtype=float)
    for doppler_idx in range(num_doppler):
        left_idx = wrap_index(
            np.arange(
                doppler_idx - cfar_cfg.guard_size - cfar_cfg.search_size,
                doppler_idx - cfar_cfg.guard_size,
            ),
            num_doppler,
        )
        right_idx = wrap_index(
            np.arange(
                doppler_idx + cfar_cfg.guard_size + 1,
                doppler_idx + cfar_cfg.guard_size + cfar_cfg.search_size + 1,
            ),
            num_doppler,
        )
        left_sum = np.sum(amp_map[:, left_idx], axis=1)
        right_sum = np.sum(amp_map[:, right_idx], axis=1)

        mode = cfar_cfg.mode.upper()
        if mode == "GO":
            noise_sum = np.maximum(left_sum, right_sum)
        elif mode == "SO":
            noise_sum = np.minimum(left_sum, right_sum)
        elif mode == "CA":
            noise_sum = left_sum + right_sum
        else:
            raise ValueError(f"Unsupported CFAR mode: {cfar_cfg.mode}")

        base_map[:, doppler_idx] = noise_sum / float(cfar_cfg.thres_div)

    return base_map


def local_doppler_peak_mask(amp_map: np.ndarray) -> np.ndarray:
    prev_map = amp_map[:, np.r_[-1, 0 : amp_map.shape[1] - 1]]
    next_map = amp_map[:, np.r_[1 : amp_map.shape[1], 0]]
    return (amp_map >= prev_map) & (amp_map > next_map)


def firmware_velocity_cfar(amp_map: np.ndarray, cfar_cfg: CfarConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_map = velocity_cfar_base_map(amp_map, cfar_cfg)
    score_map = amp_map / np.maximum(base_map, np.finfo(float).eps)
    threshold_map = cfar_cfg.mul_fac * base_map
    cfar_mask = amp_map > threshold_map
    if cfar_cfg.peak_det_enable:
        cfar_mask &= local_doppler_peak_mask(amp_map)
    return cfar_mask, threshold_map, score_map, base_map


def apply_detection_validity_mask(mask: np.ndarray, proc_cfg: ProcessConfig) -> np.ndarray:
    out = mask.copy()
    if proc_cfg.min_valid_range_bin > 0:
        out[: int(proc_cfg.min_valid_range_bin), :] = False
    return out


def signed_spatial_fft_bins(zero_based_bins: np.ndarray, num_angle: int) -> np.ndarray:
    signed_bins = np.asarray(zero_based_bins, dtype=float).copy()
    signed_bins[signed_bins > num_angle / 2 - 1] -= num_angle
    return signed_bins


def build_spatial_dbf_first_rd_map(
    rd_cube: np.ndarray,
    proc_cfg: ProcessConfig,
    array_cfg: ArrayConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the Expert branch RD statistic using azimuth DBF only.

    This implements the revised second-column algorithm:

        RD cube -> azimuth DBF peak map -> CFAR -> point cloud

    Elevation DBF is intentionally NOT used to enhance the pre-CFAR RD map.
    The returned spatial_amp_map is the azimuth DBF peak for each RD cell.
    best_el_map / el_peak_map / el_snr_map are placeholders for compatibility;
    3-D elevation is estimated only after CFAR in build_spatial_dbf_first_target_table().
    """
    if proc_cfg.expert_dbf_mode.lower() != "separable_current_array":
        raise ValueError(f"Unsupported expert_dbf_mode: {proc_cfg.expert_dbf_mode}")

    num_range, num_doppler, num_ant = rd_cube.shape
    num_cells = num_range * num_doppler
    rd_flat = rd_cube.reshape(num_cells, num_ant)

    az_grid_deg = proc_cfg.expert_az_grid_deg
    az_channels = proc_cfg.az_channels
    az_snapshot = rd_flat[:, az_channels]

    if proc_cfg.angle_estimator_mode.lower() == "firmware_fft128":
        az_beam_amp = np.abs(np.fft.fft(az_snapshot, n=proc_cfg.num_angle, axis=1))
        best_az_idx = np.argmax(az_beam_amp, axis=1)
        az_peak_flat = az_beam_amp[np.arange(num_cells), best_az_idx]
        best_az_signed = signed_spatial_fft_bins(best_az_idx, proc_cfg.num_angle)
        best_az_flat = np.degrees(np.arcsin(best_az_signed / (proc_cfg.num_angle / 2.0)))
    else:
        az_coord = array_cfg.x[az_channels]
        az_steering = np.exp(
            -1j
            * 2.0
            * np.pi
            / proc_cfg.lambda_
            * az_coord.reshape(-1, 1)
            * np.sin(np.deg2rad(az_grid_deg.reshape(1, -1)))
        )
        az_beam_amp = np.abs(az_snapshot @ np.conjugate(az_steering))
        best_az_idx = np.argmax(az_beam_amp, axis=1)
        az_peak_flat = az_beam_amp[np.arange(num_cells), best_az_idx]
        best_az_flat = az_grid_deg[best_az_idx]

    az_floor_flat = np.median(az_beam_amp, axis=1)
    az_snr_flat = az_peak_flat / np.maximum(az_floor_flat, np.finfo(float).eps)

    # Revised Expert statistic: use azimuth DBF peak as the pre-CFAR RD map.
    # No elevation DBF enhancement is applied before CFAR.
    spatial_amp_map = az_peak_flat.reshape(num_range, num_doppler)
    best_az_map = best_az_flat.reshape(num_range, num_doppler)
    az_peak_map = spatial_amp_map
    az_snr_map = az_snr_flat.reshape(num_range, num_doppler)

    # Compatibility placeholders.  These are not used for pre-CFAR enhancement.
    best_el_map = np.zeros((num_range, num_doppler), dtype=float)
    el_peak_map = np.zeros((num_range, num_doppler), dtype=float)
    el_snr_map = np.zeros((num_range, num_doppler), dtype=float)

    return spatial_amp_map, best_az_map, best_el_map, az_peak_map, el_peak_map, az_snr_map, el_snr_map

def build_spatial_dbf_first_target_table(
    peak_mask: np.ndarray,
    dbf_amp_map: np.ndarray,
    threshold_map: np.ndarray,
    best_az_map: np.ndarray,
    best_el_map: np.ndarray,
    az_peak_map: np.ndarray,
    el_peak_map: np.ndarray,
    proc_cfg: ProcessConfig,
    rd_cube: Optional[np.ndarray] = None,
    array_cfg: Optional[ArrayConfig] = None,
) -> Dict[str, np.ndarray]:
    range_idx, doppler_idx = np.nonzero(peak_mask)
    if range_idx.size == 0:
        return empty_point_table()

    amp = dbf_amp_map[range_idx, doppler_idx]
    order = np.argsort(amp)[::-1]
    keep_count = min(proc_cfg.max_targets, order.size)
    order = order[:keep_count]
    range_idx = range_idx[order]
    doppler_idx = doppler_idx[order]
    amp_sorted = amp[order]

    threshold = threshold_map[range_idx, doppler_idx]
    az_deg = best_az_map[range_idx, doppler_idx]
    az_peak = az_peak_map[range_idx, doppler_idx]

    # Elevation is deliberately estimated AFTER CFAR only for coordinate recovery.
    # It is not used to enhance the RD statistic before CFAR.
    el_deg = np.zeros(keep_count, dtype=float)
    el_peak = np.zeros(keep_count, dtype=float)
    if rd_cube is not None and array_cfg is not None:
        for i in range(keep_count):
            el_snapshot = make_coherent_elevation_snapshot(
                rd_cube,
                int(range_idx[i]),
                int(doppler_idx[i]),
                float(az_deg[i]),
                proc_cfg,
                array_cfg,
            )
            point_el_deg, _, point_el_peak, _ = estimate_elevation_angle(
                el_snapshot,
                array_cfg.el_dbf_coord,
                proc_cfg.angle_grid_deg,
                proc_cfg.lambda_,
                proc_cfg,
            )
            el_deg[i] = float(point_el_deg)
            el_peak[i] = float(point_el_peak)
    else:
        # Backward-compatible fallback.  In the azimuth-only pre-CFAR branch,
        # best_el_map is normally all zeros.
        el_deg = best_el_map[range_idx, doppler_idx]
        el_peak = el_peak_map[range_idx, doppler_idx]

    doppler_bins = np.arange(-proc_cfg.num_doppler_fft // 2, proc_cfg.num_doppler_fft // 2)
    doppler_bin = doppler_bins[doppler_idx]
    range_bin = range_idx.astype(float)
    range_m = range_bin * proc_cfg.range_resolution

    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    x_m = range_m * np.cos(el_rad) * np.cos(az_rad)
    y_m = range_m * np.cos(el_rad) * np.sin(az_rad)
    z_m = range_m * np.sin(el_rad)

    return make_table(
        RangeBin=range_bin,
        RangeM=range_m,
        DopplerBin=doppler_bin,
        Amplitude=amp_sorted,
        CfarThreshold=threshold,
        AzimuthDeg=az_deg,
        ElevationDeg=el_deg,
        X=x_m,
        Y=y_m,
        Z=z_m,
        AzPeak=az_peak,
        ElPeak=el_peak,
        AzimuthPeakIndex=np.ones(keep_count, dtype=int),
    )


def estimate_dbf_angle(snapshot: np.ndarray, coord: np.ndarray, angle_grid_deg: np.ndarray, lambda_: float) -> Tuple[float, np.ndarray, float, float]:
    snapshot = snapshot.reshape(-1)
    coord = coord.reshape(-1)
    steering = np.exp(
        -1j
        * 2.0
        * np.pi
        / lambda_
        * coord.reshape(-1, 1)
        * np.sin(np.deg2rad(angle_grid_deg.reshape(1, -1)))
    )
    beam = np.abs(np.conjugate(steering).T @ snapshot)
    peak_amp = float(np.max(beam)) if beam.size else 0.0
    noise_floor = float(np.median(beam)) if beam.size else 0.0
    spectrum = beam / peak_amp if peak_amp > 0 else np.zeros_like(beam)
    angle_deg = float(angle_grid_deg[int(np.argmax(spectrum))]) if spectrum.size else 0.0
    return angle_deg, spectrum, peak_amp, noise_floor


def estimate_pcd_aligned_elevation_angle(snapshot: np.ndarray, angle_grid_deg: np.ndarray) -> Tuple[float, np.ndarray, float, float]:
    snapshot = snapshot.reshape(-1)
    row_idx = np.arange(snapshot.size, dtype=float).reshape(-1, 1)
    steering = np.exp(1j * np.pi * row_idx * np.sin(np.deg2rad(angle_grid_deg.reshape(1, -1))))
    beam = np.abs(np.conjugate(steering).T @ snapshot)
    peak_amp = float(np.max(beam)) if beam.size else 0.0
    noise_floor = float(np.median(beam)) if beam.size else 0.0
    spectrum = beam / peak_amp if peak_amp > 0 else np.zeros_like(beam)
    angle_deg = float(angle_grid_deg[int(np.argmax(spectrum))]) if spectrum.size else 0.0
    return angle_deg, spectrum, peak_amp, noise_floor


def estimate_firmware_spatial_fft_angle(snapshot: np.ndarray, num_angle: int) -> Tuple[float, np.ndarray, float, float]:
    snapshot = snapshot.reshape(-1)
    fft_spectrum = np.abs(np.fft.fft(snapshot, n=num_angle))
    peak_amp = float(np.max(fft_spectrum)) if fft_spectrum.size else 0.0
    noise_floor = float(np.median(fft_spectrum)) if fft_spectrum.size else 0.0
    spectrum = fft_spectrum / peak_amp if peak_amp > 0 else np.zeros_like(fft_spectrum)
    max_idx = int(np.argmax(spectrum)) if spectrum.size else 0
    signed_idx = max_idx
    if signed_idx > num_angle / 2 - 1:
        signed_idx -= num_angle
    angle_deg = float(np.degrees(np.arcsin(signed_idx / (num_angle / 2.0))))
    return angle_deg, spectrum, peak_amp, noise_floor


def estimate_azimuth_angle(
    snapshot: np.ndarray,
    coord: np.ndarray,
    angle_grid_deg: np.ndarray,
    lambda_: float,
    proc_cfg: ProcessConfig,
) -> Tuple[float, np.ndarray, float, float]:
    if proc_cfg.angle_estimator_mode.lower() == "firmware_fft128":
        return estimate_firmware_spatial_fft_angle(snapshot, proc_cfg.num_angle)
    return estimate_dbf_angle(snapshot, coord, angle_grid_deg, lambda_)


def estimate_elevation_angle(
    snapshot: np.ndarray,
    coord: np.ndarray,
    angle_grid_deg: np.ndarray,
    lambda_: float,
    proc_cfg: ProcessConfig,
) -> Tuple[float, np.ndarray, float, float]:
    if proc_cfg.angle_estimator_mode.lower() == "firmware_fft128":
        return estimate_firmware_spatial_fft_angle(snapshot, proc_cfg.num_angle)
    if proc_cfg.angle_estimator_mode.lower() == "pcd_aligned_dbf":
        return estimate_pcd_aligned_elevation_angle(snapshot, angle_grid_deg)
    return estimate_dbf_angle(snapshot, coord, angle_grid_deg, lambda_)


def select_dbf_peaks(
    spectrum: np.ndarray,
    angle_grid_deg: np.ndarray,
    max_peaks: int,
    min_peak_ratio: float,
    min_separation_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    spectrum = np.asarray(spectrum).reshape(-1)
    if spectrum.size == 0 or np.max(spectrum) <= 0:
        return np.empty(0), np.empty(0)

    peak_mask = np.zeros(spectrum.size, dtype=bool)
    if spectrum.size >= 3:
        peak_mask[1:-1] = (spectrum[1:-1] >= spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])

    candidate_idx = np.flatnonzero(peak_mask & (spectrum >= min_peak_ratio * np.max(spectrum)))
    if candidate_idx.size == 0:
        candidate_idx = np.array([int(np.argmax(spectrum))])

    candidate_idx = candidate_idx[np.argsort(spectrum[candidate_idx])[::-1]]
    selected: List[int] = []
    for idx in candidate_idx:
        if not selected or np.all(np.abs(angle_grid_deg[idx] - angle_grid_deg[selected]) >= min_separation_deg):
            selected.append(int(idx))
        if len(selected) >= max_peaks:
            break

    selected_idx = np.array(selected, dtype=int)
    return angle_grid_deg[selected_idx], spectrum[selected_idx] / np.max(spectrum)


def make_coherent_elevation_snapshot(
    rd_cube: np.ndarray,
    range_idx: int,
    doppler_idx: int,
    az_deg: float,
    proc_cfg: ProcessConfig,
    array_cfg: ArrayConfig,
) -> np.ndarray:
    el_ants = array_cfg.el_coherent_ants
    mode = proc_cfg.angle_estimator_mode.lower()
    if mode == "firmware_fft128":
        az_bin_signed = int(round((proc_cfg.num_angle / 2.0) * np.sin(np.deg2rad(az_deg))))
        az_bin0 = az_bin_signed % proc_cfg.num_angle
        comp = np.exp(-1j * 2.0 * np.pi / proc_cfg.num_angle * az_bin0 * np.arange(el_ants.shape[1]))
    elif mode == "pcd_aligned_dbf":
        # Empirically aligned with PCD/C angle chain on current calibrated 1D FFT data.
        # The previous negative sign made exact-RD elevation angles flip/outlier.
        comp = np.exp(1j * np.pi * np.arange(el_ants.shape[1]) * np.sin(np.deg2rad(az_deg)))
    else:
        az_grid_idx = int(np.argmin(np.abs(proc_cfg.angle_grid_deg - az_deg)))
        comp = array_cfg.el_az_comp_lut[az_grid_idx, :]

    el_snapshot = np.zeros(el_ants.shape[0], dtype=np.complex128)
    for row_idx in range(el_ants.shape[0]):
        row_ants = el_ants[row_idx, :]
        el_snapshot[row_idx] = rd_cube[range_idx, doppler_idx, row_ants] @ comp
    return el_snapshot


def build_target_table(
    rd_cube: np.ndarray,
    rd_amp_map: np.ndarray,
    threshold_map: np.ndarray,
    peak_mask: np.ndarray,
    proc_cfg: ProcessConfig,
    array_cfg: ArrayConfig,
) -> Dict[str, np.ndarray]:
    range_idx, doppler_idx = np.nonzero(peak_mask)
    if range_idx.size == 0:
        return empty_point_table()

    amp = rd_amp_map[range_idx, doppler_idx]
    order = np.argsort(amp)[::-1]
    keep_count = min(proc_cfg.max_targets, order.size)
    order = order[:keep_count]
    range_idx = range_idx[order]
    doppler_idx = doppler_idx[order]
    amp_sorted = amp[order]
    threshold = threshold_map[range_idx, doppler_idx]

    doppler_bins = np.arange(-proc_cfg.num_doppler_fft // 2, proc_cfg.num_doppler_fft // 2)
    doppler_bin = doppler_bins[doppler_idx]
    range_bin = range_idx.astype(float)
    range_m = range_bin * proc_cfg.range_resolution

    out: Dict[str, List[float]] = {
        "RangeBin": [],
        "RangeM": [],
        "DopplerBin": [],
        "Amplitude": [],
        "CfarThreshold": [],
        "AzimuthDeg": [],
        "ElevationDeg": [],
        "X": [],
        "Y": [],
        "Z": [],
        "AzPeak": [],
        "ElPeak": [],
        "AzimuthPeakIndex": [],
    }

    dbf_count = keep_count if math.isinf(proc_cfg.max_targets_to_dbf) else min(int(proc_cfg.max_targets_to_dbf), keep_count)
    for k in range(dbf_count):
        az_snapshot = rd_cube[range_idx[k], doppler_idx[k], proc_cfg.az_channels]
        _, az_spectrum, az_peak_top, az_noise_floor = estimate_azimuth_angle(
            az_snapshot,
            array_cfg.x[proc_cfg.az_channels],
            proc_cfg.angle_grid_deg,
            proc_cfg.lambda_,
            proc_cfg,
        )
        az_snr_linear = az_peak_top / max(az_noise_floor, np.finfo(float).eps)
        if proc_cfg.dbf_az_snr_gate_enable and az_snr_linear < proc_cfg.dbf_az_snr_threshold:
            continue

        az_angles, az_peak_ratios = select_dbf_peaks(
            az_spectrum,
            proc_cfg.angle_grid_deg,
            proc_cfg.max_azimuth_peaks_per_rd,
            proc_cfg.min_azimuth_peak_ratio,
            proc_cfg.min_azimuth_peak_separation_deg,
        )

        for peak_idx, az_deg in enumerate(az_angles):
            el_snapshot = make_coherent_elevation_snapshot(rd_cube, int(range_idx[k]), int(doppler_idx[k]), float(az_deg), proc_cfg, array_cfg)
            point_el_deg, _, el_peak, _ = estimate_elevation_angle(
                el_snapshot,
                array_cfg.el_dbf_coord,
                proc_cfg.angle_grid_deg,
                proc_cfg.lambda_,
                proc_cfg,
            )

            point_range_m = float(range_m[k])
            point_x = point_range_m * np.cos(np.deg2rad(point_el_deg)) * np.cos(np.deg2rad(az_deg))
            point_y = point_range_m * np.cos(np.deg2rad(point_el_deg)) * np.sin(np.deg2rad(az_deg))
            point_z = point_range_m * np.sin(np.deg2rad(point_el_deg))

            out["RangeBin"].append(float(range_bin[k]))
            out["RangeM"].append(point_range_m)
            out["DopplerBin"].append(float(doppler_bin[k]))
            out["Amplitude"].append(float(amp_sorted[k]))
            out["CfarThreshold"].append(float(threshold[k]))
            out["AzimuthDeg"].append(float(az_deg))
            out["ElevationDeg"].append(float(point_el_deg))
            out["X"].append(float(point_x))
            out["Y"].append(float(point_y))
            out["Z"].append(float(point_z))
            out["AzPeak"].append(float(az_peak_top * az_peak_ratios[peak_idx]))
            out["ElPeak"].append(float(el_peak))
            out["AzimuthPeakIndex"].append(float(peak_idx + 1))

    if not out["RangeBin"]:
        return empty_point_table()

    return make_table(**{key: np.asarray(value) for key, value in out.items()})


def process_usb_fft_frame(
    fft_cube: np.ndarray,
    proc_cfg: ProcessConfig,
    cfar_cfg: CfarConfig,
    expert_cfar_cfg: CfarConfig,
    array_cfg: ArrayConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    mean_range = np.mean(fft_cube, axis=1, keepdims=True)
    doppler_input = fft_cube - mean_range
    rd_cube = firmware_tdm_doppler_fft(doppler_input, proc_cfg)

    # Original branch: noncoherent channel accumulation -> CFAR -> DBF DOA.
    rd_amp_map = np.sum(np.abs(rd_cube), axis=2)
    cfar_mask, threshold_map, _, _ = firmware_velocity_cfar(rd_amp_map, cfar_cfg)
    cfar_mask = apply_detection_validity_mask(cfar_mask, proc_cfg)

    # Expert branch: azimuth-DBF-enhanced RD map -> CFAR. Elevation is estimated only after CFAR for 3-D coordinates.
    (
        dbf_first_amp_map,
        dbf_first_az_map,
        dbf_first_el_map,
        dbf_first_az_peak_map,
        dbf_first_el_peak_map,
        dbf_first_az_snr_map,
        dbf_first_el_snr_map,
    ) = build_spatial_dbf_first_rd_map(rd_cube, proc_cfg, array_cfg)
    dbf_first_cfar_mask, dbf_first_threshold_map, _, _ = firmware_velocity_cfar(dbf_first_amp_map, expert_cfar_cfg)
    dbf_first_cfar_mask = apply_detection_validity_mask(dbf_first_cfar_mask, proc_cfg)

    target_table = build_target_table(rd_cube, rd_amp_map, threshold_map, cfar_mask, proc_cfg, array_cfg)
    dbf_first_target_table = build_spatial_dbf_first_target_table(
        dbf_first_cfar_mask,
        dbf_first_amp_map,
        dbf_first_threshold_map,
        dbf_first_az_map,
        dbf_first_el_map,
        dbf_first_az_peak_map,
        dbf_first_el_peak_map,
        proc_cfg,
        rd_cube,
        array_cfg,
    )

    radar_data = {
        "fftCube": fft_cube,
        "meanRange": mean_range,
        "dopplerInput": doppler_input,
        "rdCube": rd_cube,
        "rdAmpMap": rd_amp_map,
        "cfarMask": cfar_mask,
        "thresholdMap": threshold_map,
        "dbfFirstAmpMap": dbf_first_amp_map,
        "dbfFirstAzMap": dbf_first_az_map,
        "dbfFirstElMap": dbf_first_el_map,
        "dbfFirstAzPeakMap": dbf_first_az_peak_map,
        "dbfFirstElPeakMap": dbf_first_el_peak_map,
        "dbfFirstAzSnrMap": dbf_first_az_snr_map,
        "dbfFirstElSnrMap": dbf_first_el_snr_map,
        "dbfFirstCfarMask": dbf_first_cfar_mask,
        "dbfFirstDetectionMask": dbf_first_cfar_mask,
        "dbfFirstThresholdMap": dbf_first_threshold_map,
        "dbfFirstTargetTable": dbf_first_target_table,
    }
    return radar_data, target_table


def pcd_dynamic_to_rd_map(
    pcd_dynamic: Dict[str, np.ndarray],
    common_cfg: CommonConfig,
    proc_cfg: ProcessConfig,
) -> Dict[str, np.ndarray]:
    out_map = np.zeros((common_cfg.num_range_bin, proc_cfg.num_doppler_fft), dtype=float)
    occupancy_map = np.zeros_like(out_map)
    if table_len(pcd_dynamic) == 0:
        return {"map": out_map, "occupancyMap": occupancy_map, "rangeBins": np.empty(0), "dopplerBins": np.empty(0)}

    range_bin = pcd_dynamic["RangeBin"].astype(int)
    doppler_bin = pcd_dynamic["DopplerBin"]
    doppler_index = pcd_dynamic["DopplerIndex"].astype(int)
    power = pcd_dynamic.get("Power", pcd_dynamic.get("Amplitude", np.ones(range_bin.size)))

    valid = (
        (range_bin >= 0)
        & (range_bin < common_cfg.num_range_bin)
        & (doppler_index >= 0)
        & (doppler_index < proc_cfg.num_doppler_fft)
    )
    range_bin = range_bin[valid]
    doppler_index = doppler_index[valid]
    doppler_bin = doppler_bin[valid]
    power = power[valid]

    for r, d, p in zip(range_bin, doppler_index, power):
        out_map[r, d] = max(out_map[r, d], float(p))
        occupancy_map[r, d] = 1.0

    return {"map": out_map, "occupancyMap": occupancy_map, "rangeBins": range_bin, "dopplerBins": doppler_bin}

# ============================================================================
# User-facing wrappers and simple export helpers
# ============================================================================


PointCloudResult = Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]


def make_default_configs() -> Tuple[CommonConfig, ProcessConfig, CfarConfig, CfarConfig, ArrayConfig]:
    """Create default configs and pre-build array geometry.

    Returns:
        (common_cfg, proc_cfg, cfar_cfg, expert_cfar_cfg, array_cfg)
    """
    common_cfg = CommonConfig()
    proc_cfg = ProcessConfig()
    cfar_cfg = CfarConfig()
    expert_cfar_cfg = CfarConfig()
    array_cfg = build_array_geometry(ArrayConfig(), proc_cfg)
    return common_cfg, proc_cfg, cfar_cfg, expert_cfar_cfg, array_cfg


def point_table_to_xyz(point_table: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert a point table dict to an [N, 3] XYZ ndarray.

    Args:
        point_table: Dict produced by build_target_table,
            build_spatial_dbf_first_target_table, or generate_point_cloud_from_bin.

    Returns:
        ndarray [N, 3], columns are X, Y, Z in meters.  Empty result is [0, 3].
    """
    n = table_len(point_table)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack([
        np.asarray(point_table.get("X", np.empty(0)), dtype=np.float64),
        np.asarray(point_table.get("Y", np.empty(0)), dtype=np.float64),
        np.asarray(point_table.get("Z", np.empty(0)), dtype=np.float64),
    ])


def point_table_to_array(point_table: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert a point table dict to a dense numeric array.

    Args:
        point_table: Point table dict.

    Returns:
        ndarray [N, 8] with columns:
        X, Y, Z, RangeM, DopplerBin, Amplitude, AzimuthDeg, ElevationDeg.
    """
    n = table_len(point_table)
    if n == 0:
        return np.empty((0, 8), dtype=np.float64)
    cols = ["X", "Y", "Z", "RangeM", "DopplerBin", "Amplitude", "AzimuthDeg", "ElevationDeg"]
    return np.column_stack([
        np.asarray(point_table.get(name, np.full(n, np.nan)), dtype=np.float64)
        for name in cols
    ])


def generate_point_cloud_from_bin(
    bin_file: str,
    branch: str = "dbf_first",
    common_cfg: Optional[CommonConfig] = None,
    proc_cfg: Optional[ProcessConfig] = None,
    cfar_cfg: Optional[CfarConfig] = None,
    expert_cfar_cfg: Optional[CfarConfig] = None,
    array_cfg: Optional[ArrayConfig] = None,
    frame_idx: int = 1,
    return_radar_data: bool = False,
) -> PointCloudResult:
    """Generate point cloud from one USB FFT bin file.

    Args:
        bin_file: Path of one USB FFT frame, e.g. "usb_fft_xxx.bin".
        branch: "dbf_first"/"expert" for DBF-before-CFAR branch,
            "original" for original CFAR-before-DBF branch, or "both".
        common_cfg: Optional CommonConfig.  Default uses original script values.
        proc_cfg: Optional ProcessConfig.  Default uses original script values.
        cfar_cfg: Optional CFAR config for original branch.
        expert_cfar_cfg: Optional CFAR config for DBF-first branch.
        array_cfg: Optional pre-built ArrayConfig.  If omitted, geometry is built.
        frame_idx: Frame id written into the returned point table's Frame column.
        return_radar_data: If True, include intermediate radar_data and fft_cube.

    Returns:
        If branch is "dbf_first" or "original": point table dict whose columns
        include X/Y/Z, range, doppler, amplitude, azimuth and elevation.
        If branch is "both":
            {
              "original": original_branch_point_table,
              "dbf_first": dbf_first_branch_point_table,
              optionally "radar_data" and "fft_cube"
            }
    """
    if common_cfg is None:
        common_cfg = CommonConfig()
    if proc_cfg is None:
        proc_cfg = ProcessConfig()
    if cfar_cfg is None:
        cfar_cfg = CfarConfig()
    if expert_cfar_cfg is None:
        expert_cfar_cfg = CfarConfig()
    if array_cfg is None:
        array_cfg = build_array_geometry(ArrayConfig(), proc_cfg)

    fft_cube = read_usb_fft_frame(bin_file, common_cfg)
    radar_data, target_table = process_usb_fft_frame(
        fft_cube=fft_cube,
        proc_cfg=proc_cfg,
        cfar_cfg=cfar_cfg,
        expert_cfar_cfg=expert_cfar_cfg,
        array_cfg=array_cfg,
    )

    original_pc = point_table_from_target(target_table, frame_idx)
    dbf_first_pc = point_table_from_target(radar_data["dbfFirstTargetTable"], frame_idx)

    key = branch.strip().lower()
    if key in {"dbf_first", "dbf", "expert", "spatial_dbf_first"}:
        if return_radar_data:
            return {"point_cloud": dbf_first_pc, "radar_data": radar_data, "fft_cube": fft_cube}
        return dbf_first_pc
    if key in {"original", "cfar_first", "software"}:
        if return_radar_data:
            return {"point_cloud": original_pc, "radar_data": radar_data, "fft_cube": fft_cube}
        return original_pc
    if key == "both":
        out: Dict[str, object] = {"original": original_pc, "dbf_first": dbf_first_pc}
        if return_radar_data:
            out["radar_data"] = radar_data
            out["fft_cube"] = fft_cube
        return out  # type: ignore[return-value]
    raise ValueError("branch must be one of: dbf_first, original, both")


def save_point_table_csv(point_table: Dict[str, np.ndarray], csv_path: str) -> None:
    """Save a point table dict to CSV.

    Args:
        point_table: Point table dict with aligned 1-D columns.
        csv_path: Output CSV path.

    Returns:
        None.  Creates parent folder if needed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    n = table_len(point_table)
    columns = [
        "Frame", "RangeBin", "RangeM", "DopplerBin", "Amplitude", "CfarThreshold",
        "AzimuthDeg", "ElevationDeg", "AzimuthPeakIndex", "X", "Y", "Z",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(n):
            row = []
            for name in columns:
                arr = np.asarray(point_table.get(name, np.full(n, np.nan)))
                row.append(arr[i] if i < arr.size else np.nan)
            writer.writerow(row)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate radar point cloud from one USB FFT bin file")
    parser.add_argument("bin_file", help="Path to one usb_fft_*.bin frame")
    parser.add_argument(
        "--branch",
        default="dbf_first",
        choices=["dbf_first", "original", "both"],
        help="Point-cloud generation branch",
    )
    parser.add_argument("--csv", default="", help="Optional CSV output path for one branch")
    parser.add_argument("--npz", default="", help="Optional NPZ output path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    result = generate_point_cloud_from_bin(args.bin_file, branch=args.branch)

    if args.branch == "both":
        original = result["original"]  # type: ignore[index]
        dbf_first = result["dbf_first"]  # type: ignore[index]
        print(f"original points: {table_len(original)}")
        print(f"dbf_first points: {table_len(dbf_first)}")
        if args.npz:
            np.savez(
                args.npz,
                original=point_table_to_array(original),
                dbf_first=point_table_to_array(dbf_first),
            )
    else:
        point_table = result  # type: ignore[assignment]
        print(f"points: {table_len(point_table)}")
        print("columns: X,Y,Z,RangeM,DopplerBin,Amplitude,AzimuthDeg,ElevationDeg")
        print(point_table_to_array(point_table))
        if args.csv:
            save_point_table_csv(point_table, args.csv)
            print(f"saved csv: {args.csv}")
        if args.npz:
            np.savez(args.npz, point_cloud=point_table_to_array(point_table), xyz=point_table_to_xyz(point_table))
            print(f"saved npz: {args.npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
