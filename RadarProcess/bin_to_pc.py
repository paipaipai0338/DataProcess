from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from RadarProcess.utils import (
    Radar_Config,
    angle_fft,
    bin_to_cube_range_fft,
    doppler_fft,
    elevation_mvdr_with_azimuth_compensation,
    get_radar_res,
    two_dimension_cfar,
)


ArrayLikePath = Union[Path, str]


def mean_cancel_slow_time(range_fft_cube: np.ndarray) -> np.ndarray:
    """Remove static clutter by subtracting the chirp-axis mean."""
    cube = np.asarray(range_fft_cube)
    if cube.ndim != 3:
        raise RuntimeError(f"range_fft_cube must have shape (range, chirp, ant), got {cube.shape}")
    return cube - np.mean(cube, axis=1, keepdims=True)


def _empty_targets(return_dict: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    cartesian = np.empty((0, 4), dtype=np.float64)
    if not return_dict:
        return cartesian
    return {
        "cartesian coordinate": cartesian,
        "polar coordinate": np.empty((0, 4), dtype=np.float64),
        "target index": np.empty((0, 2), dtype=np.int64),
        "rd map": np.empty((0, 0), dtype=np.float64),
        "cfar mask": np.empty((0, 0), dtype=bool),
    }


def _rd_map_to_db(rd_map: np.ndarray) -> np.ndarray:
    peak = float(np.max(rd_map)) if rd_map.size else 0.0
    if peak <= 0.0:
        return np.full_like(rd_map, -120.0, dtype=np.float64)
    return 20.0 * np.log10(np.maximum(rd_map, np.finfo(float).eps) / peak)


def get_bin_pc(
    file_path: ArrayLikePath,
    ref_range: int = 10,
    ref_velocity: int = 10,
    guard_range: int = 4,
    guard_velocity: int = 2,
    alpha: float = 2.0,
    mode: str = "ca",
    *,
    mean_cancel: bool = True,
    min_range_m: float = 0.0,
    max_range_m: Optional[float] = None,
    min_abs_velocity_mps: float = 0.0,
    max_targets: Optional[int] = None,
    return_dict: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Generate a radar point cloud from one range-FFT .bin frame.

    The flow mirrors RadarProcess.utils.get_corner_data, but enables slow-time
    mean cancellation before Doppler FFT.  This suppresses static reflectors and
    leaves moving human targets easier for CFAR to pick up.

    Returns:
        Default: ndarray with shape (N, 4), columns [x, y, z, velocity].
        return_dict=True: dict with cartesian/polar coordinates plus debug maps.
    """
    radar_config = Radar_Config()
    range_fft_cube = bin_to_cube_range_fft(str(file_path), radar_config)
    if range_fft_cube is None:
        return _empty_targets(return_dict=return_dict)

    range_res, _, _, _ = get_radar_res(radar_config)
    r_axis = np.arange(range_fft_cube.shape[0], dtype=np.float64) * range_res

    doppler_input = mean_cancel_slow_time(range_fft_cube) if mean_cancel else range_fft_cube
    rd_cube, v_axis = doppler_fft(
        doppler_input,
        radar_config,
        window=True,
        doppler_mode="firmware_tdm",
    )

    rd_map = np.sum(np.abs(rd_cube), axis=-1)
    if not np.any(np.isfinite(rd_map)) or np.max(rd_map) <= 0:
        return _empty_targets(return_dict=return_dict)

    cfar_mask = two_dimension_cfar(
        data=rd_map,
        ref_range=ref_range,
        ref_velocity=ref_velocity,
        guard_range=guard_range,
        guard_velocity=guard_velocity,
        alpha=alpha,
        mode=mode.lower(),
    )

    range_indices, velocity_indices = np.where(cfar_mask)
    if range_indices.size == 0:
        return _empty_targets(return_dict=return_dict)

    valid = r_axis[range_indices] >= float(min_range_m)
    if max_range_m is not None:
        valid &= r_axis[range_indices] <= float(max_range_m)
    if min_abs_velocity_mps > 0:
        valid &= np.abs(v_axis[velocity_indices]) >= float(min_abs_velocity_mps)

    range_indices = range_indices[valid]
    velocity_indices = velocity_indices[valid]
    if range_indices.size == 0:
        return _empty_targets(return_dict=return_dict)

    amplitudes = rd_map[range_indices, velocity_indices]
    order = np.argsort(amplitudes)[::-1]
    if max_targets is not None:
        order = order[: int(max_targets)]
    range_indices = range_indices[order]
    velocity_indices = velocity_indices[order]

    polar_points = []
    cartesian_points = []
    accepted_indices = []
    channels_azi = [8, 9, 10, 11, 12, 13, 14, 15]

    for range_idx, velocity_idx in zip(range_indices, velocity_indices):
        r = float(r_axis[range_idx])
        v = float(v_axis[velocity_idx])

        az_spectrum, az_axis = angle_fft(
            rd_cube,
            radar_config,
            target_index=[int(range_idx), int(velocity_idx)],
            channel_index=channels_azi,
            type="azi",
            method="MVDR",
        )
        az_spectrum = np.abs(az_spectrum)
        if az_spectrum.size == 0 or np.max(az_spectrum) <= 0:
            continue
        azimuth_rad = float(az_axis[int(np.argmax(az_spectrum))])

        ele_spectrum, ele_axis, _ = elevation_mvdr_with_azimuth_compensation(
            rd_cube,
            radar_config,
            range_idx=int(range_idx),
            velocity_idx=int(velocity_idx),
            azimuth_rad=azimuth_rad,
            n_fft_angle=1024,
        )
        ele_spectrum = np.abs(ele_spectrum)
        if ele_spectrum.size == 0 or np.max(ele_spectrum) <= 0:
            continue
        elevation_rad = float(ele_axis[int(np.argmax(ele_spectrum))])

        azimuth_deg = float(np.rad2deg(azimuth_rad))
        elevation_deg = float(np.rad2deg(elevation_rad))
        polar_points.append([r, v, azimuth_deg, elevation_deg])

        x = r * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = r * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = r * np.sin(elevation_rad)
        cartesian_points.append([float(x), float(y), float(z), v])
        accepted_indices.append([int(range_idx), int(velocity_idx)])

    cartesian = np.asarray(cartesian_points, dtype=np.float64).reshape(-1, 4)
    if not return_dict:
        return cartesian

    polar = np.asarray(polar_points, dtype=np.float64).reshape(-1, 4)
    target_indices = np.asarray(accepted_indices, dtype=np.int64).reshape(-1, 2)

    return {
        "cartesian coordinate": cartesian,
        "polar coordinate": polar,
        "target index": target_indices,
        "rd map": rd_map,
        "rd map db": _rd_map_to_db(rd_map),
        "cfar mask": cfar_mask,
        "doppler input": doppler_input,
        "rd cube": rd_cube,
        "range axis": r_axis,
        "velocity axis": v_axis,
    }
