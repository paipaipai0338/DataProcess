#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility-style wrapper for lky.py.

Goal
----
Convert one USB 1D-FFT pseudo-float .bin frame into a radar point-cloud array
with a simple API similar to sensors_utils.py:

    pc = generate_point_cloud_array_from_bin("xxx/xxx/xx.bin")

This file intentionally reuses the validated low-level implementation in
lky.py: pseudo-float decoding, firmware TDM Doppler FFT, CFAR, DBF angle
estimation, and XYZ construction.  The main additions here are:

1) a compact dataclass config for common intermediate parameters;
2) selectable MTI / clutter cancellation before Doppler FFT;
3) wrapper functions returning ndarray directly.

Expected input
--------------
The .bin file is the USB 1D-FFT export used by lky.py, i.e. after range FFT,
with shape [range=256, chirp=64, virtual_ant=16] after decoding.

Returned array format
---------------------
Default output_format="xyz":
    ndarray [N, 3] = X, Y, Z

output_format="full":
    ndarray [N, 8] = X, Y, Z, RangeM, DopplerBin, Amplitude, AzimuthDeg, ElevationDeg

output_format="table":
    dict[str, np.ndarray] point table with named aligned columns.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union, Literal, Any

import numpy as np

# Put lky.py in the same directory as this file, or make sure it is importable.
import lky


MtiMode = Literal["none", "mean", "two_pulse", "three_pulse", "iir"]
BranchMode = Literal["dbf_first", "original", "both"]
OutputFormat = Literal["xyz", "full", "table"]


@dataclass
class RadarPointCloudConfig:
    """High-level point-cloud generation config.

    The defaults preserve lky.py's original behavior as much as possible:
    mti="mean" matches the original fixed `fft_cube - mean(axis=chirp)` step.

    Notes on channel order:
      - az_channels and el_coherent_ants are Python zero-based indices.
      - Do not reverse these casually; channel order changes phase slope sign.
    """

    # -------- main behavior --------
    branch: BranchMode = "dbf_first"
    output_format: OutputFormat = "xyz"
    frame_idx: int = 1

    # -------- MTI / clutter cancellation before Doppler FFT --------
    # lky.py originally always did mean subtraction over chirps.
    # Set mti="none" to disable this cancellation.
    mti: MtiMode = "mean"
    mti_alpha: float = 0.95  # only used by mti="iir"

    # -------- file / cube dimensions --------
    num_range_bin: int = 256
    num_chirp: int = 64
    num_ant: int = 16

    # -------- radar / processing constants --------
    num_doppler_fft: int = 64
    num_tx: int = 4
    num_rx: int = 4
    freq_start: float = 56.4e9
    bandwidth: float = 6.4453e9
    chirp_gap: float = 80e-6
    c: float = 299_792_458.0

    # -------- CFAR parameters --------
    # Your sample frame may need mul_fac around 3.0; lky original default is 10.0.
    cfar_mode: str = "SO"
    cfar_search_size: int = 8
    cfar_guard_size: int = 5
    cfar_mul_fac: float = 10.0
    cfar_thres_div: int = 8
    cfar_peak_det_enable: bool = False

    # Expert / DBF-first branch CFAR. None means use the same as cfar_*.
    expert_cfar_mode: Optional[str] = None
    expert_cfar_search_size: Optional[int] = None
    expert_cfar_guard_size: Optional[int] = None
    expert_cfar_mul_fac: Optional[float] = None
    expert_cfar_thres_div: Optional[int] = None
    expert_cfar_peak_det_enable: Optional[bool] = None

    # -------- DBF / point filtering parameters --------
    max_targets: int = 3000
    max_targets_to_dbf: float = np.inf
    min_valid_range_bin: int = 8
    max_azimuth_peaks_per_rd: int = 1
    min_azimuth_peak_ratio: float = 0.60
    min_azimuth_peak_separation_deg: float = 10.0
    dbf_az_snr_gate_enable: bool = True
    dbf_az_snr_threshold: float = 4.0
    angle_estimator_mode: str = "pcd_aligned_dbf"  # or "firmware_fft128"
    expert_dbf_mode: str = "separable_current_array"

    # Python zero-based channel index.  lky original:
    # MATLAB [16,15,14,13,12,11,10,9] -> Python [15..8]
    az_channels: np.ndarray = field(default_factory=lambda: np.arange(16, 8, -1, dtype=int) - 1)

    # Used by elevation coherent synthesis.  lky original:
    # MATLAB [[4,3,2,1],[8,7,6,5],[12,11,10,9]] -> Python -1
    el_coherent_ants: np.ndarray = field(
        default_factory=lambda: np.array(
            [[4, 3, 2, 1], [8, 7, 6, 5], [12, 11, 10, 9]], dtype=int
        ) - 1
    )
    el_channels: np.ndarray = field(default_factory=lambda: np.array([1, 5, 9], dtype=int) - 1)
    el_ref_column: int = 3

    # Angle grids.  They are arrays, so keep default_factory.
    angle_grid_deg: np.ndarray = field(default_factory=lambda: np.arange(-80.0, 80.0 + 0.25, 0.25))
    expert_az_grid_deg: np.ndarray = field(default_factory=lambda: np.arange(-80.0, 80.0 + 1.0, 1.0))
    expert_el_grid_deg: np.ndarray = field(default_factory=lambda: np.arange(-80.0, 80.0 + 1.0, 1.0))
    num_angle: int = 128


@dataclass
class RadarPointCloudOutput:
    """Structured output for debugging / visualization."""

    point_cloud: Union[np.ndarray, Dict[str, np.ndarray]]
    point_table: Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]
    radar_data: Optional[Dict[str, Any]] = None
    config: Optional[RadarPointCloudConfig] = None


def make_lky_configs(cfg: RadarPointCloudConfig) -> Tuple[
    lky.CommonConfig,
    lky.ProcessConfig,
    lky.CfarConfig,
    lky.CfarConfig,
    lky.ArrayConfig,
]:
    """Build lky.py config objects from the compact wrapper config."""
    common_cfg = lky.CommonConfig(
        num_range_bin=cfg.num_range_bin,
        num_chirp=cfg.num_chirp,
        num_ant=cfg.num_ant,
    )

    proc_cfg = lky.ProcessConfig()
    proc_cfg.num_doppler_fft = cfg.num_doppler_fft
    proc_cfg.num_tx = cfg.num_tx
    proc_cfg.num_rx = cfg.num_rx
    proc_cfg.freq_start = cfg.freq_start
    proc_cfg.bandwidth = cfg.bandwidth
    proc_cfg.chirp_gap = cfg.chirp_gap
    proc_cfg.c = cfg.c

    proc_cfg.angle_grid_deg = np.asarray(cfg.angle_grid_deg, dtype=float)
    proc_cfg.expert_az_grid_deg = np.asarray(cfg.expert_az_grid_deg, dtype=float)
    proc_cfg.expert_el_grid_deg = np.asarray(cfg.expert_el_grid_deg, dtype=float)
    proc_cfg.num_angle = int(cfg.num_angle)

    proc_cfg.az_channels = np.asarray(cfg.az_channels, dtype=int)
    proc_cfg.el_channels = np.asarray(cfg.el_channels, dtype=int)
    proc_cfg.el_coherent_ants = np.asarray(cfg.el_coherent_ants, dtype=int)
    proc_cfg.el_ref_column = int(cfg.el_ref_column)

    proc_cfg.max_targets = int(cfg.max_targets)
    proc_cfg.max_targets_to_dbf = cfg.max_targets_to_dbf
    proc_cfg.min_valid_range_bin = int(cfg.min_valid_range_bin)
    proc_cfg.max_azimuth_peaks_per_rd = int(cfg.max_azimuth_peaks_per_rd)
    proc_cfg.min_azimuth_peak_ratio = float(cfg.min_azimuth_peak_ratio)
    proc_cfg.min_azimuth_peak_separation_deg = float(cfg.min_azimuth_peak_separation_deg)
    proc_cfg.dbf_az_snr_gate_enable = bool(cfg.dbf_az_snr_gate_enable)
    proc_cfg.dbf_az_snr_threshold = float(cfg.dbf_az_snr_threshold)
    proc_cfg.angle_estimator_mode = str(cfg.angle_estimator_mode)
    proc_cfg.expert_dbf_mode = str(cfg.expert_dbf_mode)

    cfar_cfg = lky.CfarConfig(
        mode=cfg.cfar_mode,
        search_size=cfg.cfar_search_size,
        guard_size=cfg.cfar_guard_size,
        mul_fac=cfg.cfar_mul_fac,
        thres_div=cfg.cfar_thres_div,
        peak_det_enable=cfg.cfar_peak_det_enable,
    )

    expert_cfar_cfg = lky.CfarConfig(
        mode=cfg.expert_cfar_mode if cfg.expert_cfar_mode is not None else cfg.cfar_mode,
        search_size=cfg.expert_cfar_search_size if cfg.expert_cfar_search_size is not None else cfg.cfar_search_size,
        guard_size=cfg.expert_cfar_guard_size if cfg.expert_cfar_guard_size is not None else cfg.cfar_guard_size,
        mul_fac=cfg.expert_cfar_mul_fac if cfg.expert_cfar_mul_fac is not None else cfg.cfar_mul_fac,
        thres_div=cfg.expert_cfar_thres_div if cfg.expert_cfar_thres_div is not None else cfg.cfar_thres_div,
        peak_det_enable=(
            cfg.expert_cfar_peak_det_enable
            if cfg.expert_cfar_peak_det_enable is not None
            else cfg.cfar_peak_det_enable
        ),
    )

    array_cfg = lky.build_array_geometry(lky.ArrayConfig(), proc_cfg)
    return common_cfg, proc_cfg, cfar_cfg, expert_cfar_cfg, array_cfg


def apply_mti_to_range_fft_cube(
    fft_cube: np.ndarray,
    mode: MtiMode = "mean",
    alpha: float = 0.95,
) -> np.ndarray:
    """Apply optional MTI/clutter cancellation over chirp dimension.

    Args:
        fft_cube: Range-FFT cube, shape [range, chirp, virtual_ant].
        mode:
            "none": no cancellation.
            "mean": subtract mean over chirps; this matches original lky.py.
            "two_pulse": x[n] - x[n-1].
            "three_pulse": x[n] - 2x[n-1] + x[n-2].
            "iir": first-order high-pass y[n] = alpha*y[n-1] + x[n] - x[n-1].
        alpha: IIR memory coefficient for mode="iir".

    Returns:
        doppler_input, same shape as fft_cube.
    """
    x = np.asarray(fft_cube)
    if x.ndim != 3:
        raise ValueError(f"fft_cube must have shape [range, chirp, ant], got {x.shape}")
    if mode not in ("none", "mean", "two_pulse", "three_pulse", "iir"):
        raise ValueError("mti must be one of: none, mean, two_pulse, three_pulse, iir")

    if mode == "none":
        return x.copy()

    if mode == "mean":
        return x - np.mean(x, axis=1, keepdims=True)

    y = np.zeros_like(x)
    if mode == "two_pulse":
        y[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        return y

    if mode == "three_pulse":
        y[:, 2:, :] = x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]
        return y

    # mode == "iir"
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"mti_alpha should be in [0, 1), got {alpha}")
    for k in range(1, x.shape[1]):
        y[:, k, :] = alpha * y[:, k - 1, :] + x[:, k, :] - x[:, k - 1, :]
    return y


def point_table_to_output(
    point_table: Dict[str, np.ndarray],
    output_format: OutputFormat = "xyz",
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Convert lky point-table dict to requested user-facing format."""
    if output_format == "table":
        return point_table
    if output_format == "xyz":
        return lky.point_table_to_xyz(point_table)
    if output_format == "full":
        return lky.point_table_to_array(point_table)
    raise ValueError("output_format must be one of: xyz, full, table")


def process_fft_cube_to_point_cloud(
    fft_cube: np.ndarray,
    cfg: Optional[RadarPointCloudConfig] = None,
    *,
    return_radar_data: bool = False,
) -> RadarPointCloudOutput:
    """Convert decoded range-FFT cube to point cloud.

    Args:
        fft_cube: Decoded USB 1D-FFT cube [range, chirp, virtual_ant].
        cfg: RadarPointCloudConfig.
        return_radar_data: If True, include RD maps, masks, thresholds, etc.

    Returns:
        RadarPointCloudOutput.  For cfg.branch="both", point_cloud is a dict
        containing both "original" and "dbf_first" arrays/tables.
    """
    if cfg is None:
        cfg = RadarPointCloudConfig()

    _, proc_cfg, cfar_cfg, expert_cfar_cfg, array_cfg = make_lky_configs(cfg)

    doppler_input = apply_mti_to_range_fft_cube(fft_cube, mode=cfg.mti, alpha=cfg.mti_alpha)
    rd_cube = lky.firmware_tdm_doppler_fft(doppler_input, proc_cfg)

    # Original branch: noncoherent RD accumulation -> CFAR -> DBF -> XYZ.
    rd_amp_map = np.sum(np.abs(rd_cube), axis=2)
    cfar_mask, threshold_map, score_map, base_map = lky.firmware_velocity_cfar(rd_amp_map, cfar_cfg)
    cfar_mask = lky.apply_detection_validity_mask(cfar_mask, proc_cfg)
    original_target_table = lky.build_target_table(
        rd_cube, rd_amp_map, threshold_map, cfar_mask, proc_cfg, array_cfg
    )
    original_pc_table = lky.point_table_from_target(original_target_table, cfg.frame_idx)

    # DBF-first / expert branch: azimuth DBF RD statistic -> CFAR -> elevation -> XYZ.
    (
        dbf_first_amp_map,
        dbf_first_az_map,
        dbf_first_el_map,
        dbf_first_az_peak_map,
        dbf_first_el_peak_map,
        dbf_first_az_snr_map,
        dbf_first_el_snr_map,
    ) = lky.build_spatial_dbf_first_rd_map(rd_cube, proc_cfg, array_cfg)

    dbf_first_cfar_mask, dbf_first_threshold_map, dbf_first_score_map, dbf_first_base_map = lky.firmware_velocity_cfar(
        dbf_first_amp_map, expert_cfar_cfg
    )
    dbf_first_cfar_mask = lky.apply_detection_validity_mask(dbf_first_cfar_mask, proc_cfg)

    dbf_first_target_table = lky.build_spatial_dbf_first_target_table(
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
    dbf_first_pc_table = lky.point_table_from_target(dbf_first_target_table, cfg.frame_idx)

    table_map: Dict[str, Dict[str, np.ndarray]] = {
        "original": original_pc_table,
        "dbf_first": dbf_first_pc_table,
    }

    if cfg.branch == "both":
        point_table: Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]] = table_map
        point_cloud: Union[np.ndarray, Dict[str, np.ndarray]] = {
            name: point_table_to_output(table, cfg.output_format)  # type: ignore[dict-item]
            for name, table in table_map.items()
        }
    elif cfg.branch == "original":
        point_table = original_pc_table
        point_cloud = point_table_to_output(original_pc_table, cfg.output_format)
    elif cfg.branch == "dbf_first":
        point_table = dbf_first_pc_table
        point_cloud = point_table_to_output(dbf_first_pc_table, cfg.output_format)
    else:
        raise ValueError("branch must be one of: dbf_first, original, both")

    radar_data = None
    if return_radar_data:
        radar_data = {
            "fftCube": fft_cube,
            "dopplerInput": doppler_input,
            "rdCube": rd_cube,
            "rdAmpMap": rd_amp_map,
            "cfarMask": cfar_mask,
            "thresholdMap": threshold_map,
            "scoreMap": score_map,
            "baseMap": base_map,
            "dbfFirstAmpMap": dbf_first_amp_map,
            "dbfFirstAzMap": dbf_first_az_map,
            "dbfFirstElMap": dbf_first_el_map,
            "dbfFirstAzPeakMap": dbf_first_az_peak_map,
            "dbfFirstElPeakMap": dbf_first_el_peak_map,
            "dbfFirstAzSnrMap": dbf_first_az_snr_map,
            "dbfFirstElSnrMap": dbf_first_el_snr_map,
            "dbfFirstCfarMask": dbf_first_cfar_mask,
            "dbfFirstThresholdMap": dbf_first_threshold_map,
            "dbfFirstScoreMap": dbf_first_score_map,
            "dbfFirstBaseMap": dbf_first_base_map,
        }

    return RadarPointCloudOutput(
        point_cloud=point_cloud,
        point_table=point_table,
        radar_data=radar_data,
        config=cfg,
    )


def generate_point_cloud_from_bin_structured(
    bin_path: str,
    cfg: Optional[RadarPointCloudConfig] = None,
    *,
    return_radar_data: bool = False,
) -> RadarPointCloudOutput:
    """Main structured API: bin path -> point-cloud output object."""
    if cfg is None:
        cfg = RadarPointCloudConfig()
    common_cfg, _, _, _, _ = make_lky_configs(cfg)
    fft_cube = lky.read_usb_fft_frame(bin_path, common_cfg)
    return process_fft_cube_to_point_cloud(fft_cube, cfg, return_radar_data=return_radar_data)


def generate_point_cloud_array_from_bin(
    bin_path: str,
    *,
    branch: BranchMode = "dbf_first",
    output_format: OutputFormat = "xyz",
    mti: MtiMode = "none",
    cfar_mul_fac: float = 10.0,
    expert_cfar_mul_fac: Optional[float] = None,
    min_valid_range_bin: int = 8,
    max_targets: int = 10000,
    dbf_az_snr_gate_enable: bool = True,
    dbf_az_snr_threshold: float = 10.0,
    frame_idx: int = 1,
    return_radar_data: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, Dict[str, np.ndarray], RadarPointCloudOutput]:
    """Convenience API: `xxx/xxx/xx.bin` -> point-cloud ndarray.

    Args:
        bin_path: USB 1D-FFT pseudo-float .bin path.
        branch: "dbf_first", "original", or "both".
        output_format:
            "xyz"  -> [N,3] X,Y,Z.
            "full" -> [N,8] X,Y,Z,RangeM,DopplerBin,Amplitude,AzimuthDeg,ElevationDeg.
            "table" -> dict columns.
        mti: "mean" matches original lky.py; "none" disables MTI.
        cfar_mul_fac: CFAR threshold multiplier for original branch.
        expert_cfar_mul_fac: CFAR threshold multiplier for DBF-first branch;
            if None, uses cfar_mul_fac.
        kwargs: Any other RadarPointCloudConfig field override, e.g.
            cfar_guard_size=12, max_azimuth_peaks_per_rd=2,
            az_channels=np.array([...]), el_coherent_ants=np.array([...]).

    Returns:
        By default ndarray. If return_radar_data=True, returns RadarPointCloudOutput.
    """
    cfg = RadarPointCloudConfig(
        branch=branch,
        output_format=output_format,
        mti=mti,
        cfar_mul_fac=cfar_mul_fac,
        expert_cfar_mul_fac=expert_cfar_mul_fac,
        min_valid_range_bin=min_valid_range_bin,
        max_targets=max_targets,
        dbf_az_snr_gate_enable=dbf_az_snr_gate_enable,
        dbf_az_snr_threshold=dbf_az_snr_threshold,
        frame_idx=frame_idx,
    )
    for key, value in kwargs.items():
        if not hasattr(cfg, key):
            raise TypeError(f"Unknown RadarPointCloudConfig field: {key}")
        setattr(cfg, key, value)

    out = generate_point_cloud_from_bin_structured(bin_path, cfg, return_radar_data=return_radar_data)
    if return_radar_data:
        return out
    return out.point_cloud


def save_point_cloud_array(path: str, pc: Union[np.ndarray, Dict[str, np.ndarray]]) -> None:
    """Save point cloud as .npy/.npz depending on output type and suffix."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if isinstance(pc, dict):
        np.savez(path, **pc)
    else:
        np.save(path, pc)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate radar point-cloud ndarray from one lky-style USB FFT bin")
    parser.add_argument("bin_path")
    parser.add_argument("--branch", default="dbf_first", choices=["dbf_first", "original", "both"])
    parser.add_argument("--output-format", default="xyz", choices=["xyz", "full"])
    parser.add_argument("--mti", default="none", choices=["none", "mean", "two_pulse", "three_pulse", "iir"])
    parser.add_argument("--cfar-mul", type=float, default=10.0)
    parser.add_argument("--expert-cfar-mul", type=float, default=None)
    parser.add_argument("--out", default="", help="Optional .npy or .npz save path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pc = generate_point_cloud_array_from_bin(
        args.bin_path,
        branch=args.branch,
        output_format=args.output_format,
        mti=args.mti,
        cfar_mul_fac=args.cfar_mul,
        expert_cfar_mul_fac=args.expert_cfar_mul,
    )

    if isinstance(pc, dict):
        for name, arr in pc.items():
            print(f"{name}: shape={arr.shape}")
    else:
        print(f"point_cloud: shape={pc.shape}")
        if pc.size:
            print(pc)

    if args.out:
        save_point_cloud_array(args.out, pc)
        print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
