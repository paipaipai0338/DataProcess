# [[0,  0,  0,  0,  4,  3,  2,  1],
#  [0,  0,  0,  0,  8,  7,  6,  5],
# [16, 15, 14, 13, 12, 11, 10, 9]]
import os
from dataclasses import dataclass
from pathlib import Path
from typing import *

import numpy as np


@dataclass  # dataclass 可理解为结构体类，加入装饰器(@dataclass)可省略  __init__, __repr__, __eq__ 等必要方法
class Radar_Config:
    # 基础参数
    fs: int = 10_000_000  # 10 MHz，采样率
    chirp_time: float = 80e-6  # 80 μs，脉冲宽度
    B_set: float = 6.4453e9  # 6.4453 GHz，带宽
    time_B: float = 55e-6  # 55 μs，B段扫频时长
    c: float = 3e8  # 光速 m/s
    fc: float = 60e9  # 60 GHz，载频
    d_azi: float = 2170e-6  # 2170 μm，方位向孔径
    d_ele: float = 2400e-6  # 2400 μm，俯仰向孔径
    Tx: int = 4  # 发射通道数
    Rx: int = 4  # 接收通道数
    num_samp: int = 512  # 预设距离维采样点数量
    num_chirp: int = 64  # 预设一帧中chirp数量

    # 依赖其他参数的属性（使用 field(init=False)）
    slope: float = None  # 调频率
    lam: float = None  # 波长
    prf: float = None  # 脉冲重复频率

    def __post_init__(self) -> None:
        """
        根据基础雷达配置自动计算派生参数。
        输入:
          self: Radar_Config，包含 fs/chirp_time/B_set/time_B/c/fc/Tx 等标量配置。
        输出:
          None；原地更新 self.slope/self.lam/self.prf，类型均为 float，shape 均为标量。
        """
        self.slope = self.B_set / self.time_B
        self.lam = self.c / self.fc
        self.prf = 1 / (self.chirp_time * self.Tx)


def bin_to_cube(file_name: str, radar_config: Radar_Config) -> Optional[np.ndarray]:
    """
    读取原始数据 raw_data.bin 转换为 adc_data: (num_samp, num_chirp, num_ant)

    file_name: str bin 文件路径
    输入:
      file_name: str，bin 文件路径，shape 为标量字符串。
      radar_config: Radar_Config，雷达参数对象，内部 num_samp/num_chirp/Tx/Rx 为 int 标量。
    中间变量:
      raw: np.ndarray，dtype=uint8，shape=(expected_bytes,)。
      raw8: np.ndarray，dtype=uint8，shape=(expected_bytes/8, 8)。
      words_u16: np.ndarray，dtype=uint16，shape=(num_samp*num_chirp*num_ant,)。
      frame_data: np.ndarray，dtype=int16，shape=(num_samp*num_chirp*num_ant,)。
      B: np.ndarray，dtype=int16，shape=(num_samp, num_ant, num_chirp)。
    输出:
      adc_data: np.ndarray，dtype=int16，shape=(num_samp, num_chirp, num_ant)；数据不足时返回 None。
    """
    num_byte = 2
    num_samp = radar_config.num_samp
    num_chirp = radar_config.num_chirp
    num_ant = radar_config.Tx * radar_config.Rx
    expected_bytes = num_samp * num_chirp * num_ant * num_byte

    raw = np.fromfile(file_name, dtype=np.uint8)

    if raw.size < expected_bytes:
        print(f"[WARN] {os.path.basename(file_name)} 数据长度不足，跳过。"
              f"({raw.size} < {expected_bytes})")
        return None

    # 截断到一帧
    raw = raw[:expected_bytes]

    # 每8字节组成4个int16
    # (7,8)->1, (5,6)->2, (3,4)->3, (1,2)->4
    raw8 = raw.reshape(-1, 8)

    # 用 uint16 做组合然后转成有符号
    words_u16 = np.empty(raw8.shape[0] * 4, dtype=np.uint16)
    # 第1个 16bit
    words_u16[0::4] = (raw8[:, 6].astype(np.uint16) << 8) | raw8[:, 7].astype(np.uint16)
    # 第2个 16bit
    words_u16[1::4] = (raw8[:, 4].astype(np.uint16) << 8) | raw8[:, 5].astype(np.uint16)
    # 第3个 16bit
    words_u16[2::4] = (raw8[:, 2].astype(np.uint16) << 8) | raw8[:, 3].astype(np.uint16)
    # 第4个 16bit
    words_u16[3::4] = (raw8[:, 0].astype(np.uint16) << 8) | raw8[:, 1].astype(np.uint16)

    frame_data = words_u16.astype(np.int16)

    # B: (num_samp, num_ant, num_chirp)
    B = frame_data.reshape((num_samp, num_ant, num_chirp), order="F")
    # adc_data: (num_samp, num_chirp, num_ant)
    adc_data = np.transpose(B, (0, 2, 1))
    return adc_data

def bin_to_cube_range_fft(file_name: Path|str, radar_config: Radar_Config) -> Optional[np.ndarray]:
    """
    读取保存的 1D-FFT 数据 raw_data.bin 转换为 adc_data_range_FFT: (num_samp, num_chirp, num_ant)

    file_name: str bin 文件路径
    输入:
      file_name: str，bin 文件路径，shape 为标量字符串。
      radar_config: Radar_Config，雷达参数对象，内部 num_samp/num_chirp/Tx/Rx 为 int 标量。
    中间变量:
      raw: np.ndarray，dtype=uint8，shape=(expected_bytes,)。
      raw8: np.ndarray，dtype=uint8，shape=(expected_bytes,)。
      pf_u32: np.ndarray，dtype=uint32，shape=(use_range*num_chirp*num_ant,)。
      vec_cplx: np.ndarray，dtype=complex64，shape=(use_range*num_chirp*num_ant,)。
      mcu_timing: np.ndarray，dtype=complex64，shape=(use_range, num_ant, num_chirp)。
    输出:
      adc_data_range_FFT: np.ndarray，dtype=complex64，shape=(use_range, num_chirp, num_ant)；文件大小不匹配时返回 None。
    """

    def _pseudo_float_cplx_to_complex(pf_u32: np.ndarray) -> np.ndarray:
        """
        MATLAB pseudoFloatCplx2FloatCplx 的 numpy 复刻
        输入: uint32 的伪浮点复数 (exp[31:28] + imag[27:14] + real[13:0])
        输出: complex64
        输入:
          pf_u32: np.ndarray，dtype=uint32，shape=(N,)。
        中间变量:
          exp/real/imag: np.ndarray，dtype=int32，shape=(N,)。
          scale: np.ndarray，dtype=float32，shape=(N,)。
          out: np.ndarray，dtype=complex64，shape=(N,)。
        输出:
          np.ndarray，dtype=complex64，shape=(N,)。
        """
        pf = pf_u32.astype(np.uint32)

        exp = (pf >> 28).astype(np.int32)  # 4-bit exponent
        real = (pf & 0x3FFF).astype(np.int32)  # 14-bit signed
        imag = ((pf >> 14) & 0x3FFF).astype(np.int32)  # 14-bit signed

        # two's complement on 14-bit
        real[real >= (1 << 13)] -= (1 << 14)
        imag[imag >= (1 << 13)] -= (1 << 14)

        scale = np.power(2.0, exp - 13).astype(np.float32)
        out = (real.astype(np.float32) + 1j * imag.astype(np.float32)) * scale
        return out.astype(np.complex64)

    num_samp = radar_config.num_samp
    num_chirp = radar_config.num_chirp
    num_ant = radar_config.Tx * radar_config.Rx
    use_range = num_samp // 2
    expected_bytes = use_range * num_chirp * num_ant * 4  # uint32

    raw = np.fromfile(file_name, dtype=np.uint8)
    if raw.size != expected_bytes:
        print(f"[WARN] {os.path.basename(file_name)} size mismatch: "
              f"{raw.size} != {expected_bytes}, skip.")
        return None

    # MATLAB pfloat_ospi64: 每8字节翻转（temp(end:-1:1)）
    raw8 = raw.reshape(-1, 8)[:, ::-1].reshape(-1)

    # MATLAB typecast(uint8(...), 'uint32')，在小端机器上等价于 <u4
    pf_u32 = np.frombuffer(raw8.tobytes(), dtype="<u4")

    # MATLAB dataType='ReImPf'
    vec_cplx = _pseudo_float_cplx_to_complex(pf_u32)  # length = use_range*num_ant*num_chirp

    mcu_timing = vec_cplx.reshape((use_range, num_ant, num_chirp), order="F")

    # adc_data_range_FFT: (num_samp, num_chirp, num_ant)
    adc_data_range_FFT = np.transpose(mcu_timing, (0, 2, 1))

    return adc_data_range_FFT

def two_dimension_cfar(
    data: np.ndarray,
    ref_range: int,
    ref_velocity: int,
    guard_range: int,
    guard_velocity: int,
    alpha: float,
    mode: str = 'ca',
    os_rank: Optional[int] = None,
) -> np.ndarray:
    """
    2D CFAR on a range-velocity map.

    Axis convention:
      data.shape == (num_range_bins, num_velocity_bins)
      axis 0 / first index  -> range
      axis 1 / second index -> velocity

    Args:
      ref_range/ref_velocity: half-size of reference window on range/velocity axes.
      guard_range/guard_velocity: half-size of guard window on range/velocity axes.
      alpha: threshold multiplier.
      mode: 'ca', 'so', 'go', or 'os'.
      os_rank: rank used by OS-CFAR.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise RuntimeError("data must be a 2D numpy array with shape (range, velocity)")
    if ref_range <= guard_range or ref_velocity <= guard_velocity:
        raise RuntimeError("reference window must be larger than guard window on both range and velocity axes")
    if mode not in ['ca', 'so', 'go', 'os']:
        raise RuntimeError("mode must be one of 'ca'/'so'/'go'/'os'")
    if mode == 'os' and (os_rank is None or os_rank <= 0):
        raise RuntimeError("OS-CFAR requires a positive os_rank")

    num_range, num_velocity = data.shape
    mask = np.zeros_like(data, dtype=bool)

    for range_idx in range(ref_range, num_range - ref_range):
        for velocity_idx in range(ref_velocity, num_velocity - ref_velocity):
            window = data[
                range_idx - ref_range:range_idx + ref_range + 1,
                velocity_idx - ref_velocity:velocity_idx + ref_velocity + 1,
            ]

            guard_mask = np.zeros_like(window, dtype=bool)
            guard_start_range = ref_range - guard_range
            guard_end_range = ref_range + guard_range + 1
            guard_start_velocity = ref_velocity - guard_velocity
            guard_end_velocity = ref_velocity + guard_velocity + 1
            guard_mask[
                guard_start_range:guard_end_range,
                guard_start_velocity:guard_end_velocity,
            ] = True

            background = window[~guard_mask]
            if len(background) == 0:
                continue

            if mode == 'ca':
                bg_power = np.mean(background)
            elif mode == 'so':
                sorted_bg = np.sort(background)[::-1]
                bg_power = sorted_bg[1] if len(sorted_bg) >= 2 else sorted_bg[0]
            elif mode == 'go':
                k = len(background) // 2
                sorted_bg = np.sort(background)[::-1]
                bg_power = sorted_bg[k] if k < len(sorted_bg) else sorted_bg[-1]
            elif mode == 'os':
                rank = min(os_rank, len(background) - 1)
                sorted_bg = np.sort(background)
                bg_power = np.mean(sorted_bg[:rank])

            if data[range_idx, velocity_idx] > bg_power * alpha:
                mask[range_idx, velocity_idx] = True

    return mask

def range_fft(data: np.ndarray, radar_config: Radar_Config, remove_mean_fasttime: bool = False, window: bool = True, n_fft_range: int = 512, mti: str = 'none') -> Tuple[np.ndarray, np.ndarray]:
    """
    对 ADC 数据沿快时间维做距离 FFT，并生成距离轴。
    输入:
      data: np.ndarray，ADC 数据，shape=(num_samp, num_chirp, num_ant)。
      radar_config: Radar_Config，雷达参数对象，fs/c/slope 为 float 标量。
      remove_mean_fasttime: bool，是否沿采样维去均值，shape 为标量。
      window: bool，是否使用 Hann 窗，shape 为标量。
      n_fft_range: int，距离 FFT 点数，shape 为标量。
      mti: str，慢时间动目标抑制方式，取值 'none'/'mean'/'two_pulse'/'three_pulse'/'iir'，shape 为标量字符串。
    中间变量:
      w_r: np.ndarray，dtype=float32，距离窗，shape=(num_samp,)。
      rng_fft: np.ndarray，dtype=complex，距离谱，shape=(n_fft_range/2, num_chirp, num_ant)。
      f_b: np.ndarray，dtype=float，拍频轴，shape=(n_fft_range/2,)。
    输出:
      rng_fft: np.ndarray，dtype=complex，距离 FFT 结果，shape=(n_fft_range/2, num_chirp, num_ant)。
      r_axis: np.ndarray，dtype=float，距离轴，单位 m，shape=(n_fft_range/2,)。
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise RuntimeError("data must have shape (num_samp, num_chirp, num_ant)")
    if n_fft_range <= 0:
        raise RuntimeError(f"n_fft_range 必须大于 0，当前值为 {n_fft_range}")
    num_samp, num_chirp, num_ant = data.shape
    # --- optional mean removal on fast-time (helps suppress DC/leakage) ---
    if remove_mean_fasttime:
        data = data - data.mean(axis=0, keepdims=True)
    # --- Range FFT over samples (fast-time) ---
    if window:
        w_r = np.hanning(num_samp).astype(np.float32)
        data_r = data * w_r[:, None, None]
    else:
        data_r = data
    rng_fft = np.fft.fft(data_r, n=n_fft_range, axis=0)  # (n_fft_range, num_chirp, num_ant)
    half = n_fft_range // 2
    rng_fft = rng_fft[:half, :, :]

    Nr = rng_fft.shape[0]
    k = np.arange(Nr)
    f_b = k / n_fft_range
    r_axis = (radar_config.c * f_b * radar_config.fs) / (2.0 * radar_config.slope)
    # --- MTI on slow-time (chirp) axis=1, applied to range-domain data ---
    # Note: MTI will suppress static/near-zero-doppler components.
    if mti not in ("none", "mean", "two_pulse", "three_pulse", "iir"):
        raise RuntimeError(f"Unknown mti='{mti}'. Use 'none'|'mean'|'two_pulse'|'three_pulse'|'iir'.")

    if mti != "none":
        x = rng_fft  # (Nr, num_chirp, num_ant), complex
        if mti == "mean":
            # simplest clutter removal: subtract mean across chirps
            x = x - x.mean(axis=1, keepdims=True)
        elif mti == "two_pulse":
            # y[n] = x[n] - x[n-1]
            y = np.zeros_like(x)
            y[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
            x = y
        elif mti == "three_pulse":
            # y[n] = x[n] - 2x[n-1] + x[n-2]
            y = np.zeros_like(x)
            y[:, 2:, :] = x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]
            x = y
        rng_fft = x
    return rng_fft, r_axis[:half]

# def doppler_fft(
#     data: np.ndarray,
#     radar_config: Radar_Config,
#     window: bool = True,
#     n_fft_doppler: int = 1024,
#     doppler_mode: Literal["normal", "firmware_tdm"] = "firmware_tdm",
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     doppler_mode:
#       "normal":
#         普通 Doppler FFT。输出 shape=(range, n_fft_doppler, ant)。
#
#       "firmware_tdm":
#         Firmware 风格 TDM Doppler FFT。
#         假设 4Tx TDM-MIMO: Tx0, Tx1, Tx2, Tx3, Tx0...
#         会把每个 TX 的 64 个 chirp 插回 256 点 TDM 时间轴，
#         做 256 点 FFT，再按固件卸载顺序取 64 个 Doppler bin。
#         输出 shape=(range, num_chirp, ant)。
#         此模式下 n_fft_doppler 不再控制输出点数。
#     """
#     data = np.asarray(data)
#     if data.ndim != 3:
#         raise RuntimeError("data must have shape (num_samp_or_range, num_chirp, num_ant)")
#     if n_fft_doppler <= 0:
#         raise RuntimeError(f"n_fft_doppler 必须大于 0，当前值为 {n_fft_doppler}")
#
#     num_samp, num_chirp, num_ant = data.shape
#
#     if doppler_mode == "normal":
#         if window:
#             w_d = np.hanning(num_chirp).astype(np.float32)
#             dop_in = data * w_d[None, :, None]
#         else:
#             dop_in = data
#
#         dop_fft = np.fft.fft(dop_in, n=n_fft_doppler, axis=1)
#         dop_fft = np.fft.fftshift(dop_fft, axes=1)
#
#         Nd = n_fft_doppler
#         k = np.arange(Nd) - Nd // 2
#         f_d = k / Nd * radar_config.prf
#         v_axis = (radar_config.lam / 2.0) * f_d
#         return dop_fft, v_axis
#
#     if doppler_mode != "firmware_tdm":
#         raise RuntimeError(f"未知 doppler_mode={doppler_mode}")
#
#     # -------- Firmware 风格 TDM Doppler FFT --------
#     num_tx = int(getattr(radar_config, "num_tx", getattr(radar_config, "Tx", 4)))
#     num_rx = int(getattr(radar_config, "num_rx", getattr(radar_config, "Rx", num_ant // num_tx)))
#
#     if num_tx <= 0 or num_rx <= 0:
#         raise RuntimeError(f"num_tx/num_rx 非法: num_tx={num_tx}, num_rx={num_rx}")
#
#     if num_ant != num_tx * num_rx:
#         raise RuntimeError(
#             f"firmware_tdm 要求 num_ant == num_tx * num_rx，"
#             f"当前 num_ant={num_ant}, num_tx={num_tx}, num_rx={num_rx}"
#         )
#
#     # 例如 64 chirps * 4 Tx = 256 点固件 Doppler FFT
#     firmware_fft_size = num_chirp * num_tx
#
#     # 固件 unload 顺序：
#     # 对 256 点 FFT，取 224..255, 0..31，共 64 个 bin。
#     use_a = (num_tx - 1) * num_chirp + num_chirp // 2
#     use_b = num_chirp // 2 - 1
#     unload_bins = np.r_[use_a:firmware_fft_size, 0:use_b + 1]
#
#     dop_fft = np.zeros((num_samp, num_chirp, num_ant), dtype=np.complex128)
#
#     if window:
#         # 这里用 Hann 近似固件速度窗。
#         # 如果要和 lky.py / C 固件完全一致，可以把 C_WINVEL_PRE 表搬进来替换这里。
#         full_win = np.hanning(firmware_fft_size).astype(np.float32)
#         tx_windows = np.stack([
#             full_win[tx_idx:firmware_fft_size:num_tx][:num_chirp]
#             for tx_idx in range(num_tx)
#         ], axis=0)
#     else:
#         tx_windows = np.ones((num_tx, num_chirp), dtype=np.float32)
#
#     for ant_idx in range(num_ant):
#         tx_idx = ant_idx // num_rx
#
#         # 该虚拟天线属于某个 TX，把它的 64 个 chirp 插回真实 TDM 时序槽
#         slot_idx = tx_idx + np.arange(num_chirp) * num_tx
#
#         tdm_input = np.zeros((num_samp, firmware_fft_size), dtype=np.complex128)
#         tdm_input[:, slot_idx] = data[:, :, ant_idx] * tx_windows[tx_idx][None, :]
#
#         fft_out = np.fft.fft(tdm_input, n=firmware_fft_size, axis=1)
#         dop_fft[:, :, ant_idx] = fft_out[:, unload_bins]
#
#     # 固件输出的 64 个 bin 对应 [-32, ..., 31]
#     k = np.arange(-num_chirp // 2, num_chirp // 2)
#
#     # 优先用 chirp_gap/chirp_time 算，和 lky.py 一致：
#     # v_res = lam / (2 * num_chirp * num_tx * chirp_gap)
#     chirp_gap = getattr(radar_config, "chirp_gap", getattr(radar_config, "chirp_time", None))
#     if chirp_gap is not None:
#         f_d = k / firmware_fft_size / float(chirp_gap)
#     else:
#         # 如果 radar_config.prf 已经是每个 TX 的有效 PRF，即 1/(chirp_time*num_tx)，这个等价。
#         f_d = k / num_chirp * radar_config.prf
#
#     v_axis = (radar_config.lam / 2.0) * f_d
#     return dop_fft, v_axis

def doppler_fft(
    data: np.ndarray,
    radar_config: Radar_Config,
    window: bool = True,
    n_fft_doppler: int = 1024,
    doppler_mode: Literal["normal", "firmware_tdm"] = "firmware_tdm",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    doppler_mode:
      "normal":
        普通 Doppler FFT。输出 shape=(range, n_fft_doppler, ant)。

      "firmware_tdm":
        Firmware 风格 TDM Doppler FFT。
        假设 4Tx TDM-MIMO: Tx0, Tx1, Tx2, Tx3, Tx0...
        会把每个 TX 的 64 个 chirp 插回 256 点 TDM 时间轴，
        做 256 点 FFT，再按固件卸载顺序取 64 个 Doppler bin。
        输出 shape=(range, num_chirp, ant)。
        此模式下 n_fft_doppler 不再控制输出点数。
    """
    data = np.asarray(data)

    if data.ndim != 3:
        raise RuntimeError(
            "data must have shape "
            "(num_samp_or_range, num_chirp, num_ant), "
            f"actual shape={data.shape}"
        )

    if n_fft_doppler <= 0:
        raise RuntimeError(
            f"n_fft_doppler 必须大于 0，当前值为 "
            f"{n_fft_doppler}"
        )

    num_samp, num_chirp, num_ant = data.shape

    # ============================================================
    # 均值对消分支
    #
    # 对每个 range bin、每根虚拟天线，
    # 沿慢时间 chirp 维减去复数 IQ 均值。
    #
    # data:       [R, C, A]
    # mean:       [R, 1, A]
    # data_clean: [R, C, A]
    # ============================================================
    data_clean = (
        data
        - np.mean(
            data,
            axis=1,
            keepdims=True,
        )
    )

    # ============================================================
    # 普通 Doppler FFT
    # ============================================================
    if doppler_mode == "normal":
        if window:
            w_d = np.hanning(
                num_chirp
            ).astype(np.float32)

            dop_in = (
                data
                * w_d[None, :, None]
            )

            dop_in_clean = (
                data_clean
                * w_d[None, :, None]
            )
        else:
            dop_in = data
            dop_in_clean = data_clean

        # 原始分支
        dop_fft = np.fft.fft(
            dop_in,
            n=n_fft_doppler,
            axis=1,
        )

        dop_fft = np.fft.fftshift(
            dop_fft,
            axes=1,
        )

        # 均值对消分支
        dop_fft_clean = np.fft.fft(
            dop_in_clean,
            n=n_fft_doppler,
            axis=1,
        )

        dop_fft_clean = np.fft.fftshift(
            dop_fft_clean,
            axes=1,
        )

        Nd = n_fft_doppler

        k = (
            np.arange(Nd)
            - Nd // 2
        )

        f_d = (
            k
            / Nd
            * radar_config.prf
        )

        v_axis = (
            radar_config.lam
            / 2.0
        ) * f_d

        return (
            dop_fft,
            dop_fft_clean,
            v_axis,
        )

    if doppler_mode != "firmware_tdm":
        raise RuntimeError(
            f"未知 doppler_mode={doppler_mode}"
        )

    # ============================================================
    # Firmware 风格 TDM Doppler FFT
    # ============================================================
    num_tx = int(
        getattr(
            radar_config,
            "num_tx",
            getattr(
                radar_config,
                "Tx",
                4,
            ),
        )
    )

    num_rx = int(
        getattr(
            radar_config,
            "num_rx",
            getattr(
                radar_config,
                "Rx",
                num_ant // num_tx,
            ),
        )
    )

    if num_tx <= 0 or num_rx <= 0:
        raise RuntimeError(
            f"num_tx/num_rx 非法: "
            f"num_tx={num_tx}, "
            f"num_rx={num_rx}"
        )

    if num_ant != num_tx * num_rx:
        raise RuntimeError(
            "firmware_tdm 要求 "
            "num_ant == num_tx * num_rx，"
            f"当前 num_ant={num_ant}, "
            f"num_tx={num_tx}, "
            f"num_rx={num_rx}"
        )

    firmware_fft_size = (
        num_chirp
        * num_tx
    )

    # 固件 unload 顺序：
    # 例如 256 点 FFT，取 224..255 和 0..31。
    use_a = (
        (num_tx - 1) * num_chirp
        + num_chirp // 2
    )

    use_b = (
        num_chirp // 2
        - 1
    )

    unload_bins = np.r_[
        use_a:firmware_fft_size,
        0:use_b + 1,
    ]

    # 原始分支和均值对消分支
    dop_fft = np.zeros(
        (
            num_samp,
            num_chirp,
            num_ant,
        ),
        dtype=np.complex128,
    )

    dop_fft_clean = np.zeros(
        (
            num_samp,
            num_chirp,
            num_ant,
        ),
        dtype=np.complex128,
    )

    if window:
        full_win = np.hanning(
            firmware_fft_size
        ).astype(np.float32)

        tx_windows = np.stack(
            [
                full_win[
                    tx_idx:
                    firmware_fft_size:
                    num_tx
                ][:num_chirp]
                for tx_idx in range(num_tx)
            ],
            axis=0,
        )
    else:
        tx_windows = np.ones(
            (
                num_tx,
                num_chirp,
            ),
            dtype=np.float32,
        )

    for ant_idx in range(num_ant):
        tx_idx = (
            ant_idx
            // num_rx
        )

        # 当前虚拟天线对应的真实 TDM 时间槽
        slot_idx = (
            tx_idx
            + np.arange(num_chirp)
            * num_tx
        )

        # --------------------------------------------------------
        # 原始数据分支
        # --------------------------------------------------------
        tdm_input = np.zeros(
            (
                num_samp,
                firmware_fft_size,
            ),
            dtype=np.complex128,
        )

        tdm_input[:, slot_idx] = (
            data[:, :, ant_idx]
            * tx_windows[
                tx_idx
            ][None, :]
        )

        fft_out = np.fft.fft(
            tdm_input,
            n=firmware_fft_size,
            axis=1,
        )

        dop_fft[
            :,
            :,
            ant_idx,
        ] = fft_out[:, unload_bins]

        # --------------------------------------------------------
        # 均值对消分支
        # --------------------------------------------------------
        tdm_input_clean = np.zeros(
            (
                num_samp,
                firmware_fft_size,
            ),
            dtype=np.complex128,
        )

        tdm_input_clean[:, slot_idx] = (
            data_clean[:, :, ant_idx]
            * tx_windows[
                tx_idx
            ][None, :]
        )

        fft_out_clean = np.fft.fft(
            tdm_input_clean,
            n=firmware_fft_size,
            axis=1,
        )

        dop_fft_clean[
            :,
            :,
            ant_idx,
        ] = fft_out_clean[
            :,
            unload_bins,
        ]

    # 固件输出的 num_chirp 个 bin：
    # [-num_chirp/2, ..., num_chirp/2 - 1]
    k = np.arange(
        -num_chirp // 2,
        num_chirp // 2,
    )

    chirp_gap = getattr(
        radar_config,
        "chirp_gap",
        getattr(
            radar_config,
            "chirp_time",
            None,
        ),
    )

    if chirp_gap is not None:
        f_d = (
            k
            / firmware_fft_size
            / float(chirp_gap)
        )
    else:
        f_d = (
            k
            / num_chirp
            * radar_config.prf
        )

    v_axis = (
        radar_config.lam
        / 2.0
    ) * f_d

    return (
        dop_fft,
        dop_fft_clean,
        v_axis,
    )

def angle_fft(data: np.ndarray, radar_config: Radar_Config, window: bool = True, n_fft_angle: int = 1024, target_index: List = None, channel_index: List = None, type: str = 'ele', method: str = 'fft') -> Tuple[np.ndarray, np.ndarray]:
    """
    对指定距离-多普勒单元和通道做角度谱估计。
    输入:
      data: np.ndarray，距离-多普勒-通道数据，shape=(Nr, Nd, Nch)。
      radar_config: Radar_Config，雷达参数对象，lam/d_ele/d_azi 为 float 标量。
      window: bool，FFT 方法下是否使用 Hann 窗，shape 为标量。
      n_fft_angle: int，角度网格/FFT 点数，shape 为标量。
      target_index: List[int]，目标 [range_index, doppler_index]，shape=(2,)。
      channel_index: List[int]，参与角度估计的通道索引，shape=(M,)。
      type: str，角度类型，'ele' 表示俯仰，'azi' 表示方位，shape 为标量字符串。
      method: str，角度估计方法，'fft' 或 'MVDR'，shape 为标量字符串。
    中间变量:
      snap_data: np.ndarray，快拍数据，FFT 时 shape=(1, M)，MVDR 时 shape=(S, M)。
      sin_theta: np.ndarray，角度正弦网格，shape=(n_fft_angle,)。
      A: np.ndarray，dtype=complex，MVDR 导向矢量矩阵，shape=(M, n_fft_angle)。
      R/Rinv: np.ndarray，dtype=complex，协方差矩阵及逆矩阵，shape=(M, M)。
    输出:
      ang_result: np.ndarray，角度谱，FFT 时 dtype=complex，MVDR 时 dtype=float，shape=(n_fft_angle,)。
      az_axis: np.ndarray，dtype=float，角度轴，单位 rad，shape=(n_fft_angle,)。
    """

    # --- 1. 参数与配置 ---
    if type == 'ele':
        d = radar_config.d_ele
    elif type == 'azi':
        d = radar_config.d_azi
    else:
        raise RuntimeError(f"Unknown type: '{type}'. Use 'ele' or 'azi'.")

    data = np.asarray(data)
    if data.ndim != 3:
        raise RuntimeError("data must have shape (Nr, Nd, Nch)")
    if n_fft_angle <= 0:
        raise RuntimeError(f"n_fft_angle 必须大于 0，当前值为 {n_fft_angle}")
    if len(target_index) != 2:
        raise RuntimeError("target_index 必须为 [range_index, doppler_index]")
    if not channel_index:
        raise RuntimeError("channel_index 不能为空")
    if method not in ('fft', 'MVDR'):
        raise RuntimeError("method 必须为 'fft' 或 'MVDR'")
    Nr, Nd, Nch = data.shape
    if target_index[0] < 0 or target_index[0] >= Nr or target_index[1] < 0 or target_index[1] >= Nd:
        raise RuntimeError(f"target_index 越界，当前值为 {target_index}，data shape 为 {data.shape}")
    if min(channel_index) < 0 or max(channel_index) >= Nch:
        raise RuntimeError(f"channel_index 越界，当前值为 {channel_index}，通道数为 {Nch}")
    # data shape假设: [Range, Doppler, Antenna] 或类似
    # 确保拿到正确维度

    num_ant = len(channel_index)

    # --- 2. 数据切片与快拍选取 ---
    if method == 'fft':
        # FFT 模式：通常只取单快拍，或者多快拍相干累加（视具体需求）
        # 这里保持你原有的逻辑：单点切片
        # Shape: (1, num_ant)
        snap_data = data[target_index[0], target_index[1], channel_index].reshape(1, -1)
    else:  # MVDR
        # MVDR 模式：需要多个快拍来估计协方差矩阵 R
        K = 64
        Nd = data.shape[1]
        # 获取快拍索引
        snap_indices = np.arange(target_index[1] - K, target_index[1] + K + 1)
        snap_indices = np.clip(snap_indices, 0, Nd - 1)
        snap_indices = np.unique(snap_indices)

        # 取数据 Shape: (S, num_ant)
        snap_data = data[target_index[0], snap_indices, :]
        snap_data = snap_data[:, channel_index]

    # --- 3. 角度网格生成 (通用) ---
    # 使用 linspace 生成均匀的角度正弦空间 [-1, 1]，比 fftfreq 更直观用于 MVDR
    if method == 'MVDR':
        # MVDR 通常不需要像 FFT 那样凑 2 的幂次，但也可用 n_fft_angle 控制精度
        sin_theta = np.linspace(-1, 1, n_fft_angle)
        az_axis = np.arcsin(sin_theta)  # 注意：这里 sin_theta 为 1/-1 时可能产生极小误差，arcsin 没问题
    else:
        # FFT 模式保持原逻辑，与频率对齐
        u = np.fft.fftshift(np.fft.fftfreq(n_fft_angle, d=1.0))
        sin_theta = (u * radar_config.lam) / d
        # 裁剪以防数值误差导致 arcsin nan
        mask = np.abs(sin_theta) <= 1.0
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        az_axis = np.arcsin(sin_theta)

    # --- 4. 核心算法 ---

    if method == 'fft':
        # === FFT 流程 ===
        # 仅在 FFT 模式下加窗
        process_data = snap_data
        if window:
            w_a = np.hanning(num_ant).astype(process_data.dtype)
            # 广播乘法 (1, M) * (M,)
            process_data = process_data * w_a

        # Zero padding
        # axis=1 是天线维度
        ang_spec = np.fft.fft(process_data, n=n_fft_angle, axis=1)
        ang_spec = np.fft.fftshift(ang_spec, axes=1)

        # 如果是单快拍，去掉第一维
        ang_result = ang_spec[0]

    elif method == 'MVDR':
        # === MVDR 流程 ===
        # 1. 绝对不要加窗 (Keep Raw Data)
        X = snap_data  # Shape (S, M)
        S, M = X.shape

        # 2. 计算协方差矩阵 R = E[x x^H]
        # X.T 是 (M, S), X.conj() 是 (S, M)
        # R 应该是 (M, M)
        # 正确公式: R = (X.conj().T @ X) / S
        R = (X.conj().T @ X) / S

        # 3. 对角加载 (Diagonal Loading)
        # 增强鲁棒性，防止 R 不可逆
        tr = np.trace(R).real
        dl_factor = 1e-3  # 加载因子，可调
        R = R + (dl_factor * tr / M) * np.eye(M, dtype=R.dtype)

        # 4. 求逆
        Rinv = np.linalg.inv(R)

        # 5. 构建导向矢量矩阵 A
        # A shape: (M, n_fft_angle)
        # 假设天线是均匀线阵 (ULA)，索引 0..M-1
        m = np.arange(M).reshape(-1, 1)  # (M, 1)

        # Steering Vector: a(theta) = exp(-j * 2pi * d/lam * m * sin(theta))
        # 注意：这里 sin_theta 使用上面生成的网格
        phase = -2.0j * np.pi * (d / radar_config.lam) * m * sin_theta.reshape(1, -1)
        A = np.exp(phase)

        # 6. 计算 MVDR 空间谱
        # P = 1 / (a^H * Rinv * a)
        # 分母计算技巧：
        # Rinv @ A -> (M, N)
        # conj(A) * (Rinv @ A) -> 逐元素乘
        # sum(..., axis=0) -> 对天线维求和

        den = np.sum(np.conj(A) * (Rinv @ A), axis=0).real
        ang_result = 1.0 / np.maximum(den, 1e-12)  # 避免除以0

        # 如果之前为了 sin_theta 做了 mask (仅针对 FFT 频率超范围情况)，这里其实不需要
        # 因为 linspace(-1, 1) 保证了都在范围内

    return ang_result, az_axis

def mvdr_spectrum_from_snapshots(
    snapshots: np.ndarray,
    radar_config: Radar_Config,
    type: str = 'ele',
    n_fft_angle: int = 1024,
    diagonal_loading: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ???????? MVDR ??????

    snapshots.shape == (num_snapshots, num_ant)??????????
    ????????????????????????
    """
    if type == 'ele':
        d = radar_config.d_ele
    elif type == 'azi':
        d = radar_config.d_azi
    else:
        raise RuntimeError(f"Unknown type: '{type}'. Use 'ele' or 'azi'.")

    X = np.asarray(snapshots, dtype=np.complex128)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise RuntimeError("snapshots must have shape (num_snapshots, num_ant)")
    if n_fft_angle <= 0:
        raise RuntimeError(f"n_fft_angle ???? 0????? {n_fft_angle}")

    num_snapshots, num_ant = X.shape
    if num_snapshots <= 0 or num_ant < 2:
        raise RuntimeError(f"snapshots shape ??: {X.shape}")

    sin_theta = np.linspace(-1.0, 1.0, n_fft_angle)
    angle_axis = np.arcsin(sin_theta)

    R = (X.conj().T @ X) / float(num_snapshots)
    tr = np.trace(R).real
    if tr <= 0:
        tr = 1.0
    R = R + (diagonal_loading * tr / num_ant) * np.eye(num_ant, dtype=R.dtype)
    Rinv = np.linalg.inv(R)

    m = np.arange(num_ant, dtype=float).reshape(-1, 1)
    phase = -2.0j * np.pi * (d / radar_config.lam) * m * sin_theta.reshape(1, -1)
    A = np.exp(phase)

    den = np.sum(np.conj(A) * (Rinv @ A), axis=0).real
    spectrum = 1.0 / np.maximum(den, 1e-12)
    return spectrum, angle_axis

def elevation_mvdr_with_azimuth_compensation(
    rd_cube: np.ndarray,
    radar_config: Radar_Config,
    range_idx: int,
    velocity_idx: int,
    azimuth_rad: float,
    n_fft_angle: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ??????????? 4 ?????????????????

    ??????? lky.py ? el_coherent_ants ???
      rows = [[3,2,1,0], [7,6,5,4], [11,10,9,8]]

    ??? [3,7,11] / [2,6,10] / [1,5,9] / [0,4,8]
    ???? 3 ????????? azimuth_rad ??????????
    ?? 4 ???????? 3 ?????????? MVDR?
    """
    rd_cube = np.asarray(rd_cube)
    if rd_cube.ndim != 3:
        raise RuntimeError("rd_cube must have shape (range, velocity, ant)")

    row_groups = np.array(
        [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]],
        dtype=int,
    )
    if np.max(row_groups) >= rd_cube.shape[2]:
        raise RuntimeError(f"rd_cube only has {rd_cube.shape[2]} channels")

    # shape: (3 elevation rows, 4 azimuth columns)
    row_data = rd_cube[range_idx, velocity_idx, row_groups]

    # The columns are ordered [3,2,1,0], matching the azimuth-reversed channel order.
    col_idx = np.arange(row_groups.shape[1], dtype=float)
    az_comp = np.exp(
        1.0j
        * 2.0
        * np.pi
        * (radar_config.d_azi / radar_config.lam)
        * col_idx
        * np.sin(azimuth_rad)
    )

    compensated_columns = (row_data * az_comp.reshape(1, -1)).T  # shape: (4, 3)
    coherent_ele_snapshot = np.sum(compensated_columns, axis=0)  # shape: (3,)

    ele_spectrum, ele_axis = mvdr_spectrum_from_snapshots(
        coherent_ele_snapshot,
        radar_config,
        type='ele',
        n_fft_angle=n_fft_angle,
    )
    return ele_spectrum, ele_axis, coherent_ele_snapshot

def get_radar_res(
    radar_config: Radar_Config,
    doppler_mode: Literal["normal", "firmware_tdm"] = "firmware_tdm",
    azi_num_ant: int = 8,
    ele_num_ant: int = 3,
    aperture_mode: Literal["effective", "physical"] = "effective",
) -> Tuple[float, float, float, float]:
    """
    返回:
      range_res: 距离分辨率, m
      velocity_res: 速度分辨率, m/s
      azi_angle_res_deg: 方位理论角分辨率, deg
      ele_angle_res_deg: 俯仰理论角分辨率, deg
    """

    def _array_angle_res_deg(
            lam: float,
            d: float,
            num_ant: int,
            theta_deg: float = 0.0,
            aperture_mode: Literal["effective", "physical"] = "effective",
    ) -> float:
        """
        阵列孔径理论角分辨率，单位 deg。

        aperture_mode:
          "effective":
            使用常见雷达角分辨率公式 A = N * d。
            对应 delta_sin ≈ λ / (N*d)。

          "physical":
            使用物理孔径 A = (N-1) * d。
            更保守一些。
        """
        if num_ant < 2:
            raise RuntimeError(f"num_ant 必须 >= 2，当前为 {num_ant}")
        if d <= 0:
            raise RuntimeError(f"阵元间距 d 必须 > 0，当前为 {d}")

        if aperture_mode == "effective":
            aperture = num_ant * d
        elif aperture_mode == "physical":
            aperture = (num_ant - 1) * d
        else:
            raise RuntimeError(f"未知 aperture_mode={aperture_mode}")

        theta = np.deg2rad(theta_deg)

        # 在 sin(theta) 空间的分辨率
        delta_sin = lam / aperture

        # theta=0° 附近 cos(theta)=1；这里保留一般角度写法
        delta_theta_rad = delta_sin / max(np.cos(theta), 1e-12)

        return float(np.rad2deg(delta_theta_rad))

    # 距离分辨率
    range_res = radar_config.c * radar_config.fs / (
        2.0 * radar_config.slope * radar_config.num_samp
    )

    # 速度分辨率
    if doppler_mode == "firmware_tdm":
        velocity_res = radar_config.lam / (
            2.0 * radar_config.num_chirp * radar_config.Tx * radar_config.chirp_time
        )
    else:
        velocity_res = radar_config.lam * radar_config.prf / (
            2.0 * radar_config.num_chirp
        )

    # 阵列孔径理论角分辨率，0° 附近
    azi_angle_res_deg = _array_angle_res_deg(
        lam=radar_config.lam,
        d=radar_config.d_azi,
        num_ant=azi_num_ant,
        theta_deg=0.0,
        aperture_mode=aperture_mode,
    )

    ele_angle_res_deg = _array_angle_res_deg(
        lam=radar_config.lam,
        d=radar_config.d_ele,
        num_ant=ele_num_ant,
        theta_deg=0.0,
        aperture_mode=aperture_mode,
    )

    return range_res, velocity_res, azi_angle_res_deg, ele_angle_res_deg

def get_corner_data(file_path: Path|str, ref_range: int=6, ref_velocity: int=4, guard_range:int=4, guard_velocity: int=2, alpha: float=2.0, mode: str='so') -> Dict[str, np.ndarray]:
    radar_config = Radar_Config()
    # (256, 64, 16)
    data = bin_to_cube_range_fft(str(file_path), radar_config)  # 读取函数
    range_res, _, _, _ = get_radar_res(radar_config)
    r_axis = np.arange(data.shape[0], dtype=float) * range_res

    # 固件风格 TDM Doppler FFT
    rd_cube, v_axis = doppler_fft(
        data,
        radar_config,
        window=True,
        doppler_mode="firmware_tdm",
    )
    rd_map_abs_sum = np.sum(np.abs(rd_cube), axis=-1)
    cfar_params = dict(
        ref_range=ref_range,
        ref_velocity=ref_velocity,
        guard_range=guard_range,
        guard_velocity=guard_velocity,
        alpha=alpha,
        mode=mode,
    )
    # normal下参数
    # cfar_params = dict(
    #     ref_range=10,
    #     ref_velocity=40,
    #     guard_range=3,
    #     guard_velocity=5,
    #     alpha=10.0,
    #     mode='ca',
    # )
    mask = two_dimension_cfar(data=rd_map_abs_sum, **cfar_params)
    range_indices, velocity_indices = np.where(mask)
    targets_indices = (np.vstack((range_indices, velocity_indices))).T
    # print(f"CFAR params: {cfar_params}, detections={int(mask.sum())}")

    targets = {
        "polar coordinate": [],  # 极坐标
        "cartesian coordinate": []  # 笛卡尔坐标
    }

    for idx, target in enumerate(targets_indices):
        range_idx, velocity_idx = target
        r, v = r_axis[range_idx], v_axis[velocity_idx]

        channels_azi = [8, 9, 10, 11, 12, 13, 14, 15]
        az_spectrum, az_axis = angle_fft(
            rd_cube,
            radar_config,
            target_index=[range_idx, velocity_idx],
            channel_index=channels_azi,
            type='azi',
            method='MVDR',
        )
        az_spectrum = np.abs(az_spectrum)
        az_spectrum = az_spectrum / max(np.max(az_spectrum), np.finfo(float).eps)
        az_peak_idx = int(np.argmax(az_spectrum))
        azimuth_rad = float(az_axis[az_peak_idx])
        azimuth_deg = float(np.rad2deg(azimuth_rad))

        ele_spectrum, ele_axis, ele_snapshot = elevation_mvdr_with_azimuth_compensation(
            rd_cube,
            radar_config,
            range_idx=range_idx,
            velocity_idx=velocity_idx,
            azimuth_rad=azimuth_rad,
            n_fft_angle=1024,
        )
        ele_spectrum = np.abs(ele_spectrum)
        ele_spectrum = ele_spectrum / max(np.max(ele_spectrum), np.finfo(float).eps)
        ele_peak_idx = int(np.argmax(ele_spectrum))
        elevation_deg = float(np.rad2deg(ele_axis[ele_peak_idx]))

        targets["polar coordinate"].append([r, v, azimuth_deg, elevation_deg])

        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)

        # Radar coordinate
        #   X: 前向
        #   Y: 左侧
        #   Z: 上方
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        targets["cartesian coordinate"].append([x, y, z, v])

        # print(
        #     f'Target {idx}: range={r:.3f} m, velocity={v:.3f} m/s, '
        #     f'azimuth={azimuth_deg:.2f} deg, elevation={elevation_deg:.2f} deg, '
        # )


    targets["cartesian coordinate"] = np.array(targets["cartesian coordinate"])
    targets["polar coordinate"] = np.array(targets["polar coordinate"])

    return targets

def get_pc_data(file_path: Path|str):
    data = np.load(file_path)
    return data