import os
from pathlib import Path
from typing import *
from datetime import datetime, timedelta

def timestamp_to_ms(ts_str):
    parts = ts_str.split('_')
    seconds = int(parts[0])
    nanoseconds = int(parts[1])
    return (seconds * 1000) + (nanoseconds // 1_000_000)

def files_to_time_list(files: List) -> List:
    """
    将 aaa_bbb.ccc 文件转化为时间 list, 前提条件 aaa, bbb 分别为 unix时间戳的 s, ns
    输入:
      files: List[str]，文件名或路径列表，shape=(N,)。
    中间变量:
      base: str，单个文件名去后缀后的时间戳字符串，shape 为标量字符串。
      sec/ns: int，Unix 秒和纳秒，shape 均为标量。
      unix_ts: float，秒级 Unix 时间戳，shape 为标量。
    输出:
      times: List[datetime]，datetime 对象列表，shape=(N,)。
    """
    times = []
    for file in files:
        base = file.split('\\')[-1].split('.')[0]
        sec_str, ns_str = base.split('_')
        sec = int(sec_str)
        ns = int(ns_str)
        unix_ts = sec + ns * 1e-9
        times.append(unix_to_datetime(unix_ts))
    return times

def unix_to_datetime(unix_ts: float) -> datetime:
    """
    将 Unix 浮点时间戳转换为本地 datetime 对象。
    unix_ts: 例如 1719999999.123456789 这种秒级+纳秒的小数
    输入:
      unix_ts: float，Unix 时间戳，单位为秒，shape 为标量。
    输出:
      datetime，Python datetime 对象，shape 为标量对象。
    """
    return datetime.fromtimestamp(unix_ts)

def find_nearest_file(
    lidar_dir: Optional[str],
    radar_pc_dir: Optional[str],
    radar_bin_dir: Optional[str],
    max_delta_sec: Optional[float] = None,
    one_to_one: bool = True
)-> Tuple[List[Optional[str]], List[Optional[str]], List[Optional[str]]]:
    """
    按文件名时间戳对齐激光雷达、雷达点云和雷达 bin 文件。
    返回:
      lidar_list, radar_pc_list, radar_bin_list
    其中每个 list 元素是完整路径或 None，且三者长度一致。

    对齐策略：
      选一个“基准源”(优先 lidar -> bin -> pc) 作为时间轴；
      对每个基准时间，去其它源找最近文件；
      若 max_delta_sec 指定，超过阈值的匹配会被置为 None；
      若 one_to_one=True，同一源的同一文件最多被匹配一次（按遍历顺序贪心）。
    输入:
      lidar_dir: Optional[str]，激光雷达 .pcd 目录或 None，shape 为标量路径。
      radar_pc_dir: Optional[str]，雷达点云 .npy 目录或 None，shape 为标量路径。
      radar_bin_dir: Optional[str]，雷达 .bin 目录或 None，shape 为标量路径。
      max_delta_sec: Optional[float]，最大允许时间差，单位秒，shape 为标量。
      one_to_one: bool，是否限制一对一匹配，shape 为标量。
    中间变量:
      *_files: List[str]，对应目录下文件名列表，shape=(N,)。
      *_times: List[datetime]，与文件名同序的时间列表，shape=(N,)。
      used_*: Optional[set[int]]，已匹配索引集合。
    输出:
      Tuple[List[Optional[str]], List[Optional[str]], List[Optional[str]]]；
      三个列表分别为 lidar/radar_pc/radar_bin 的完整路径或 None，shape 均为 (N_base,)。
    """

    def _list_files(dir_path: Optional[str], suffix: str) -> Tuple[List[str], List[datetime]]:
        """
        列出指定目录中匹配后缀的文件并解析时间戳。
        返回 (files, times)；若 dir_path 为 None 则返回空列表
        files 仅文件名，不含路径；times 与 files 同序
        输入:
          dir_path: Optional[str]，目录路径或 None，shape 为标量路径。
          suffix: str，文件后缀，例如 ".pcd"/".npy"/".bin"，shape 为标量字符串。
        输出:
          files: List[str]，排序后的文件名列表，shape=(N,)。
          times: List[datetime]，与 files 同序的时间列表，shape=(N,)。
        """
        if not dir_path:
            return [], []
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(suffix)]
        # 为了可重复性，建议按名字排序（名字是时间戳）
        files.sort()
        times = files_to_time_list(files)
        return files, times

    def _abs_seconds(dt1: datetime, dt2: datetime) -> float:
        """
        计算两个 datetime 之间的绝对秒差。
        输入:
          dt1: datetime，第一个时间点，shape 为标量对象。
          dt2: datetime，第二个时间点，shape 为标量对象。
        输出:
          float，两个时间点的绝对时间差，单位秒，shape 为标量。
        """
        return abs((dt1 - dt2).total_seconds())

    def _find_nearest_index(
            target_time: datetime,
            times: List[datetime],
            used: Optional[set] = None,
            max_delta_sec: Optional[float] = None
    ) -> Optional[int]:
        """
        在候选时间列表中查找距离目标时间最近且满足约束的索引。
        在 times 中找与 target_time 最近的 index。
        used: 已占用 index 集合（one_to_one 时用）
        max_delta_sec: 超过则返回 None
        输入:
          target_time: datetime，目标时间，shape 为标量对象。
          times: List[datetime]，候选时间列表，shape=(N,)。
          used: Optional[set[int]]，已占用索引集合或 None，shape 最大为 (N,)。
          max_delta_sec: Optional[float]，最大允许秒差或 None，shape 为标量。
        输出:
          Optional[int]，最近文件索引，shape 为标量；无可用匹配时返回 None。
        """
        if not times:
            return None

        best_i = None
        best_d = None
        for i, t in enumerate(times):
            if used is not None and i in used:
                continue
            d = _abs_seconds(target_time, t)
            if best_d is None or d < best_d:
                best_d = d
                best_i = i

        if best_i is None:
            return None
        if max_delta_sec is not None and best_d is not None and best_d > max_delta_sec:
            return None
        return best_i
    # 读三类文件
    lidar_files, lidar_times = _list_files(lidar_dir, ".pcd")
    pc_files, pc_times       = _list_files(radar_pc_dir, ".npy")
    bin_files, bin_times     = _list_files(radar_bin_dir, ".bin")

    # 选基准时间轴（必须有至少一个源非空）
    if lidar_times:
        base_name = "lidar"
        base_times = lidar_times
        base_files = lidar_files
    elif bin_times:
        base_name = "bin"
        base_times = bin_times
        base_files = bin_files
    elif pc_times:
        base_name = "pc"
        base_times = pc_times
        base_files = pc_files
    else:
        # 全 None 或全空目录
        return [], [], []

    # one_to_one: 记录已使用的 index
    used_lidar = set() if one_to_one else None
    used_pc    = set() if one_to_one else None
    used_bin   = set() if one_to_one else None

    # 输出（长度一致）
    out_lidar: List[Optional[str]] = []
    out_pc: List[Optional[str]] = []
    out_bin: List[Optional[str]] = []

    for bi, t in enumerate(base_times):
        # 基准帧本身
        lidar_path = None
        pc_path = None
        bin_path = None

        if base_name == "lidar":
            lidar_path = os.path.join(lidar_dir, base_files[bi]) if lidar_dir else None
            if used_lidar is not None:
                used_lidar.add(bi)
        elif base_name == "pc":
            pc_path = os.path.join(radar_pc_dir, base_files[bi]) if radar_pc_dir else None
            if used_pc is not None:
                used_pc.add(bi)
        else:  # base_name == "bin"
            bin_path = os.path.join(radar_bin_dir, base_files[bi]) if radar_bin_dir else None
            if used_bin is not None:
                used_bin.add(bi)

        # 给其它源找最近
        if lidar_path is None and lidar_times:
            li = _find_nearest_index(t, lidar_times, used_lidar, max_delta_sec)
            if li is not None and lidar_dir:
                lidar_path = os.path.join(lidar_dir, lidar_files[li])
                if used_lidar is not None:
                    used_lidar.add(li)

        if pc_path is None and pc_times:
            pi = _find_nearest_index(t, pc_times, used_pc, max_delta_sec)
            if pi is not None and radar_pc_dir:
                pc_path = os.path.join(radar_pc_dir, pc_files[pi])
                if used_pc is not None:
                    used_pc.add(pi)

        if bin_path is None and bin_times:
            bi2 = _find_nearest_index(t, bin_times, used_bin, max_delta_sec)
            if bi2 is not None and radar_bin_dir:
                bin_path = os.path.join(radar_bin_dir, bin_files[bi2])
                if used_bin is not None:
                    used_bin.add(bi2)

        out_lidar.append(lidar_path)
        out_pc.append(pc_path)
        out_bin.append(bin_path)

    return out_lidar, out_pc, out_bin


def align_multi_sensor_files(
    sources: Dict[str, Optional[Path]],
    max_delta_sec: Optional[float] = None,
    one_to_one: bool = True,
    base_source: Optional[str] = None,
    suffix_map: Optional[Dict[str, str]] = None,
    time_offsets_sec: Optional[Dict[str, float]] = None
) -> Dict[str, List[Optional[str]]]:
    """
    多传感器文件时间戳对齐

    Args:
        sources: 传感器字典，格式为 {'传感器名称': '目录路径', ...}
                 路径为None的传感器会被忽略
        max_delta_sec: 最大允许时间差（秒），超过则匹配置为None
        one_to_one: 是否一对一匹配（每个文件最多被匹配一次）
        base_source: 基准传感器名称，None则自动选择第一个非空传感器
        suffix_map: 自定义文件后缀映射，如 {'lidar': '.pcd', 'gt': '.txt'}
                    默认规则：名称含'pc'用.npy，含'bin'用.bin，含'pcd'用.pcd

    Returns:
        字典，键为传感器名称，值为对应的文件路径列表（所有列表长度相同）

    Example:
        sources = {
            'lidar': '/data/lidar',
            'radar1_pc': '/data/radar1/pc',
            'radar1_bin': '/data/radar1/bin',
            'radar2_pc': '/data/radar2/pc',
            'radar2_bin': '/data/radar2/bin',
            'gt': '/data/ground_truth',
            'realsense_pc': '/data/realsense'
        }
        result = align_multi_sensor_files(sources, max_delta_sec=0.1)
    """

    # 默认后缀映射规则
    def get_default_suffix(name: str) -> str:
        if 'pc' in name.lower():
            return '.npy'
        if 'bin' in name.lower():
            return '.bin'
        if 'pcd' in name.lower():
            return '.pcd'
        return ''

    def list_files(dir_path: Optional[str], suffix: str) -> Tuple[List[str], List[datetime]]:
        """列出目录中匹配后缀的文件并解析时间戳"""
        if not dir_path or not suffix:
            return [], []
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(suffix)]
        files.sort()
        # 假设 files_to_time_list 函数已存在
        times = files_to_time_list(files)
        return files, times

    def time_diff(dt1: datetime, dt2: datetime) -> float:
        return abs((dt1 - dt2).total_seconds())

    def find_nearest(target_time: datetime, times: List[datetime],
                     used: Optional[set] = None) -> Optional[int]:
        if not times:
            return None
        best_i, best_d = None, None
        for i, t in enumerate(times):
            if used is not None and i in used:
                continue
            d = time_diff(target_time, t)
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        if best_i is not None and max_delta_sec is not None and best_d > max_delta_sec:
            return None
        return best_i

    # 过滤掉路径为None的传感器
    sources = {k: v for k, v in sources.items() if v is not None}
    if not sources:
        return {}

    # 读取所有传感器的文件列表和时间戳
    sensor_data = {}
    for name, path in sources.items():
        suffix = suffix_map.get(name, get_default_suffix(name)) if suffix_map else get_default_suffix(name)
        files, times = list_files(path, suffix)
        if time_offsets_sec and name in time_offsets_sec:
            offset = timedelta(seconds=float(time_offsets_sec[name]))
            times = [t + offset for t in times]
        if files:  # 只保留非空的传感器
            sensor_data[name] = {
                'path': path,
                'files': files,
                'times': times
            }

    if not sensor_data:
        return {name: [] for name in sources.keys()}

    # 选择基准传感器
    if base_source is None or base_source not in sensor_data:
        base_source = list(sensor_data.keys())[0]

    base_times = sensor_data[base_source]['times']
    base_files = sensor_data[base_source]['files']

    # 已使用索引记录
    used_indices = {name: (set() if one_to_one else None) for name in sensor_data.keys()}

    # 对齐结果
    result = {name: [] for name in sources.keys()}

    for bi, base_time in enumerate(base_times):
        for name, data in sensor_data.items():
            if name == base_source:
                # 基准传感器直接使用当前文件
                file_path = os.path.join(data['path'], base_files[bi])
                result[name].append(file_path)
                if used_indices[name] is not None:
                    used_indices[name].add(bi)
            else:
                # 其他传感器查找最近匹配
                idx = find_nearest(base_time, data['times'], used_indices[name])
                if idx is not None:
                    file_path = os.path.join(data['path'], data['files'][idx])
                    result[name].append(file_path)
                    if used_indices[name] is not None:
                        used_indices[name].add(idx)
                else:
                    result[name].append(None)

    # 为没有数据的传感器填充None
    for name in sources.keys():
        if name not in result:
            result[name] = [None] * len(base_times)

    return result
