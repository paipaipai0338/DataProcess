# DataProcess

本项目用于多传感器数据处理，重点包括相机、毫米波雷达、激光雷达和 RealSense 数据的读取、时间对齐、标定、关键点提取、点云处理与可视化。

## 项目结构

```text
DataProcess/
├── Calib/              # 标定相关脚本，例如雷达与相机之间的外参估计
├── Img2Keypoint/       # 图像人体检测、2D/3D关键点处理与后处理
├── Img2Points/         # 图像点位、角点等辅助处理
├── LidarProcess/       # 激光雷达点云读取与处理
├── RadarProcess/       # 雷达原始数据读取、FFT、CFAR和点云生成
├── RealSenseProcess/   # RealSense深度数据读取与处理
├── TimeProcess/        # 多源传感器时间戳转换与数据对齐
└── Vis/                # 多传感器结果可视化
```

## 主要功能

- 雷达 `.bin` 原始数据解析与点云生成
- LiDAR `.pcd` 点云读取
- RealSense 深度数据读取
- 多传感器文件按时间戳对齐
- 相机图像中的人体检测与关键点提取
- 雷达、相机、点云之间的标定和坐标变换
- 多传感器点云结果可视化

## 环境依赖

建议使用 Python 3.8+。项目中常用依赖包括：

```bash
pip install numpy opencv-python open3d tqdm torch
```

如果需要运行图像关键点检测流程，还需要安装并配置：

```bash
pip install mmdet mmpose mmengine
```

同时需要准备对应的检测和姿态估计模型配置文件与权重文件。

## 使用示例

运行图像关键点提取：

```bash
python Img2Keypoint/pipeline.py --root_path E:\your_dataset\camera --device cuda:0
```

运行多传感器可视化：

```bash
python Vis/display_sensor_results.py
```

运行前请根据实际数据路径修改脚本中的 `root_path`、`calib_path` 等配置。

## 数据目录约定

项目脚本通常按传感器类型读取数据，常见目录包括：

```text
dataset_root/
├── camera/
├── calib/
├── robosense/
├── realsense/
├── dpct低位机/
└── dpct高位机/
```

具体目录名称和文件后缀可在对应脚本中调整，例如 `Vis/display_sensor_results.py` 中的 `sensors` 和 `suffix_map`。

## 输出结果

处理结果一般保存到输入数据目录下的 `results/` 子目录中，例如：

```text
camera/
└── results/
    └── 2D/
```

其中关键点、检测框等中间结果通常以 `.npz`、`.pkl` 等格式保存。

## 说明

本项目以数据处理为核心，代码按传感器和处理阶段拆分。使用时建议先确认数据路径、时间戳格式和标定文件，再运行对应模块。
