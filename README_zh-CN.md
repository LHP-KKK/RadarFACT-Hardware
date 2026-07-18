# RadarFACT-Hardware

[English](README.md)

面向 RadarFACT 研究的雷达—相机—LiDAR 硬件采集、标定、同步与数据准备仓库。

> [!IMPORTANT]
> 本项目的建立与推进得益于 GRT 团队扎实而系统的开源贡献。GRT 论文、项目网站、
> 研究代码、Red Rover 采集系统以及 NRDK 工具链，为本项目提供了关键基础。
> 请优先阅读、引用并支持原始的
> [GRT 项目主页](https://wiselabcmu.github.io/grt/)、
> [论文](https://arxiv.org/abs/2509.12482)、
> [研究代码](https://github.com/WiseLabCMU/grt)、
> [Red Rover](https://radarml.github.io/red-rover/) 和
> [NRDK](https://radarml.github.io/nrdk/)。

本仓库主要覆盖 RadarFACT 的**硬件与数据前端**，不包含完整的 RadarFACT 训练、
推理后端或预训练权重。仓库会持续更新，后续将逐步补充硬同步验证、里程计、
在线雷达频谱处理和 RadarFACT 端侧部署。

## 系统组成

- Jetson Orin Nano 级别的边缘计算平台；
- TI xWR18xx 雷达与 DCA1000 原始 I/Q 采集；
- Livox Mid-360 与其 IMU；
- 海康工业相机；
- STM32 时间同步与触发板；
- ROS 2 Humble、Red Rover 与 I/Q-1M-like 数据导出工具。

现有资料中同时出现 IWR1834Boost 与 AWR1843 类板卡描述。正式采集时必须在每条
trace 的元数据中记录实际板卡、固件、调制参数和天线配置，不能仅凭项目名推断。

## 目录

```text
docs/                       架构、数据流程、限制和上游致谢
firmware/stm32-timesync/    STM32F10x 同步固件
hardware/cad/printable/     STL 与 3MF 打印文件
hardware/cad/source/        SolidWorks 源文件
hardware/docs/              硬件清单、接线与同步说明
software/acquisition/       ROS 2 与 Red Rover 采集脚本
software/calibration/       ChArUco 标定与雷达可视化工具
software/iq1m_tools/        导出、时间匹配、投影和缓存工具
```

## 数据流程

```text
Radar I/Q trace + ROS 2 bag
        ↓
保留设备时间、主机时间和 Mid-360 每点 offset_time
        ↓
I/Q-1M-like 场景导出
        ↓
Radar–Camera–LiDAR 时间匹配
        ↓
标定投影与 Camera Region Cache
        ↓
RadarFACT / GRT 兼容输入
```

详见 [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)。

## 快速开始

脚本以 Ubuntu 22.04 和 ROS 2 Humble 为目标环境。运行前需要分别安装并验证：

- Livox-SDK2 与 `livox_ros_driver2`；
- 海康 MVS SDK 与 ROS 2 相机节点；
- Red Rover、雷达固件和 DCA1000 配置；
- NumPy、OpenCV、Matplotlib、PyYAML、ReportLab 等 Python 依赖。

三个传感器后端正常后，可用环境变量覆盖公开脚本中的示例路径：

```bash
export DATA_ROOT=/mnt/sensor_data/demo_traces
export IMAGE_TOPIC=/image_raw
export LIDAR_TOPIC=/livox/lidar
export IMU_TOPIC=/livox/imu

bash software/acquisition/record_indoor_demo.sh indoor_forward_01
```

## 使用前必须确认

- 示例相机内参只能用于对应的相机、镜头、焦距、分辨率和 ROI；
- `extrinsic_R/T`、`Rcl/Pcl` 等矩阵必须确认坐标系方向、单位和存储顺序；
- 同步板固件只是工程起点，必须用示波器验证电压、极性、脉宽、频率和相位；
- 图像、雷达和 LiDAR 的时间戳必须与实际采样/曝光时刻对应；
- 旧数据依赖最近邻时间匹配，不能当作已经完成严格硬同步；
- 当前仓库不提供原始采集数据、完整 RadarFACT 后端或模型权重。

## 后续计划

- 验证相机、雷达和 Mid-360 的硬同步；
- 建立 FAST-LIO2/FAST-LIVO2 基线并替换占位 pose；
- 增加可复现的雷达—相机和 LiDAR—相机标定样例；
- 将实时 RadarFrame 接入内存 4-D FFT；
- 增加 RadarFACT 在线推理、状态诊断和降级机制；
- 发布示例序列与量化评估工具。

## 致谢与贡献归属

本项目是在 GRT 开源全栈基础上的独立复现和扩展。GRT 团队针对单芯片毫米波雷达
建立的大规模原始数据采集方法、I/Q-1M 数据集、Red Rover、NRDK 和 GRT 模型，
构成了本项目最重要的技术基础。仓库中的改动主要围绕本地硬件适配、Mid-360、
海康相机、标定、同步、I/Q-1M-like 导出和未来 RadarFACT 实验展开。

本仓库不是 GRT 官方仓库，也不代表卡内基梅隆大学、Bosch Research、
威斯康星大学麦迪逊分校或 GRT 作者的官方立场。

仓库将持续更新。若本项目对你的研究有帮助，请首先引用原始 GRT 论文，引用格式见
[README.md](README.md#citation)。

