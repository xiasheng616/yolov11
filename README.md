# 计算机视觉实践：行人检测跟踪与 RGB-D 目标点云

基于 Ultralytics YOLO、ByteTrack 和 Open3D 的个人学习与实验仓库。目前包含两个方向：

- **行人检测与跟踪**：CrowdHuman 数据转换、YOLO11s 微调、ByteTrack 参数对比、TrackEval 评估，以及 ONNX 视频推理实践。
- **RGB-D 目标感知**：像素反投影、目标表面点定位、彩色点云生成、检测框区域提取、深度筛选和实例分割掩码提取。

本仓库保留了 Ultralytics 上游源码和目录。个人实验代码主要位于根目录下的训练、跟踪、评估及 `yolo-3d*.py` 脚本中；检测、跟踪、分割模型与第三方库归各自原项目所有。

## 1. 行人检测与多目标跟踪

### 实验内容

1. 将 CrowdHuman 标注转换为 YOLO 数据集格式。
2. 使用 YOLO11s 预训练权重，在 CrowdHuman 上进行微调。
3. 使用 ByteTrack 进行视频多目标跟踪，对比关联阈值与轨迹保留时间。
4. 将跟踪输出转换为 MOT 格式，使用 TrackEval 评估 HOTA、IDF1、MOTA、IDSW、FP 和 FN。
5. 实践 ONNX 视频推理。

### 核心文件

| 文件 | 用途 |
|---|---|
| [crowdhuman_to_yolo.py](crowdhuman_to_yolo.py) | CrowdHuman 标注转换 |
| [train_crowdhuman.py](train_crowdhuman.py) | YOLO11s 微调 |
| [track_crowdhuman.py](track_crowdhuman.py) | 视频读取与 ByteTrack 跟踪 |
| [evaluate_abc.py](evaluate_abc.py) | A/B/C 三个序列的本地评估流程 |
| [evaluate_c_night.py](evaluate_c_night.py) | C 序列本地评估流程 |

### ByteTrack 对比结果

以下为本地 A/B/C 三个序列的 TrackEval 合并结果。A、B、C 分别取自 MOT17-09-DPM、MOT17-02-DPM、MOT17-04-DPM 的训练序列；本地视频预处理与分辨率不同于官方原始序列，因此结果仅用于本项目内的受控对比，不是 MOTChallenge 官方排行榜成绩。

| 配置 | `match_thresh` | `track_buffer` | HOTA ↑ | IDF1 ↑ | MOTA ↑ | IDSW ↓ | FP ↓ | FN ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.8 | 30 | **46.847** | **59.797** | **54.652** | 339 | 6171 | 25897 |
| match07 | 0.7 | 30 | 44.010 | 53.828 | 53.737 | 646 | **4591** | 27824 |
| match09 | 0.9 | 30 | 46.567 | 59.154 | 54.642 | 341 | 6632 | **25441** |
| buffer60 | 0.8 | 60 | 46.728 | 59.621 | 54.632 | **333** | 6229 | 25859 |

在这组本地实验中，默认配置的 HOTA、IDF1、MOTA 综合表现更好。降低关联阈值虽然减少了 FP，但 ID 切换和漏检增加；延长轨迹保留时间仅小幅减少 IDSW，没有带来整体指标提升。结果不表示某组参数在所有场景下最优。

## 2. RGB-D 目标定位与彩色点云

### 数据与几何约定

使用 Open3D 提供的 TUM RGB-D 样例，输入为已对齐的彩色图与原始深度 PNG。

- 图像分辨率：`640 × 480`。
- 内参：`fx = fy = 525.0`，`cx = 319.5`，`cy = 239.5`。
- TUM 原始深度换算：`Z = depth_raw / 5000.0`，单位为米；`0` 表示无效深度。
- 相机坐标约定：X 向右、Y 向下、Z 向前。

反投影公式：

```text
Z = depth_raw / 5000.0
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

### 核心文件

| 文件 | 用途 |
|---|---|
| [yolo-3d1.py](yolo-3d1.py) | 对比检测框中心单像素深度与中心 5×5 邻域有效深度中位数，计算表面点 XYZ |
| [yolo-3d2.py](yolo-3d2.py) | 提取置信度最高椅子检测框中的彩色点云 |
| [yolo-3d3.py](yolo-3d3.py) | 按参考深度筛选区域点云，统计保留比例，并从固定相机视角显示不同点云 |
| [yolo-3d4.py](yolo-3d4.py) | 使用 YOLO11s-seg 掩码与有效深度联合提取椅子点云 |

### 已完成的验证

- 使用 NumPy 手动反投影整幅深度图，在相同内参、深度比例和截断范围下，与 Open3D 自动生成的点云逐点比较。
- 一次验证中，两边均为 **227933 个有效点**，坐标误差在打印精度下为 0，最大颜色误差为 0。
- 对比中心单像素与 5×5 邻域中位数，处理无效深度及邻域有效比例不足的情况。
- 对比检测框区域、深度范围筛选、分割掩码三种点云提取方式。

### 单帧样例记录

| 提取方式 | 有效点数 | 观察 |
|---|---:|---|
| 检测框区域 | 46390 | 椅子之外混入地面、桌面等背景 |
| 参考深度前后 0.3 米筛选 | 20443 | 背景减少，同时椅腿、底座等不同深度部位损失较多 |
| 实例分割掩码 | 25808 | 当前样例中背景较少，可见椅子结构保留相对完整 |

这些是当前单帧的定性观察，点数不代表分割准确率。检测与分割模型可能给出不同的目标框，严格比较时需统一实例、ROI、深度条件与相机视角。

输出点云只包含深度图能够观测到的有效表面点，不代表完整物体模型。中心/邻域反投影结果是表面位置估计，不是完整物体的真实质心或 6DoF 姿态。

## 3. 环境与运行

主要依赖：Python、PyTorch、Ultralytics、OpenCV、NumPy、Open3D；跟踪评估另需 TrackEval。

在已配置好匹配 PyTorch/CUDA 的环境中，从仓库根目录安装本地代码和基础依赖：

```bash
pip install -e .
pip install opencv-python numpy open3d
```

当前代码仍是实验脚本形式。运行前请检查：

1. `train_crowdhuman.py` 的 `ROOT` 默认指向原训练服务器路径，需改为当前机器的项目目录。
2. 跟踪、评估脚本中的数据、视频、GT、跟踪配置及输出路径，需按本地实际目录设置。
3. RGB-D 脚本中的 `data_root` 当前为 `D:\study\open3d_data`，需改为可用缓存目录。首次获取样例需要联网。
4. 将相应的 `yolo11s.pt`、`yolo11s-seg.pt` 放到脚本指定位置。当前脚本的权重路径尚未统一：前三个 RGB-D 脚本使用 `yolo11s.pt`，第四个使用 `weights/yolo11s-seg.pt`。
5. 图像和点云交互显示需要本地图形桌面环境。

### 行人检测训练和跟踪

准备好转换后的 CrowdHuman 数据及 YAML，修改脚本中的目录后运行：

```bash
python train_crowdhuman.py
python track_crowdhuman.py
```

TrackEval 评估前，需要对应序列的 GT、帧号和分辨率映射，以及转换后的 MOT 跟踪文件。仅有保存的视频不能直接得到 HOTA、IDF1、MOTA。

### RGB-D 实验

```bash
python yolo-3d1.py
python yolo-3d2.py
python yolo-3d4.py
```

`yolo-3d3.py` 显示对照时还会读取当前工作目录中的 `chair_mask_cloud.ply`。请先生成这份文件：在 `yolo-3d4.py` 创建 `pcd` 后、显示前，加入：

```python
o3d.io.write_point_cloud("chair_mask_cloud.ply", pcd)
```

再执行：

```bash
python yolo-3d3.py
```

## 4. 当前限制

- RGB-D 验证主要基于 TUM 的一对样例图像，尚未证明多场景泛化性能。
- 参数对比结果依赖当前数据和评估协议，不宣称普遍最优或官方榜单成绩。
- 部分路径、类别、内参和显示设置仍写在脚本中，后续可整理为配置或命令行参数。
- 预训练模型可能误检；掩码边缘误差和无效深度会影响目标点云完整性。
- 当前尚未完成 Jetson/TensorRT 板端部署及相应性能评测。

## 5. 参考项目与数据

- [Ultralytics](https://github.com/ultralytics/ultralytics)：YOLO 训练、检测、分割与跟踪接口。
- [ByteTrack](https://github.com/ifzhang/ByteTrack)：多目标跟踪方法。
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)：跟踪评估工具。
- [Open3D](https://github.com/isl-org/Open3D)：RGB-D 样例、点云数据结构与可视化。
- [CrowdHuman](https://www.crowdhuman.org/)：行人检测数据集。
- [MOTChallenge](https://motchallenge.net/)：跟踪序列与评估数据。
- [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)：RGB-D 数据与相机/深度格式说明。

第三方源码、模型及数据集的许可和署名以对应上游项目为准。仓库中的上游文档保留在 [docs/README.md](docs/README.md)。
