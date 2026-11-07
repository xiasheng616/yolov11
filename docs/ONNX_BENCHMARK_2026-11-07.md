# PyTorch / ONNX FP32 后端对照

记录日期：2026-11-07。跟踪指标来自本地 `evaluation/backend_compare/comparison.json`；时延来自用户提供的 benchmark.py 终端输出。已检查当前计时代码和评估 manifest 中的轨迹哈希，没有重新运行性能测试。

## 条件与计时范围

- YOLO11s CrowdHuman 微调权重及其静态 FP32 ONNX 导出。
- 输入为 A_street.mp4；对应 MOT17-09-DPM 的本地 960×540 压缩预览，共 525 帧。
- 两后端检测输入均为 640×640，`rect=False`、batch=1、conf=0.1、IoU=0.7、classes=[0]、half=False、device=0。
- 时延测试预热 20 次，对同一视频前 200 帧重复 3 轮，共 600 次计时；不是 600 张不同图片。
- 计时边界为同步 GPU 后调用 `model.predict()`，并在结束前再次同步。包括该调用中的预处理、推理和后处理/API 开销；不包括模型加载、视频解码、ByteTrack、画框及视频写入。
- ONNX 日志显示会话 providers 为 `CUDAExecutionProvider`、`CPUExecutionProvider`。这证明启用了 CUDA provider，不表示已逐节点分析全部算子的执行位置。
- 当前未记录 GPU 型号、逐帧原始时延和逐轮统计，不据此给出跨设备结论或统计显著性结论。

## 检测流水线时延

| 后端 | 平均 ms | 中位数 ms | P95 ms |
|---|---:|---:|---:|
| PyTorch FP32 | 11.15 | 10.60 | 14.89 |
| ONNX FP32 / ORT CUDA | 11.10 | 10.97 | 12.05 |

按已显示的四舍五入数值计算，ONNX 平均时延降低约 0.45%，中位数反而增加约 3.49%，P95 降低约 19.07%。平均时延基本相当，不能宣传显著加速。P95 更低是本次测量的观察，尚不足以证明在其他视频/设备上都更稳定。

这里没有报告完整视频跟踪系统的端到端 FPS。

## A 序列跟踪评估

两组跟踪输出均使用 baseline ByteTrack 配置、相同预览视频和方形输入，转换到 GT 坐标后使用 TrackEval 评估。轨迹哈希已与 manifest 核对一致。

| 后端 | HOTA | IDF1 | MOTA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 52.809 | 62.283 | 67.718 | 45 | 584 | 1090 |
| ONNX FP32 | 52.798 | 62.256 | 67.681 | 45 | 584 | 1092 |

ONNX 相比 PyTorch：HOTA 低 0.011 个点、IDF1 低 0.027 个点、MOTA 低 0.037 个点，IDSW/FP 相同，FN 多 2。当前单序列的指标接近；不能推广为所有场景的无损转换。

这些是同一基准下的跟踪指标，不是独立的检测 mAP，也不是 MOTChallenge 原始图像协议的官方榜单结果。不能与之前 rect=True 的 baseline 直接作为同输入条件的速度/精度对照。

## 可用于 README 的结论

> 完成 YOLO11s FP32 ONNX 导出和 ORT CUDA 推理，在统一 640×640 输入及后处理条件下，与 PyTorch 进行了检测流水线时延和跟踪效果对照。A 序列 HOTA 为 52.809 / 52.798；平均时延为 11.15 / 11.10 ms，基本相当。本次测试 ONNX P95 更低，尚未证明普遍的速度或鲁棒性收益。

## 复现命令

```powershell
python benchmark.py --model runs/detect/crowdhuman_30/weights/best.pt --source data/videos/A_street.mp4 --device 0 --warmup 20 --frames 200 --repeat 3
python benchmark.py --model exports/yolo11s_crowdhuman.onnx --source data/videos/A_street.mp4 --device 0 --warmup 20 --frames 200 --repeat 3
python evaluate_tracks.py --runs runs --videos data/videos --resources evaluation_resources --output evaluation/backend_compare --scenes A --variants pt_square onnx_square
```

## 独立实现验收

当前 benchmark.py 已正确排除预热、每轮重新读取相同帧段、检查不足帧数、在 GPU 同步边界计时，并验证计时样本数等于 frames×repeat。ONNX 会话检查及日志打印在计时循环之外。该阶段可以收尾，不要求重复实现配置 JSON 或为得到加速结论而继续调参。

ONNX 的自动猜测任务警告可通过 `YOLO(str(args.model), task="detect")` 明确任务类型来消除；当前日志已默认 detect，不据此认定推理失败。
