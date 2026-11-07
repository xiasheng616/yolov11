# 第三方代码、模型与数据

- Ultralytics YOLO：本仓库保留 `ultralytics/` 及其 `pyproject.toml`，包版本为 8.3.185；上游 https://github.com/ultralytics/ultralytics 。源码作者与许可证属于上游，根目录 [LICENSE](../LICENSE) 保留相应 AGPL-3.0 文本。
- ByteTrack：通过 Ultralytics 跟踪接口使用；方法来源 https://github.com/ifzhang/ByteTrack 。
- TrackEval：独立下载的第三方评估工具，使用 commit `12c8791b303e0a0b50f753af204249e622d0281a`；https://github.com/JonathonLuiten/TrackEval 。
- CrowdHuman：https://www.crowdhuman.org/ 。
- MOTChallenge：https://motchallenge.net/data/MOT17/ 。
- ONNX Runtime：https://onnxruntime.ai/ 。
- Open3D：https://github.com/isl-org/Open3D 。
- TUM RGB-D：https://cvg.cit.tum.de/data/datasets/rgbd-dataset 。

个人工作集中在数据转换、实验入口、轨迹/配置记录、坐标映射与评估集成、基准计时、RGB-D 数值处理。没有将第三方网络、跟踪方法或评估指标认领为原创算法。

精选图片为本人实际运行产物，底层输入来自相应公开数据；输入数据的使用与分享仍应遵循原数据集条款。未重新分发完整数据、预训练权重或训练输出。
