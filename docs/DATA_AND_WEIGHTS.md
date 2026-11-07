# 数据、权重和评估资源

大型输入和权重保留在本地，不随 Git 下载。下列路径均相对仓库根目录。

## 跟踪演示

任意普通视频可通过 `--source` 输入。公开预训练 YOLO11s 可由 `YOLO('yolo11s.pt')` 获取；它仅用于入口演示，不代表 CrowdHuman 微调结果。

本项目指标使用本地 `runs/detect/crowdhuman_30/weights/best.pt`。当前没有发布该微调文件的下载附件。严格复现需获取相同权重，或按下述数据流程自行训练；自行训练不保证逐位复现已有指标。

## CrowdHuman

官方：https://www.crowdhuman.org/

```text
datasets/CrowdHuman/
├── annotation_train.odgt
├── annotation_val.odgt
├── train/Images/*.jpg
└── val/Images/*.jpg
```

下载 ZIP 后先用 `ZipFile.testzip()` 或 `unzip -t` 检查完整性。转换脚本固定核对官方 train=15000、val=4370 的记录数。

`--image-mode symlink` 避免复制原始图片，适合支持符号链接的环境；Windows 无相应权限时使用 `--image-mode copy`，需为复制后的数据预留磁盘空间。已有普通目标文件不会被静默覆盖。

转换过滤 mask/ignore 框的正标签并保存元信息，不实现 ignore-aware YOLO loss。

## MOT17 评估

官方归档：https://motchallenge.net/data/MOT17/

仅需 `MOT17Labels.zip` 标注包和对应的视频/序列。TrackEval 来源：https://github.com/JonathonLuiten/TrackEval

```text
evaluation_resources/
├── MOT17Labels.zip
└── TrackEval-master/trackeval/
```

准备命令（需要网络；系统是否需要代理由当地环境决定）：

```powershell
New-Item -ItemType Directory -Force evaluation_resources
curl.exe -L --fail -o evaluation_resources/MOT17Labels.zip https://motchallenge.net/data/MOT17Labels.zip
git clone https://github.com/JonathonLuiten/TrackEval.git evaluation_resources/TrackEval-master
git -C evaluation_resources/TrackEval-master checkout 12c8791b303e0a0b50f753af204249e622d0281a
```

旧指标使用下列本地预览，帧序不变，960×540、30 FPS：

| 本地文件 | 对应序列 | 帧数 |
|---|---|---:|
| data/videos/A_street.mp4 | MOT17-09-DPM | 525 |
| data/videos/B_crowd.mp4 | MOT17-02-DPM | 600 |
| data/videos/C_night.mp4 | MOT17-04-DPM | 1050 |

A 预览 SHA256：`44472faa5643b173fa9b0d4837d3018cd1f5c4a3f13ab4bc9f865cf552022f9d`。

本仓库不重新分发视频，也未验证当前官网重新获取的压缩视频是否字节一致。严格比较需相同输入；可使用官方序列制作预览，但不能假设重编码后会得到完全相同的模型输出。

轨迹为 1-based 视频像素坐标，映射到 GT 时使用 `(x-1)*scale+1`。现有评估适配整幅缩放，不能直接用于裁剪、抽帧或改变播放顺序的输入。脚本会输出首/中/末帧 GT 对齐图，应检查目标是否对应。

这些结果不代表 MOTChallenge 官方排行榜成绩。
