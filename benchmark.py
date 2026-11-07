import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
import torch
import time
import numpy as np

parser = argparse.ArgumentParser(description="性能基准＋跟踪指标对照")
parser.add_argument("--source", required=True, help="视频路径")
parser.add_argument("--model", required=True, help="模型路径")
parser.add_argument("--warmup", type=int, default=20, help="先预热多少次")
parser.add_argument("--frames", type=int, default=200)
parser.add_argument("--repeat", type=int, default=3, help="测量几轮")
parser.add_argument("--device", default="0", help="例如 0 或 cpu")
args = parser.parse_args()
if args.warmup < 1 or args.frames < 1 or args.repeat < 1:
    parser.error("warmup、frames 和 repeat 必须大于零")

model = YOLO(str(args.model), task="detect")

predict_args = {
    "imgsz": 640,
    "rect": False,
    "batch": 1,
    "conf": 0.1,
    "iou": 0.7,
    "classes": [0],
    "half": False,
    "device": args.device,
    "verbose": False,
    "save": False,
}

video_path = str(args.source)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"无法打开视频：{video_path}")

success, warm_frame = cap.read()
if not success:
    raise RuntimeError("无法读取预热图片")

for _ in range(args.warmup):
    model.predict(warm_frame, **predict_args)

if Path(args.model).suffix.lower() == ".onnx":
    session = model.predictor.model.session
    providers = session.get_providers()
    print("ONNX实际会话 providers：", providers)

    if args.device != "cpu" and "CUDAExecutionProvider" not in providers:
        raise RuntimeError("本次ONNX会话没有启用CUDA")

print("预热已完成")

cap.release()
times_ms=[]

for repeat_index in range(args.repeat):
    # 每轮重新打开，保证测量的是相同的前200帧
    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            raise RuntimeError("无法打开视频")

        for frame_index in range(args.frames):
            success, frame = cap.read()

            if not success:
                raise RuntimeError(
                    f"视频不足 {args.frames} 帧，"
                    f"本轮只读取了 {frame_index} 帧"
                )

            if args.device != "cpu":
                torch.cuda.synchronize(int(args.device))

            start = time.perf_counter()

            result = model.predict(frame, **predict_args)[0]

            if args.device != "cpu":
                torch.cuda.synchronize(int(args.device))

            elapsed_ms = (time.perf_counter() - start) * 1000

            times_ms.append(elapsed_ms)

        print(f"第 {repeat_index + 1} 轮完成")

    finally:
        cap.release()


expected_samples = args.frames * args.repeat

if len(times_ms) != expected_samples:
    raise ValueError(
        f"计时样本不足：期望 {expected_samples}，"
        f"实际 {len(times_ms)}"
    )

mean_ms = float(np.mean(times_ms))
median_ms = float(np.median(times_ms))
p95_ms = float(np.percentile(times_ms, 95))

print(  f"每帧耗时平均值:{mean_ms:.2f}ms,"
        f"每帧耗时中位数:{median_ms:.2f}ms,"
        f"每帧耗时P95:{p95_ms:.2f}ms"
      )
