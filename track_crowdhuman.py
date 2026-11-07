import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
import json
import shutil
import platform
import torch
import ultralytics
from ultralytics.utils.checks import check_yaml


parser = argparse.ArgumentParser(description="yolov11+ByteTrack 行人检测跟踪")
parser.add_argument("--source", required=True, help="视频路径")
parser.add_argument("--model", required=True, help="权重路径")
parser.add_argument("--output", required=True, help="输出目录")
parser.add_argument("--tracker", required=True, help="跟踪配置")
parser.add_argument("--conf", type=float, default=0.1)
parser.add_argument("--show", action="store_true", help="显示跟踪窗口")
parser.add_argument("--device", default="0", help="例如 0 或 cpu")
parser.add_argument(
    "--square",
    action="store_true",
    help="固定使用640×640输入",
)
args = parser.parse_args()
output_dir = Path(args.output).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
conf=args.conf
tracker=args.tracker
device=args.device

model = YOLO(str(args.model), task="detect")

video_path = str(args.source)

cap = cv2.VideoCapture(video_path)
writer = None
tracks_file = None

try:
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    # 保存视频需要知道原视频的尺寸和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("视频尺寸或帧率无效")

    output_path = output_dir / "tracked.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    frame_count = 0

    track_rows = 0
    stop_reason = "read_finished"

    tracks_file = (output_dir / "tracks.txt").open(
        "w",
        encoding="utf-8",
        newline="",
    )

    # 找到实际使用的 YAML，并保存本次配置快照
    tracker_path = Path(check_yaml(tracker)).resolve()
    tracker_snapshot = output_dir / "tracker.yaml"

    if tracker_path != tracker_snapshot.resolve():
        shutil.copy2(tracker_path, tracker_snapshot)
    tracker = str(tracker_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            tracker=str(tracker),
            conf=conf,
            imgsz=640,
            classes=[0],
            verbose=False,
            device=device,
            rect=not args.square,
        )

        if frame_count == 1 and Path(args.model).suffix.lower() == ".onnx":
            session = model.predictor.model.session

            print("ONNX实际会话 providers：", session.get_providers())
            print("ONNX模型输入尺寸：", session.get_inputs()[0].shape)

        result = results[0]
        boxes = result.boxes

        # 只保存已经分配跟踪 ID 的框
        if boxes is not None and boxes.id is not None:
            coordinates = boxes.xyxy.cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist()
            scores = boxes.conf.cpu().tolist()

            for xyxy, track_id, score in zip(
                coordinates, track_ids, scores
            ):
                x1, y1, x2, y2 = xyxy
                box_width = x2 - x1
                box_height = y2 - y1

                if box_width <= 0 or box_height <= 0:
                    continue

                # MOTChallenge 使用从1开始的左上角坐标
                left = x1 + 1
                top = y1 + 1

                tracks_file.write(
                    f"{frame_count},{track_id},"
                    f"{left:.3f},{top:.3f},"
                    f"{box_width:.3f},{box_height:.3f},"
                    f"{score:.6f},-1,-1,-1\n"
                )
                track_rows += 1

        annotated_frame = results[0].plot()

        # 每一帧都写入，包括没有检测到人的帧
        writer.write(annotated_frame)

        # 只有传入 --show 才显示窗口和读取按键
        if args.show:
            cv2.imshow("YOLO11 + ByteTrack", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_reason = "user_quit"
                break

    print("已处理帧数：", frame_count)
    print("视频已保存：", output_path)

    run_config = {
        "source": str(Path(args.source).resolve()),
        "model": str(Path(args.model).resolve()),
        "tracker_source": str(tracker_path),
        "tracker_snapshot": "tracker.yaml",
        "device": str(device),
        "conf": conf,
        "imgsz": 640,
        "classes": [0],
        "rect": not args.square,
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "processed_frames": frame_count,
        },
        "track_rows": track_rows,
        "stop_reason": stop_reason,
        "coordinate_format": "MOT xywh, 1-based, input-video pixels",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
        },
    }

    with (output_dir / "run_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)


finally:
    # 正常结束、按 q 或运行中报错，都执行资源释放
    cap.release()

    if writer is not None:
        writer.release()

    if args.show:
        cv2.destroyAllWindows()

    if tracks_file is not None:
        tracks_file.close()
