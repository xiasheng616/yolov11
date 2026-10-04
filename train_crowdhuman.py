from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/root/autodl-tmp/yolov11/yolov11")


def main():
    # 查找已有的官方预训练权重
    candidates = [
        ROOT / "weights/yolo11s.pt",
        ROOT / "yolo11s.pt",
    ]
    weights = next((p for p in candidates if p.is_file()), None)

    if weights is None:
        raise FileNotFoundError(
            "没有找到 yolo11s.pt，请将预训练权重放到项目的 weights 文件夹"
        )

    model = YOLO(str(weights))

    model.train(
        data=str(ROOT / "datasets/CrowdHuman_yolo/crowdhuman.yaml"),
        epochs=30,
        patience=8,
        imgsz=640,
        batch=16,
        device=0,
        amp=True,
        workers=4,
        cache=False,
        close_mosaic=5,
        save=True,
        project=str(ROOT / "runs/detect"),
        name="crowdhuman_30",
    )

    print("训练结果保存在：", model.trainer.save_dir)


if __name__ == "__main__":
    main()