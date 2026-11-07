from pathlib import Path
from ultralytics import YOLO
import argparse


parser = argparse.ArgumentParser(description="yolov11 crowdhuman训练")
parser.add_argument("--data", required=True, help="数据集 YAML")
parser.add_argument("--model", required=True, help="初始权重")
parser.add_argument("--output", required=True, help="训练输出目录（其下创建 crowdhuman_30）")
parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
parser.add_argument("--batch", type=int, default=16, help="batch size")
parser.add_argument("--device", default="0", help="设备")
args = parser.parse_args()
data = Path(args.data).resolve()
output_dir = Path(args.output).resolve()
epochs=args.epochs
batch=args.batch
device=args.device


def main():

    model = YOLO(str(args.model), task="detect")

    model.train(
        data=str(data),
        epochs=epochs,
        patience=8,
        imgsz=640,
        batch=batch,
        device=device,
        amp=True,
        workers=4,
        cache=False,
        close_mosaic=5,
        save=True,
        project=str(output_dir),
        name="crowdhuman_30",
    )

    print("训练结果保存在：", model.trainer.save_dir)


if __name__ == "__main__":
    main()
