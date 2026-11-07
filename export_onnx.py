import argparse
import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO


parser = argparse.ArgumentParser(description="导出 YOLO11 ONNX")
parser.add_argument("--model", required=True, help="训练后的PT权重")
parser.add_argument("--output", required=True, help="导出目录")
args = parser.parse_args()

model_path = Path(args.model).resolve()
output_dir = Path(args.output).resolve()

if not model_path.is_file():
    raise FileNotFoundError(model_path)

output_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(str(model_path))

# 先固定一个易于验证的 FP32、静态输入版本
exported = model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    half=False,
    nms=False,
    simplify=False,
    opset=17,
    device="cpu",
)

# Ultralytics 默认导出到权重旁边，再复制到指定目录
exported_path = Path(exported).resolve()
output_path = output_dir / "yolo11s_crowdhuman.onnx"

if exported_path != output_path.resolve():
    shutil.copy2(exported_path, output_path)

# 检查文件能否解析，以及ONNX结构是否合法
onnx_model = onnx.load(str(output_path))
onnx.checker.check_model(onnx_model)

input_shape = [
    dimension.dim_value
    for dimension in onnx_model.graph.input[0].type.tensor_type.shape.dim
]

if input_shape != [1, 3, 640, 640]:
    raise ValueError(f"输入尺寸不符合预期：{input_shape}")

print("ONNX已保存：", output_path)
print("输入尺寸：", input_shape)
