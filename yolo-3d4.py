from ultralytics import YOLO
import cv2 as cv
import open3d as o3d
import numpy as np

fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5
tum_rgbd = o3d.data.SampleTUMRGBDImage(
    data_root=r"D:\study\open3d_data"
)

rgb = cv.imread(tum_rgbd.color_path)
depth = cv.imread(tum_rgbd.depth_path, cv.IMREAD_UNCHANGED)

if depth.ndim == 3 and depth.shape[2] == 1:
    depth = depth[:, :, 0]

model = YOLO("weights/yolo11s-seg.pt")

result = model.predict(
    source=rgb,
    conf=0.25,
    retina_masks=True,
    verbose=False
)[0]


chair_indices = []

for i, box in enumerate(result.boxes):
    name = result.names[int(box.cls.item())]
    if name == "chair":
        chair_indices.append(i)


if result.masks is None or not chair_indices:
    raise RuntimeError("没有获得椅子的分割结果")

best_index = max(
    chair_indices,
    key=lambda i: result.boxes[i].conf.item()
)

mask = result.masks.data[best_index].cpu().numpy() > 0.5

print("掩码尺寸：", mask.shape)
print("深度图尺寸：", depth.shape)


assert mask.shape == depth.shape

cv.imwrite("chair_mask.png", mask.astype(np.uint8) * 255)
cv.imwrite("chair_segmentation.jpg", result.plot())


height, width = depth.shape
v, u = np.indices((height, width))

Z_map = depth.astype(np.float32) / 5000.0
X_map = (u - cx) * Z_map / fx
Y_map = (v - cy) * Z_map / fy

# 同时满足：模型认为属于椅子、深度有效
valid = mask & (Z_map > 0)

points = np.stack((X_map, Y_map, Z_map), axis=-1)[valid]

color_rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
colors = color_rgb[valid].astype(np.float64) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

print("有效点数:", points.shape[0])


if not o3d.io.write_point_cloud("chair_mask_cloud.ply", pcd):
    raise OSError("保存 chair_mask_cloud.ply 失败")

o3d.visualization.draw_geometries([pcd])
