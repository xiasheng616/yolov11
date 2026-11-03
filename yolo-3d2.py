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

model = YOLO("yolo11s.pt")
result = model.predict(source=rgb, conf=0.25, verbose=False)[0]

# 1. 找出所有 chair 框
chair_boxes = []

for box in result.boxes:
    class_id = int(box.cls.item())
    name = result.names[class_id]

    if name == "chair":
        chair_boxes.append(box)

# 2. 选择置信度最高的 chair 框
if not chair_boxes:
    print("没有检测到 chair")
else:
    best_chair = max(
        chair_boxes,
        key=lambda box: box.conf.item()
    )

    # 转成整数，map(int, ...) 表示对四个坐标分别调用 int()
    x1, y1, x2, y2 = map(
        int, best_chair.xyxy[0].cpu().tolist()
    )

    height, width = depth.shape

    # 限制在图片范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    depth_roi = depth[y1:y2, x1:x2]
    color_roi = rgb[y1:y2, x1:x2]

    v_roi, u_roi = np.indices(depth_roi.shape)
    u_roi = u_roi + x1
    v_roi = v_roi + y1

    roi_height, roi_width = depth_roi.shape
    print(f"检测框区域尺寸：宽 {roi_width}，高 {roi_height}")

    Z = depth_roi.astype(np.float32) / 5000.0
    X = (u_roi - cx) * Z / fx
    Y = (v_roi - cy) * Z / fy
    valid = Z > 0
    points = np.stack((X, Y, Z), axis=-1)[valid]
    color_rgb = cv.cvtColor(color_roi, cv.COLOR_BGR2RGB)
    colors = color_rgb[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    cv.imwrite("test.jpg", color_roi)

    success = o3d.io.write_point_cloud("chair_region.ply", pcd)



    print("有效点数：", points.shape[0])
    o3d.visualization.draw_geometries([pcd])



