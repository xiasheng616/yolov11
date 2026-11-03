#YOLO 检测物体，再用检测框中心的深度计算 XYZ


from ultralytics import YOLO
import cv2 as cv
import open3d as o3d
import numpy as np

tum_rgbd = o3d.data.SampleTUMRGBDImage(
    data_root=r"D:\study\open3d_data"
)


# 读取原始图像，深度还没有转换成米
rgb = cv.imread(tum_rgbd.color_path)
depth = cv.imread(tum_rgbd.depth_path, cv.IMREAD_UNCHANGED)

if depth.ndim == 3 and depth.shape[2] == 1:
    depth = depth[:, :, 0]







height, width = depth.shape

fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5
height, width = depth.shape




# 检测物体
model = YOLO("yolo11s.pt")
result = model.predict(source=rgb, conf=0.25, verbose=False)[0]

# 带有检测框和类别的图片
output = result.plot()

for index, box in enumerate(result.boxes, start=1):
    # 检测框在原图上的像素坐标
    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
    class_id = int(box.cls.item())
    name = result.names[class_id]

    # 检测框中心，限制在图像范围内
    u = max(0, min(width - 1, int((x1 + x2) / 2)))
    v = max(0, min(height - 1, int((y1 + y2) / 2)))

    r = 2  # 中心向四周各取 2 个像素，共 5×5

    left = max(0, u - r)
    right = min(width, u + r + 1)
    top = max(0, v - r)
    bottom = min(height, v + r + 1)

    patch = depth[top:bottom, left:right]

    valid_depths = patch[patch > 0]
    valid_ratio = valid_depths.size / patch.size

    depth_raw = depth[v, u]

    # 在结果图 output 上画出取深度的区域
    cv.rectangle(
        output,
        (left, top),
        (right - 1, bottom - 1),
        (0, 255, 255),  # 黄色，BGR
        1  # 线宽
    )
    cv.circle(output, (u, v), 1, (0, 0, 255), -1)


    if depth_raw == 0:
        print(f"[{index}] {name}：中心深度无效，")
    else:
        Z = float(depth_raw) / 5000.0
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        print(
            f"[{index}] {name}：中心深度={Z:.3f} 米，"
        )
    if valid_ratio < 0.5:
        print(f"[{index}] {name}：邻域有效深度不足")
    else:
        Z_median = float(np.median(valid_depths)) / 5000.0
        X_median = (u - cx) * Z_median / fx
        Y_median = (v - cy) * Z_median / fy

        print(
            f"[{index}] {name}：邻域中位数={Z_median:.3f} 米，"
            f"有效比例={valid_ratio:.1%}，"
            f"XYZ=({X_median:.3f}, {Y_median:.3f}, {Z_median:.3f}) 米"
        )

cv.imshow("Detection", output)
cv.waitKey(0)
cv.destroyAllWindows()