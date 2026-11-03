from ultralytics import YOLO
import cv2 as cv
import open3d as o3d
import numpy as np

def show_from_camera(pcd, title="Camera view"):
    # 当前 TUM 图片的原始尺寸和内参
    width, height = 640, 480

    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        525.0, 525.0, 319.5, 239.5
    )

    # 点云已经处于相机坐标系，无需额外旋转、平移
    camera.extrinsic = np.eye(4)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=title,
        width=width,
        height=height
    )
    vis.add_geometry(pcd)
    vis.poll_events()

    # 添加点云之后，再固定显示相机
    control = vis.get_view_control()
    control.convert_from_pinhole_camera_parameters(
        camera, allow_arbitrary=True
    )

    # 统一近、远裁剪距离，单位为米
    control.set_constant_z_near(0.01)
    control.set_constant_z_far(100.0)

    vis.get_render_option().point_size = 2.0

    vis.run()
    vis.destroy_window()




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

    if valid_ratio < 0.5:
        print("邻域有效深度不足")

    else:

        depth_roi = depth[y1:y2, x1:x2]
        color_roi = rgb[y1:y2, x1:x2]

        v_roi, u_roi = np.indices(depth_roi.shape)
        u_roi = u_roi + x1
        v_roi = v_roi + y1

        z_ref = float(np.median(valid_depths)) / 5000.0
        Z = depth_roi.astype(np.float32) / 5000.0
        X = (u_roi - cx) * Z / fx
        Y = (v_roi - cy) * Z / fy
        valid = Z > 0
        points = np.stack((X, Y, Z), axis=-1)[valid]
        color_rgb = cv.cvtColor(color_roi, cv.COLOR_BGR2RGB)
        colors = color_rgb[valid].astype(np.float64) / 255.0

        threshold = 0.3  # 先尝试允许前后相差 30 厘米

        keep = np.abs(points[:, 2] - z_ref) < threshold

        filtered_points = points[keep]
        filtered_colors = colors[keep]


        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points)
        pcd1.colors = o3d.utility.Vector3dVector(colors)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(filtered_points)
        pcd2.colors = o3d.utility.Vector3dVector(filtered_colors)

        print("参考深度:",z_ref)


        saved_box = o3d.io.write_point_cloud("chair1.ply", pcd1)
        saved_filtered = o3d.io.write_point_cloud("chair2.ply", pcd2)
        pcd_box = o3d.io.read_point_cloud("chair1.ply")
        pcd_filtered = o3d.io.read_point_cloud("chair2.ply")
        pcd_mask = o3d.io.read_point_cloud("chair_mask_cloud.ply")

        print("筛选前点数:", points.shape[0])
        print("筛选后点数:", filtered_points.shape[0])
        print("保留比例:", filtered_points.shape[0] / points.shape[0])

        show_from_camera(pcd_box, "Box")
        show_from_camera(pcd_filtered, "Depth filter")
        show_from_camera(pcd_mask, "Segmentation mask")


