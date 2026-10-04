import cv2
from ultralytics import YOLO

model = YOLO(
    r"runs\detect\crowdhuman_30\weights\best.pt"
)

video_path = r"data\videos\A_street.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open video. Check video_path.")

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.1,
        imgsz=640,
        classes=[0],
        verbose=False,
    )

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO11 + ByteTrack", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()