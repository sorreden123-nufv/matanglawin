import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture("traffic.mp4")

counted_ids = set()
vehicle_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    if results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            track_id = int(box.id[0]) if box.id is not None else None

            if track_id is not None and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

    cv2.putText(frame, f"Count: {vehicle_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Traffic", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Total vehicles:", vehicle_count)