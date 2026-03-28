from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run on video
results = model.predict(
    source="traffic.mp4",
    save=True,
    conf=0.25
)

print("Done!")