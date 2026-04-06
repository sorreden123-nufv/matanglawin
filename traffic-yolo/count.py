import cv2
from ultralytics import YOLO

# Load your model and video
model = YOLO("runs/detect/train8/weights/best.pt")
cap = cv2.VideoCapture("../../traffic1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize for a smaller window
    frame = cv2.resize(frame, (1280, 720))
    
    # Run YOLO prediction on the current frame
    results = model.predict(frame, verbose=False) # verbose=False stops it from spamming your terminal

    # Reset the count to 0 for this frame
    current_count = 0

    # If YOLO finds any boxes, count how many there are
    if results[0].boxes is not None:
        current_count = len(results[0].boxes)

    # Draw the text on the screen showing the current live count
    cv2.putText(frame, f"Currently on screen: {current_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video
    cv2.imshow("Traffic", frame)

    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()