import cv2
from ultralytics import YOLO

# Load your model and video
model = YOLO("runs/detect/train12/weights/best.pt")
cap = cv2.VideoCapture("../../traffic1.mp4")

# Extract the class names from your custom model (e.g., Car, Truck, Bus, Motorcycle)
class_names = model.names 

# Set your desired confidence level here (0.5 means 50% confidence)
CONFIDENCE_THRESHOLD = 0.6 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize for a smaller window
    frame = cv2.resize(frame, (1280, 720))
    
    # persist=True tells the model to remember objects across frames
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Generate a new frame with YOLO's bounding boxes and labels drawn on it
    annotated_frame = results[0].plot()

    # Initialize a dictionary to hold the count for all classes (setting them to 0 initially)
    class_counts = {name: 0 for name in class_names.values()}
    current_count = 0

    # If YOLO finds any boxes, count them and identify their classes
    if results[0].boxes is not None:
        boxes = results[0].boxes
        current_count = len(boxes)
        
        # Iterate through each detected box to find out what class it is
        for box in boxes:
            class_id = int(box.cls[0].item()) # Get the class ID
            class_name = class_names[class_id] # Map ID to the actual string name
            class_counts[class_name] += 1      # Increment the tally for that specific class

    # --- COMMAND LINE OUTPUT ---
    # Print the in-depth reading to the terminal
    print(f"Total on screen: {current_count} | Breakdown: {class_counts}")

    # --- SCREEN DRAWING ---
    # Draw the total count text on the *annotated* screen
    cv2.putText(annotated_frame, f"Total currently on screen: {current_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the specific four classes dynamically underneath the total count
    y_offset = 90
    for name, count in class_counts.items():
        # Text format: "ClassName: Count"
        text = f"{name}: {count}"
        cv2.putText(annotated_frame, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_offset += 35 # Push the next class down by 35 pixels so they don't overlap

    # Show the video with both the bounding boxes and your custom text
    cv2.imshow("Traffic", annotated_frame)

    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Made by Sorrera, Raiden G.