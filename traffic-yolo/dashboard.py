import cv2
import streamlit as st
import pandas as pd
import os
from ultralytics import YOLO

# --- Dashboard Setup ---
st.set_page_config(page_title="Traffic Dashboard", layout="wide")
st.title("🚦 Real-Time Traffic Monitor")

# Sidebar for controls
st.sidebar.header("Settings")

# 1. Added a text input for the video filename/path
video_path = st.sidebar.text_input("Video File Path:", "../../traffic2.mp4")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

# Create layout columns for the live feed
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Video Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("Live Statistics")
    stats_placeholder = st.empty()
    chart_placeholder = st.empty()

# Placeholder for the final summary to appear at the bottom later
summary_placeholder = st.container()

# --- YOLO Setup ---
@st.cache_resource
def load_model():
    # Make sure this points to your latest trained weights!
    return YOLO("runs/detect/train9/weights/best.pt")

model = load_model()
class_names = model.names

def run_traffic_cam(source_path):
    # 2. Check if the file actually exists before running
    if not os.path.exists(source_path):
        st.error(f"Error: Could not find the file at '{source_path}'. Please check the path and try again.")
        return

    cap = cv2.VideoCapture(source_path)
    
    # Dictionary to keep track of unique vehicles across the whole video
    # Format will be: {track_id: "class_name"}
    unique_tracked_objects = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (1280, 720))
        
        # Run tracking
        results = model.track(frame, persist=True, conf=confidence, verbose=False)
        annotated_frame = results[0].plot()

        # Frame-by-frame counts for the live bar chart
        live_class_counts = {name: 0 for name in class_names.values()}
        
        # 3. Log unique IDs for the final summary
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            # zip() lets us loop through the boxes and their unique IDs at the same time
            for box, track_id in zip(boxes, boxes.id):
                class_id = int(box.cls[0].item())
                class_name = class_names[class_id]
                t_id = int(track_id.item())
                
                # If we haven't seen this ID before, save it to our unique dictionary
                if t_id not in unique_tracked_objects:
                    unique_tracked_objects[t_id] = class_name
                
                # Tally the live count for this specific frame
                live_class_counts[class_name] += 1

        # Update the Video Feed
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update the Live Text Stats
        with stats_placeholder.container():
            st.write(f"**Currently on Screen:** {sum(live_class_counts.values())}")
            for name, count in live_class_counts.items():
                st.write(f"- {name}: {count}")

        # Update the Live Bar Chart
        df = pd.DataFrame(list(live_class_counts.items()), columns=['Vehicle', 'Count'])
        chart_placeholder.bar_chart(df.set_index('Vehicle'))

    cap.release()
    
    # 4. Generate the Final Summary after the video finishes
    with summary_placeholder:
        st.success("✅ Video processing complete! Here is your final summary:")
        
        # Tally up all the unique objects we saved
        final_totals = {name: 0 for name in class_names.values()}
        for obj_class in unique_tracked_objects.values():
            final_totals[obj_class] += 1
            
        st.write(f"### Total Unique Vehicles Detected: {len(unique_tracked_objects)}")
        
        # Display the breakdown, skipping classes that had 0 detections
        for name, count in final_totals.items():
            if count > 0:
                st.write(f"- **{name}:** {count}")

# Run the app
if st.button("Start Camera"):
    # Pass the text input path into the function
    run_traffic_cam(video_path)