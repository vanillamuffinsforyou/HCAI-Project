import streamlit as st
import pandas as pd
import os
import time
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# Page setup
st.set_page_config(page_title="Sorting Algorithm Visualizer with Engagement", layout="wide")

# Title
st.title("🔢 Sorting Algorithm Visualizer with Engagement Detection")

# Initialize session state
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "camera_started" not in st.session_state:
    st.session_state.camera_started = False

# Dropdown
topic = st.selectbox("Select a Sorting Algorithm", [
    "Bubble Sort", "Insertion Sort", "Selection Sort",
    "Merge Sort", "Quick Sort", "Heap Sort", "Radix Sort"
])

# Extract keyword
keyword = topic.split()[0].lower()

# Load images
image_folder = "sorting_images"
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if keyword in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
image_paths.sort()

# Load algorithm info
try:
    info_df = pd.read_csv("algorithm_info.csv")
except FileNotFoundError:
    st.error("Error: algorithm_info.csv not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading algorithm_info.csv: {e}")
    st.stop()

# Description
desc_row = info_df[info_df["Title"].str.lower() == topic.lower()]

# Load videos info
try:
    video_df = pd.read_csv("video_links.csv")
except FileNotFoundError:
    video_df = pd.DataFrame({"Topic Name": [], "Link": []})
except Exception as e:
    st.error(f"Error loading videos.csv: {e}")
    st.stop()

# Camera Feed and Engagement Tracking
cap = cv2.VideoCapture(0)  # Access camera feed
content_type = "text"  # Start with text
last_switch_time = time.time()
not_engaged_dominant_since = None
switch_interval = 120  # seconds
content_displayed = False
engagement_scores = deque(maxlen=30)  # Using a shorter window for faster responsiveness

# Load YOLO model for person detection
try:
    det_model = YOLO("yolov8n.pt")
except Exception as e:
    st.error(f"Error loading person detection model: {e}")
    st.stop()

# Load classification model for engagement (replace with your model path)
try:
    cls_model = YOLO("best.pt")
except Exception as e:
    st.error(f"Error loading engagement classification model: {e}")
    st.stop()
class_names = ["Engaged", "Not Engaged"]  # Ensure this matches your classification model's output

# Function to retrieve algorithm details
def retrieve_algorithm_details(topic):
    if not desc_row.empty:
        row = desc_row.iloc[0]  # Get the first match
        details = {}
        if pd.notna(row["Title"]):
            details["Title"] = row["Title"]
        if pd.notna(row["Description"]):
            details["Description"] = row["Description"]
        if pd.notna(row["Steps"]):
            details["Steps"] = row["Steps"]
        if pd.notna(row["Code Example"]):
            details["Code Example"] = row["Code Example"]
        if pd.notna(row["Time Complexity (Worst)"]):
            details["Time Complexity (Worst)"] = row["Time Complexity (Worst)"]
        if pd.notna(row["Space Complexity"]):
            details["Space Complexity"] = row["Space Complexity"]
        return details
    return None

# Function to retrieve video link
def retrieve_video_link(topic):
    try:
        video_row = video_df[video_df["Topic Name"].astype(str).str.lower() == topic.lower()]
        if not video_row.empty:
            return video_row.iloc[0]["Link"]
    except AttributeError:
        st.error("Error: 'Topic Name' column in videos.csv contains non-string values.")
    return None

# Streamlit UI placeholders for content
placeholder = st.empty()  # Placeholder for camera feed display
text_placeholder = st.empty()
image_placeholder = st.empty()
video_placeholder = st.empty()

# Start button
if st.button("▶ Start Engagement Tracking"):
    st.session_state.run_camera = True
    st.session_state.camera_started = True
    # Display initial text content after the button is clicked
    algorithm_details = retrieve_algorithm_details(topic)
    with text_placeholder.container():
        if algorithm_details:
            st.subheader(f"📘 {algorithm_details.get('Title', 'Description')}")
            if "Description" in algorithm_details:
                st.write(algorithm_details["Description"])
            if "Steps" in algorithm_details:
                st.subheader("Steps:")
                st.write(algorithm_details["Steps"])
            if "Code Example" in algorithm_details:
                st.subheader("Code Example:")
                st.code(algorithm_details["Code Example"], language="python")
            if "Time Complexity (Worst)" in algorithm_details:
                st.markdown(f"**Time Complexity (Worst):** {algorithm_details['Time Complexity (Worst)']}")
            if "Space Complexity" in algorithm_details:
                st.markdown(f"**Space Complexity:** {algorithm_details['Space Complexity']}")

while st.session_state.run_camera:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    height, width, _ = display_frame.shape
    engagement_this_frame = []
    results = det_model(display_frame)[0]

    for det in results.boxes:
        cls_id = int(det.cls[0])
        if det_model.names[cls_id] == "person":
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cropped = display_frame[y1:y2, x1:x2]
            try:
                resized = cv2.resize(cropped, (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pred = cls_model(rgb)[0].probs
                label = class_names[pred.top1]
                confidence = pred.data[pred.top1].item()
                score = confidence if label == "Engaged" else (1 - confidence)
                engagement_scores.append(score)
                color = (0, 255, 0) if label == "Engaged" else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                continue

    avg_engagement = np.mean(list(engagement_scores)) * 100 if engagement_scores else 0
    cv2.putText(display_frame, f"Avg Engagement (Last {engagement_scores.maxlen} frames): {avg_engagement:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Content Switching Logic based on average engagement
    if avg_engagement < 50:  # Adjust threshold as needed
        if not_engaged_dominant_since is None:
            not_engaged_dominant_since = time.time()
        elif time.time() - not_engaged_dominant_since >= switch_interval:
            if content_type == "text":
                with image_placeholder.container():
                    st.subheader("🖼️ Visual Steps")
                    if image_paths:
                        for img_path in image_paths:
                            try:
                                img = Image.open(img_path)
                                st.image(img, use_column_width=True)
                            except Exception as e:
                                st.error(f"Error displaying image {os.path.basename(img_path)}: {e}")
                text_placeholder.empty()
                video_placeholder.empty()
                content_type = "image"
            elif content_type == "image":
                with video_placeholder.container():
                    video_link = retrieve_video_link(topic)
                    if video_link:
                        st.subheader("🎥 Video Tutorial")
                        st.video(video_link)
                text_placeholder.empty()
                image_placeholder.empty()
                content_type = "video"
            elif content_type == "video":
                with text_placeholder.container():
                    algorithm_details = retrieve_algorithm_details(topic)
                    if algorithm_details:
                        st.subheader(f"📘 {algorithm_details.get('Title', 'Description')}")
                        if "Description" in algorithm_details:
                            st.write(algorithm_details["Description"])
                        if "Steps" in algorithm_details:
                            st.subheader("Steps:")
                            st.write(algorithm_details["Steps"])
                        if "Code Example" in algorithm_details:
                            st.subheader("Code Example:")
                            st.code(algorithm_details["Code Example"], language="python")
                        if "Time Complexity (Worst)" in algorithm_details:
                            st.markdown(f"**Time Complexity (Worst):** {algorithm_details['Time Complexity (Worst)']}")
                        if "Space Complexity" in algorithm_details:
                            st.markdown(f"**Space Complexity:** {algorithm_details['Space Complexity']}")
                image_placeholder.empty()
                video_placeholder.empty()
                content_type = "text"
            not_engaged_dominant_since = None  # Reset timer
    else:
        not_engaged_dominant_since = None  # Reset if engagement improves
        if content_type != "text":
            with text_placeholder.container():
                algorithm_details = retrieve_algorithm_details(topic)
                if algorithm_details:
                    st.subheader(f"📘 {algorithm_details.get('Title', 'Description')}")
                    if "Description" in algorithm_details:
                        st.write(algorithm_details["Description"])
                    if "Steps" in algorithm_details:
                        st.subheader("Steps:")
                        st.write(algorithm_details["Steps"])
                    if "Code Example" in algorithm_details:
                        st.subheader("Code Example:")
                        st.code(algorithm_details["Code Example"], language="python")
                    if "Time Complexity (Worst)" in algorithm_details:
                        st.markdown(f"**Time Complexity (Worst):** {algorithm_details['Time Complexity (Worst)']}")
                    if "Space Complexity" in algorithm_details:
                        st.markdown(f"**Space Complexity:** {algorithm_details['Space Complexity']}")
            image_placeholder.empty()
            video_placeholder.empty()
            content_type = "text"

    # Display the camera feed with engagement annotations
    frame_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    placeholder.image(frame_image, channels="RGB")

cap.release()