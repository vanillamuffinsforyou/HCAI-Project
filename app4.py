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
st.title("🔢 Sorting Algorithm Visualizer with Engagement Detection")

# Initialize session state
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "content_type" not in st.session_state:
    st.session_state.content_type = "text"
if "last_switch_time" not in st.session_state:
    st.session_state.last_switch_time = time.time()
if "not_engaged_since" not in st.session_state:
    st.session_state.not_engaged_since = None
if "engagement_scores" not in st.session_state:
    st.session_state.engagement_scores = deque(maxlen=30)

# Dropdown for algorithm selection
topic = st.selectbox("Select a Sorting Algorithm", [
    "Bubble Sort", "Insertion Sort", "Selection Sort",
    "Merge Sort", "Quick Sort", "Heap Sort", "Radix Sort"
])
keyword = topic.split()[0].lower()

# Load images and CSV
image_folder = "sorting_images"
image_paths = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if keyword in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

try:
    info_df = pd.read_csv("algorithm_info.csv")
except Exception as e:
    st.error(f"Error loading algorithm_info.csv: {e}")
    st.stop()

# Load YOLO models
det_model = YOLO("yolov8n.pt")
cls_model = YOLO("best.pt")
class_names = ["Engaged", "Not Engaged"]

# Algorithm details
def retrieve_algorithm_details(topic):
    row = info_df[info_df["Title"].str.lower() == topic.lower()]
    if not row.empty:
        row = row.iloc[0]
        return {
            "Title": row["Title"],
            "Description": row["Description"],
            "Steps": row["Steps"],
            "Code Example": row["Code Example"],
            "Time Complexity (Worst)": row["Time Complexity (Worst)"],
            "Space Complexity": row["Space Complexity"]
        }
    return None

# UI placeholders
cam_placeholder = st.empty()
content_placeholder = st.empty()

# Start button
if st.button("▶ Start Engagement Tracking"):
    st.session_state.run_camera = True

# Display content
def display_content(content_type, algorithm_details):
    content_placeholder.empty()
    if content_type == "text":
        content_placeholder.subheader(f"📚 {algorithm_details['Title']}")
        content_placeholder.markdown(f"**Description:** {algorithm_details['Description']}")
        content_placeholder.markdown(f"**Steps:** {algorithm_details['Steps']}")
        content_placeholder.code(algorithm_details['Code Example'], language="python")
        content_placeholder.markdown(f"**Time Complexity (Worst):** {algorithm_details['Time Complexity (Worst)']}")
        content_placeholder.markdown(f"**Space Complexity:** {algorithm_details['Space Complexity']}")

        # Static output shown here
        arr = [64, 34, 25, 12, 22, 11, 90]
        sorted_arr = sorted(arr)
        content_placeholder.markdown(f"**Input:** `{arr}`")
        content_placeholder.markdown(f"**Output:** `{sorted_arr}`")

    elif content_type == "image":
        content_placeholder.subheader("🖼️ Visual Steps")
        for img_path in image_paths:
            content_placeholder.image(img_path, use_column_width=True)

    elif content_type == "video":
        try:
            video_df = pd.read_csv("videos.csv")
            row = video_df[video_df["Topic Name"].str.lower() == topic.lower()]
            if not row.empty:
                content_placeholder.subheader("🎥 Video Tutorial")
                content_placeholder.video(row.iloc[0]["Link"])
        except:
            content_placeholder.warning("Video not available.")

# Display initial content
algorithm_details = retrieve_algorithm_details(topic)
if algorithm_details:
    display_content("text", algorithm_details)

# Camera + Engagement Loop
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)
    frame_window = cam_placeholder.image([])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
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
                    st.session_state.engagement_scores.append(score)
                    color = (0, 255, 0) if label == "Engaged" else (0, 0, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except:
                    continue

        avg_engagement = np.mean(st.session_state.engagement_scores) * 100 if st.session_state.engagement_scores else 0
        cv2.putText(display_frame, f"Avg Engagement: {avg_engagement:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if avg_engagement < 50:
            if st.session_state.not_engaged_since is None:
                st.session_state.not_engaged_since = time.time()
            elif time.time() - st.session_state.not_engaged_since >= 5:
                # Switch content
                if st.session_state.content_type == "text":
                    st.session_state.content_type = "image"
                elif st.session_state.content_type == "image":
                    st.session_state.content_type = "video"
                else:
                    st.session_state.content_type = "text"

                display_content(st.session_state.content_type, algorithm_details)
                st.session_state.not_engaged_since = None
        else:
            st.session_state.not_engaged_since = None

        # Update frame
        frame_window.image(Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)))
