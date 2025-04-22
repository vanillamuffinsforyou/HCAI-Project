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
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pickle
import faiss

# =========================== CONFIG ===========================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm_model = genai.GenerativeModel("gemini-2.0-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ======================== DATA LOAD ===========================
try:
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load("embeddings.npy")
    index = faiss.read_index("faiss.index")
except Exception as e:
    st.error("Error loading preprocessed data. Have you run preprocess.py?")
    st.stop()

try:
    algo_df = pd.read_csv("algorithm_info.csv")
except:
    algo_df = pd.DataFrame({
        "Title": ["Radix Sort"],
        "Description": ["Sorts numbers by processing individual digits."],
        "Steps": ["1. Group elements\n2. Sort each group"],
        "Code Example": ["def radix_sort(arr):\n    pass"],
        "Time Complexity (Worst)": ["O(nk)"],
        "Space Complexity": ["O(n+k)"]
    })

try:
    video_df = pd.read_csv("video_links.csv")
except:
    video_df = pd.DataFrame({
        "Topic Name": ["Radix Sort"],
        "Link": ["https://www.youtube.com/embed/dQw4w9WgXcQ"]
    })

# ======================== LLM LOGIC ===========================
def extract_keywords(query):
    prompt = f"""
    Extract important keywords for information retrieval based on this question:
    Query: "{query}"
    Return a comma-separated list of keywords.
    """
    response = llm_model.generate_content(prompt)
    return [kw.strip() for kw in response.text.strip().split(",") if kw.strip()]

def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def generate_response(user_input):
    context = "\n\n".join(retrieve_relevant_chunks(user_input))
    prompt = (
        f"You are a helpful assistant for students.\n\nHere is the topic context:\n{context}\n\n"
        f"User: {user_input}\n\nAssistant:"
    )
    response = llm_model.generate_content(prompt)
    return response.text.strip()

def retrieve_text_data_llm(topic):
    keywords = extract_keywords(topic)
    best_row, best_score = None, 0
    for _, row in algo_df.iterrows():
        score = sum(1 for kw in keywords if kw.lower() in row["Title"].lower())
        if score > best_score:
            best_score, best_row = score, row
    if best_row is not None:
        return (
            f"**Description:** {best_row['Description']}\n\n"
            f"**Steps:** {' | '.join(best_row['Steps'].splitlines())}\n\n"
            f"**Code Example:**\n```\n{best_row['Code Example']}\n```\n\n"
            f"**Time Complexity (Worst):** {best_row['Time Complexity (Worst)']}\n\n"
            f"**Space Complexity:** {best_row['Space Complexity']}"
        )
    return "No matching algorithm details found."

def retrieve_image_data_llm(topic):
    folder = "sorting_images"
    if not os.path.exists(folder): return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if topic.lower().replace(" ", "") in f.lower().replace("_", "").replace(" ", "")
    ])

def retrieve_video_data_llm(topic):
    keywords = extract_keywords(topic)
    best_row, best_score = None, 0
    for _, row in video_df.iterrows():
        score = sum(1 for kw in keywords if kw.lower() in row["Topic Name"].lower())
        if score > best_score:
            best_score, best_row = score, row
    return best_row["Link"] if best_row is not None and best_score > 0 else None

# ====================== STREAMLIT APP =========================
# Page setup
st.set_page_config(page_title="Sorting Algorithm Visualizer with Engagement", layout="wide")

# Title
st.title("🔢 Sorting Algorithm Visualizer with Engagement Detection")

# Initialize session state
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "camera_started" not in st.session_state:
    st.session_state.camera_started = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Dropdown
topic = st.selectbox("Select a Sorting Algorithm", [
    "Bubble Sort", "Insertion Sort", "Selection Sort",
    "Merge Sort", "Quick Sort", "Heap Sort", "Radix Sort"
])

# Extract keyword
keyword = topic.split()[0].lower()

# Load images (using the simpler method if LLM-based retrieval isn't immediately needed here)
image_folder = "sorting_images"
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if keyword in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
image_paths.sort()

# Load algorithm info (already loaded)
# Load videos info (already loaded)

# Camera Feed and Engagement Tracking
cap = cv2.VideoCapture(0)  # Access camera feed
content_type = "text"  # Start with text
last_switch_time = time.time()
not_engaged_dominant_since = None
switch_interval = 10  # Reduced for testing, set back to 120
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

# Streamlit UI columns setup
left_column, right_column = st.columns(2)

# Placeholder for camera feed display in the left column
placeholder = left_column.empty()

# Streamlit UI placeholders for dynamic content in the right column
text_placeholder = right_column.empty()
image_placeholder = right_column.empty()
video_placeholder = right_column.empty()

# LLM Chat Assistant (defined outside the conditional start)
with right_column:
    st.markdown("---")
    st.subheader(" LLM Chat Assistant")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    chat_input_key = "llm_chat_input"
    if user_prompt := st.chat_input("Ask a question about the algorithm...", key=chat_input_key):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            response = generate_response(user_prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

# Start button (remains outside the LLM chat definition)
if st.button("▶ Start Engagement Tracking"):
    st.session_state.run_camera = True
    st.session_state.camera_started = True
    # Display initial text content after the button is clicked
    algorithm_details = retrieve_text_data_llm(topic) # Using LLM for initial text
    with text_placeholder.container():
        st.subheader(f"📘 {topic}")
        st.markdown(algorithm_details)

while st.session_state.run_camera:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    height, width, _ = display_frame.shape
    engagement_this_frame = []
    results = det_model(display_frame)[0]
    person_detected = False

    for det in results.boxes:
        cls_id = int(det.cls[0])
        if det_model.names[cls_id] == "person":
            person_detected = True
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

    avg_engagement = np.mean(list(engagement_scores)) * 100 if engagement_scores and person_detected else 0

    cv2.putText(display_frame, f"Avg Engagement (Last {engagement_scores.maxlen} frames): {avg_engagement:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Content Switching Logic based on average engagement
    if person_detected and avg_engagement < 50:  # Only switch if a person is detected and not engaged
        if not_engaged_dominant_since is None:
            not_engaged_dominant_since = time.time()
        elif time.time() - not_engaged_dominant_since >= switch_interval:
            if content_type == "text":
                with image_placeholder.container():
                    st.subheader("🖼️ Visual Steps")
                    img_paths_llm = retrieve_image_data_llm(topic) # Use LLM for image retrieval
                    if img_paths_llm:
                        for img_path in img_paths_llm:
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
                    video_link_llm = retrieve_video_data_llm(topic) # Use LLM for video retrieval
                    if video_link_llm:
                        st.subheader("🎥 Video Tutorial")
                        st.video(video_link_llm)
                text_placeholder.empty()
                image_placeholder.empty()
                content_type = "video"
            elif content_type == "video":
                with text_placeholder.container():
                    algorithm_details_llm = retrieve_text_data_llm(topic) # Use LLM for text retrieval
                    st.subheader(f"📘 {topic}")
                    st.markdown(algorithm_details_llm)
                image_placeholder.empty()
                video_placeholder.empty()
                content_type = "text"
            not_engaged_dominant_since = None  # Reset timer
    elif person_detected and avg_engagement >= 50:
        not_engaged_dominant_since = None  # Reset the timer if engagement improves
    elif not person_detected:
        not_engaged_dominant_since = None # Reset if no person is detected

    # Display the camera feed with engagement annotations on the left
    frame_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    placeholder.image(frame_image, channels="RGB")

cap.release()