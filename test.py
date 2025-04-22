import os
import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
import av
from collections import deque
from transformers import pipeline
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -------------------------------------
# CONFIGURATION & GLOBAL VARIABLES
# -------------------------------------
# Set content mode: "text", "images", or "videos"
content_mode = "text"  # Change this variable to switch between modes

# Replace with your actual Llama 3.2 3B model ID
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  

# Initialize the text-generation pipeline using Llama 3.2 3B for both extraction and chat
llama_generator = pipeline(
    "text-generation", 
    model=LLAMA_MODEL_ID,
    token="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",
    trust_remote_code=True  # If needed for custom models
)

def llama_generate(prompt: str, max_new_tokens: int = 64, temperature: float = 0.7) -> str:
    result = llama_generator(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    return result[0]['generated_text'].strip()

# -------------------------------------
# DYNAMIC CONTENT FUNCTIONS
# -------------------------------------
def extract_keywords(query: str) -> list:
    prompt = (f"Extract relevant keywords for the algorithm topic '{query}'. "
              f"Return them as a comma-separated list.")
    result = llama_generate(prompt, max_new_tokens=64, temperature=0.3)
    keywords = [kw.strip() for kw in result.split(",") if kw.strip()]
    return keywords

# Load algorithm info CSV (fallback to sample data if missing)
try:
    algo_df = pd.read_csv("algorithm_info.csv")
except Exception:
    algo_df = pd.DataFrame({
        "Title": ["Radix Sort"],
        "Description": ["Sorts numbers by processing individual digits."],
        "Steps": ["1. Group elements\n2. Sort each group"],
        "Code Example": ["def radix_sort(arr):\n    pass"],
        "Time Complexity (Worst)": ["O(nk)"],
        "Space Complexity": ["O(n+k)"]
    })

# Load video information CSV (fallback sample)
try:
    video_df = pd.read_csv("videos.csv")
except Exception:
    video_df = pd.DataFrame({
        "Topic": ["Radix Sort"],
        "VideoLink": ["https://www.youtube.com/embed/dQw4w9WgXcQ"]
    })

def retrieve_text_data(topic: str) -> str:
    keywords = extract_keywords(topic)
    best_row, best_score = None, 0
    for _, row in algo_df.iterrows():
        score = sum(1 for kw in keywords if kw.lower() in row["Title"].lower())
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is not None and best_score > 0:
        content = (
            f"**Description:** {best_row['Description']}\n\n"
            f"**Steps:** {' | '.join(best_row['Steps'].splitlines())}\n\n"
            f"**Code Example:**\n```\n{best_row['Code Example']}\n```\n\n"
            f"**Time Complexity (Worst):** {best_row['Time Complexity (Worst)']}\n\n"
            f"**Space Complexity:** {best_row['Space Complexity']}"
        )
        return content
    return "No matching algorithm details found."

def retrieve_image_data(topic: str) -> list:
    folder = "images"
    if not os.path.exists(folder): 
        return []
    # Retrieve and sort images whose filenames contain the topic name
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if topic.lower() in f.lower()]
    )
    return files

def retrieve_video_data(topic: str) -> str:
    keywords = extract_keywords(topic)
    best_row, best_score = None, 0
    for _, row in video_df.iterrows():
        score = sum(1 for kw in keywords if kw.lower() in row["Topic"].lower())
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is not None and best_score > 0:
        return best_row["VideoLink"]
    return None

def update_dynamic_box(topic: str):
    if content_mode == "text":
        st.session_state.dynamic_box.markdown(retrieve_text_data(topic))
    elif content_mode == "images":
        imgs = retrieve_image_data(topic)
        if imgs:
            st.session_state.dynamic_box.image(imgs, width=300)
        else:
            st.session_state.dynamic_box.markdown("No images found for this topic.")
    elif content_mode == "videos":
        link = retrieve_video_data(topic)
        if link:
            st.session_state.dynamic_box.video(link)
        else:
            st.session_state.dynamic_box.markdown("No videos found for this topic.")
    else:
        st.session_state.dynamic_box.markdown("Invalid content mode selected.")

def generate_response(prompt: str) -> str:
    full_prompt = f"You are a helpful assistant for students.\nUser: {prompt}\nAssistant:"
    response_text = llama_generate(full_prompt, max_new_tokens=200, temperature=0.7)
    return response_text

# -------------------------------------
# REAL-TIME ENGAGEMENT DETECTION SETUP
# -------------------------------------
# Initialize YOLO models; adjust model paths as needed
det_model = YOLO("yolov8n.pt")
cls_model = YOLO("best.pt")
class_names = ["Engaged", "Not Engaged"]
engagement_scores = deque(maxlen=300)  # Approximately 10 seconds at 30 FPS

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = det_model(img)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = det_model.names[cls_id]
            if label != "person":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img[y1:y2, x1:x2]
            try:
                resized = cv2.resize(cropped, (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                cls_result = cls_model(rgb)
                pred_idx = int(cls_result[0].probs.top1)
                cls_label = class_names[pred_idx]
                cls_conf = cls_result[0].probs.data[pred_idx].item()
                score = cls_conf if cls_label == "Engaged" else (1 - cls_conf)
                engagement_scores.append(score)
                color = (0, 255, 0) if cls_label == "Engaged" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text = f"{cls_label} ({cls_conf*100:.1f}%)"
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception:
                continue
        if engagement_scores:
            avg_engagement = np.mean(engagement_scores) * 100
            cv2.putText(img, f"Avg Engagement (10s): {avg_engagement:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------------
# STREAMLIT LAYOUT & APP
# -------------------------------------
st.set_page_config(page_title="YOLO + LLM Assistant", layout="wide")
col1, col2 = st.columns(2)

# Left column: Real-time Engagement Detection
with col1:
    st.header("Real-time Engagement Detection")
    webrtc_streamer(
        key="engagement",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# Right column: Dynamic Content and LLM Chat Assistant
with col2:
    st.header("Dynamic Content")
    if "dynamic_box" not in st.session_state:
        st.session_state.dynamic_box = st.empty()
    # Input for topic; change the default value as needed
    topic = st.text_input("Enter topic for dynamic content", value="Radix Sort")
    update_dynamic_box(topic)

    st.markdown("---")
    st.header("LLM Chat Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Chat input and response generation
    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response_text = generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.markdown(response_text)
