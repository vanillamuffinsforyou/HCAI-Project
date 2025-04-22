import os
import cv2
import time
import av
import faiss
import pickle
import numpy as np
import streamlit as st
from av import VideoFrame
from dotenv import load_dotenv
from ultralytics import YOLO
from collections import deque
from sentence_transformers import SentenceTransformer
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("No API key found. Please set GOOGLE_API_KEY in your .env file.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

try:
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embeddings = np.load("embeddings.npy")
    index = faiss.read_index("faiss.index")
except Exception as e:
    st.error("Error loading preprocessed data. Have you run preprocess.py?")
    st.stop()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def generate_response(user_input):
    context = "\n\n".join(retrieve_relevant_chunks(user_input))
    prompt = (
        f"You are a helpful assistant for students. Here is the topic context from a PDF:\n\n{context}\n\n"
        f"User: {user_input}\n\nAI:"
    )
    response = model.generate_content(prompt)
    return response.text.strip()

det_model = YOLO("yolov8n.pt")
cls_model = YOLO("best.pt")
class_names = ["Engaged", "Not Engaged"]
engagement_scores = deque(maxlen=300)  # 10s at 30 FPS

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: VideoFrame) -> VideoFrame:
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
                pred_idx = cls_result[0].probs.top1
                cls_label = class_names[pred_idx]
                cls_conf = cls_result[0].probs.data[pred_idx].item()
                score = cls_conf if cls_label == "Engaged" else (1 - cls_conf)
                engagement_scores.append(score)

                color = (0, 255, 0) if cls_label == "Engaged" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text = f"{cls_label} ({cls_conf*100:.1f}%)"
                cv2.putText(img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except:
                continue

        if engagement_scores:
            avg_engagement = np.mean(engagement_scores) * 100
            cv2.putText(img, f"Avg Engagement (10s): {avg_engagement:.1f}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        return VideoFrame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="YOLO + LLM Assistant", layout="wide")
col1, col2 = st.columns(2)

# Webcam + Engagement Detection
with col1:
    st.header(" Real-time Engagement Detection")
    webrtc_streamer(
        key="engagement",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# LLM Chatbot
with col2:
    st.header(" LLM Chat Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_text = generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.markdown(response_text)
