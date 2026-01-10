# -----------------------------
# stress_detector_app.py
# -----------------------------

import os
# -----------------------------
# Suppress TensorFlow and OpenCV logs
# -----------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TensorFlow INFO/WARNING logs

import streamlit as st
import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.ERROR)

import numpy as np
from fer import FER
from collections import deque
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    RTCConfiguration
)

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Real-Time Stress Detector",
    layout="centered"
)

st.title("Real-Time Stress Detector 💛")
st.write("Detect stress from facial expressions using your webcam or an uploaded image.")

# -----------------------------
# Initialize FER detector
# -----------------------------
detector = FER(mtcnn=False)  # more stable in headless/cloud environments

# -----------------------------
# Stress history (for graph)
# -----------------------------
MAX_POINTS = 50
if "stress_history" not in st.session_state:
    st.session_state.stress_history = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)

# -----------------------------
# Helper function: calculate stress
# -----------------------------
def calculate_stress(img):
    results = detector.detect_emotions(img)
    stress_score = 0
    dominant_emotion = "Neutral"

    if results:
        stress_values = []
        for face in results:
            emotions = face["emotions"]
            # Weighted stress score
            stress = 0.4*emotions["angry"] + 0.35*emotions["fear"] + 0.25*emotions["sad"]
            stress_values.append(stress)
            dominant_emotion = max(emotions, key=emotions.get)
        stress_score = np.mean(stress_values)

    # Normalize to 0-100
    stress_score = min(int(stress_score*100), 100)
    st.session_state.stress_history.append(stress_score)
    smooth_stress = int(np.mean(st.session_state.stress_history))

    # Stress level
    if smooth_stress > 70:
        level = "High Stress"
        color = (0,0,255)
    elif smooth_stress > 40:
        level = "Moderate Stress"
        color = (0,165,255)
    else:
        level = "Low Stress"
        color = (0,255,0)

    # Draw text on frame if image is colored
    if img.ndim == 3:
        cv2.putText(
            img,
            f"{dominant_emotion} | Stress: {smooth_stress}% ({level})",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    return img, smooth_stress, level, dominant_emotion

# -----------------------------
# Mode selection
# -----------------------------
st.sidebar.subheader("Mode selection")
mode = st.sidebar.radio("Choose input method:", ["Webcam (local only)", "Upload Image"])

# -----------------------------
# Webcam mode
# -----------------------------
if mode == "Webcam (local only)":
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img, _, _, _ = calculate_stress(img)
            return img

    webrtc_streamer(
        key="stress-detector",
        video_processor_factory=VideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# -----------------------------
# Upload Image mode (cloud)
# -----------------------------
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        processed_img, stress, level, emotion = calculate_stress(img)
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption=f"{level} | {stress}% | {emotion}")
        st.success(f"Detected Stress: {stress}% ({level}), Dominant Emotion: {emotion}")

# -----------------------------
# Stress trend chart
# -----------------------------
st.subheader("Stress Trend")
st.line_chart(list(st.session_state.stress_history))

st.markdown(
    "**Note:** This estimates stress from facial emotions and is not medical advice."
)
