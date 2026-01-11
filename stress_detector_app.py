# -----------------------------
# stress_detector_app.py
# -----------------------------

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF logs

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import cv2
import numpy as np
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Real-Time Stress Detector",
    layout="centered"
)

st.title("Real-Time Stress Detector 💛")
st.write(
    "Detect stress from facial expressions using your webcam, an uploaded image, "
    "or a demo mode (no webcam required)."
)

# -----------------------------
# Cached FER detector (CRITICAL)
# -----------------------------
@st.cache_resource
def load_detector():
    from fer import FER
    return FER(mtcnn=False)  # stable in cloud environments

detector = load_detector()

# -----------------------------
# Stress history
# -----------------------------
MAX_POINTS = 50
if "stress_history" not in st.session_state:
    st.session_state.stress_history = deque([0] * MAX_POINTS, maxlen=MAX_POINTS)

# -----------------------------
# Helper: calculate stress
# -----------------------------
def calculate_stress(img):
    results = detector.detect_emotions(img)
    stress_score = 0
    dominant_emotion = "Neutral"

    if results:
        stress_values = []
        for face in results:
            emotions = face["emotions"]
            stress = (
                0.4 * emotions.get("angry", 0) +
                0.35 * emotions.get("fear", 0) +
                0.25 * emotions.get("sad", 0)
            )
            stress_values.append(stress)
            dominant_emotion = max(emotions, key=emotions.get)

        stress_score = np.mean(stress_values)
        st.session_state.stress_history.append(stress_score)  # only update if face detected

    # Smooth & normalize
    smooth_stress = int(np.mean(st.session_state.stress_history))
    stress_score = min(int(stress_score*100), 100) if results else 0

    # Stress level & color
    if smooth_stress > 70:
        level = "High Stress"
        color = (0, 0, 255)
    elif smooth_stress > 40:
        level = "Moderate Stress"
        color = (0, 165, 255)
    else:
        level = "Low Stress"
        color = (0, 255, 0)

    # Face status
    face_status = "Face detected" if results else "No face"

    # Overlay text
    if img.ndim == 3:
        cv2.putText(
            img,
            f"{dominant_emotion} | Stress: {smooth_stress}% ({level}) | {face_status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

    return img, smooth_stress, level, dominant_emotion

# -----------------------------
# Mode selection (cloud-friendly)
# -----------------------------
# default to Upload Image on cloud
default_mode = 1 if st.runtime.exists() else 0
st.sidebar.subheader("Mode Selection")
mode = st.sidebar.radio(
    "Choose input method:",
    ["Webcam (local only)", "Upload Image", "Demo Mode (No Webcam)"],
    index=default_mode
)

# -----------------------------
# Webcam mode
# -----------------------------
if mode == "Webcam (local only)":

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_count = 0

        def transform(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Process every 2nd frame (CPU-friendly)
            if self.frame_count % 2 == 0:
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
# Upload Image mode
# -----------------------------
elif mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        processed_img, stress, level, emotion = calculate_stress(img)

        st.image(
            cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
            caption=f"{level} | {stress}% | {emotion}"
        )
        st.success(f"Detected Stress: {stress}% ({level}) | Dominant Emotion: {emotion}")

# -----------------------------
# Demo Mode (No Webcam)
# -----------------------------
else:
    st.info("Demo Mode uses preloaded images so the app works without a webcam.")
    demo_choice = st.radio(
        "Select demo scenario:",
        ["Low Stress", "Moderate Stress", "High Stress"]
    )

    demo_images = {
        "Low Stress": "assets/calm.jpg",
        "Moderate Stress": "assets/moderate.jpg",
        "High Stress": "assets/stressed.jpg"
    }

    img = cv2.imread(demo_images[demo_choice])
    if img is None:
        st.error("Demo images missing in assets folder.")
    else:
        processed_img, stress, level, emotion = calculate_stress(img)
        st.image(
            cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
            caption=f"{level} | {stress}% | {emotion}"
        )
        st.success(f"Demo Result → Stress: {stress}% ({level}), Dominant Emotion: {emotion}")

# -----------------------------
# Stress trend chart
# -----------------------------
st.subheader("Stress Trend")
st.line_chart(list(st.session_state.stress_history))

# -----------------------------
# Disclaimer
# -----------------------------
st.markdown(
    "**Disclaimer:** This application estimates stress from facial expressions "
    "and is **not medical advice**."
)
