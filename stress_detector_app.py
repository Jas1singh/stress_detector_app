import streamlit as st
import cv2
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
st.write("Detect stress from facial expressions using your webcam.")

# -----------------------------
# Initialize FER detector
# -----------------------------
# mtcnn=False is more stable in Docker / headless environment
detector = FER(mtcnn=False)

# -----------------------------
# Stress history
# -----------------------------
MAX_POINTS = 50
if "stress_history" not in st.session_state:
    st.session_state.stress_history = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)

# -----------------------------
# WebRTC config
# -----------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------
# Video Transformer
# -----------------------------
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = detector.detect_emotions(img)

        stress_score = 0
        dominant_emotion = "Neutral"

        if results:
            stress_values = []
            for face in results:
                emotions = face["emotions"]
                # Weighted stress calculation
                stress = 0.4*emotions["angry"] + 0.35*emotions["fear"] + 0.25*emotions["sad"]
                stress_values.append(stress)
                dominant_emotion = max(emotions, key=emotions.get)
            stress_score = np.mean(stress_values)

        # Normalize and smooth
        stress_score = min(int(stress_score*100), 100)
        history = st.session_state.stress_history
        history.append(stress_score)
        smooth_stress = int(np.mean(history))

        # Determine stress level
        if smooth_stress > 70:
            level = "High Stress"
            color = (0,0,255)
        elif smooth_stress > 40:
            level = "Moderate Stress"
            color = (0,165,255)
        else:
            level = "Low Stress"
            color = (0,255,0)

        # Draw text on video
        cv2.putText(
            img,
            f"{dominant_emotion} | Stress: {smooth_stress}% ({level})",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        return img

# -----------------------------
# Start WebRTC webcam stream
# -----------------------------
webrtc_streamer(
    key="stress-detector",
    video_processor_factory=VideoTransformer,  # updated for v0.47+
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# -----------------------------
# Display stress trend chart
# -----------------------------
st.subheader("Stress Trend")
st.line_chart(list(st.session_state.stress_history))

st.markdown(
    "**Note:** This estimates stress from facial emotions and is not medical advice."
)
