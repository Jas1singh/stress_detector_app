import streamlit as st
import cv2
from fer import FER
import numpy as np
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize FER detector
detector = FER(mtcnn=True)

# Deque to store stress history
max_points = 50
stress_history = deque([0]*max_points, maxlen=max_points)

st.title("Real-Time Stress Detector 💛")
st.write("Detect stress from facial expressions in real-time using your webcam.")

# Video transformer class for Streamlit
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = detector.detect_emotions(img)
        
        stress_score = 0
        dominant_emotion = "Neutral"
        if result:
            emotions = result[0]["emotions"]
            stress_score = emotions.get("angry", 0) + emotions.get("fear", 0) + emotions.get("sad", 0)
            dominant_emotion = max(emotions, key=emotions.get)
        
        # Add stress to history
        stress_history.append(stress_score)
        
        # Draw info on frame
        cv2.putText(img, f"{dominant_emotion} Stress: {stress_score:.2f}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        return img

# Start webcam stream
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Show stress graph
st.line_chart(list(stress_history))
