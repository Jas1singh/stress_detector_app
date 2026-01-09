#!/bin/bash
# Install ffmpeg (required for streamlit-webrtc)
apt-get update
apt-get install -y ffmpeg

pip install --upgrade pip setuptools wheel