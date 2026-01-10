# Use official slim Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        wget \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy Python dependencies
# -----------------------------
COPY requirements.txt .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies (no cache)
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy the rest of the app
# -----------------------------
COPY . .

# -----------------------------
# Expose Streamlit port
# -----------------------------
EXPOSE 10000

# -----------------------------
# Run Streamlit app
# -----------------------------
CMD ["streamlit", "run", "stress_detector_app.py", "--server.port=10000", "--server.address=0.0.0.0", "--server.headless=true"]
