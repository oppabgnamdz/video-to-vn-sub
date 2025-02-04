# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libportaudio2 \
    python3-dev \
    build-essential \
    portaudio19-dev \
    gcc \
    git \
    flac \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Verify installations
RUN python -c "import streamlit; import openai; import moviepy; import speech_recognition; import pysrt; import langdetect; print('All required packages installed successfully')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 8501

# Command to run the application
CMD ["python", "-m", "streamlit", "run", "app/main.py"]