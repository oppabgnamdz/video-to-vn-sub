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
RUN python -c "import openai; import moviepy; import whisper; import pysrt; import langdetect; import telegram; print('All required packages installed successfully')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p output/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Command to run the application (default entry point, can be overridden)
CMD ["python", "app/telegram_bot.py"]