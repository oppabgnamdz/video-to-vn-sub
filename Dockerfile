# Sử dụng multi-stage build để tối ưu
FROM python:3.9-slim AS builder

# Cài đặt chỉ các dependencies cần thiết để build
RUN apt-get update && apt-get install --no-install-recommends -y \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập môi trường
WORKDIR /build
COPY requirements.txt .

# Cài đặt dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage chính
FROM python:3.9-slim

# Cài đặt chỉ runtime dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    ffmpeg \
    flac \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập môi trường
WORKDIR /app

# Copy các dependencies từ builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy mã nguồn
COPY . .

# Tạo thư mục cần thiết
RUN mkdir -p output/temp

# Thiết lập biến môi trường
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Tải trước mô hình Whisper để tránh tải khi chạy
RUN python -c "import whisper; whisper.load_model('small')"

# Command mặc định
CMD ["python", "app/telegram_bot.py"]