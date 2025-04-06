#!/bin/bash

# Kiểm tra nếu Docker đã cài đặt
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo "Lỗi: Docker hoặc Docker Compose chưa được cài đặt."
    echo "Vui lòng cài đặt bằng lệnh: apt update && apt install -y docker.io docker-compose"
    exit 1
fi

# Nhập Telegram Bot Token
echo "Nhập Telegram Bot Token:"
read -p "> " TELEGRAM_BOT_TOKEN

if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "Lỗi: Token không được để trống"
    exit 1
fi

# Nhập OpenAI API Key (tùy chọn)
echo "Nhập OpenAI API Key (có thể bỏ qua nếu không sử dụng):"
read -p "> " OPENAI_API_KEY

# Hiển thị thông tin
echo "====================================="
echo "Token Bot Telegram: $TELEGRAM_BOT_TOKEN"
echo "OpenAI API Key: ${OPENAI_API_KEY:-Không sử dụng}"
echo "====================================="

# Xác nhận
read -p "Xác nhận thông tin này? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Hủy bỏ"
    exit 0
fi

# Tạo file .env
echo "TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN" > .env
if [ ! -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> .env
fi

echo "Đã tạo file .env"

# Kiểm tra xem Docker đang chạy chưa
if ! docker info &> /dev/null; then
    echo "Lỗi: Docker service chưa chạy"
    echo "Vui lòng chạy: systemctl start docker"
    exit 1
fi

# Build và chạy container
echo "Đang build và chạy container..."
docker-compose down 2>/dev/null
docker-compose build
docker-compose up -d

# Kiểm tra trạng thái
echo "====================================="
echo "Trạng thái container:"
docker-compose ps
echo "====================================="
echo "Xem logs bằng lệnh: docker-compose logs -f"
echo "====================================="
echo "Bot Telegram đã sẵn sàng sử dụng!"
echo "=====================================" 