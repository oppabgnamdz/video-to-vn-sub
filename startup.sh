#!/bin/bash

# Kiểm tra nếu Docker đã cài đặt
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo "Lỗi: Docker hoặc Docker Compose chưa được cài đặt."
    echo "Vui lòng cài đặt bằng lệnh: apt update && apt install -y docker.io docker-compose"
    exit 1
fi

# Kiểm tra RAM
if command -v free &> /dev/null; then
    total_ram=$(free -m | awk '/^Mem:/{print $2}')
    echo "====================================="
    echo "Tổng RAM: ${total_ram}MB"
    
    # Đề xuất mô hình dựa trên RAM
    if [ "$total_ram" -lt 2048 ]; then
        suggested_model="tiny"
        echo "⚠️ RAM quá thấp, chỉ nên dùng mô hình tiny"
    elif [ "$total_ram" -lt 4096 ]; then
        suggested_model="base"
        echo "🔶 RAM trung bình, nên dùng mô hình base"
    elif [ "$total_ram" -lt 8192 ]; then
        suggested_model="small"
        echo "✅ RAM khá tốt, có thể dùng mô hình small"
    else
        suggested_model="medium"
        echo "🔷 RAM cao, có thể dùng mô hình medium"
    fi
    echo "====================================="
else
    suggested_model="small"
fi

# Chọn mô hình Whisper
echo "Chọn mô hình Whisper (mặc định: $suggested_model):"
echo "1. tiny   - Nhẹ nhất, yêu cầu ít RAM (~1GB)"
echo "2. base   - Nhẹ, độ chính xác tạm được (~2GB RAM)"
echo "3. small  - Cân bằng giữa chính xác và RAM (~4GB RAM)"
echo "4. medium - Chính xác nhất, cần nhiều RAM (~5GB RAM)"
read -p "> (1/2/3/4, Enter để dùng $suggested_model): " model_choice

case $model_choice in
    1) whisper_model="tiny" ;;
    2) whisper_model="base" ;;
    3) whisper_model="small" ;;
    4) whisper_model="medium" ;;
    *) whisper_model="$suggested_model" ;;
esac

echo "Đã chọn mô hình: $whisper_model"

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
echo "Mô hình Whisper: $whisper_model"
echo "====================================="

# Xác nhận
read -p "Xác nhận thông tin này? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Hủy bỏ"
    exit 0
fi

# Tạo file .env
echo "TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN" > .env
echo "WHISPER_MODEL=$whisper_model" >> .env
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