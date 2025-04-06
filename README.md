# Video to Subtitles Converter (Telegram Bot)

Ứng dụng chuyển đổi video thành phụ đề SRT tự động qua Telegram Bot, sử dụng Whisper - mô hình AI nhận dạng giọng nói hiệu quả.

## Tính năng

- Trích xuất phụ đề từ file video qua Telegram
- Hỗ trợ nhiều nguồn video: upload qua Telegram, URL, M3U8 streaming
- Sử dụng Whisper với mô hình medium để nhận dạng giọng nói và tạo phụ đề chính xác
- Tự động phát hiện ngôn ngữ
- Xuất phụ đề dạng SRT

## Yêu cầu

- Python 3.8+
- ffmpeg
- Token Telegram Bot (từ BotFather)
- Thư viện Python (xem requirements.txt)

## Cài đặt

1. Clone repository:

   ```
   git clone <repository-url>
   cd video-to-vn-sub
   ```

2. Cài đặt dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Đảm bảo đã cài đặt ffmpeg:

   ```
   apt-get update && apt-get install -y ffmpeg
   ```

4. Tạo Bot Telegram:
   - Liên hệ [@BotFather](https://t.me/BotFather) trên Telegram
   - Tạo bot mới với lệnh `/newbot`
   - Lưu lại token được cung cấp

## Cách sử dụng

### Chạy trực tiếp

```bash
# Thiết lập token bot
export TELEGRAM_BOT_TOKEN=your_token_here

# Khởi động bot
python app/telegram_bot.py
```

### Sử dụng Docker

1. Tạo file `.env` với nội dung:

   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   OPENAI_API_KEY=your_openai_key_here   # Nếu cần
   ```

2. Build và chạy:
   ```
   docker-compose up -d
   ```

## Sử dụng Bot

1. Tìm bot của bạn trên Telegram theo username đã đăng ký
2. Bắt đầu chat với `/start`
3. Gửi video hoặc URL video để bot tạo phụ đề
4. Bot sẽ xử lý và trả về file SRT

## Triển khai

Để triển khai trên VPS, chỉ cần vài bước đơn giản:

```bash
# Cài Docker và Docker Compose
apt update
apt install -y docker.io docker-compose

# Clone repository
git clone <repository-url>
cd video-to-vn-sub

# Tạo file .env với token bot
echo "TELEGRAM_BOT_TOKEN=your_token_here" > .env

# Build và khởi động dịch vụ
docker-compose up -d
```

## Lưu ý

- Mô hình Whisper medium cần khoảng 5GB RAM khi chạy
- Thời gian xử lý phụ thuộc vào độ dài của video
- Telegram giới hạn kích thước file upload là 50MB
- Để đạt hiệu suất cao nhất, nên chạy trên máy có GPU
