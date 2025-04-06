import os
import tempfile
import logging
from pathlib import Path
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from config import OUTPUT_DIR, TEMP_DIR
from utils.file_utils import ensure_directories, delete_file
from services.video_processor import VideoProcessor
from models.data_models import ProcessingResult

# Thiết lập logging
logger = logging.getLogger("TelegramBot")


class VideoToSubBot:
    def __init__(self, token):
        self.token = token
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.setup_handlers()
        ensure_directories(OUTPUT_DIR, TEMP_DIR)
        self.video_processor = VideoProcessor(OUTPUT_DIR)

    def setup_handlers(self):
        # Đăng ký các command
        self.dispatcher.add_handler(
            CommandHandler("start", self.start_command))
        self.dispatcher.add_handler(CommandHandler("help", self.help_command))

        # Xử lý video
        self.dispatcher.add_handler(MessageHandler(
            Filters.video | Filters.document.video, self.handle_video))

        # Xử lý url
        self.dispatcher.add_handler(MessageHandler(
            Filters.text & ~Filters.command, self.handle_url))

        # Xử lý lỗi
        self.dispatcher.add_error_handler(self.error_handler)

    def start_command(self, update: Update, context: CallbackContext):
        update.message.reply_text(
            "Chào mừng đến với Video to Subtitle Bot!\n\n"
            "Gửi video hoặc URL video để tôi tạo phụ đề SRT.\n"
            "Bạn có thể gửi video trực tiếp hoặc URL video/m3u8.\n\n"
            "Gõ /help để xem trợ giúp."
        )

    def help_command(self, update: Update, context: CallbackContext):
        update.message.reply_text(
            "📋 *Hướng dẫn sử dụng* 📋\n\n"
            "*1. Gửi video:*\n"
            "- Tải lên video từ thiết bị của bạn\n\n"
            "*2. Gửi URL:*\n"
            "- URL video trực tiếp (mp4, avi,...)\n"
            "- URL m3u8 (playlist stream)\n\n"
            "Bot sẽ xử lý video và trả về file phụ đề .SRT\n"
            "Ngôn ngữ sẽ được tự động phát hiện.",
            parse_mode=ParseMode.MARKDOWN
        )

    def error_handler(self, update: Update, context: CallbackContext):
        logger.error(f"Lỗi: {context.error} - Update: {update}")
        if update and update.effective_message:
            update.effective_message.reply_text(
                "Đã xảy ra lỗi, vui lòng thử lại sau.")

    def progress_callback(self, percentage, message, update: Update):
        # Chỉ cập nhật khi tỷ lệ % thay đổi đáng kể để tránh spam
        if percentage % 10 == 0 or percentage in [25, 45, 75, 95]:
            update.effective_message.reply_text(f"🔄 {message}")

    def handle_video(self, update: Update, context: CallbackContext):
        message = update.message
        message.reply_text("🎬 Đã nhận video, đang xử lý...")

        # Lấy file từ message
        if message.video:
            file = message.video.get_file()
            file_name = message.video.file_name if message.video.file_name else "video.mp4"
        else:  # Document
            file = message.document.get_file()
            file_name = message.document.file_name if message.document.file_name else "video.mp4"

        # Tạo tên file tạm
        temp_file = TEMP_DIR / f"telegram_{message.chat_id}_{file_name}"

        try:
            # Tải file về
            file.download(custom_path=temp_file)
            message.reply_text(
                "📥 Đã tải video xong, đang trích xuất phụ đề...")

            # Tạo wrapper cho file để tương thích với VideoProcessor
            class UploadedFile:
                def __init__(self, path):
                    self.path = Path(path)
                    self.name = self.path.name
                    self.size = self.path.stat().st_size

                def getbuffer(self):
                    return self.path.read_bytes()

            uploaded_file = UploadedFile(temp_file)

            # Xử lý video
            def update_progress_wrapper(percentage, msg):
                self.progress_callback(percentage, msg, update)

            result = self.video_processor.process_video(
                "upload", uploaded_file, update_progress_wrapper)

            if result.success:
                # Gửi file SRT
                srt_file = Path(result.srt_path)
                with open(srt_file, 'rb') as f:
                    message.reply_document(
                        document=f,
                        filename=f"{file_name.rsplit('.', 1)[0]}.srt",
                        caption=f"✅ Đã tạo phụ đề thành công!\nNgôn ngữ phát hiện: {result.detected_language}"
                    )
            else:
                message.reply_text(f"❌ Lỗi: {result.error_message}")

        except Exception as e:
            logger.error(f"Lỗi khi xử lý video: {str(e)}")
            message.reply_text(f"❌ Đã xảy ra lỗi: {str(e)}")
        finally:
            # Dọn dẹp file
            delete_file(temp_file)

    def handle_url(self, update: Update, context: CallbackContext):
        message = update.message
        url = message.text.strip()

        # Kiểm tra URL
        if not (url.startswith('http://') or url.startswith('https://')):
            message.reply_text(
                "❌ URL không hợp lệ. Vui lòng gửi URL bắt đầu bằng http:// hoặc https://")
            return

        message.reply_text("🔍 Đã nhận URL, đang xử lý...")

        # Xác định loại URL
        source_type = "m3u8" if url.endswith('.m3u8') else "url"

        try:
            # Xử lý video
            def update_progress_wrapper(percentage, msg):
                self.progress_callback(percentage, msg, update)

            result = self.video_processor.process_video(
                source_type, url, update_progress_wrapper)

            if result.success:
                # Gửi file SRT
                srt_file = Path(result.srt_path)
                with open(srt_file, 'rb') as f:
                    message.reply_document(
                        document=f,
                        filename=f"subtitles_{source_type}.srt",
                        caption=f"✅ Đã tạo phụ đề thành công!\nNgôn ngữ phát hiện: {result.detected_language}"
                    )
            else:
                message.reply_text(f"❌ Lỗi: {result.error_message}")

        except Exception as e:
            logger.error(f"Lỗi khi xử lý URL: {str(e)}")
            message.reply_text(f"❌ Đã xảy ra lỗi: {str(e)}")

    def run(self):
        # Khởi động bot
        self.updater.start_polling()
        logger.info("Bot đã khởi động và đang lắng nghe...")
        self.updater.idle()


def main():
    # Lấy token từ biến môi trường
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("Không tìm thấy TELEGRAM_BOT_TOKEN trong biến môi trường")
        return

    # Khởi tạo và chạy bot
    bot = VideoToSubBot(token)
    bot.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()
