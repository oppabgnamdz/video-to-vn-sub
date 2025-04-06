#!/usr/bin/env python3
import sys
import logging
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from config import OUTPUT_DIR, TEMP_DIR
from utils.file_utils import ensure_directories
from services.video_processor import VideoProcessor
from models.data_models import ProcessingResult


def setup_logging():
    """Thiết lập logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def progress_callback(percentage, message):
    """Callback để hiển thị tiến trình"""
    tqdm.write(f"{message}")


def process_video(source, source_type):
    """Xử lý video và tạo phụ đề"""
    processor = VideoProcessor(OUTPUT_DIR)

    with tqdm(total=100, desc="Xử lý video") as pbar:
        def update_progress(percentage, message):
            pbar.n = percentage
            pbar.set_description(message)
            pbar.refresh()

        result = processor.process_video(source_type, source, update_progress)

    if result.success:
        print(f"\nXử lý thành công! Phụ đề đã được tạo tại: {result.srt_path}")
        print(f"Ngôn ngữ phát hiện: {result.detected_language}")
        return result
    else:
        print(f"\nXử lý thất bại: {result.error_message}")
        return None


def main():
    """Hàm chính của chương trình"""
    parser = argparse.ArgumentParser(
        description='Chuyển đổi video sang phụ đề')
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('-f', '--file', type=str,
                              help='Đường dẫn đến file video')
    source_group.add_argument('-u', '--url', type=str, help='URL của video')
    source_group.add_argument('-m', '--m3u8', type=str,
                              help='URL của M3U8 playlist')

    args = parser.parse_args()

    # Đảm bảo thư mục tồn tại
    ensure_directories(OUTPUT_DIR, TEMP_DIR)

    # Xác định nguồn video và loại
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Lỗi: File {args.file} không tồn tại")
            return 1

        # Tạo class giả lập UploadedFile để tương thích với API cũ
        class UploadedFile:
            def __init__(self, path):
                self.path = Path(path)
                self.name = self.path.name
                self.size = self.path.stat().st_size

            def getbuffer(self):
                return self.path.read_bytes()

        source = UploadedFile(args.file)
        source_type = "upload"
    elif args.url:
        source = args.url
        source_type = "url"
    elif args.m3u8:
        source = args.m3u8
        source_type = "m3u8"

    # Xử lý video
    result = process_video(source, source_type)
    return 0 if result and result.success else 1


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
