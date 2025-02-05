import streamlit as st
import os
from datetime import datetime
from pathlib import Path
from services.telegram_service import TelegramService

from config import OUTPUT_DIR, TEMP_DIR
from models.data_models import TranslationStats
from services.cache_service import CacheService
from services.video_processor import VideoProcessor
from services.translation.google_translator import GoogleTranslator
from services.translation.openai_translator import OpenAITranslator
from utils.internet_utils import check_internet
from utils.file_utils import (
    validate_file_size,
    save_uploaded_file,
    cleanup_old_files,
    get_unique_filename
)


class StreamlitApp:
    def __init__(self):
        self.cache_service = CacheService()
        self.stats = TranslationStats()
        self.video_processor = VideoProcessor(OUTPUT_DIR)

    def run(self):
        """Khởi chạy ứng dụng Streamlit"""

        # Ẩn footer và menu của Streamlit
        hide_streamlit_style = ''' 
        <style>
            footer {visibility: hidden;}
            #MainMenu {visibility: hidden;}
        </style>'''
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

        # Sidebar - Cài đặt
        with st.sidebar:
            st.title("⚙️ Cài đặt")
            st.info("🎥 Ứng dụng chuyển đổi video thành phụ đề và dịch tự động!")

            openai_key = st.text_input(
                '🔑 OpenAI API Key (để sử dụng OpenAI)', type='password'
            )

            if openai_key.strip() == "abc":
                openai_key = os.getenv("OPENAI_API_KEY", "")
            translation_style = st.selectbox(
                "✍️ Phong cách dịch:",
                ["📖 Trang trọng", "💬 Thông dụng"]
            )
            if st.button("🗑️ Xóa Cache"):
                self.cache_service.clear()
                st.success("✅ Đã xóa cache!")

        if not check_internet():
            return

        uploaded_file = st.file_uploader(
            "📤 Chọn file (video hoặc phụ đề)",
            type=['mp4', 'avi', 'mkv', 'mov', 'srt']
        )

        # Tạo tabs cho các loại input khác nhau
        tab1, tab2 = st.tabs(["🔗 URL Thông thường", "📺 M3U8 Stream"])

        with tab1:
            direct_url = st.text_input("🔗 Nhập URL video trực tiếp:")
            url_to_process = direct_url if direct_url else None
            source_type = "url" if direct_url else None

        with tab2:
            m3u8_url = st.text_input("📺 Nhập URL M3U8:")
            if m3u8_url:
                url_to_process = m3u8_url
                source_type = "m3u8"

        # Kiểm tra có input hợp lệ không
        has_input = uploaded_file or (
            url_to_process and url_to_process.strip())

        process_button = st.button(
            "🚀 Bắt đầu xử lý",
            disabled=not has_input or st.session_state.get(
                "is_processing", False),
            key="process_button"
        )

        if process_button:
            st.session_state["is_processing"] = True

            use_openai = bool(openai_key.strip())
            translate_service = "🌟 OpenAI (Chất lượng cao)" if use_openai else "🆓 Google Translate (Miễn phí)"

            st.info(
                f"🔍 Đang sử dụng **{translate_service}** để dịch phụ đề...")

            self._process_request(
                openai_key if openai_key else None,
                uploaded_file,
                url_to_process,
                source_type,
                translation_style
            )

            st.session_state["is_processing"] = False

        self._display_history()

    def _process_request(self, api_key, uploaded_file, url, source_type, intensity):
        try:
            st.subheader("🔄 Đang xử lý...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress, message):
                progress_bar.progress(progress / 100)
                status_text.text(message)

            update_progress(10, "📥 Đang chuẩn bị dữ liệu...")

            use_openai = api_key is not None
            translator = (
                OpenAITranslator(self.cache_service,
                                 self.stats, api_key, intensity)
                if use_openai else
                GoogleTranslator(self.cache_service, self.stats)
            )

            if uploaded_file:
                file_extension = uploaded_file.name.split(".")[-1]
                if file_extension.lower() == "srt":
                    self._process_srt_file(
                        uploaded_file, translator, update_progress)
                else:
                    self._process_video_file(
                        uploaded_file, None, "upload", translator, update_progress)

            elif url and url.strip():
                self._process_video_file(
                    None, url, source_type, translator, update_progress)

        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

    # Các phương thức khác giữ nguyên...
    def _process_video_file(self, uploaded_file, url, source_type, translator, progress_callback):
        progress_callback(15, "🎬 Đang xử lý video...")

        source_data = uploaded_file if uploaded_file else url

        result = self.video_processor.process_video(
            source_type, source_data, progress_callback
        )

        if result.success and result.srt_path:
            translated_srt = translator.translate_srt(
                result.srt_path,
                source_language=result.detected_language,
                progress_callback=progress_callback
            )

            if translated_srt:
                self._handle_success(
                    result.srt_path, translated_srt, result.source_name, result.detected_language)

    def _display_history(self):
        """Hiển thị lịch sử xử lý"""
        st.subheader("📋 Lịch sử xử lý")

        if "processed_files" in st.session_state and st.session_state["processed_files"]:
            for file in reversed(st.session_state["processed_files"]):
                with st.expander(f"{file['source']} - {file['timestamp']}"):
                    st.code(open(file['translated_srt']
                                 ).read(), language='text')
        else:
            st.info("📭 Chưa có lịch sử xử lý.")

    def _handle_success(self, original_srt, translated_srt, source_name, detected_language):
        """Xử lý khi hoàn thành việc xử lý video và dịch phụ đề"""
        st.success("✅ Hoàn thành! Tải file phụ đề:")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button("⬇️ Tải phụ đề gốc", open(
                original_srt, 'rb'), file_name="original.srt")

        with col2:
            st.download_button("⬇️ Tải phụ đề đã dịch", open(
                translated_srt, 'rb'), file_name="translated.srt")

        # Lưu vào lịch sử
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = []

        st.session_state["processed_files"].append({
            'source': source_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'translated_srt': translated_srt,
            'language': detected_language
        })

        # Giới hạn lịch sử lưu trữ
        if len(st.session_state["processed_files"]) > 10:
            st.session_state["processed_files"] = st.session_state["processed_files"][-10:]

        # Gửi file phụ đề đến Telegram nếu có cấu hình
        bot_token = os.getenv("BOT_TOKEN", "")
        chat_id = os.getenv("CHAT_ID", "")
        if bot_token and chat_id:
            telegram_service = TelegramService(bot_token, chat_id)
            short_source_name = (
                source_name[:10] + "...") if len(source_name) > 10 else source_name
            telegram_service.send_file(
                translated_srt, f"📄 Phụ đề tiếng Việt cho {short_source_name}")
