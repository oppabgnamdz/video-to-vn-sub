import streamlit as st
import os
from datetime import datetime
from pathlib import Path

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
        # st.set_page_config(
        #     page_title="🎬 Video Subtitle Translator",
        #     page_icon="🌍",
        #     layout="wide"
        # )

        # Sidebar - Cài đặt và thông tin cache
        with st.sidebar:
            st.title("⚙️ Cài đặt")
            st.info("🎥 Ứng dụng chuyển đổi video thành phụ đề và dịch tự động!")

            # Chọn dịch vụ dịch
            translation_service = st.radio(
                "Chọn dịch vụ dịch:",
                ["🌟 OpenAI (Chất lượng cao)", "🆓 Google Translate (Miễn phí)"]
            )
            use_openai = "OpenAI" in translation_service

            openai_key = ""
            if use_openai:
                openai_key = st.text_input(
                    '🔑 OpenAI API Key:', type='password')
                if not openai_key:
                    st.warning(
                        "⚠️ Vui lòng nhập OpenAI API Key để sử dụng dịch vụ OpenAI")

            # Xóa cache
            if st.button("🗑️ Xóa Cache"):
                self.cache_service.clear()
                st.success("✅ Đã xóa cache!")

        # Kiểm tra kết nối Internet
        if not check_internet():
            return

        # Layout chính
        hide_streamlit_style = ''' <style>
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>'''
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        # st.title("🎬 Video Subtitle Translator")
        # st.subheader("Chuyển đổi video sang phụ đề và dịch tự động 🌍")

        # Tabs lựa chọn
        # tab1, tab2 = st.tabs(["📂 Tải file", "🔧 Cài đặt"])

        # with tab1:
        # col1, col2 = st.columns([2, 1])

        # with col1:
        uploaded_file = st.file_uploader(
            "📤 Chọn file video",
            type=['mp4', 'avi', 'mkv', 'mov']
        )
        url = st.text_input("🔗 Hoặc nhập URL video:")
        uploaded_srt = st.file_uploader(
            "📄 Hoặc tải lên file phụ đề SRT",
            type=['srt']
        )

        # Kiểm tra có input hợp lệ không
        has_input = uploaded_file or (
            url and url.strip()) or uploaded_srt

        # Chọn phong cách dịch
        translation_style = st.selectbox(
            "✍️ Phong cách dịch:",
            ["📖 Trang trọng", "💬 Thông dụng"]
        )

        # Nút xử lý
        can_process = has_input and (
            not use_openai or (use_openai and openai_key))
        process_button = st.button(
            "🚀 Bắt đầu xử lý",
            disabled=not can_process or st.session_state.get(
                "is_processing", False),
            key="process_button"
        )

        if process_button:
            st.session_state["is_processing"] = True
            self._process_request(
                openai_key if use_openai else None,
                uploaded_file,
                url,
                uploaded_srt,
                translation_style,
                use_openai
            )
            st.session_state["is_processing"] = False

        # Hiển thị lịch sử xử lý
        self._display_history()

    def _process_request(self, api_key, uploaded_file, url, uploaded_srt, intensity, use_openai):
        try:
            st.subheader("🔄 Đang xử lý...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress, message):
                progress_bar.progress(progress / 100)
                status_text.text(message)

            update_progress(10, "📥 Đang chuẩn bị dữ liệu...")

            # Khởi tạo translator
            translator = (
                OpenAITranslator(self.cache_service,
                                 self.stats, api_key, intensity)
                if use_openai else
                GoogleTranslator(self.cache_service, self.stats)
            )

            if uploaded_srt:
                self._process_srt_file(
                    uploaded_srt, translator, update_progress)
            else:
                self._process_video_file(
                    uploaded_file, url, translator, update_progress)

        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

    def _process_srt_file(self, uploaded_srt, translator, progress_callback):
        progress_callback(30, "📜 Đang xử lý file phụ đề...")

        srt_path = get_unique_filename(OUTPUT_DIR, "original", "srt")

        if save_uploaded_file(uploaded_srt, srt_path):
            progress_callback(50, "🎯 Đã xử lý phụ đề. Bắt đầu dịch...")

            translated_srt = translator.translate_srt(
                str(srt_path),
                source_language=None,
                progress_callback=progress_callback
            )

            if translated_srt:
                self._handle_success(
                    srt_path, translated_srt, uploaded_srt.name, None)

    def _process_video_file(self, uploaded_file, url, translator, progress_callback):
        progress_callback(15, "🎬 Đang xử lý video...")

        source_type = "upload" if uploaded_file else "url"
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

    def _handle_success(self, original_srt, translated_srt, source_name, detected_language):
        st.success("✅ Hoàn thành! Tải file phụ đề:")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button("⬇️ Tải phụ đề gốc", open(
                original_srt, 'rb'), file_name="original.srt")

        with col2:
            st.download_button("⬇️ Tải phụ đề đã dịch", open(
                translated_srt, 'rb'), file_name="translated.srt")

    def _display_history(self):
        st.subheader("📋 Lịch sử xử lý")

        if "processed_files" in st.session_state and st.session_state["processed_files"]:
            for file in reversed(st.session_state["processed_files"]):
                with st.expander(f"{file['source']} - {file['timestamp']}"):
                    st.code(open(file['translated_srt']
                                 ).read(), language='text')
        else:
            st.info("📭 Chưa có lịch sử xử lý.")
