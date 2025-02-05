import streamlit as st
import os
from datetime import datetime
from typing import Optional, Callable
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
        if not check_internet():
            return

        # Chọn dịch vụ dịch
        translation_service = st.radio(
            "Chọn dịch vụ dịch:",
            ["OpenAI (Chất lượng cao)", "Google Translate (Miễn phí)"]
        )
        use_openai = translation_service == "OpenAI (Chất lượng cao)"

        # Xử lý OpenAI API key nếu cần
        openai_key = ""
        if use_openai:
            openai_key = st.text_input('OpenAI API Key:', type='password')
            if openai_key == 'abc':
                openai_key = os.getenv('OPENAI_API_KEY', '')
            if not openai_key:
                st.warning(
                    "⚠️ Vui lòng nhập OpenAI API Key để sử dụng dịch vụ OpenAI")

        # Tabs cho các loại input
        tab1, tab2, tab3, tab4 = st.tabs([
            "Tải Video", "URL Video", "Tải File Phụ Đề", "⚙️ Cài đặt"
        ])

        with tab1:
            uploaded_file = st.file_uploader(
                "Chọn file video",
                type=['mp4', 'avi', 'mkv', 'mov']
            )
            if uploaded_file:
                st.info(
                    f"Dung lượng file: {uploaded_file.size / 1024 / 1024:.1f}MB")

        with tab2:
            url = st.text_input('Nhập URL video:')

        with tab3:
            uploaded_srt = st.file_uploader(
                "Chọn file phụ đề SRT", type=['srt'])

        with tab4:
            st.header("⚙️ Cài đặt")

            # Hiển thị thông tin cache
            st.subheader("Cache và Bộ nhớ")
            cache_size = len(self.cache_service._cache)
            st.info(f"Số lượng bản dịch trong cache: {cache_size}")

            # Nút xóa cache
            if st.button("🗑️ Xóa Cache"):
                try:
                    self.cache_service.clear()
                    st.session_state.processed_files = []
                    st.success("✅ Đã xóa cache và lịch sử!")
                except Exception as e:
                    st.error(f"Lỗi khi xóa cache: {str(e)}")

            # Tùy chọn xóa riêng
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Chỉ xóa Cache"):
                    self.cache_service.clear()
                    st.success("✅ Đã xóa cache!")
            with col2:
                if st.button("🗑️ Chỉ xóa Lịch sử"):
                    st.session_state.processed_files = []
                    st.success("✅ Đã xóa lịch sử!")

            # Hiển thị thông tin bổ sung
            st.divider()
            st.subheader("Thông tin lưu trữ")
            if 'processed_files' in st.session_state:
                st.info(
                    f"Số file trong lịch sử: {len(st.session_state.processed_files)}")
        # Tùy chọn phong cách dịch
        intensity = st.selectbox(
            'Phong cách dịch:',
            ['trang trọng', 'thông dụng']
        )

        # Kiểm tra có input không
        has_input = bool(
            uploaded_file or
            (url and url.strip()) or
            uploaded_srt
        )

        # Hiển thị trạng thái xử lý
        if st.session_state.is_processing:
            st.info("🔄 Đang xử lý... Vui lòng đợi")

        # Nút xử lý
        can_process = has_input and (
            not use_openai or (use_openai and openai_key))
        process_button = st.button(
            'Xử lý',
            disabled=not can_process or st.session_state.is_processing,
            key='process_button'
        )

        if process_button:
            st.session_state.is_processing = True
            try:
                cleanup_old_files(TEMP_DIR)
                self._process_request(
                    openai_key if use_openai else None,
                    uploaded_file,
                    url,
                    uploaded_srt,
                    intensity,
                    use_openai
                )
            finally:
                st.session_state.is_processing = False

        # Hiển thị lịch sử
        self._display_history()

    def _process_request(self, api_key: Optional[str], uploaded_file,
                         url: str, uploaded_srt, intensity: str, use_openai: bool):
        try:
            # Khởi tạo translator
            translator = (
                OpenAITranslator(self.cache_service,
                                 self.stats, api_key, intensity)
                if use_openai else
                GoogleTranslator(self.cache_service, self.stats)
            )

            # Tạo progress bar và status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Định nghĩa hàm update progress
            def update_progress(progress: int, message: str):
                progress_bar.progress(progress / 100)
                status_text.write(message)

            update_progress(10, "📥 Bắt đầu xử lý... (10%)")

            if uploaded_srt:
                self._process_srt_file(
                    uploaded_srt, translator, update_progress)
            else:
                self._process_video_file(
                    uploaded_file, url, translator, update_progress)

        except Exception as e:
            st.error(f"Lỗi trong quá trình xử lý: {str(e)}")

    def _process_srt_file(self, uploaded_srt, translator, progress_callback: Callable):
        try:
            if progress_callback:
                progress_callback(30, "📜 Đang xử lý file phụ đề... (30%)")

            srt_path = get_unique_filename(OUTPUT_DIR, "original", "srt")

            if save_uploaded_file(uploaded_srt, srt_path):
                if progress_callback:
                    progress_callback(
                        50, "🎯 Đã xử lý phụ đề. Bắt đầu dịch... (50%)")

                translated_srt = translator.translate_srt(
                    str(srt_path),
                    source_language=None,
                    progress_callback=progress_callback
                )

                if translated_srt:
                    self._handle_success(
                        str(srt_path),
                        translated_srt,
                        uploaded_srt.name,
                        None
                    )
        except Exception as e:
            st.error(f"Lỗi khi xử lý file phụ đề: {str(e)}")

    def _process_video_file(self, uploaded_file, url, translator, progress_callback: Callable):
        try:
            source_type = "upload" if uploaded_file else "url"
            source_data = uploaded_file if uploaded_file else url

            result = self.video_processor.process_video(
                source_type,
                source_data,
                progress_callback
            )

            if result.success and result.srt_path:
                translated_srt = translator.translate_srt(
                    result.srt_path,
                    source_language=result.detected_language,
                    progress_callback=progress_callback
                )

                if translated_srt:
                    self._handle_success(
                        result.srt_path,
                        translated_srt,
                        result.source_name,
                        result.detected_language
                    )
            else:
                st.error(f"Lỗi xử lý video: {result.error_message}")
        except Exception as e:
            st.error(f"Lỗi khi xử lý video: {str(e)}")

    def _handle_success(self, original_srt: str, translated_srt: str,
                        source_name: str, detected_language: Optional[str]):
        try:
            # Lưu vào lịch sử
            st.session_state.processed_files.append({
                'original_srt': original_srt,
                'translated_srt': translated_srt,
                'source': source_name,
                'source_language': detected_language,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Hiển thị nút tải xuống
            self._display_download_buttons(
                original_srt,
                translated_srt,
                detected_language
            )
        except Exception as e:
            st.error(f"Lỗi khi lưu kết quả: {str(e)}")

    def _display_download_buttons(self, original_srt: str,
                                  translated_srt: str,
                                  source_language: Optional[str]):
        try:
            st.success("✅ Hoàn thành! Tải file phụ đề:")

            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                    if os.path.exists(original_srt):
                        with open(original_srt, 'rb') as f:
                            st.download_button(
                                label=f"⬇️ Tải phụ đề gốc ({source_language or 'không xác định'})",
                                data=f,
                                file_name=f"goc_{timestamp}.srt",
                                mime="text/srt",
                                use_container_width=True
                            )

                    st.write("")

                    if os.path.exists(translated_srt):
                        with open(translated_srt, 'rb') as f:
                            st.download_button(
                                label="⬇️ Tải phụ đề tiếng Việt",
                                data=f,
                                file_name=f"tiengviet_{timestamp}.srt",
                                mime="text/srt",
                                use_container_width=True
                            )
        except Exception as e:
            st.error(f"Lỗi khi tạo nút tải xuống: {str(e)}")

    def _display_history(self):
        try:
            if st.session_state.processed_files:
                st.subheader("📋 Lịch sử xử lý")

                for idx, file in enumerate(reversed(st.session_state.processed_files)):
                    with st.expander(f"{idx + 1}. {file['source']} ({file['timestamp']})"):
                        st.markdown(
                            f"**Ngôn ngữ gốc:** {file['source_language'] or 'Không xác định'}")

                        with st.expander("Xem phụ đề gốc"):
                            try:
                                if os.path.exists(file['original_srt']):
                                    with open(file['original_srt'], 'r', encoding='utf-8') as f:
                                        st.code(f.read(), language='text')
                                else:
                                    st.warning(
                                        "File phụ đề gốc không còn tồn tại")
                            except Exception:
                                st.error("❌ Không thể đọc file phụ đề gốc")

                        with st.expander("Xem phụ đề đã dịch"):
                            try:
                                if os.path.exists(file['translated_srt']):
                                    with open(file['translated_srt'], 'r', encoding='utf-8') as f:
                                        st.code(f.read(), language='text')
                                else:
                                    st.warning(
                                        "File phụ đề đã dịch không còn tồn tại")
                            except Exception:
                                st.error("❌ Không thể đọc file phụ đề đã dịch")
        except Exception as e:
            st.error(f"Lỗi khi hiển thị lịch sử: {str(e)}")
