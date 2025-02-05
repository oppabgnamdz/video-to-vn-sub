import streamlit as st
from pathlib import Path
import os
from config import OUTPUT_DIR, TEMP_DIR
from utils.file_utils import ensure_directories
from ui.app import StreamlitApp


def create_app():
    # Cấu hình trang
    st.set_page_config(
        page_title="Chuyển Đổi Video sang Phụ Đề",
        page_icon="🎥",
        layout="centered"
    )

    st.title('🎥 Chuyển Đổi Video sang Phụ Đề')

    # Khởi tạo state
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

    # Đảm bảo thư mục tồn tại
    ensure_directories(OUTPUT_DIR, TEMP_DIR)

    # Tạo và chạy app
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    create_app()
