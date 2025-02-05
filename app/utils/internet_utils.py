import requests
import streamlit as st
from pathlib import Path
from typing import Tuple, Optional
from urllib.parse import urlparse
import socket
from requests.exceptions import RequestException
import time


def check_internet(timeout: int = 3) -> bool:
    """
    Kiểm tra kết nối internet.

    Args:
        timeout (int): Thời gian chờ tối đa (giây)

    Returns:
        bool: True nếu có kết nối internet, False nếu không
    """
    try:
        # Thử kết nối đến nhiều host khác nhau
        hosts = ['http://www.google.com',
                 'http://www.cloudflare.com',
                 'http://www.amazon.com']

        for host in hosts:
            try:
                requests.get(host, timeout=timeout)
                return True
            except RequestException:
                continue

        st.error("❌ Không có kết nối internet")
        return False
    except Exception:
        st.error("❌ Lỗi khi kiểm tra kết nối internet")
        return False


def download_file(url: str, output_path: Path,
                  chunk_size: int = 8192,
                  progress_callback=None) -> Tuple[bool, Optional[str]]:
    """
    Tải file từ URL với progress bar.

    Args:
        url (str): URL của file cần tải
        output_path (Path): Đường dẫn để lưu file
        chunk_size (int): Kích thước mỗi chunk khi tải
        progress_callback: Hàm callback để cập nhật tiến độ

    Returns:
        Tuple[bool, Optional[str]]: (Thành công/thất bại, thông báo lỗi)
    """
    try:
        # Tạo session với retry
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        # Gửi request với stream
        response = session.get(url, stream=True)
        response.raise_for_status()

        # Lấy kích thước file
        total_size = int(response.headers.get('content-length', 0))

        # Tạo progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        downloaded = 0
        start_time = time.time()

        # Mở file để ghi
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Cập nhật tiến độ
                    if total_size:
                        progress = downloaded / total_size
                        elapsed_time = time.time() - start_time
                        speed = downloaded / \
                            (1024 * elapsed_time) if elapsed_time > 0 else 0

                        # Hiển thị tiến độ
                        status = f"Đang tải: {(progress * 100):.1f}% ({speed:.1f} KB/s)"
                        if progress_callback:
                            progress_callback(progress * 100, status)
                        else:
                            progress_bar.progress(progress)
                            progress_text.text(status)

        return True, None

    except Exception as e:
        error_msg = f"Lỗi khi tải file: {str(e)}"
        st.error(error_msg)
        return False, error_msg


def validate_url(url: str) -> bool:
    """
    Kiểm tra URL có hợp lệ và có thể truy cập không.

    Args:
        url (str): URL cần kiểm tra

    Returns:
        bool: True nếu URL hợp lệ và có thể truy cập
    """
    try:
        # Kiểm tra format URL
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False

        # Kiểm tra có thể kết nối không
        response = requests.head(url, timeout=5)
        return response.status_code == 200

    except Exception:
        return False


def get_content_type(url: str) -> Optional[str]:
    """
    Lấy content-type của URL.

    Args:
        url (str): URL cần kiểm tra

    Returns:
        Optional[str]: Content-type của URL hoặc None nếu có lỗi
    """
    try:
        response = requests.head(url, timeout=5)
        return response.headers.get('content-type')
    except Exception:
        return None


def get_file_size(url: str) -> Optional[int]:
    """
    Lấy kích thước file từ URL.

    Args:
        url (str): URL cần kiểm tra

    Returns:
        Optional[int]: Kích thước file (bytes) hoặc None nếu có lỗi
    """
    try:
        response = requests.head(url, timeout=5)
        return int(response.headers.get('content-length', 0))
    except Exception:
        return None


def test_connection_speed() -> Tuple[float, float]:
    """
    Kiểm tra tốc độ kết nối mạng.

    Returns:
        Tuple[float, float]: (Download speed (MB/s), Upload speed (MB/s))
    """
    try:
        # Test download speed
        start_time = time.time()
        response = requests.get(
            'http://speedtest.ftp.otenet.gr/files/test1Mb.db')
        download_size = len(response.content)
        download_time = time.time() - start_time
        download_speed = download_size / (1024 * 1024 * download_time)

        # Test upload speed (giả lập bằng cách gửi một lượng data nhỏ)
        data = 'x' * 1024 * 1024  # 1MB of data
        start_time = time.time()
        response = requests.post('https://httpbin.org/post', data=data)
        upload_time = time.time() - start_time
        upload_speed = 1 / upload_time  # 1MB / time taken

        return download_speed, upload_speed
    except Exception:
        return 0.0, 0.0
