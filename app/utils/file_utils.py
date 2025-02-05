import os
import time
import shutil
import streamlit as st
from pathlib import Path
from typing import Optional, List, Tuple
from config import MAX_FILE_SIZE, FILE_RETENTION_DAYS


def validate_file_size(file) -> bool:
    """
    Kiểm tra xem kích thước file có nằm trong giới hạn cho phép không.

    Args:
        file: File đã upload (từ st.file_uploader)

    Returns:
        bool: True nếu file hợp lệ, False nếu quá lớn
    """
    if file.size > MAX_FILE_SIZE:
        st.error(
            f"❌ File quá lớn. Giới hạn {MAX_FILE_SIZE / (1024*1024):.1f}MB")
        return False
    return True


def save_uploaded_file(uploaded_file, save_path: Path) -> bool:
    """
    Lưu file đã upload vào đường dẫn chỉ định.

    Args:
        uploaded_file: File đã upload
        save_path (Path): Đường dẫn để lưu file

    Returns:
        bool: True nếu lưu thành công, False nếu có lỗi
    """
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Lỗi khi lưu file: {str(e)}")
        return False


def cleanup_old_files(directory: Path) -> None:
    """
    Xóa các file cũ trong thư mục.

    Args:
        directory (Path): Thư mục cần dọn dẹp
    """
    try:
        current_time = time.time()
        for file in directory.iterdir():
            if file.is_file():
                file_age = current_time - file.stat().st_mtime
                if file_age > FILE_RETENTION_DAYS * 24 * 3600:
                    delete_file(file)
    except Exception as e:
        st.warning(f"⚠️ Lỗi khi dọn dẹp file tạm: {str(e)}")


def ensure_directories(*directories: Path) -> None:
    """
    Đảm bảo các thư mục tồn tại.

    Args:
        directories: Danh sách các thư mục cần tạo
    """
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            st.error(f"Lỗi khi tạo thư mục {directory}: {str(e)}")


def delete_file(file_path: Path) -> None:
    """
    Xóa file an toàn.

    Args:
        file_path (Path): Đường dẫn file cần xóa
    """
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        st.warning(f"⚠️ Lỗi khi xóa file {file_path}: {str(e)}")


def get_unique_filename(directory: Path, base_name: str, extension: str) -> Path:
    """
    Tạo tên file không trùng lặp.

    Args:
        directory (Path): Thư mục chứa file
        base_name (str): Tên cơ bản của file
        extension (str): Phần mở rộng của file

    Returns:
        Path: Đường dẫn file không trùng lặp
    """
    counter = 1
    file_path = directory / f"{base_name}.{extension}"
    while file_path.exists():
        file_path = directory / f"{base_name}_{counter}.{extension}"
        counter += 1
    return file_path


def get_file_info(file_path: Path) -> dict:
    """
    Lấy thông tin về file.

    Args:
        file_path (Path): Đường dẫn đến file

    Returns:
        dict: Thông tin về file (kích thước, thời gian tạo, etc.)
    """
    try:
        stat = file_path.stat()
        return {
            'size': stat.st_size,
            'created': time.ctime(stat.st_ctime),
            'modified': time.ctime(stat.st_mtime),
            'extension': file_path.suffix,
            'name': file_path.name
        }
    except Exception:
        return {}


def copy_file(source: Path, destination: Path) -> bool:
    """
    Copy file từ source sang destination.

    Args:
        source (Path): Đường dẫn file nguồn
        destination (Path): Đường dẫn đích

    Returns:
        bool: True nếu copy thành công, False nếu thất bại
    """
    try:
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        st.error(f"Lỗi khi copy file: {str(e)}")
        return False


def get_all_files(directory: Path, pattern: str = "*") -> List[Path]:
    """
    Lấy danh sách tất cả các file trong thư mục theo pattern.

    Args:
        directory (Path): Thư mục cần quét
        pattern (str): Pattern để lọc file (ví dụ: "*.mp4")

    Returns:
        List[Path]: Danh sách các file tìm thấy
    """
    try:
        return list(directory.glob(pattern))
    except Exception:
        return []
