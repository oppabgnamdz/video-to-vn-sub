from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path


@dataclass
class ProcessingResult:
    """Kết quả xử lý video/audio thành phụ đề"""
    success: bool
    srt_path: Optional[str] = None  # Đường dẫn tới file phụ đề đã tạo
    source_name: Optional[str] = None  # Tên file/URL nguồn
    detected_language: Optional[str] = None  # Ngôn ngữ phát hiện được
    error_message: Optional[str] = None  # Thông báo lỗi nếu có


@dataclass
class TranslationStats:
    """Thống kê về quá trình dịch"""
    total_lines: int = 0  # Tổng số dòng cần dịch
    processed_lines: int = 0  # Số dòng đã dịch xong
    total_cost: float = 0  # Chi phí dịch (cho OpenAI)
    total_tokens: int = 0  # Tổng số tokens đã sử dụng
    start_time: Optional[float] = None  # Thời điểm bắt đầu dịch

    def get_progress(self) -> float:
        """Tính phần trăm tiến độ"""
        if self.total_lines == 0:
            return 0
        return (self.processed_lines / self.total_lines) * 100

    def get_cost_per_line(self) -> float:
        """Tính chi phí trung bình mỗi dòng"""
        if self.processed_lines == 0:
            return 0
        return self.total_cost / self.processed_lines


@dataclass
class ProcessedFile:
    """Thông tin về file đã xử lý"""
    original_srt: str  # Đường dẫn tới file phụ đề gốc
    translated_srt: str  # Đường dẫn tới file phụ đề đã dịch
    source: str  # Tên file/URL nguồn
    source_language: Optional[str]  # Ngôn ngữ gốc
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def get_original_content(self) -> Optional[str]:
        """Đọc nội dung file phụ đề gốc"""
        try:
            with open(self.original_srt, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def get_translated_content(self) -> Optional[str]:
        """Đọc nội dung file phụ đề đã dịch"""
        try:
            with open(self.translated_srt, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None


@dataclass
class TranslationSettings:
    """Cấu hình cho quá trình dịch"""
    source_language: str = 'auto'  # Ngôn ngữ nguồn
    target_language: str = 'vi'  # Ngôn ngữ đích
    use_openai: bool = True  # Sử dụng OpenAI hay Google Translate
    openai_api_key: Optional[str] = None  # API key cho OpenAI
    translation_style: str = 'normal'  # Phong cách dịch (normal/formal)
    batch_size: int = 10  # Số dòng dịch mỗi lần (cho OpenAI)

    def validate(self) -> bool:
        """Kiểm tra tính hợp lệ của cấu hình"""
        if self.use_openai and not self.openai_api_key:
            return False
        return True


@dataclass
class VideoMetadata:
    """Thông tin về video"""
    duration: float  # Độ dài video (giây)
    has_audio: bool  # Có audio không
    file_size: int  # Kích thước file (bytes)
    format: str  # Định dạng file

    @property
    def duration_formatted(self) -> str:
        """Định dạng độ dài video dễ đọc"""
        hours = int(self.duration // 3600)
        minutes = int((self.duration % 3600) // 60)
        seconds = int(self.duration % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def file_size_formatted(self) -> str:
        """Định dạng kích thước file dễ đọc"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.file_size < 1024:
                return f"{self.file_size:.1f}{unit}"
            self.file_size /= 1024
        return f"{self.file_size:.1f}TB"


@dataclass
class TranslationResult:
    """Kết quả của quá trình dịch"""
    success: bool
    original_text: str  # Văn bản gốc
    translated_text: Optional[str] = None  # Văn bản đã dịch
    error_message: Optional[str] = None  # Thông báo lỗi
    duration: Optional[float] = None  # Thời gian dịch (giây)
    cost: Optional[float] = None  # Chi phí dịch
    tokens_used: Optional[int] = None  # Số tokens sử dụng

    def to_dict(self) -> Dict:
        """Chuyển đổi sang dictionary"""
        return {
            'success': self.success,
            'original_text': self.original_text,
            'translated_text': self.translated_text,
            'error_message': self.error_message,
            'duration': self.duration,
            'cost': self.cost,
            'tokens_used': self.tokens_used
        }
