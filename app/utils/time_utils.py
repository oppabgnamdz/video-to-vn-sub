from datetime import datetime, timedelta
from typing import Union, Optional
import time


def format_duration(seconds: Union[int, float]) -> str:
    """
    Định dạng thời gian từ giây thành chuỗi dễ đọc.

    Args:
        seconds: Số giây cần định dạng

    Returns:
        str: Chuỗi thời gian định dạng (VD: "2 giờ 30 phút")
    """
    if seconds < 60:
        return f"{int(seconds)} giây"

    minutes = seconds // 60
    if minutes < 60:
        return f"{int(minutes)} phút {int(seconds % 60)} giây"

    hours = minutes // 60
    minutes = minutes % 60
    if hours < 24:
        return f"{int(hours)} giờ {int(minutes)} phút"

    days = hours // 24
    hours = hours % 24
    return f"{int(days)} ngày {int(hours)} giờ"


def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Lấy timestamp hiện tại theo định dạng chỉ định.

    Args:
        format (str): Định dạng timestamp mong muốn

    Returns:
        str: Timestamp định dạng
    """
    return datetime.now().strftime(format)


def calculate_duration(start_time: float) -> float:
    """
    Tính khoảng thời gian từ start_time đến hiện tại.

    Args:
        start_time (float): Thời điểm bắt đầu (từ time.time())

    Returns:
        float: Số giây đã trôi qua
    """
    return time.time() - start_time


def parse_duration(duration_str: str) -> Optional[timedelta]:
    """
    Chuyển đổi chuỗi thời gian thành timedelta.
    VD: "2h30m" -> timedelta(hours=2, minutes=30)

    Args:
        duration_str (str): Chuỗi thời gian cần parse

    Returns:
        Optional[timedelta]: timedelta object hoặc None nếu không parse được
    """
    try:
        total_seconds = 0
        current_number = ""

        for char in duration_str:
            if char.isdigit():
                current_number += char
            elif char in ['s', 'm', 'h', 'd']:
                if not current_number:
                    continue

                value = int(current_number)
                if char == 's':
                    total_seconds += value
                elif char == 'm':
                    total_seconds += value * 60
                elif char == 'h':
                    total_seconds += value * 3600
                elif char == 'd':
                    total_seconds += value * 86400

                current_number = ""

        return timedelta(seconds=total_seconds)
    except Exception:
        return None


def format_timestamp(timestamp: float, include_date: bool = True) -> str:
    """
    Định dạng timestamp Unix thành chuỗi dễ đọc.

    Args:
        timestamp (float): Unix timestamp
        include_date (bool): Có bao gồm ngày tháng không

    Returns:
        str: Chuỗi thời gian định dạng
    """
    dt = datetime.fromtimestamp(timestamp)
    if include_date:
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    return dt.strftime("%H:%M:%S")


def get_time_ago(timestamp: float) -> str:
    """
    Tính thời gian đã trôi qua từ timestamp đến hiện tại.
    VD: "2 giờ trước", "5 phút trước"

    Args:
        timestamp (float): Unix timestamp

    Returns:
        str: Chuỗi mô tả thời gian đã trôi qua
    """
    now = datetime.now()
    dt = datetime.fromtimestamp(timestamp)
    diff = now - dt

    seconds = diff.total_seconds()

    if seconds < 60:
        return "Vừa xong"
    if seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} phút trước"
    if seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} giờ trước"
    if seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} ngày trước"
    if seconds < 2592000:
        weeks = int(seconds / 604800)
        return f"{weeks} tuần trước"

    return dt.strftime("%d/%m/%Y")
