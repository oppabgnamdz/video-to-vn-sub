import json
import streamlit as st
from typing import Dict, List, Optional
from config import MAX_CACHE_SIZE, CACHE_FILE


class CacheService:
    def __init__(self):
        self._cache: Dict[str, List[str]] = {}
        self.load()

    def load(self) -> None:
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                st.info(
                    f"📂 Đã tải {len(self._cache)} bản dịch từ bộ nhớ cache")
        except Exception as e:
            st.warning(f"⚠️ Lỗi khi tải cache: {str(e)}")
            self._cache = {}

    def save(self) -> None:
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False)
            st.info(f"💾 Đã lưu {len(self._cache)} bản dịch vào cache")
        except Exception as e:
            st.warning(f"⚠️ Lỗi khi lưu cache: {str(e)}")

    def get(self, key: str) -> Optional[List[str]]:
        return self._cache.get(key)

    def set(self, key: str, value: List[str]) -> None:
        try:
            cache_size = len(json.dumps(self._cache).encode('utf-8'))
            if cache_size > MAX_CACHE_SIZE:
                self._cache.clear()
                st.warning("🗑️ Đã xóa cache do vượt quá dung lượng")
            self._cache[key] = value
            self.save()
        except Exception as e:
            st.warning(f"⚠️ Lỗi khi cập nhật cache: {str(e)}")

    def clear(self) -> None:
        self._cache.clear()
        self.save()
        st.info("🗑️ Đã xóa toàn bộ cache")
