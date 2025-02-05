import time
import random
import hashlib
import streamlit as st
from googletrans import Translator
from typing import List, Optional
import pysrt
from langdetect import detect
from .base import BaseTranslator


class GoogleTranslator(BaseTranslator):
    def __init__(self, cache_service, stats):
        super().__init__(cache_service, stats)
        self.translator = Translator()
        self.max_retries = 3
        self.retry_delay = 2

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        for attempt in range(self.max_retries):
            try:
                result = self.translator.translate(
                    text,
                    src=source_lang if source_lang != 'auto' else 'auto',
                    dest=target_lang
                )
                return result.text
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (attempt + 1) +
                           random.uniform(0, 1))

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        results = []
        for text in texts:
            results.append(self.translate_text(text, source_lang, target_lang))
            time.sleep(0.5 + random.uniform(0, 0.5))  # Rate limiting
        return results

    def _detect_language(self, subs: pysrt.SubRipFile) -> str:
        if len(subs) > 0:
            sample_text = "\n".join([sub.text for sub in subs[:5]])
            try:
                detected_lang = detect(sample_text)
                if detected_lang:
                    st.info(f"🎯 Ngôn ngữ phát hiện: {detected_lang}")
                    return detected_lang
            except:
                pass
        st.warning(
            "⚠️ Không phát hiện được ngôn ngữ, dùng tiếng Nhật làm mặc định")
        return 'ja'

    def _process_subtitles(self, subs: pysrt.SubRipFile,
                           source_language: str, target_language: str,
                           progress_callback=None) -> None:
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, sub in enumerate(subs):
            try:
                cache_key = hashlib.md5(
                    f"{sub.text}{source_language}-{target_language}".encode()
                ).hexdigest()

                if cached_result := self.cache_service.get(cache_key):
                    sub.text = cached_result[0]
                    st.info(f"🎯 Sử dụng dòng từ cache")
                else:
                    translated_text = self.translate_text(
                        sub.text,
                        source_language,
                        target_language
                    )
                    sub.text = translated_text
                    self.cache_service.set(cache_key, [translated_text])

                self.stats.processed_lines += 1

                if progress_callback:
                    progress = (i + 1) / self.stats.total_lines
                    progress_callback(progress * 100,
                                      f"Tiến độ dịch: {(progress * 100):.1f}%")

                time.sleep(0.5 + random.uniform(0, 0.5))

            except Exception as e:
                st.warning(f"⚠️ Lỗi dịch dòng {i+1}: {str(e)}")
                continue
