import hashlib
import streamlit as st
import openai
from typing import List
import pysrt
from langdetect import detect
from config import OPENAI_MODEL, OPENAI_PRICING, BATCH_SIZE
from .base import BaseTranslator


class OpenAITranslator(BaseTranslator):
    def __init__(self, cache_service, stats, api_key: str, intensity: str = "normal"):
        super().__init__(cache_service, stats)
        self.api_key = api_key
        self.intensity = intensity
        openai.api_key = api_key

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        return self.translate_batch([text], source_lang, target_lang)[0]

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        # 🔹 Kiểm tra API Key trước khi gọi OpenAI API
        if not self.api_key or not self.api_key.startswith("sk-"):
            st.error("❌ OpenAI API Key không hợp lệ! Dừng tiến trình dịch.")
            st.stop()  # 🔥 Dừng ngay lập tức, không tiếp tục xử lý

        numbered_texts = [f"{i+1}. {text}" for i, text in enumerate(texts)]
        combined_text = "\n".join(numbered_texts)

        style = "dịch tự nhiên, phù hợp với bối cảnh phim người lớn,phim khiêu dâm, phim 18 +, hấp dẫn và thú vị." \
            if self.intensity == "mild" else "văn phong thông dụng, tự nhiên"

        temperature = 0.6 if self.intensity == "mild" else 0.7

        system_message = f"Bạn là một dịch giả chuyên nghiệp. Hãy dịch với {style}. Dịch từ {source_lang} sang tiếng Việt."

        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Dịch đoạn hội thoại sau sang tiếng Việt, giữ nguyên số thứ tự và trình tự:\n\n{combined_text}"}
                ],
                max_tokens=4000,
                temperature=temperature
            )

            # 🔹 Kiểm tra nếu OpenAI trả về lỗi
            if "error" in response:
                st.error(f"❌ Lỗi OpenAI: {response['error']['message']}")
                st.stop()  # 🔥 Dừng toàn bộ tiến trình

            self._update_stats(response['usage'])

            translated_text = response['choices'][0]['message']['content'].strip(
            )
            translated_sentences = translated_text.split("\n")
            translated_sentences = [
                sent.split(". ", 1)[-1] if ". " in sent else sent
                for sent in translated_sentences
            ]

            if len(translated_sentences) != len(texts):
                st.warning(
                    f"⚠️ Số dòng dịch không khớp ({len(translated_sentences)} vs {len(texts)})")
                translated_sentences = self._normalize_output(
                    translated_sentences, len(texts))

            return translated_sentences

        except openai.error.OpenAIError as e:
            st.error(f"❌ Lỗi từ OpenAI API: {str(e)}")
            st.stop()  # 🔥 Dừng nếu gặp lỗi từ OpenAI API

        except Exception as e:
            st.error(f"❌ Lỗi hệ thống: {str(e)}")
            st.stop()  # 🔥 Dừng nếu gặp lỗi khác
            return [""] * len(texts)

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
        batch = []
        indices = []
        batch_size = self._calculate_optimal_batch_size(subs)

        for i, sub in enumerate(subs):
            batch.append(sub.text)
            indices.append(i)

            if len(batch) >= batch_size or i == len(subs) - 1:
                cache_key = hashlib.md5(
                    f"{''.join(batch)}{source_language}-{target_language}{self.intensity}".encode()
                ).hexdigest()

                if cached_result := self.cache_service.get(cache_key):
                    for idx, translated_text in zip(indices, cached_result):
                        subs[idx].text = translated_text
                    st.info(f"🎯 Sử dụng {len(batch)} dòng từ cache")
                else:
                    translated_batch = self.translate_batch(
                        batch, source_language, target_language
                    )

                    for idx, translated_text in zip(indices, translated_batch):
                        if translated_text:
                            subs[idx].text = translated_text

                    self.cache_service.set(cache_key, translated_batch)

                self.stats.processed_lines += len(batch)

                if progress_callback:
                    progress = (i + 1) / self.stats.total_lines
                    progress_callback(progress * 100,
                                      f"Tiến độ dịch: {(progress * 100):.1f}%")

                batch = []
                indices = []

    def _calculate_optimal_batch_size(self, subs: pysrt.SubRipFile) -> int:
        avg_length = sum(len(sub.text) for sub in subs) / len(subs)
        if avg_length < 50:
            return BATCH_SIZE["large"]
        elif avg_length < 100:
            return BATCH_SIZE["medium"]
        return BATCH_SIZE["small"]

    def _normalize_output(self, sentences: List[str], target_length: int) -> List[str]:
        if len(sentences) < target_length:
            sentences.extend([""] * (target_length - len(sentences)))
        return sentences[:target_length]

    def _update_stats(self, usage) -> None:
        input_tokens = usage['prompt_tokens']
        output_tokens = usage['completion_tokens']

        pricing = OPENAI_PRICING[OPENAI_MODEL]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        self.stats.total_cost += input_cost + output_cost
        self.stats.total_tokens += input_tokens + output_tokens
