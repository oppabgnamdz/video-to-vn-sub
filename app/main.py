import streamlit as st
import os
import requests
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import openai
import json
import hashlib
import time
from datetime import datetime
import pysrt
from langdetect import detect, LangDetectException
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from pathlib import Path


@dataclass
class ProcessingResult:
    success: bool
    srt_path: Optional[str] = None
    source_name: Optional[str] = None
    detected_language: Optional[str] = None
    error_message: Optional[str] = None


def validate_api_key(api_key: str) -> bool:
    try:
        openai.api_key = api_key
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return True
    except:
        st.error(f"❌ API key không hợp lệ: {api_key}")
        return False


def check_internet() -> bool:
    try:
        requests.get("http://www.google.com", timeout=3)
        return True
    except:
        st.error("❌ Không có kết nối internet")
        return False


def validate_video_file(file) -> bool:
    if file.size > 500 * 1024 * 1024:  # 500MB
        st.error("❌ File quá lớn. Giới hạn 500MB")
        return False
    return True


class TranslationCache:
    MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB

    def __init__(self, cache_file: str = "translation_cache.json"):
        self.cache_file = cache_file
        self._cache: Dict[str, List[str]] = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                st.info(
                    f"📂 Đã tải {len(self._cache)} bản dịch từ bộ nhớ cache")
            except Exception as e:
                st.warning(f"⚠️ Lỗi khi tải cache: {str(e)}")
                self._cache = {}

    def save(self) -> None:
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, ensure_ascii=False)
        st.info(f"💾 Đã lưu {len(self._cache)} bản dịch vào cache")

    def get(self, key: str) -> Optional[List[str]]:
        return self._cache.get(key)

    def set(self, key: str, value: List[str]) -> None:
        cache_size = len(json.dumps(self._cache).encode('utf-8'))
        if cache_size > self.MAX_CACHE_SIZE:
            self._cache.clear()
            st.warning("🗑️ Đã xóa cache do vượt quá dung lượng")
        self._cache[key] = value
        self.save()


class CostTracker:
    PRICE_PER_1K_TOKENS = {
        "gpt-3.5-turbo": {"input": 0.0010, "output": 0.0020}
    }

    def __init__(self):
        self.total_cost = 0
        self.total_tokens = 0
        self.start_time = None
        self.processed_lines = 0
        self.total_lines = 0

    def start(self) -> None:
        self.start_time = time.time()

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if model not in self.PRICE_PER_1K_TOKENS:
            return 0
        pricing = self.PRICE_PER_1K_TOKENS[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def update_stats(self, input_tokens: int, output_tokens: int, model: str = "gpt-3.5-turbo") -> float:
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        self.total_tokens += input_tokens + output_tokens
        return cost

    @staticmethod
    def format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours} giờ {minutes} phút {seconds} giây"
        elif minutes > 0:
            return f"{minutes} phút {seconds} giây"
        return f"{seconds} giây"


class VideoProcessor:
    LANGUAGE_CODES = {
        'en': 'en-US', 'ja': 'ja-JP', 'ko': 'ko-KR',
        'zh': 'zh-CN', 'vi': 'vi-VN', 'th': 'th-TH'
    }

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def process_video(self, source_type: str, source_data: any, progress_callback) -> ProcessingResult:
        video_path = self.temp_dir / "video.mp4"
        audio_path = self.temp_dir / "audio.wav"
        srt_path = self.output_dir / "output.srt"

        try:
            if source_type == "upload":
                source_name = source_data.name
                progress_callback(15, "📥 Đang lưu video... (15%)")
                if not self._save_uploaded_file(source_data, video_path):
                    return ProcessingResult(False, error_message="Lỗi khi lưu file")
            else:
                source_name = source_data
                progress_callback(15, "📥 Đang tải video từ URL... (15%)")
                if not self._download_video(source_data, video_path):
                    return ProcessingResult(False, error_message="Lỗi khi tải video")

            progress_callback(30, "🎵 Đang trích xuất âm thanh... (30%)")
            if not self._extract_audio(video_path, audio_path):
                return ProcessingResult(False, error_message="Lỗi khi trích xuất âm thanh")

            progress_callback(
                45, "🔍 Đang chuyển đổi âm thanh thành văn bản... (45%)")
            success, detected_language = self._speech_to_srt(
                audio_path, srt_path)
            if not success:
                return ProcessingResult(False, error_message="Lỗi khi tạo phụ đề")

            return ProcessingResult(True, str(srt_path), source_name, detected_language)

        finally:
            self._cleanup_temp_files([video_path, audio_path])

    def _save_uploaded_file(self, uploaded_file: any, save_path: Path) -> bool:
        try:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return True
        except Exception as e:
            st.error(f"Lỗi khi lưu file: {str(e)}")
            return False

    def _download_video(self, url: str, output_path: Path) -> bool:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            progress_bar = st.progress(0)
            progress_text = st.empty()

            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            progress_text.text(
                                f"Đang tải: {(progress * 100):.1f}%")
            return True
        except Exception as e:
            st.error(f"Lỗi khi tải video: {str(e)}")
            return False

    def _extract_audio(self, video_path: Path, audio_path: Path) -> bool:
        try:
            video = VideoFileClip(str(video_path))
            if video.audio is None:
                raise Exception("Video không có âm thanh")
            video.audio.write_audiofile(str(audio_path), logger=None)
            video.close()
            return True
        except Exception as e:
            st.error(f"Lỗi khi trích xuất âm thanh: {str(e)}")
            return False

    def _speech_to_srt(self, audio_path: Path, output_srt: Path) -> Tuple[bool, Optional[str]]:
        recognizer = sr.Recognizer()
        detected_language = None

        try:
            with sr.AudioFile(str(audio_path)) as source:
                audio_sample = recognizer.record(
                    source, duration=min(10, source.DURATION))
                detected_language = self._detect_language(
                    audio_sample, recognizer)

                source = sr.AudioFile(str(audio_path))
                with source as audio_file:
                    self._process_audio_chunks(
                        audio_file, recognizer, detected_language, output_srt)

            return True, detected_language
        except Exception as e:
            st.error(f"Lỗi khi tạo phụ đề: {str(e)}")
            return False, None

    def _detect_language(self, audio_sample: sr.AudioData, recognizer: sr.Recognizer) -> str:
        for lang_code in self.LANGUAGE_CODES.values():
            try:
                sample_text = recognizer.recognize_google(
                    audio_sample, language=lang_code)
                detected_lang = detect(sample_text)
                if detected_lang:
                    st.info(f"🎯 Ngôn ngữ phát hiện: {detected_lang}")
                    return detected_lang
            except:
                continue
        st.warning(
            "⚠️ Không phát hiện được ngôn ngữ, dùng tiếng Nhật làm mặc định")
        return 'ja'

    def _process_audio_chunks(self, audio_file: sr.AudioFile, recognizer: sr.Recognizer,
                              detected_language: str, output_srt: Path) -> None:
        chunk_duration = 10
        audio_length = audio_file.DURATION
        progress_bar = st.progress(0)
        progress_text = st.empty()

        with open(output_srt, 'w', encoding='utf-8') as srt_file:
            subtitle_count = 1
            for i in range(0, int(audio_length), chunk_duration):
                audio = recognizer.record(
                    audio_file, duration=min(chunk_duration, audio_length-i))
                try:
                    lang_code = self.LANGUAGE_CODES.get(
                        detected_language, 'ja-JP')
                    text = recognizer.recognize_google(
                        audio, language=lang_code)
                except sr.UnknownValueError:
                    text = "..."
                except sr.RequestError as e:
                    st.warning(f"Lỗi API Google Speech Recognition: {str(e)}")
                    continue

                self._write_subtitle(srt_file, subtitle_count, i,
                                     min(i + chunk_duration, audio_length), text)
                subtitle_count += 1

                if audio_length:
                    progress = i / audio_length
                    progress_bar.progress(progress)
                    progress_text.text(f"Đang xử lý: {(progress * 100):.1f}%")

    @staticmethod
    def _write_subtitle(srt_file: any, count: int, start_seconds: float,
                        end_seconds: float, text: str) -> None:
        start_time = VideoProcessor._create_srt_timestamp(start_seconds)
        end_time = VideoProcessor._create_srt_timestamp(end_seconds)
        srt_file.write(f"{count}\n{start_time} --> {end_time}\n{text}\n\n")

    @staticmethod
    def _create_srt_timestamp(seconds: float) -> str:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = int(seconds % 60)
        msecs = int((seconds * 1000) % 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

    @staticmethod
    def _cleanup_temp_files(files: List[Path]) -> None:
        for file in files:
            try:
                if file.exists():
                    file.unlink()
            except Exception as e:
                st.warning(f"Lỗi khi xóa file tạm {file}: {str(e)}")


class SubtitleTranslator:
    def __init__(self, cache: TranslationCache, cost_tracker: CostTracker):
        self.cache = cache
        self.cost_tracker = cost_tracker

    def translate_srt(self, input_file: str, source_language: Optional[str],
                      target_language: str = 'vi', intensity: str = "normal",
                      progress_callback=None) -> Optional[str]:
        subs = pysrt.open(input_file)
        self.cost_tracker.total_lines = len(subs)
        output_file = f"{os.path.splitext(input_file)[0]}-{target_language}.srt"

        st.info(f"🚀 Bắt đầu dịch: {input_file}")
        st.info(f"📝 Tổng số dòng: {self.cost_tracker.total_lines}")

        if not source_language:
            source_language = self._detect_source_language(subs)

        batch_size = self._calculate_optimal_batch_size(subs)
        self._process_subtitles(
            subs, batch_size, source_language, target_language, intensity)

        subs.save(output_file)
        self._display_translation_stats(output_file)

        if progress_callback:
            progress_callback(100, "✅ Đã hoàn thành dịch! (100%)")

        return output_file

    def _detect_source_language(self, subs: pysrt.SubRipFile) -> str:
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

    @staticmethod
    def _calculate_optimal_batch_size(subs: pysrt.SubRipFile) -> int:
        avg_length = sum(len(sub.text) for sub in subs) / len(subs)
        if avg_length < 50:
            return 15
        elif avg_length < 100:
            return 10
        return 5

    def _process_subtitles(self, subs: pysrt.SubRipFile, batch_size: int,
                           source_language: str, target_language: str, intensity: str) -> None:
        batch = []
        indices = []
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, sub in enumerate(subs):
            batch.append(sub.text)
            indices.append(i)

            if len(batch) >= batch_size or i == len(subs) - 1:
                translated_batch = self._translate_batch(
                    batch, source_language, target_language, intensity)

                for j, translated_text in zip(indices, translated_batch):
                    if translated_text:
                        subs[j].text = translated_text

                batch = []
                indices = []

                progress = (i + 1) / self.cost_tracker.total_lines
                progress_bar.progress(progress)
                progress_text.text(f"Tiến độ dịch: {(progress * 100):.1f}%")

    def _translate_batch(self, texts: List[str], source_language: str,
                         target_language: str, intensity: str) -> List[str]:
        try:
            cache_key = hashlib.md5(
                f"{''.join(texts)}{source_language}-{target_language}{intensity}".encode()
            ).hexdigest()

            if cached_result := self.cache.get(cache_key):
                self.cost_tracker.processed_lines += len(texts)
                st.info(f"🎯 Sử dụng {len(texts)} dòng từ cache")
                return cached_result

            style = "dịch tự nhiên, phù hợp với bối cảnh người lớn, hấp dẫn và thú vị." if intensity == "mild" else "văn phong thông dụng, tự nhiên"
            temperature = 0.6 if intensity == "mild" else 0.9

            numbered_texts = [f"{i+1}. {text}" for i, text in enumerate(texts)]
            combined_text = "\n".join(numbered_texts)

            system_message = f"Bạn là một dịch giả chuyên nghiệp. Hãy dịch với {style}. Dịch từ {source_language} sang tiếng Việt."

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Dịch đoạn hội thoại sau sang tiếng Việt, giữ nguyên số thứ tự và trình tự:\n\n{combined_text}"}
                ],
                max_tokens=4000,
                temperature=temperature
            )

            batch_cost = self.cost_tracker.update_stats(
                response['usage']['prompt_tokens'],
                response['usage']['completion_tokens']
            )

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

            self.cache.set(cache_key, translated_sentences)

            self.cost_tracker.processed_lines += len(texts)
            st.success(f"✅ Đã dịch: {len(texts)} dòng (${batch_cost:.4f})")

            return translated_sentences

        except Exception as e:
            st.error(f"❌ Lỗi dịch: {e}")
            return [None] * len(texts)

    def _normalize_output(self, sentences: List[str], target_length: int) -> List[str]:
        if len(sentences) < target_length:
            sentences.extend([""] * (target_length - len(sentences)))
        return sentences[:target_length]

    def _display_translation_stats(self, output_file: str) -> None:
        elapsed_time = time.time() - self.cost_tracker.start_time
        st.success(f"""
        📊 THỐNG KÊ DỊCH
        {'=' * 48}
        ✅ File đã lưu: {output_file}
        ⏱️ Tổng thời gian: {self.cost_tracker.format_time(elapsed_time)}
        📝 Số dòng xử lý: {self.cost_tracker.total_lines}
        💰 Tổng chi phí: ${self.cost_tracker.total_cost:.4f}
        🔤 Tổng tokens: {self.cost_tracker.total_tokens:,}
        💵 Chi phí/dòng: ${(self.cost_tracker.total_cost/self.cost_tracker.total_lines):.4f}
        """)

def create_app():
    st.set_page_config(
        page_title="Chuyển Đổi Video sang Phụ Đề",
        page_icon="🎥",
        layout="centered"
    )

    st.title('🎥 Chuyển Đổi Video sang Phụ Đề')

    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

    return StreamlitApp()


class StreamlitApp:
    def __init__(self):
        self.output_dir = Path(os.getcwd()) / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        self.cache = TranslationCache()
        self.cost_tracker = CostTracker()
        self.video_processor = VideoProcessor(str(self.output_dir))
        self.translator = SubtitleTranslator(self.cache, self.cost_tracker)

    def cleanup_old_files(self):
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if time.time() - os.path.getctime(file_path) > 24*3600:  # 24 giờ
                    os.remove(file_path)
        except Exception as e:
            st.warning(f"⚠️ Lỗi dọn dẹp file tạm: {str(e)}")

    def run(self):
        if not check_internet():
            return

        openai_key = st.text_input('OpenAI API Key:', type='password')
        if openai_key == 'abc':
            openai_key = os.getenv('OPENAI_API_KEY', '')

        tab1, tab2, tab3 = st.tabs(
            ["Tải Video", "URL Video", "Tải File Phụ Đề"])

        with tab1:
            uploaded_file = st.file_uploader("Chọn file video", type=[
                                             'mp4', 'avi', 'mkv', 'mov'])
            if uploaded_file:
                st.info(
                    f"Dung lượng file: {uploaded_file.size / 1024 / 1024:.1f}MB")

        with tab2:
            url = st.text_input('Nhập URL video:')

        with tab3:
            uploaded_srt = st.file_uploader(
                "Chọn file phụ đề SRT", type=['srt'])

        intensity = st.selectbox('Phong cách dịch:', [
                                 'trang trọng', 'thông dụng'])

        has_input = bool(uploaded_file or (
            url and url.strip()) or uploaded_srt)

        if st.session_state.is_processing:
            st.info("🔄 Đang xử lý... Vui lòng đợi")

        process_button = st.button(
            'Xử lý',
            disabled=not (
                has_input and openai_key) or st.session_state.is_processing,
            key='process_button'
        )

        if process_button:
            if not validate_api_key(openai_key):
                return

            if uploaded_file and not validate_video_file(uploaded_file):
                return

            st.session_state.is_processing = True
            try:
                self.cleanup_old_files()
                self._process_video_request(
                    openai_key, uploaded_file, url, uploaded_srt,
                    'mild' if intensity == 'trang trọng' else 'hot'
                )
            finally:
                st.session_state.is_processing = False

        self._display_history()

    def _process_video_request(self, openai_key: str, uploaded_file: any,
                               url: str, uploaded_srt: any, intensity: str) -> None:
        try:
            openai.api_key = openai_key
            self.cost_tracker.start()

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress: int, message: str):
                progress_bar.progress(progress / 100)
                status_text.write(message)

            update_progress(10, "📥 Bắt đầu xử lý... (10%)")

            if uploaded_srt:
                update_progress(30, "📜 Đang xử lý file phụ đề... (30%)")
                original_srt = self._save_uploaded_file(
                    uploaded_srt, self.output_dir / "original.srt")

                if original_srt:
                    update_progress(
                        50, "🎯 Đã xử lý phụ đề. Bắt đầu dịch... (50%)")
                    translated_srt = self.translator.translate_srt(
                        str(original_srt),
                        source_language=None,
                        intensity=intensity,
                        progress_callback=update_progress
                    )

                    if translated_srt:
                        update_progress(100, "✅ Hoàn thành! (100%)")
                        self._add_to_history(str(original_srt), translated_srt,
                                             uploaded_srt.name, None)
                        self._display_download_buttons(
                            str(original_srt), translated_srt, None)
            else:
                source_type = "upload" if uploaded_file else "url"
                source_data = uploaded_file if uploaded_file else url

                result = self.video_processor.process_video(
                    source_type, source_data,
                    progress_callback=update_progress
                )

                if result.success and result.srt_path:
                    update_progress(
                        50, "🎯 Đã xử lý video. Bắt đầu dịch... (50%)")
                    translated_srt = self.translator.translate_srt(
                        result.srt_path,
                        source_language=result.detected_language,
                        intensity=intensity,
                        progress_callback=update_progress
                    )

                    if translated_srt:
                        update_progress(100, "✅ Hoàn thành! (100%)")
                        self._add_to_history(result.srt_path, translated_srt,
                                             result.source_name, result.detected_language)
                        self._display_download_buttons(result.srt_path, translated_srt,
                                                       result.detected_language)
                else:
                    st.error(f"Lỗi xử lý video: {result.error_message}")

        except Exception as e:
            st.error(f"Lỗi trong quá trình xử lý: {str(e)}")
        finally:
            st.session_state.is_processing = False

    def _save_uploaded_file(self, uploaded_file: any, save_path: Path) -> Optional[str]:
        try:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return str(save_path)
        except Exception as e:
            st.error(f"Lỗi khi lưu file: {str(e)}")
            return None

    def _add_to_history(self, original_srt: str, translated_srt: str,
                        source: str, source_language: str) -> None:
        st.session_state.processed_files.append({
            'original_srt': original_srt,
            'translated_srt': translated_srt,
            'source': source,
            'source_language': source_language,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def _display_download_buttons(self, original_srt: str, translated_srt: str,
                                  source_language: str) -> None:
        st.success("✅ Hoàn thành! Tải file phụ đề:")

        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                with open(original_srt, 'rb') as f:
                    st.download_button(
                        label=f"⬇️ Tải phụ đề gốc ({source_language or 'không xác định'})",
                        data=f,
                        file_name=f"goc_{timestamp}.srt",
                        mime="text/srt",
                        use_container_width=True,
                        disabled=st.session_state.is_processing
                    )

                st.write("")

                with open(translated_srt, 'rb') as f:
                    st.download_button(
                        label="⬇️ Tải phụ đề tiếng Việt",
                        data=f,
                        file_name=f"tiengviet_{timestamp}.srt",
                        mime="text/srt",
                        use_container_width=True,
                        disabled=st.session_state.is_processing
                    )

    def _display_history(self) -> None:
        if st.session_state.processed_files:
            st.subheader("📋 Lịch sử xử lý")

            for idx, file in enumerate(reversed(st.session_state.processed_files)):
                with st.expander(f"{idx + 1}. {file['source']} ({file['timestamp']})"):
                    st.markdown(
                        f"**Ngôn ngữ gốc:** {file['source_language'] or 'Không xác định'}")

                    with st.expander("Xem phụ đề gốc"):
                        try:
                            with open(file['original_srt'], 'r', encoding='utf-8') as f:
                                st.code(f.read(), language='text')
                        except Exception:
                            st.error("❌ Không thể đọc file phụ đề gốc")

                    with st.expander("Xem phụ đề đã dịch"):
                        try:
                            with open(file['translated_srt'], 'r', encoding='utf-8') as f:
                                st.code(f.read(), language='text')
                        except Exception:
                            st.error("❌ Không thể đọc file phụ đề đã dịch")

                    col1, col2 = st.columns(2)
                    with col1:
                        try:
                            with open(file['original_srt'], 'rb') as f:
                                st.download_button(
                                    label=f"⬇️ Tải phụ đề gốc",
                                    data=f,
                                    file_name=f"goc_{idx}.srt",
                                    mime="text/srt",
                                    key=f"orig_{idx}",
                                    disabled=st.session_state.is_processing
                                )
                        except Exception:
                            st.error("❌ File gốc không khả dụng")

                    with col2:
                        try:
                            with open(file['translated_srt'], 'rb') as f:
                                st.download_button(
                                    label="⬇️ Tải phụ đề Việt",
                                    data=f,
                                    file_name=f"tiengviet_{idx}.srt",
                                    mime="text/srt",
                                    key=f"trans_{idx}",
                                    disabled=st.session_state.is_processing
                                )
                        except Exception:
                            st.error("❌ File dịch không khả dụng")


if __name__ == "__main__":
    app = create_app()
    app.run()
