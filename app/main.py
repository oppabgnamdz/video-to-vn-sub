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

class TranslationCache:
    def __init__(self, cache_file: str = "translation_cache.json"):
        self.cache_file = cache_file
        self._cache: Dict[str, List[str]] = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                st.info(f"📂 Đã tải {len(self._cache)} bản dịch từ cache")
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
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

class VideoProcessor:
    LANGUAGE_CODES = {
        'en': 'en-US', 'ja': 'ja-JP', 'ko': 'ko-KR',
        'zh': 'zh-CN', 'vi': 'vi-VN', 'th': 'th-TH'
    }

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def process_video(self, source_type: str, source_data: any) -> ProcessingResult:
        video_path = self.temp_dir / "video.mp4"
        audio_path = self.temp_dir / "audio.wav"
        srt_path = self.output_dir / "output.srt"

        try:
            # Handle video source
            if source_type == "upload":
                source_name = source_data.name
                if not self._save_uploaded_file(source_data, video_path):
                    return ProcessingResult(False, error_message="Failed to save uploaded file")
            else:  # URL
                source_name = source_data
                if not self._download_video(source_data, video_path):
                    return ProcessingResult(False, error_message="Failed to download video")

            # Extract audio and create SRT
            if not self._extract_audio(video_path, audio_path):
                return ProcessingResult(False, error_message="Failed to extract audio")

            success, detected_language = self._speech_to_srt(audio_path, srt_path)
            if not success:
                return ProcessingResult(False, error_message="Failed to create SRT")

            return ProcessingResult(True, str(srt_path), source_name, detected_language)

        finally:
            # Cleanup temporary files
            self._cleanup_temp_files([video_path, audio_path])

    def _save_uploaded_file(self, uploaded_file: any, save_path: Path) -> bool:
        try:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return True
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
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
                            progress_text.text(f"Tải video: {(progress * 100):.1f}%")
            return True
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")
            return False

    def _extract_audio(self, video_path: Path, audio_path: Path) -> bool:
        try:
            video = VideoFileClip(str(video_path))
            if video.audio is None:
                raise Exception("Video has no audio")
            video.audio.write_audiofile(str(audio_path), logger=None)
            video.close()
            return True
        except Exception as e:
            st.error(f"Error extracting audio: {str(e)}")
            return False

    def _speech_to_srt(self, audio_path: Path, output_srt: Path) -> Tuple[bool, Optional[str]]:
        recognizer = sr.Recognizer()
        detected_language = None

        try:
            with sr.AudioFile(str(audio_path)) as source:
                # Language detection from sample
                audio_sample = recognizer.record(source, duration=min(10, source.DURATION))
                detected_language = self._detect_language(audio_sample, recognizer)
                
                # Process full audio
                source = sr.AudioFile(str(audio_path))
                with source as audio_file:
                    self._process_audio_chunks(audio_file, recognizer, detected_language, output_srt)
                
            return True, detected_language
        except Exception as e:
            st.error(f"Error creating SRT: {str(e)}")
            return False, None

    def _detect_language(self, audio_sample: sr.AudioData, recognizer: sr.Recognizer) -> str:
        for lang_code in self.LANGUAGE_CODES.values():
            try:
                sample_text = recognizer.recognize_google(audio_sample, language=lang_code)
                detected_lang = detect(sample_text)
                if detected_lang:
                    st.info(f"🎯 Detected language: {detected_lang}")
                    return detected_lang
            except:
                continue
        
        st.warning("⚠️ Could not detect language, using Japanese as default")
        return 'ja'

    def _process_audio_chunks(self, audio_file: sr.AudioFile, recognizer: sr.Recognizer, 
                            detected_language: str, output_srt: Path) -> None:
        chunk_duration = 10
        audio_length = audio_file.DURATION
        progress_bar = st.progress(0)
        
        with open(output_srt, 'w', encoding='utf-8') as srt_file:
            subtitle_count = 1
            for i in range(0, int(audio_length), chunk_duration):
                audio = recognizer.record(audio_file, duration=min(chunk_duration, audio_length-i))
                try:
                    lang_code = self.LANGUAGE_CODES.get(detected_language, 'ja-JP')
                    text = recognizer.recognize_google(audio, language=lang_code)
                except sr.UnknownValueError:
                    text = "..."
                except sr.RequestError as e:
                    st.warning(f"Google Speech Recognition API error: {str(e)}")
                    continue

                self._write_subtitle(srt_file, subtitle_count, i, 
                                  min(i + chunk_duration, audio_length), text)
                subtitle_count += 1
                
                if audio_length:
                    progress = i / audio_length
                    progress_bar.progress(progress)

    @staticmethod
    def _write_subtitle(srt_file: any, count: int, start_seconds: float, 
                       end_seconds: float, text: str) -> None:
        start_time = VideoProcessor._create_srt_timestamp(start_seconds)
        end_time = VideoProcessor._create_srt_timestamp(end_seconds)
        
        srt_file.write(f"{count}\n")
        srt_file.write(f"{start_time} --> {end_time}\n")
        srt_file.write(f"{text}\n\n")

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
                st.warning(f"Error deleting temporary file {file}: {str(e)}")

class SubtitleTranslator:
    def __init__(self, cache: TranslationCache, cost_tracker: CostTracker):
        self.cache = cache
        self.cost_tracker = cost_tracker

    def translate_srt(self, input_file: str, source_language: Optional[str], 
                     target_language: str = 'vi', intensity: str = "normal") -> Optional[str]:
        subs = pysrt.open(input_file)
        self.cost_tracker.total_lines = len(subs)
        output_file = f"{os.path.splitext(input_file)[0]}-{target_language}.srt"

        st.info(f"🚀 Starting translation: {input_file}")
        st.info(f"📝 Total lines: {self.cost_tracker.total_lines}")

        # Auto-detect source language if not provided
        if not source_language:
            source_language = self._detect_source_language(subs)

        # Optimize batch size based on average line length
        batch_size = self._calculate_optimal_batch_size(subs)
        self._process_subtitles(subs, batch_size, source_language, target_language, intensity)
        
        subs.save(output_file)
        self._display_translation_stats(output_file)
        
        return output_file

    def _detect_source_language(self, subs: pysrt.SubRipFile) -> str:
        if len(subs) > 0:
            sample_text = "\n".join([sub.text for sub in subs[:5]])
            try:
                detected_lang = detect(sample_text)
                if detected_lang:
                    st.info(f"🎯 Detected source language: {detected_lang}")
                    return detected_lang
            except:
                pass
        
        st.warning("⚠️ Could not detect source language, using Japanese as default")
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
                progress_text.text(f"Progress: {(progress * 100):.1f}%")

    def _translate_batch(self, texts: List[str], source_language: str,
                        target_language: str, intensity: str) -> List[str]:
        """Translate a batch of texts using OpenAI API with caching"""
        try:
            # Check cache first
            cache_key = hashlib.md5(
                f"{''.join(texts)}{source_language}-{target_language}{intensity}".encode()
            ).hexdigest()
            
            if cached_result := self.cache.get(cache_key):
                self.cost_tracker.processed_lines += len(texts)
                st.info(f"🎯 Using {len(texts)} lines from cache")
                return cached_result

            # Prepare translation parameters
            style = "natural and polite translation" if intensity == "mild" else "casual and colloquial translation"
            temperature = 0.6 if intensity == "mild" else 0.9

            # Format input texts with numbers
            numbered_texts = [f"{i+1}. {text}" for i, text in enumerate(texts)]
            combined_text = "\n".join(numbered_texts)

            # Create translation request
            system_message = f"You are a professional translator. Provide {style}. Translate from {source_language} to Vietnamese."
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Translate the following dialogue to Vietnamese, maintaining line numbers and order:\n\n{combined_text}"}
                ],
                max_tokens=4000,
                temperature=temperature
            )

            # Process response
            batch_cost = self.cost_tracker.update_stats(
                response['usage']['prompt_tokens'],
                response['usage']['completion_tokens']
            )

            translated_text = response['choices'][0]['message']['content'].strip()
            translated_sentences = translated_text.split("\n")
            translated_sentences = [
                sent.split(". ", 1)[-1] if ". " in sent else sent 
                for sent in translated_sentences
            ]

            # Validate output
            if len(translated_sentences) != len(texts):
                st.warning(f"⚠️ Translation count mismatch ({len(translated_sentences)} vs {len(texts)})")
                translated_sentences = self._normalize_output(translated_sentences, len(texts))

            # Cache result
            self.cache.set(cache_key, translated_sentences)
            
            self.cost_tracker.processed_lines += len(texts)
            st.success(f"✅ Translated batch: {len(texts)} lines (${batch_cost:.4f})")

            return translated_sentences

        except Exception as e:
            st.error(f"❌ Translation error: {e}")
            return [None] * len(texts)

    def _normalize_output(self, sentences: List[str], target_length: int) -> List[str]:
        """Ensure output matches input length by truncating or padding"""
        if len(sentences) < target_length:
            sentences.extend([""] * (target_length - len(sentences)))
        return sentences[:target_length]

    def _display_translation_stats(self, output_file: str) -> None:
        """Display final translation statistics"""
        elapsed_time = time.time() - self.cost_tracker.start_time
        st.success(f"""
        📊 TRANSLATION SUMMARY
        {'=' * 48}
        ✅ File saved to: {output_file}
        ⏱️ Total time: {self.cost_tracker.format_time(elapsed_time)}
        📝 Lines processed: {self.cost_tracker.total_lines}
        💰 Total cost: ${self.cost_tracker.total_cost:.4f}
        🔤 Total tokens: {self.cost_tracker.total_tokens:,}
        💵 Average cost per line: ${(self.cost_tracker.total_cost/self.cost_tracker.total_lines):.4f}
        """)

def create_app():
    """Create and configure the Streamlit application"""
    st.set_page_config(
        page_title="Video to SRT Converter",
        page_icon="🎥",
        layout="centered"
    )
    
    st.title('🎥 Video to Subtitle Converter')
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

    return StreamlitApp()

class StreamlitApp:
    def __init__(self):
        self.output_dir = Path(os.getcwd()) / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        self.cache = TranslationCache()
        self.cost_tracker = CostTracker()
        self.video_processor = VideoProcessor(str(self.output_dir))
        self.translator = SubtitleTranslator(self.cache, self.cost_tracker)

    def run(self):
        """Run the Streamlit application"""
        openai_key = st.text_input('OpenAI API Key:', type='password')
        
        tab1, tab2 = st.tabs(["Upload Video", "URL Video"])
        
        with tab1:
            uploaded_file = st.file_uploader("Choose video file", type=['mp4', 'avi', 'mkv', 'mov'])
            
        with tab2:
            url = st.text_input('Enter video URL:')
        
        intensity = st.selectbox('Translation style:', ['mild', 'hot'])
        
        has_video = bool(uploaded_file or (url and url.strip()))
        
        if st.button('Process Video', disabled=not (has_video and openai_key)):
            self._process_video_request(openai_key, uploaded_file, url, intensity)

        self._display_history()

    def _process_video_request(self, openai_key: str, uploaded_file: any, 
                             url: str, intensity: str) -> None:
        """Process video and create subtitles"""
        try:
            openai.api_key = openai_key
            self.cost_tracker.start()
            
            # Determine source type and data
            source_type = "upload" if uploaded_file else "url"
            source_data = uploaded_file if uploaded_file else url
            
            # Process video
            result = self.video_processor.process_video(source_type, source_data)
            
            if result.success and result.srt_path:
                st.success("✅ Created SRT file successfully!")
                
                # Translate the SRT
                translated_srt = self.translator.translate_srt(
                    result.srt_path,
                    source_language=result.detected_language,
                    intensity=intensity
                )
                
                if translated_srt:
                    self._add_to_history(result.srt_path, translated_srt, 
                                       result.source_name, result.detected_language)
                    self._display_download_buttons(result.srt_path, translated_srt, 
                                                result.detected_language)
            else:
                st.error(f"Failed to process video: {result.error_message}")
                
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

    def _add_to_history(self, original_srt: str, translated_srt: str, 
                       source: str, source_language: str) -> None:
        """Add processed file to history"""
        st.session_state.processed_files.append({
            'original_srt': original_srt,
            'translated_srt': translated_srt,
            'source': source,
            'source_language': source_language,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def _display_download_buttons(self, original_srt: str, translated_srt: str, 
                                source_language: str) -> None:
        """Display download buttons for SRT files"""
        st.success("✅ Processing complete! Download subtitle files:")
        
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                with open(original_srt, 'rb') as f:
                    st.download_button(
                        label=f"⬇️ Download original subtitles ({source_language})",
                        data=f,
                        file_name=f"original_{timestamp}.srt",
                        mime="text/srt",
                        use_container_width=True
                    )
                
                st.write("")
                
                with open(translated_srt, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Vietnamese subtitles",
                        data=f,
                        file_name=f"translated_{timestamp}.srt",
                        mime="text/srt",
                        use_container_width=True
                    )

    def _display_history(self) -> None:
        """Display processing history"""
        if st.session_state.processed_files:
            st.subheader("Processing History:")
            for idx, file in enumerate(reversed(st.session_state.processed_files)):
                with st.expander(f"{idx + 1}. {file['source']} ({file['timestamp']})"):
                    st.text(f"Source language: {file['source_language']}")
                    
                    st.text("Original SRT:")
                    with open(file['original_srt'], 'r', encoding='utf-8') as f:
                        st.code(f.read(), language='text')
                    
                    st.text("Translated SRT:")
                    with open(file['translated_srt'], 'r', encoding='utf-8') as f:
                        st.code(f.read(), language='text')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(file['original_srt'], 'rb') as f:
                            st.download_button(
                                label=f"Download original ({file['source_language']})",
                                data=f,
                                file_name=f"original_{idx}.srt",
                                mime="text/srt",
                                key=f"orig_{idx}"
                            )
                    with col2:
                        with open(file['translated_srt'], 'rb') as f:
                            st.download_button(
                                label="Download Vietnamese",
                                data=f,
                                file_name=f"translated_{idx}.srt",
                                mime="text/srt",
                                key=f"trans_{idx}"
                            )

if __name__ == "__main__":
    app = create_app()
    app.run()