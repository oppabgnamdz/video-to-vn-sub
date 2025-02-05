from pathlib import Path
import streamlit as st
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from typing import List, Optional, Tuple
from langdetect import detect, LangDetectException
from models.data_models import ProcessingResult
from utils.internet_utils import download_file
from utils.file_utils import save_uploaded_file, delete_file
from config import LANGUAGE_CODES


class VideoProcessor:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def process_video(self, source_type: str, source_data: any,
                      progress_callback) -> ProcessingResult:
        video_path = self.temp_dir / "video.mp4"
        audio_path = self.temp_dir / "audio.wav"
        srt_path = self.output_dir / "output.srt"

        try:
            # Lưu hoặc tải video
            if source_type == "upload":
                source_name = source_data.name
                progress_callback(15, "📥 Đang lưu video... (15%)")
                if not save_uploaded_file(source_data, video_path):
                    return ProcessingResult(False, error_message="Lỗi khi lưu file")
            else:
                source_name = source_data
                progress_callback(15, "📥 Đang tải video từ URL... (15%)")
                success, error = download_file(source_data, video_path)
                if not success:
                    return ProcessingResult(False, error_message=error)

            # Trích xuất audio
            progress_callback(30, "🎵 Đang trích xuất âm thanh... (30%)")
            if not self._extract_audio(video_path, audio_path):
                return ProcessingResult(False, error_message="Lỗi khi trích xuất âm thanh")

            # Chuyển đổi speech to text
            progress_callback(
                45, "🔍 Đang chuyển đổi âm thanh thành văn bản... (45%)")
            success, detected_language = self._speech_to_srt(
                audio_path, srt_path)
            if not success:
                return ProcessingResult(False, error_message="Lỗi khi tạo phụ đề")

            return ProcessingResult(
                success=True,
                srt_path=str(srt_path),
                source_name=source_name,
                detected_language=detected_language
            )

        finally:
            # Dọn dẹp file tạm
            delete_file(video_path)
            delete_file(audio_path)

    def _extract_audio(self, video_path: Path, audio_path: Path) -> bool:
        """Trích xuất âm thanh từ video"""
        try:
            with VideoFileClip(str(video_path)) as video:
                if video.audio is None:
                    raise Exception("Video không có âm thanh")
                video.audio.write_audiofile(
                    str(audio_path),
                    logger=None
                )
            return True
        except Exception as e:
            st.error(f"Lỗi khi trích xuất âm thanh: {str(e)}")
            return False

    def _speech_to_srt(self, audio_path: Path, output_srt: Path) -> Tuple[bool, Optional[str]]:
        """Chuyển đổi âm thanh thành phụ đề SRT"""
        recognizer = sr.Recognizer()
        detected_language = None

        try:
            # Phát hiện ngôn ngữ từ đoạn mẫu
            with sr.AudioFile(str(audio_path)) as source:
                audio_sample = recognizer.record(
                    source,
                    duration=min(10, source.DURATION)
                )
                detected_language = self._detect_language(
                    audio_sample,
                    recognizer
                )

            # Xử lý toàn bộ file audio
            with sr.AudioFile(str(audio_path)) as source:
                self._process_audio_chunks(
                    source,
                    recognizer,
                    detected_language,
                    output_srt
                )

            return True, detected_language
        except Exception as e:
            st.error(f"Lỗi khi tạo phụ đề: {str(e)}")
            return False, None

    def _detect_language(self, audio_sample: sr.AudioData,
                         recognizer: sr.Recognizer) -> str:
        """Phát hiện ngôn ngữ từ mẫu âm thanh"""
        for lang_code in LANGUAGE_CODES.values():
            try:
                sample_text = recognizer.recognize_google(
                    audio_sample,
                    language=lang_code
                )
                detected_lang = detect(sample_text)
                if detected_lang:
                    st.info(f"🎯 Ngôn ngữ phát hiện: {detected_lang}")
                    return detected_lang
            except:
                continue

        st.warning(
            "⚠️ Không phát hiện được ngôn ngữ, dùng tiếng Nhật làm mặc định")
        return 'ja'

    def _process_audio_chunks(self, audio_file: sr.AudioFile,
                              recognizer: sr.Recognizer,
                              detected_language: str,
                              output_srt: Path) -> None:
        """Xử lý file âm thanh theo từng đoạn"""
        chunk_duration = 10  # seconds
        audio_length = audio_file.DURATION
        progress_bar = st.progress(0)
        progress_text = st.empty()

        with open(output_srt, 'w', encoding='utf-8') as srt_file:
            subtitle_count = 1
            for i in range(0, int(audio_length), chunk_duration):
                # Ghi nhận âm thanh theo chunk
                audio = recognizer.record(
                    audio_file,
                    duration=min(chunk_duration, audio_length-i)
                )

                try:
                    # Nhận dạng text
                    lang_code = LANGUAGE_CODES.get(detected_language, 'ja-JP')
                    text = recognizer.recognize_google(
                        audio,
                        language=lang_code
                    )
                except sr.UnknownValueError:
                    text = "..."
                except sr.RequestError as e:
                    st.warning(f"Lỗi API Google Speech Recognition: {str(e)}")
                    continue

                # Ghi phụ đề
                self._write_subtitle(
                    srt_file,
                    subtitle_count,
                    i,
                    min(i + chunk_duration, audio_length),
                    text
                )
                subtitle_count += 1

                # Cập nhật progress
                if audio_length:
                    progress = i / audio_length
                    progress_bar.progress(progress)
                    progress_text.text(f"Đang xử lý: {(progress * 100):.1f}%")

    @staticmethod
    def _write_subtitle(srt_file: any, count: int,
                        start_seconds: float, end_seconds: float,
                        text: str) -> None:
        """Ghi một phụ đề vào file SRT"""
        start_time = VideoProcessor._format_timestamp(start_seconds)
        end_time = VideoProcessor._format_timestamp(end_seconds)
        srt_file.write(f"{count}\n{start_time} --> {end_time}\n{text}\n\n")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Định dạng thời gian theo chuẩn SRT (HH:MM:SS,mmm)"""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = int(seconds % 60)
        msecs = int((seconds * 1000) % 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"
