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
import m3u8
import requests
from urllib.parse import urljoin
import subprocess


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
            # Xử lý nguồn video dựa trên source_type
            if source_type == "upload":
                source_name = source_data.name
                progress_callback(15, "📥 Đang lưu video... (15%)")
                if not save_uploaded_file(source_data, video_path):
                    return ProcessingResult(False, error_message="Lỗi khi lưu file")

            elif source_type == "m3u8":
                source_name = "m3u8_video.mp4"
                progress_callback(15, "📥 Đang tải video từ M3U8... (15%)")
                success, error = self._process_m3u8(
                    source_data, video_path, progress_callback)
                if not success:
                    return ProcessingResult(False, error_message=error)

            else:  # URL thông thường
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

    def _process_m3u8(self, m3u8_url: str, output_path: Path,
                      progress_callback) -> Tuple[bool, Optional[str]]:
        """Xử lý và tải video từ M3U8 playlist"""
        try:
            # Tạo thư mục tạm cho segments
            segments_dir = self.temp_dir / "segments"
            segments_dir.mkdir(exist_ok=True)

            # Parse M3U8 playlist
            try:
                playlist = m3u8.load(m3u8_url)
            except Exception as e:
                return False, f"Không thể đọc M3U8 playlist: {str(e)}"

            if not playlist.segments:
                return False, "Không tìm thấy phân đoạn video trong playlist"

            # Xác định base URL
            if m3u8_url.startswith('http'):
                base_url = m3u8_url.rsplit('/', 1)[0] + '/'
            else:
                base_url = ''

            total_segments = len(playlist.segments)
            downloaded_segments = []

            # Tải từng segment với progress
            for i, segment in enumerate(playlist.segments):
                try:
                    segment_url = urljoin(base_url, segment.uri)
                    response = requests.get(segment_url, timeout=30)
                    if response.status_code == 200:
                        segment_path = segments_dir / f"segment_{i:04d}.ts"
                        with open(segment_path, 'wb') as f:
                            f.write(response.content)
                        downloaded_segments.append(str(segment_path))
                        progress_callback(
                            15 + (i / total_segments * 10),
                            f"📥 Đang tải segment {i + 1}/{total_segments}..."
                        )
                    else:
                        st.warning(
                            f"Không thể tải segment {i}: HTTP {response.status_code}")
                except Exception as e:
                    st.warning(f"Lỗi khi tải segment {i}: {str(e)}")
                    continue

            if not downloaded_segments:
                return False, "Không tải được segments video"

            # Tạo file danh sách segments
            concat_file = self.temp_dir / "concat.txt"
            with open(concat_file, 'w', encoding='utf-8') as f:
                for segment in sorted(downloaded_segments):
                    f.write(f"file '{segment}'\n")

            # Sử dụng ffmpeg để ghép file
            progress_callback(25, "🔄 Đang ghép video... (25%)")

            try:
                # Ghép segments thành TS
                temp_ts = self.temp_dir / "temp_output.ts"
                cmd = [
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                    '-i', str(concat_file),
                    '-c', 'copy',
                    str(temp_ts)
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                # Chuyển đổi TS sang MP4 với codec phù hợp
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(temp_ts),
                    '-c:v', 'libx264',  # Sử dụng H.264 codec
                    '-c:a', 'aac',      # Sử dụng AAC codec cho audio
                    '-movflags', '+faststart',  # Tối ưu cho streaming
                    str(output_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                # Kiểm tra file MP4
                cmd = ['ffmpeg', '-v', 'error', '-i',
                       str(output_path), '-f', 'null', '-']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.stderr:
                    raise Exception(f"Invalid MP4 file: {result.stderr}")

            except subprocess.CalledProcessError as e:
                return False, f"Lỗi khi ghép video: {e.stderr.decode()}"
            except Exception as e:
                return False, f"Lỗi khi xử lý video: {str(e)}"

            finally:
                # Dọn dẹp files
                if concat_file.exists():
                    concat_file.unlink()
                if temp_ts.exists():
                    temp_ts.unlink()
                for file in segments_dir.glob("*"):
                    file.unlink()
                segments_dir.rmdir()

            return True, None

        except Exception as e:
            return False, f"Lỗi khi xử lý M3U8: {str(e)}"

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
