import streamlit as st
import os
import requests
from moviepy import VideoFileClip
import speech_recognition as sr
import openai
import json
import hashlib
import time
from datetime import datetime
import pysrt

# Constants for OpenAI API costs
PRICE_PER_1K_TOKENS = {
    "gpt-3.5-turbo": {
        "input": 0.0010,
        "output": 0.0020
    }
}

# Global variables for translation tracking
total_cost = 0
total_tokens = 0
start_time = None
processed_lines = 0
total_lines = 0

# Translation cache
_translation_cache = {}
_cache_file = "translation_cache.json"

def load_cache():
    global _translation_cache
    if os.path.exists(_cache_file):
        try:
            with open(_cache_file, 'r', encoding='utf-8') as f:
                _translation_cache = json.load(f)
            st.info(f"📂 Đã tải {len(_translation_cache)} bản dịch từ cache")
        except:
            _translation_cache = {}
            st.warning("⚠️ Không thể tải cache, tạo cache mới")

def save_cache():
    with open(_cache_file, 'w', encoding='utf-8') as f:
        json.dump(_translation_cache, f, ensure_ascii=False)
    st.info(f"💾 Đã lưu {len(_translation_cache)} bản dịch vào cache")

def get_cache_key(text, target_language, intensity):
    return hashlib.md5(f"{text}{target_language}{intensity}".encode()).hexdigest()

def calculate_cost(model, input_tokens, output_tokens):
    if model not in PRICE_PER_1K_TOKENS:
        return 0
    
    pricing = PRICE_PER_1K_TOKENS[model]
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return input_cost + output_cost

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"

def batch_translate_text(texts, target_language='vi', intensity="normal"):
    global total_cost, total_tokens, processed_lines
    
    try:
        cache_key = get_cache_key("\n".join(texts), target_language, intensity)
        if cache_key in _translation_cache:
            processed_lines += len(texts)
            st.info(f"🎯 Sử dụng {len(texts)} dòng từ cache")
            return _translation_cache[cache_key]

        if intensity == "mild":
            style = "dịch tự nhiên, phù hợp với bối cảnh phim người lớn, hấp dẫn và thú vị."
            temperature = 0.6
        elif intensity == "hot":
            style = "dịch một cách khiêu gợi, gợi cảm và thô tục."
            temperature = 0.9
        else:
            style = "dịch tự nhiên, phù hợp với bối cảnh phim người lớn, hấp dẫn và thú vị."
            temperature = 0.75

        numbered_texts = [f"{i+1}. {text}" for i, text in enumerate(texts)]
        combined_text = "\n".join(numbered_texts)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Bạn là một dịch giả chuyên nghiệp. Hãy {style}"},
                {"role": "user", "content": f"Dịch đoạn hội thoại sau từ tiếng Nhật sang tiếng Việt, giữ nguyên số dòng và đúng thứ tự:\n\n{combined_text}"}
            ],
            max_tokens=4000,
            temperature=temperature
        )

        input_tokens = response['usage']['prompt_tokens']
        output_tokens = response['usage']['completion_tokens']
        total_tokens += input_tokens + output_tokens
        batch_cost = calculate_cost("gpt-3.5-turbo", input_tokens, output_tokens)
        total_cost += batch_cost

        translated_text = response['choices'][0]['message']['content'].strip()
        translated_sentences = translated_text.split("\n")
        translated_sentences = [sent.split(". ", 1)[-1] if ". " in sent else sent for sent in translated_sentences]

        if len(translated_sentences) != len(texts):
            st.warning(f"⚠️ Số câu dịch ({len(translated_sentences)}) không khớp với câu gốc ({len(texts)})")
            while len(translated_sentences) < len(texts):
                translated_sentences.append("")
            while len(translated_sentences) > len(texts):
                translated_sentences.pop()

        _translation_cache[cache_key] = translated_sentences
        save_cache()

        processed_lines += len(texts)
        st.success(f"✅ Đã dịch xong batch: {len(texts)} dòng (${batch_cost:.4f})")

        return translated_sentences
    except Exception as e:
        st.error(f"❌ Lỗi khi dịch: {e}")
        return [None] * len(texts)

def translate_srt(input_file, target_language='vi', intensity="normal", batch_size=10):
    global start_time, total_lines
    start_time = time.time()
    
    load_cache()

    subs = pysrt.open(input_file)
    total_lines = len(subs)
    output_file = f"{os.path.splitext(input_file)[0]}-vi.srt"

    st.info(f"🚀 Bắt đầu dịch file: {input_file}")
    st.info(f"📝 Tổng số dòng: {total_lines}")

    avg_length = sum(len(sub.text) for sub in subs) / len(subs)
    if avg_length < 50:
        batch_size = 15
    elif avg_length < 100:
        batch_size = 10
    else:
        batch_size = 5

    batch = []
    indices = []

    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, sub in enumerate(subs):
        batch.append(sub.text)
        indices.append(i)

        if len(batch) >= batch_size or i == len(subs) - 1:
            translated_batch = batch_translate_text(batch, target_language, intensity)

            for j, translated_text in zip(indices, translated_batch):
                if translated_text:
                    subs[j].text = translated_text

            batch = []
            indices = []
            
            # Update progress
            progress = (i + 1) / total_lines
            progress_bar.progress(progress)
            progress_text.text(f"Tiến độ: {(progress * 100):.1f}%")

    subs.save(output_file)
    
    elapsed_time = time.time() - start_time
    st.success(f"""
    📊 THỐNG KÊ TỔNG KẾT
    ={'=' * 48}
    ✅ File đã được dịch và lưu vào: {output_file}
    ⏱️ Tổng thời gian: {format_time(elapsed_time)}
    📝 Số dòng đã xử lý: {total_lines}
    💰 Tổng chi phí: ${total_cost:.4f}
    🔤 Tổng số token: {total_tokens:,}
    💵 Chi phí trung bình mỗi dòng: ${(total_cost/total_lines):.4f}
    """)
    
    return output_file

def download_video(url, output_path):
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
        st.error(f"Lỗi khi tải video: {str(e)}")
        return False

def extract_audio(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            raise Exception("Video không có audio")
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        return True
    except Exception as e:
        st.error(f"Lỗi khi tách audio: {str(e)}")
        return False

def create_srt_timestamp(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = int(seconds % 60)
    msecs = int((seconds * 1000) % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

def speech_to_srt(audio_path, output_srt):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_length = source.DURATION if hasattr(source, 'DURATION') else 0
            chunk_duration = 10
            
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            with open(output_srt, 'w', encoding='utf-8') as srt_file:
                subtitle_count = 1
                
                for i in range(0, int(audio_length), chunk_duration):
                    audio = recognizer.record(source, duration=min(chunk_duration, audio_length-i))
                    try:
                        text = recognizer.recognize_google(audio, language='ja-JP')  # Changed to Japanese
                    except sr.UnknownValueError:
                        text = "..."
                    except sr.RequestError as e:
                        st.warning(f"Lỗi API Google Speech Recognition: {str(e)}")
                        continue
                    
                    start_time = create_srt_timestamp(i)
                    end_time = create_srt_timestamp(min(i + chunk_duration, audio_length))
                    
                    srt_file.write(f"{subtitle_count}\n")
                    srt_file.write(f"{start_time} --> {end_time}\n")
                    srt_file.write(f"{text}\n\n")
                    
                    subtitle_count += 1
                    if audio_length:
                        progress = i / audio_length
                        progress_bar.progress(progress)
                        progress_text.text(f"Xử lý audio: {(progress * 100):.1f}%")
        return True
    except Exception as e:
        st.error(f"Lỗi khi tạo file SRT: {str(e)}")
        return False

def process_video(video_path, output_dir):
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    audio_path = os.path.join(temp_dir, "audio.wav")
    srt_path = os.path.join(output_dir, "output.srt")
    
    try:
            
        with st.spinner('Đang tách audio...'):
            if not extract_audio(video_path, audio_path):
                return False, None
            
        with st.spinner('Đang tạo phụ đề...'):
            if not speech_to_srt(audio_path, srt_path):
                return False, None
            
        return True, srt_path
        
    finally:
        # Cleanup
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            st.warning(f"Lỗi khi xóa file tạm: {str(e)}")

def save_uploadedfile(uploadedfile, save_path):
    try:
        with open(save_path, "wb") as f:
            f.write(uploadedfile.getbuffer())
        return True
    except Exception as e:
        st.error(f"Lỗi khi lưu file: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="Video to SRT Converter",
        page_icon="🎥",
        layout="centered"
    )
    
    st.title('🎥 Chuyển Video thành Phụ đề')
    
    # Initialize session state
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    # Input fields
    openai_key = st.text_input('OpenAI API Key:', type='password')
    
    # Tạo tab cho 2 phương thức input
    tab1, tab2 = st.tabs(["Upload Video", "URL Video"])
    
    with tab1:
        uploaded_file = st.file_uploader("Chọn file video", type=['mp4', 'avi', 'mkv', 'mov'])
        
    with tab2:
        url = st.text_input('Nhập URL video:')
    
    intensity = st.selectbox('Chọn mức độ dịch:', ['mild', 'hot'])
    
    # Check điều kiện để enable nút xử lý
    has_video = (uploaded_file is not None) or (url and url.strip())
    
    if st.button('Xử lý Video', disabled=not (has_video and openai_key)):
        try:
            # Set OpenAI API key
            openai.api_key = openai_key
            
            # Create output directory
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create temp directory for processing
            temp_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            video_path = os.path.join(temp_dir, "video.mp4")
            
            # Handle either uploaded file or URL
            if uploaded_file is not None:
                st.info("Đang xử lý file đã upload...")
                success = save_uploadedfile(uploaded_file, video_path)
                if success:
                    source = uploaded_file.name
                else:
                    st.error("Không thể lưu file video")
                    return
            else:
                st.info("Đang tải video từ URL...")
                success = download_video(url, video_path)
                if not success:
                    st.error("Không thể tải video từ URL")
                    return
                source = url
            
            # Process video to create initial SRT
            success, srt_path = process_video(video_path, output_dir)
            if success and srt_path:
                st.success("Đã tạo file SRT thành công!")
                
                # Translate the SRT
                translated_srt = translate_srt(srt_path, intensity=intensity)
                if translated_srt:
                    # Add to processed files
                    st.session_state.processed_files.append({
                        'original_srt': srt_path,
                        'translated_srt': translated_srt,
                        'source': source,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Display completion message
                    st.success("✅ Đã xử lý xong! Tải xuống các file phụ đề:")
                    
                    # Create a container for download buttons with custom styling
                    download_container = st.container()
                    with download_container:
                        # Center-align the buttons using columns
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            # First download button
                            with open(srt_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Tải file phụ đề gốc (Tiếng Nhật)",
                                    data=f,
                                    file_name=f"original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
                                    mime="text/srt",
                                    use_container_width=True,
                                )
                            
                            st.write("")  # Add some spacing
                            
                            # Second download button
                            with open(translated_srt, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Tải file phụ đề đã dịch (Tiếng Việt)",
                                    data=f,
                                    file_name=f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
                                    mime="text/srt",
                                    use_container_width=True,
                                )
        except Exception as e:
            st.error(f"Lỗi khi xử lý: {str(e)}")

    # Display history
    if st.session_state.processed_files:
        st.subheader("Lịch sử xử lý:")
        for idx, file in enumerate(reversed(st.session_state.processed_files)):
            with st.expander(f"{idx + 1}. {file['source']} ({file['timestamp']})"):
                st.text("File SRT gốc:")
                with open(file['original_srt'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='text')
                    
                st.text("File SRT đã dịch:")
                with open(file['translated_srt'], 'r', encoding='utf-8') as f:
                    st.code(f.read(), language='text')
                
                col1, col2 = st.columns(2)
                with col1:
                    with open(file['original_srt'], 'rb') as f:
                        st.download_button(
                            label="Tải lại file gốc",
                            data=f,
                            file_name=f"original_{idx}.srt",
                            mime="text/srt",
                            key=f"orig_{idx}"
                        )
                with col2:
                    with open(file['translated_srt'], 'rb') as f:
                        st.download_button(
                            label="Tải lại file đã dịch",
                            data=f,
                            file_name=f"translated_{idx}.srt",
                            mime="text/srt",
                            key=f"trans_{idx}"
                        )

if __name__ == "__main__":
    main()