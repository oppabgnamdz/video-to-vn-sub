from pathlib import Path
import os

# Paths
BASE_DIR = Path(os.getcwd())
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"
CACHE_FILE = OUTPUT_DIR / "translation_cache.json"

# OpenAI Config
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_PRICING = {
    "gpt-3.5-turbo": {
        "input": 0.0010,
        "output": 0.0020
    }
}

# Translation Config
MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_FILE_SIZE = 500 * 1024 * 1024    # 500MB
BATCH_SIZE = {
    "small": 5,
    "medium": 10,
    "large": 15
}

# Language Codes
LANGUAGE_CODES = {
    'en': 'en-US',
    'ja': 'ja-JP',
    'ko': 'ko-KR',
    'zh': 'zh-CN',
    'vi': 'vi-VN',
    'th': 'th-TH'
}

# File retention
FILE_RETENTION_DAYS = 1  # Delete temp files older than 1 day