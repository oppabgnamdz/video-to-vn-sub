"""
Video Subtitle Generator and Translator
=====================================

A Streamlit application that converts video to subtitles and translates them.
"""

from .config import *
from .models import *
from .services import *
from .utils import *
from .ui import StreamlitApp

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

__all__ = [
    'StreamlitApp',
    'create_app'  # from main.py
]
