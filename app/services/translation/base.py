from abc import ABC, abstractmethod
from typing import List, Optional
import pysrt
from models.data_models import TranslationStats


class BaseTranslator(ABC):
    def __init__(self, cache_service, stats: TranslationStats):
        self.cache_service = cache_service
        self.stats = stats

    @abstractmethod
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single piece of text"""
        pass

    @abstractmethod
    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate a batch of texts"""
        pass

    def translate_srt(self,
                      input_file: str,
                      source_language: Optional[str] = None,
                      target_language: str = 'vi',
                      progress_callback=None) -> Optional[str]:
        """Translate an entire SRT file"""
        subs = pysrt.open(input_file)
        self.stats.total_lines = len(subs)

        if not source_language:
            source_language = self._detect_language(subs)

        self._process_subtitles(subs, source_language,
                                target_language, progress_callback)

        output_file = f"{input_file.rsplit('.', 1)[0]}-{target_language}.srt"
        subs.save(output_file)

        return output_file

    @abstractmethod
    def _detect_language(self, subs: pysrt.SubRipFile) -> str:
        """Detect the language of the subtitles"""
        pass

    @abstractmethod
    def _process_subtitles(self,
                           subs: pysrt.SubRipFile,
                           source_language: str,
                           target_language: str,
                           progress_callback=None) -> None:
        """Process and translate all subtitles"""
        pass
