from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel


@dataclass
class WhisperConfig:
    model: str
    device: str
    compute_type: str

    beam_size: int
    temperature: float
    condition_on_previous_text: bool

    source_language: str
    target_language: str

    no_speech_threshold: float
    log_prob_threshold: float
    compression_ratio_threshold: float
    repetition_penalty: float


class WhisperRunner:
    def __init__(self, whisper_config: WhisperConfig) -> None:
        self.whisper_config = whisper_config
        self.model = WhisperModel(
            whisper_config.model,
            device=whisper_config.device,
            compute_type=whisper_config.compute_type,
        )

    def _language_arg(self) -> Optional[str]:
        lang = self.whisper_config.source_language.strip()
        if not lang:
            return None
        return lang

    def run(self, audio_16k_float32: np.ndarray, task: str) -> str:
        if audio_16k_float32.size == 0:
            return ""

        language_arg = self._language_arg()

        segments, _info = self.model.transcribe(
            audio_16k_float32,
            task=task,
            language=language_arg,
            beam_size=self.whisper_config.beam_size,
            temperature=self.whisper_config.temperature,
            condition_on_previous_text=self.whisper_config.condition_on_previous_text,
            vad_filter=False,
            no_speech_threshold=self.whisper_config.no_speech_threshold,
            log_prob_threshold=self.whisper_config.log_prob_threshold,
            compression_ratio_threshold=self.whisper_config.compression_ratio_threshold,
            repetition_penalty=self.whisper_config.repetition_penalty,
        )

        text_parts: List[str] = []
        for seg in segments:
            part = (seg.text or "").strip()
            if part:
                text_parts.append(part)

        return " ".join(text_parts).strip()
