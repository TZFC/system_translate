import asyncio
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import pyaudiowpatch as pyaudio
import soxr
import webrtcvad

from ollama_connect import OllamaClient
from run_whisper import WhisperRunner


@dataclass
class AudioConfig:
    frame_ms: int
    vad_aggressiveness: int
    min_speech_ms: int
    end_silence_ms: int
    max_segment_s: float
    partial_interval_s: float


@dataclass
class FeatureToggles:
    transcribe: bool
    english_translate: bool
    ai_translate: bool


class LiveOverlay:
    def __init__(
        self,
        max_lines: int,
        audio_config: AudioConfig,
        feature_toggles: FeatureToggles,
        whisper_runner: WhisperRunner,
        ollama_client: OllamaClient,
    ) -> None:
        self.max_lines = max_lines
        self.audio_config = audio_config
        self.feature_toggles = feature_toggles
        self.whisper_runner = whisper_runner
        self.ollama_client = ollama_client
        self.ollama_context_lines = ollama_client.ollama_config.context_lines

        self.sample_rate_model = 16000

        self.vad = webrtcvad.Vad(audio_config.vad_aggressiveness)

        self.clients: Set = set()
        self.broadcast_queue: asyncio.Queue[str] = asyncio.Queue()

        self.audio_queue: "asyncio.Queue[np.ndarray]" = asyncio.Queue(maxsize=400)

        self.current_segment: List[np.ndarray] = []
        self.current_segment_start_time: Optional[float] = None
        self.last_voice_time: Optional[float] = None
        self.last_partial_time: float = 0.0

        self.final_transcript_lines: Deque[str] = deque(maxlen=max_lines)
        self.final_english_translation_lines: Deque[str] = deque(maxlen=max_lines)
        self.final_ai_translation_lines: Deque[str] = deque(maxlen=max_lines)

        self.history_english_lines: Deque[str] = deque(maxlen=max(64, self.ollama_context_lines + 1))


        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

    def attach_event_loop(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def clamp_int16(self, audio_float32: np.ndarray) -> np.ndarray:
        audio_float32 = np.clip(audio_float32, -1.0, 1.0)
        return (audio_float32 * 32767.0).astype(np.int16)

    def split_into_frames_int16(self, audio_int16: np.ndarray, sample_rate: int, frame_ms: int) -> List[bytes]:
        frame_size = int(sample_rate * frame_ms / 1000)
        total = (len(audio_int16) // frame_size) * frame_size
        audio_int16 = audio_int16[:total]
        frames: List[bytes] = []
        for i in range(0, total, frame_size):
            frames.append(audio_int16[i : i + frame_size].tobytes())
        return frames

    def vad_has_voice(self, audio_float32_16k: np.ndarray) -> bool:
        audio_int16 = self.clamp_int16(audio_float32_16k)
        frames = self.split_into_frames_int16(audio_int16, self.sample_rate_model, self.audio_config.frame_ms)
        if not frames:
            return False
        voiced = 0
        for frame in frames:
            if self.vad.is_speech(frame, self.sample_rate_model):
                voiced += 1
        return voiced >= max(1, int(0.2 * len(frames)))

    def start_audio_capture_thread(self) -> None:
        audio_interface = pyaudio.PyAudio()

        wasapi_host_info = audio_interface.get_host_api_info_by_type(pyaudio.paWASAPI)
        wasapi_host_index = int(wasapi_host_info["index"])
        default_output_device_index = int(wasapi_host_info["defaultOutputDevice"])

        default_output_info = audio_interface.get_device_info_by_index(default_output_device_index)
        output_name = str(default_output_info["name"])
        output_sample_rate = int(float(default_output_info["defaultSampleRate"]))

        loopback_device_index: Optional[int] = None

        for device_index in range(audio_interface.get_device_count()):
            dev = audio_interface.get_device_info_by_index(device_index)
            if int(dev.get("hostApi")) != wasapi_host_index:
                continue
            name = str(dev.get("name", ""))
            if "loopback" not in name.lower():
                continue
            if output_name.lower() in name.lower():
                loopback_device_index = int(device_index)
                break

        if loopback_device_index is None:
            for device_index in range(audio_interface.get_device_count()):
                dev = audio_interface.get_device_info_by_index(device_index)
                if int(dev.get("hostApi")) != wasapi_host_index:
                    continue
                name = str(dev.get("name", ""))
                if "loopback" in name.lower():
                    loopback_device_index = int(device_index)
                    break

        if loopback_device_index is None:
            raise RuntimeError("No WASAPI loopback device found via PyAudio.")

        loopback_info = audio_interface.get_device_info_by_index(loopback_device_index)

        channels = int(loopback_info["maxInputChannels"])
        if channels <= 0:
            channels = 2
        if channels > 2:
            channels = 2

        print(f"[audio] default output: {default_output_device_index} | {output_name} | {output_sample_rate} Hz")
        print(f"[audio] loopback input: {loopback_device_index} | {loopback_info['name']} | channels={channels}")

        frames_per_buffer = int(output_sample_rate * (self.audio_config.frame_ms / 1000.0))
        if frames_per_buffer < 64:
            frames_per_buffer = 64

        stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=output_sample_rate,
            input=True,
            input_device_index=loopback_device_index,
            frames_per_buffer=frames_per_buffer,
        )

        def capture_loop(audio_stream: pyaudio.Stream) -> None:
            while True:
                data = audio_stream.read(frames_per_buffer, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, dtype=np.int16)

                if channels > 1:
                    audio_int16 = audio_int16.reshape(-1, channels)
                    audio_float32 = (audio_int16.astype(np.float32) / 32768.0).mean(axis=1)
                else:
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                if output_sample_rate != self.sample_rate_model:
                    audio_float32 = soxr.resample(audio_float32, output_sample_rate, self.sample_rate_model).astype(
                        np.float32
                    )

                try:
                    self.audio_queue.put_nowait(audio_float32)
                except asyncio.QueueFull:
                    pass

        threading.Thread(target=capture_loop, args=(stream,), daemon=True).start()

    def _payload(self, partial: bool, transcript_lines: List[str], english_lines: List[str], ai_lines: List[str]) -> str:
        data = {
            "ts": time.time(),
            "partial": partial,
            "features": {
                "transcribe": self.feature_toggles.transcribe,
                "english_translate": self.feature_toggles.english_translate,
                "ai_translate": self.feature_toggles.ai_translate,
            },
            "transcript": "\n".join(transcript_lines[-self.max_lines:]),
            "translation": "\n".join(english_lines[-self.max_lines:]),
            "ai_translation": "\n".join(ai_lines[-self.max_lines:]),
        }
        return json.dumps(data, ensure_ascii=False)

    async def broadcast_loop(self) -> None:
        while True:
            msg = await self.broadcast_queue.get()
            dead: List = []
            for ws in list(self.clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

    async def send_state(self, partial: bool) -> None:
        msg = self._payload(
            partial=partial,
            transcript_lines=list(self.final_transcript_lines),
            english_lines=list(self.final_english_translation_lines),
            ai_lines=list(self.final_ai_translation_lines),
        )
        await self.broadcast_queue.put(msg)

    def _threadsafe_broadcast(self, msg: str) -> None:
        if self.event_loop is None:
            return

        async def _put() -> None:
            await self.broadcast_queue.put(msg)

        self.event_loop.call_soon_threadsafe(lambda: asyncio.create_task(_put()))

    def _concat_segment(self) -> np.ndarray:
        if not self.current_segment:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(self.current_segment).astype(np.float32)

    def _reset_segment_state(self) -> None:
        self.current_segment = []
        self.current_segment_start_time = None
        self.last_voice_time = None
        self.last_partial_time = 0.0

    async def _emit_partial(self, audio_full: np.ndarray) -> None:
        transcribed = ""
        english_translated = ""

        if self.feature_toggles.transcribe:
            transcribed = self.whisper_runner.run(audio_full, task="transcribe")

        if self.feature_toggles.english_translate:
            english_translated = self.whisper_runner.run(audio_full, task="translate")

        preview_transcript = list(self.final_transcript_lines)
        preview_english = list(self.final_english_translation_lines)
        preview_ai = list(self.final_ai_translation_lines)

        if transcribed:
            preview_transcript = (preview_transcript + [transcribed])[-self.max_lines:]

        if english_translated:
            preview_english = (preview_english + [english_translated])[-self.max_lines:]

        msg = self._payload(True, preview_transcript, preview_english, preview_ai)
        await self.broadcast_queue.put(msg)

    async def _finalize_segment(self, audio_full: np.ndarray) -> None:
        transcribed = ""
        english_translated = ""

        if self.feature_toggles.transcribe:
            transcribed = self.whisper_runner.run(audio_full, task="transcribe")

        if self.feature_toggles.english_translate:
            english_translated = self.whisper_runner.run(audio_full, task="translate")

        if transcribed:
            self.final_transcript_lines.append(transcribed)

        if english_translated:
            self.final_english_translation_lines.append(english_translated)
            self.history_english_lines.append(english_translated)

        await self.send_state(partial=False)

        if not self.feature_toggles.ai_translate:
            return

        latest_english = english_translated.strip()
        if not latest_english:
            latest_english = transcribed.strip()

        if not latest_english:
            return

        context_lines = list(self.history_english_lines)
        if context_lines and context_lines[-1].strip() == latest_english:
            context_lines = context_lines[:-1]

        context_lines = context_lines[-self.ollama_context_lines:]

        preview_transcript_lines = list(self.final_transcript_lines)
        preview_english_lines = list(self.final_english_translation_lines)
        preview_ai_lines = list(self.final_ai_translation_lines)

        def on_partial_text(partial_text: str) -> None:
            msg = self._payload(
                True,
                preview_transcript_lines,
                preview_english_lines,
                (preview_ai_lines + [partial_text])[-self.max_lines:],
            )
            self._threadsafe_broadcast(msg)

        ai_text = await self.ollama_client.stream_chat_translate(
            event_loop=self.event_loop,
            context_english_lines=context_lines,
            latest_english=latest_english,
            on_partial_text=on_partial_text,
        )

        if ai_text.strip():
            self.final_ai_translation_lines.append(ai_text.strip())

        await self.send_state(partial=False)

    async def processing_loop(self) -> None:
        min_speech_s = self.audio_config.min_speech_ms / 1000.0
        end_silence_s = self.audio_config.end_silence_ms / 1000.0

        while True:
            audio_chunk = await self.audio_queue.get()
            now = time.time()

            has_voice = self.vad_has_voice(audio_chunk)

            if has_voice:
                if self.current_segment_start_time is None:
                    self.current_segment_start_time = now

                self.last_voice_time = now
                self.current_segment.append(audio_chunk)

                segment_duration = (now - self.current_segment_start_time) if self.current_segment_start_time else 0.0

                if segment_duration >= self.audio_config.max_segment_s:
                    audio_full = self._concat_segment()
                    await self._finalize_segment(audio_full)
                    self._reset_segment_state()
                    continue

                if (now - self.last_partial_time) >= self.audio_config.partial_interval_s and segment_duration >= min_speech_s:
                    audio_full = self._concat_segment()
                    await self._emit_partial(audio_full)
                    self.last_partial_time = now
            else:
                if self.current_segment_start_time is None:
                    await asyncio.sleep(0)
                    continue

                segment_duration = now - self.current_segment_start_time
                time_since_voice = (now - self.last_voice_time) if self.last_voice_time else 999.0

                if segment_duration >= min_speech_s and time_since_voice >= end_silence_s:
                    audio_full = self._concat_segment()
                    await self._finalize_segment(audio_full)
                    self._reset_segment_state()

            await asyncio.sleep(0)
