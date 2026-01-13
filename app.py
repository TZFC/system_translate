import argparse
import asyncio
import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, List, Optional, Set, Tuple

import numpy as np
import pyaudiowpatch as pyaudio
import soxr
import tomli
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def load_toml_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomli.load(f)


def get_nested(config: dict, keys: List[str], default):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def clamp_int16(audio_float32: np.ndarray) -> np.ndarray:
    audio_float32 = np.clip(audio_float32, -1.0, 1.0)
    return (audio_float32 * 32767.0).astype(np.int16)


def split_into_frames_int16(audio_int16: np.ndarray, sample_rate: int, frame_ms: int) -> List[bytes]:
    frame_size = int(sample_rate * frame_ms / 1000)
    total = (len(audio_int16) // frame_size) * frame_size
    audio_int16 = audio_int16[:total]
    frames: List[bytes] = []
    for i in range(0, total, frame_size):
        frames.append(audio_int16[i : i + frame_size].tobytes())
    return frames


class LiveOverlayServer:
    def __init__(
        self,
        host: str,
        port: int,
        model_name: str,
        device: str,
        compute_type: str,
        mode: str,
        source_language: str,
        target_language: str,
        beam_size: int,
        temperature: float,
        condition_on_previous_text: bool,
        frame_ms: int,
        vad_aggressiveness: int,
        min_speech_ms: int,
        end_silence_ms: int,
        max_segment_s: float,
        partial_interval_s: float,
        max_lines: int,
    ) -> None:
        self.host = host
        self.port = port

        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

        self.mode = mode
        self.source_language = source_language
        self.target_language = target_language

        self.beam_size = beam_size
        self.temperature = temperature
        self.condition_on_previous_text = condition_on_previous_text

        self.sample_rate_model = 16000
        self.frame_ms = frame_ms

        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.max_segment_s = max_segment_s
        self.partial_interval_s = partial_interval_s

        self.clients: Set[WebSocket] = set()
        self.broadcast_queue: asyncio.Queue[str] = asyncio.Queue()

        self.audio_queue: "asyncio.Queue[np.ndarray]" = asyncio.Queue(maxsize=400)

        self.current_segment: List[np.ndarray] = []
        self.current_segment_start_time: Optional[float] = None
        self.last_voice_time: Optional[float] = None
        self.last_partial_time: float = 0.0

        self.final_transcript_lines: Deque[str] = deque(maxlen=max_lines)
        self.final_translation_lines: Deque[str] = deque(maxlen=max_lines)

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

        frames_per_buffer = int(output_sample_rate * (self.frame_ms / 1000.0))
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

    def vad_has_voice(self, audio_float32_16k: np.ndarray) -> bool:
        audio_int16 = clamp_int16(audio_float32_16k)
        frames = split_into_frames_int16(audio_int16, self.sample_rate_model, self.frame_ms)
        if not frames:
            return False
        voiced = 0
        for frame in frames:
            if self.vad.is_speech(frame, self.sample_rate_model):
                voiced += 1
        return voiced >= max(1, int(0.2 * len(frames)))

    def concat_segment(self) -> np.ndarray:
        if not self.current_segment:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(self.current_segment).astype(np.float32)

    def whisper_run(self, audio_16k_float32: np.ndarray, task: str) -> str:
        if audio_16k_float32.size == 0:
            return ""

        language_arg = self.source_language if self.source_language.strip() else None

        segments, _info = self.model.transcribe(
            audio_16k_float32,
            task=task,
            language=language_arg,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=self.condition_on_previous_text,
            temperature=self.temperature,
        )

        text_parts: List[str] = []
        for seg in segments:
            part = (seg.text or "").strip()
            if part:
                text_parts.append(part)
        return " ".join(text_parts).strip()

    async def broadcast_loop(self) -> None:
        while True:
            msg = await self.broadcast_queue.get()
            dead: List[WebSocket] = []
            for ws in list(self.clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

    async def send_state(self, is_partial: bool) -> None:
        payload = {
            "ts": time.time(),
            "partial": is_partial,
            "mode": self.mode,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "transcript": "\n".join(self.final_transcript_lines),
            "translation": "\n".join(self.final_translation_lines),
        }
        await self.broadcast_queue.put(json.dumps(payload, ensure_ascii=False))

    async def processing_loop(self) -> None:
        min_speech_s = self.min_speech_ms / 1000.0
        end_silence_s = self.end_silence_ms / 1000.0

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
                if segment_duration >= self.max_segment_s:
                    audio_full = self.concat_segment()
                    await self.finalize_segment(audio_full)
                    self.reset_segment_state()
                    continue

                if (now - self.last_partial_time) >= self.partial_interval_s and segment_duration >= min_speech_s:
                    audio_full = self.concat_segment()
                    await self.emit_partial(audio_full)
                    self.last_partial_time = now
            else:
                if self.current_segment_start_time is None:
                    await asyncio.sleep(0)
                    continue

                segment_duration = now - self.current_segment_start_time
                time_since_voice = (now - self.last_voice_time) if self.last_voice_time else 999.0

                if segment_duration >= min_speech_s and time_since_voice >= end_silence_s:
                    audio_full = self.concat_segment()
                    await self.finalize_segment(audio_full)
                    self.reset_segment_state()

            await asyncio.sleep(0)

    def reset_segment_state(self) -> None:
        self.current_segment = []
        self.current_segment_start_time = None
        self.last_voice_time = None
        self.last_partial_time = 0.0

    async def emit_partial(self, audio_full: np.ndarray) -> None:
        transcribed = ""
        translated = ""

        if self.mode in ("transcribe", "both"):
            transcribed = self.whisper_run(audio_full, task="transcribe")

        if self.mode in ("translate", "both"):
            translated = self.whisper_run(audio_full, task="translate")

        preview_transcript = list(self.final_transcript_lines)
        preview_translation = list(self.final_translation_lines)

        if transcribed:
            preview_transcript = (preview_transcript + [transcribed])[-self.final_transcript_lines.maxlen:]

        if translated:
            preview_translation = (preview_translation + [translated])[-self.final_translation_lines.maxlen:]


        payload = {
            "ts": time.time(),
            "partial": True,
            "mode": self.mode,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "transcript": "\n".join(preview_transcript),
            "translation": "\n".join(preview_translation),
        }
        await self.broadcast_queue.put(json.dumps(payload, ensure_ascii=False))

    async def finalize_segment(self, audio_full: np.ndarray) -> None:
        transcribed = ""
        translated = ""

        if self.mode in ("transcribe", "both"):
            transcribed = self.whisper_run(audio_full, task="transcribe")

        if self.mode in ("translate", "both"):
            translated = self.whisper_run(audio_full, task="translate")

        if transcribed:
            self.final_transcript_lines.append(transcribed)
        if translated:
            self.final_translation_lines.append(translated)

        await self.send_state(is_partial=False)


def build_app(server: LiveOverlayServer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        server.start_audio_capture_thread()
        broadcast_task = asyncio.create_task(server.broadcast_loop())
        processing_task = asyncio.create_task(server.processing_loop())
        try:
            yield
        finally:
            broadcast_task.cancel()
            processing_task.cancel()

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def root() -> HTMLResponse:
        with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        server.clients.add(websocket)
        await server.send_state(is_partial=False)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            server.clients.discard(websocket)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    args = parser.parse_args()

    config = load_toml_config(args.config)

    max_lines = int(get_nested(config, ["display", "max_lines"], 2))
    if max_lines <= 0:
        raise RuntimeError("display.max_lines must be >= 1")

    host = str(get_nested(config, ["server", "host"], "127.0.0.1"))
    port = int(get_nested(config, ["server", "port"], 8765))

    frame_ms = int(get_nested(config, ["audio", "frame_ms"], 20))
    vad_aggressiveness = int(get_nested(config, ["audio", "vad_aggressiveness"], 2))
    min_speech_ms = int(get_nested(config, ["audio", "min_speech_ms"], 250))
    end_silence_ms = int(get_nested(config, ["audio", "end_silence_ms"], 550))
    max_segment_s = float(get_nested(config, ["audio", "max_segment_s"], 12.0))
    partial_interval_s = float(get_nested(config, ["audio", "partial_interval_s"], 0.9))

    model_name = str(get_nested(config, ["whisper", "model"], "small"))
    device = str(get_nested(config, ["whisper", "device"], "cuda"))
    compute_type = str(get_nested(config, ["whisper", "compute_type"], "int8_float16"))
    beam_size = int(get_nested(config, ["whisper", "beam_size"], 1))
    temperature = float(get_nested(config, ["whisper", "temperature"], 0.0))
    condition_on_previous_text = bool(get_nested(config, ["whisper", "condition_on_previous_text"], True))

    mode = str(get_nested(config, ["whisper", "mode"], "both")).strip().lower()
    if mode not in ("transcribe", "translate", "both"):
        raise RuntimeError('whisper.mode must be one of: "transcribe", "translate", "both"')

    source_language = str(get_nested(config, ["whisper", "source_language"], "")).strip()
    target_language = str(get_nested(config, ["whisper", "target_language"], "en")).strip().lower()

    if mode in ("translate", "both") and target_language != "en":
        raise RuntimeError('Whisper built-in translate outputs English only. Set whisper.target_language = "en".')

    server = LiveOverlayServer(
        host=host,
        port=port,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        mode=mode,
        source_language=source_language,
        target_language=target_language,
        beam_size=beam_size,
        temperature=temperature,
        condition_on_previous_text=condition_on_previous_text,
        frame_ms=frame_ms,
        vad_aggressiveness=vad_aggressiveness,
        min_speech_ms=min_speech_ms,
        end_silence_ms=end_silence_ms,
        max_segment_s=max_segment_s,
        partial_interval_s=partial_interval_s,
        max_lines=max_lines,
    )

    app = build_app(server)

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
