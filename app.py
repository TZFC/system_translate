import argparse
import asyncio
import json
import os
import time
from collections import deque
from typing import Deque, List, Optional, Set, Tuple

import numpy as np
import soxr
import sounddevice as sd
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def clamp_int16(audio_float32: np.ndarray) -> np.ndarray:
    audio_float32 = np.clip(audio_float32, -1.0, 1.0)
    return (audio_float32 * 32767.0).astype(np.int16)


def float32_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.ndim == 2:
        return np.mean(audio.astype(np.float32), axis=1)
    raise RuntimeError("unexpected audio shape")


def split_into_frames_int16(audio_int16: np.ndarray, sample_rate: int, frame_ms: int) -> List[bytes]:
    frame_size = int(sample_rate * frame_ms / 1000)
    total = (len(audio_int16) // frame_size) * frame_size
    audio_int16 = audio_int16[:total]
    frames = []
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
        target_translate_language: str,
        do_transcribe: bool,
        do_translate: bool,
        vad_aggressiveness: int,
        min_speech_ms: int,
        end_silence_ms: int,
        max_segment_s: float,
        partial_interval_s: float,
    ) -> None:
        self.host = host
        self.port = port

        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

        self.target_translate_language = target_translate_language
        self.do_transcribe = do_transcribe
        self.do_translate = do_translate

        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.max_segment_s = max_segment_s
        self.partial_interval_s = partial_interval_s

        self.sample_rate_model = 16000
        self.frame_ms = 20

        self.clients: Set[WebSocket] = set()
        self.broadcast_queue: asyncio.Queue[str] = asyncio.Queue()

        self.loopback_device_index: Optional[int] = None

        self.audio_queue: "asyncio.Queue[np.ndarray]" = asyncio.Queue(maxsize=200)

        self.current_segment: List[np.ndarray] = []
        self.current_segment_start_time: Optional[float] = None
        self.last_voice_time: Optional[float] = None
        self.last_partial_time: float = 0.0

        self.final_transcript_lines: Deque[str] = deque(maxlen=8)
        self.final_translation_lines: Deque[str] = deque(maxlen=8)

    def find_default_output_device(self) -> int:
        default_output = sd.default.device[1]
        if default_output is None or default_output < 0:
            raise RuntimeError("no default output device found")
        return int(default_output)

    def start_audio_capture_thread(self) -> None:
        output_device = self.find_default_output_device()
        self.loopback_device_index = output_device

        device_info = sd.query_devices(output_device)
        default_samplerate = int(device_info["default_samplerate"])

        wasapi_settings = sd.WasapiSettings(loopback=True)

        def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            audio = float32_mono(indata.copy())
            if default_samplerate != self.sample_rate_model:
                audio = soxr.resample(audio, default_samplerate, self.sample_rate_model).astype(np.float32)
            try:
                self.audio_queue.put_nowait(audio)
            except asyncio.QueueFull:
                pass

        sd.InputStream(
            samplerate=default_samplerate,
            channels=2,
            dtype="float32",
            device=output_device,
            extra_settings=wasapi_settings,
            blocksize=0,
            callback=audio_callback,
        ).start()

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

        segments, _info = self.model.transcribe(
            audio_16k_float32,
            task=task,
            language=None,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=True,
            temperature=0.0,
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

        if self.do_transcribe:
            transcribed = self.whisper_run(audio_full, task="transcribe")
        if self.do_translate:
            translated = self.whisper_run(audio_full, task="translate")

        preview_transcript = list(self.final_transcript_lines)
        preview_translation = list(self.final_translation_lines)

        if transcribed:
            preview_transcript = preview_transcript + [transcribed]
        if translated:
            preview_translation = preview_translation + [translated]

        payload = {
            "ts": time.time(),
            "partial": True,
            "transcript": "\n".join(preview_transcript[-8:]),
            "translation": "\n".join(preview_translation[-8:]),
        }
        await self.broadcast_queue.put(json.dumps(payload, ensure_ascii=False))

    async def finalize_segment(self, audio_full: np.ndarray) -> None:
        transcribed = ""
        translated = ""

        if self.do_transcribe:
            transcribed = self.whisper_run(audio_full, task="transcribe")
        if self.do_translate:
            translated = self.whisper_run(audio_full, task="translate")

        if transcribed:
            self.final_transcript_lines.append(transcribed)
        if translated:
            self.final_translation_lines.append(translated)

        await self.send_state(is_partial=False)


def build_app(server: LiveOverlayServer) -> FastAPI:
    app = FastAPI()

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

    @app.on_event("startup")
    async def startup_event() -> None:
        server.start_audio_capture_thread()
        asyncio.create_task(server.broadcast_loop())
        asyncio.create_task(server.processing_loop())

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=str, default="small")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_type", type=str, default="int8_float16")

    parser.add_argument("--translate_language", type=str, default="en")
    parser.add_argument("--transcribe", action="store_true", default=True)
    parser.add_argument("--no_transcribe", action="store_true", default=False)
    parser.add_argument("--translate", action="store_true", default=True)
    parser.add_argument("--no_translate", action="store_true", default=False)

    parser.add_argument("--vad_aggressiveness", type=int, default=2)
    parser.add_argument("--min_speech_ms", type=int, default=250)
    parser.add_argument("--end_silence_ms", type=int, default=550)
    parser.add_argument("--max_segment_s", type=float, default=12.0)
    parser.add_argument("--partial_interval_s", type=float, default=0.9)

    args = parser.parse_args()

    do_transcribe = bool(args.transcribe) and not bool(args.no_transcribe)
    do_translate = bool(args.translate) and not bool(args.no_translate)

    server = LiveOverlayServer(
        host=args.host,
        port=args.port,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        target_translate_language=args.translate_language,
        do_transcribe=do_transcribe,
        do_translate=do_translate,
        vad_aggressiveness=args.vad_aggressiveness,
        min_speech_ms=args.min_speech_ms,
        end_silence_ms=args.end_silence_ms,
        max_segment_s=args.max_segment_s,
        partial_interval_s=args.partial_interval_s,
    )

    app = build_app(server)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
