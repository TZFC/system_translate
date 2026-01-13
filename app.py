import argparse
import os
from contextlib import asynccontextmanager
from typing import Dict

import tomli
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from live_overlay import AudioConfig, FeatureToggles, LiveOverlay
from ollama_connect import OllamaClient, OllamaConfig
from run_whisper import WhisperConfig, WhisperRunner

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def load_toml_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomli.load(f)


def get_nested(config: dict, keys: list, default):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    args = parser.parse_args()

    config = load_toml_config(args.config)

    host = str(get_nested(config, ["server", "host"], "127.0.0.1"))
    port = int(get_nested(config, ["server", "port"], 8765))

    max_lines = int(get_nested(config, ["display", "max_lines"], 2))
    if max_lines <= 0:
        raise RuntimeError("display.max_lines must be >= 1")

    audio_config = AudioConfig(
        frame_ms=int(get_nested(config, ["audio", "frame_ms"], 20)),
        vad_aggressiveness=int(get_nested(config, ["audio", "vad_aggressiveness"], 2)),
        min_speech_ms=int(get_nested(config, ["audio", "min_speech_ms"], 400)),
        end_silence_ms=int(get_nested(config, ["audio", "end_silence_ms"], 700)),
        max_segment_s=float(get_nested(config, ["audio", "max_segment_s"], 12.0)),
        partial_interval_s=float(get_nested(config, ["audio", "partial_interval_s"], 0.9)),
    )

    feature_toggles = FeatureToggles(
        transcribe=bool(get_nested(config, ["features", "transcribe"], True)),
        english_translate=bool(get_nested(config, ["features", "english_translate"], True)),
        ai_translate=bool(get_nested(config, ["features", "ai_translate"], False)),
    )

    whisper_config = WhisperConfig(
        model=str(get_nested(config, ["whisper", "model"], "small")),
        device=str(get_nested(config, ["whisper", "device"], "cuda")),
        compute_type=str(get_nested(config, ["whisper", "compute_type"], "int8_float16")),
        beam_size=int(get_nested(config, ["whisper", "beam_size"], 1)),
        temperature=float(get_nested(config, ["whisper", "temperature"], 0.0)),
        condition_on_previous_text=bool(get_nested(config, ["whisper", "condition_on_previous_text"], False)),
        source_language=str(get_nested(config, ["whisper", "source_language"], "")).strip(),
        target_language=str(get_nested(config, ["whisper", "target_language"], "en")).strip().lower(),
        no_speech_threshold=float(get_nested(config, ["whisper", "no_speech_threshold"], 0.6)),
        log_prob_threshold=float(get_nested(config, ["whisper", "log_prob_threshold"], -1.0)),
        compression_ratio_threshold=float(get_nested(config, ["whisper", "compression_ratio_threshold"], 2.4)),
        repetition_penalty=float(get_nested(config, ["whisper", "repetition_penalty"], 1.1)),
    )

    if feature_toggles.english_translate and whisper_config.target_language != "en":
        raise RuntimeError('Whisper built-in translate outputs English only. Set whisper.target_language = "en".')

    whisper_runner = WhisperRunner(whisper_config)

    ollama_options: Dict = get_nested(config, ["ollama", "options"], {})
    if not isinstance(ollama_options, dict):
        raise RuntimeError("ollama.options must be a table/dict")

    ollama_config = OllamaConfig(
        base_url=str(get_nested(config, ["ollama", "base_url"], "http://127.0.0.1:11434")).strip(),
        model=str(get_nested(config, ["ollama", "model"], "qwen2.5:3b-instruct")).strip(),
        system_prompt=str(get_nested(config, ["ollama", "system_prompt"], "")).strip(),
        options=ollama_options,
        context_lines=int(get_nested(config, ["ollama", "context_lines"], 6)),
    )

    ollama_client = OllamaClient(ollama_config)

    live_overlay = LiveOverlay(
        max_lines=max_lines,
        audio_config=audio_config,
        feature_toggles=feature_toggles,
        whisper_runner=whisper_runner,
        ollama_client=ollama_client,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        live_overlay.attach_event_loop(asyncio.get_running_loop())
        live_overlay.start_audio_capture_thread()

        broadcast_task = asyncio.create_task(live_overlay.broadcast_loop())
        processing_task = asyncio.create_task(live_overlay.processing_loop())

        try:
            yield
        finally:
            broadcast_task.cancel()
            processing_task.cancel()

    import asyncio

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def root() -> HTMLResponse:
        with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        live_overlay.clients.add(websocket)
        await live_overlay.send_state(partial=False)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            live_overlay.clients.discard(websocket)

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
