@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv" (
  python -m venv .venv
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip

REM Core deps
python -m pip install faster-whisper numpy soxr webrtcvad fastapi "uvicorn[standard]" pyaudiowpatch

REM Run
python app.py --host 127.0.0.1 --port 8765 --model small --device cuda --compute_type int8_float16

pause
