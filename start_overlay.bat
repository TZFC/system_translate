@echo off
setlocal

cd /d "%~dp0"

REM ---------- Virtual environment ----------
if not exist ".venv" (
  python -m venv .venv
)

call .venv\Scripts\activate.bat

REM ---------- Dependencies ----------
python -m pip install --upgrade pip

python -m pip install ^
  faster-whisper ^
  numpy ^
  soxr ^
  webrtcvad ^
  fastapi ^
  "uvicorn[standard]" ^
  pyaudiowpatch ^
  tomli

REM ---------- Start server ----------
start "" cmd /c ^
  ".venv\Scripts\python app.py --config config.toml"

REM ---------- Wait for server ----------
timeout /t 3 /nobreak >nul

REM ---------- Open overlay in default browser ----------
start "" "http://127.0.0.1:8765/"

REM ---------- Keep this window open ----------
echo.
echo Server running.
echo Overlay opened in browser.
echo Close this window to stop everything.
pause
