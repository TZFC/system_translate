@echo off
setlocal ENABLEDELAYEDEXPANSION

cd /d "%~dp0"

echo ============================================================
echo system_translate launcher
echo ============================================================
echo.

REM ------------------------------------------------------------
REM Start Ollama API server in background
REM ------------------------------------------------------------
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Ollama is not installed or not in PATH.
    echo Please install Ollama from https://ollama.com
    pause
    exit /b 1
)

echo Starting Ollama API server...
start "" /B ollama serve

REM ------------------------------------------------------------
REM Wait for Ollama API to become ready
REM ------------------------------------------------------------
echo Waiting for Ollama API...

set OLLAMA_READY=0
for /L %%i in (1,1,30) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 > $null; exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! EQU 0 (
        set OLLAMA_READY=1
        goto ollama_ready
    )
    timeout /t 1 >nul
)

:ollama_ready
if "%OLLAMA_READY%" NEQ "1" (
    echo ERROR: Ollama API did not become ready.
    echo Make sure no firewall or permission issue is blocking it.
    pause
    exit /b 1
)

echo Ollama API is running.
echo.

REM ------------------------------------------------------------
REM Python virtual environment
REM ------------------------------------------------------------
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 goto fatal
)

call ".venv\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 goto fatal

REM ------------------------------------------------------------
REM Install dependencies (idempotent)
REM ------------------------------------------------------------
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 goto fatal

python -m pip install ^
    faster-whisper ^
    numpy ^
    soxr ^
    webrtcvad ^
    fastapi ^
    "uvicorn[standard]" ^
    pyaudiowpatch ^
    tomli
if %ERRORLEVEL% NEQ 0 goto fatal

REM ------------------------------------------------------------
REM Start application
REM ------------------------------------------------------------
echo Starting overlay server...
start "" /B cmd /c ".venv\Scripts\python app.py --config config.toml"

REM ------------------------------------------------------------
REM Open browser
REM ------------------------------------------------------------
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8765"

echo.
echo ============================================================
echo System running:
echo  - Ollama API
echo  - system_translate server
echo  - Browser overlay opened
echo ============================================================
pause
exit /b 0

:fatal
echo.
echo ============================================================
echo LAUNCH FAILED
echo Copy the output above and send it for debugging.
echo ============================================================
pause
exit /b 1
