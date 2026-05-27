@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=C:\Users\user\Tektronix\TekScope\python3\bin\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

"%PYTHON_EXE%" -m streamlit run oas_realtime\app.py
if errorlevel 1 (
  echo.
  echo Failed to start OAS realtime dashboard.
  pause
)
