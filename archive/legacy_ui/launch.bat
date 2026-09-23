@echo off
REM  EPR Sampling Tool launcher
REM  Run setup.ps1 first if you haven't already.
cd /d "%~dp0"

if not exist "%~dp0venv\Scripts\pythonw.exe" (
    echo Setup has not been run yet.
    echo Please right-click setup.ps1 and choose "Run with PowerShell" first.
    pause
    exit /b 1
)

start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0app.py"
