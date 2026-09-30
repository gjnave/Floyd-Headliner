@echo off
setlocal EnableExtensions
title Floyd Headliner
cd /d "%~dp0"

if exist "%~dp0assets\about.nfo" type "%~dp0assets\about.nfo"

if not exist ".venv\Scripts\python.exe" (
    echo Floyd Headliner is not installed yet. Run the installer first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "app.py"
if errorlevel 1 (
    echo.
    echo The app stopped with an error. Review the message above.
    pause
    exit /b 1
)
