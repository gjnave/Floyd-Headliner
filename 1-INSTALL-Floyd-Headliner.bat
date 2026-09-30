@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Get Going Fast - Floyd Headliner Likeness Transfer Installer
cd /d "%~dp0"
IF EXIST "disclaimer.md" (
   TYPE "disclaimer.md"
   pause
)
IF EXIST "about.nfo" TYPE "about.nfo"

set "APP_DIR=%~dp0Floyd-Headliner"
set "VENV_DIR=%APP_DIR%\.venv"
set "CACHE_DIR=%~dp0installer-cache"
set "BACKUP_ROOT=%~dp0app-backups"
set "PYTHON_CMD="
set "UPDATE_ONLY=0"
if /I "%~1"=="/update" set "UPDATE_ONLY=1"

echo ============================================================
echo  Get Going Fast - Floyd Headliner Likeness Transfer Installer
echo  Codeberg app source + Qwen Image 2.1 - NO COMFYUI
echo ============================================================
echo.

if "%UPDATE_ONLY%"=="1" if not exist "%APP_DIR%\app.py" (
    echo ERROR: No installed app was found. Run this installer without /update first.
    pause
    exit /b 1
)

echo [1/6] Fetching the current Floyd Headliner app from Codeberg...
if not exist "%CACHE_DIR%" (
    mkdir "%CACHE_DIR%"
    if errorlevel 1 goto :source_failed
)
set "STAGE_DIR=%CACHE_DIR%\stage-%RANDOM%-%RANDOM%"
if exist "%STAGE_DIR%" goto :source_failed
mkdir "%STAGE_DIR%"
if errorlevel 1 goto :source_failed
curl.exe --fail --location --retry 3 --output "%STAGE_DIR%\main.zip" "https://codeberg.org/Cognibuild/Floyd-Headliner/archive/main.zip"
if errorlevel 1 goto :source_failed
tar.exe -xf "%STAGE_DIR%\main.zip" -C "%STAGE_DIR%"
if errorlevel 1 goto :source_failed
set "SOURCE_DIR=%STAGE_DIR%\floyd-headliner"
if not exist "%SOURCE_DIR%\app.py" goto :source_failed
if not exist "%SOURCE_DIR%\download_models.py" goto :source_failed
if not exist "%SOURCE_DIR%\requirements.txt" goto :source_failed
if not exist "%SOURCE_DIR%\inpaint_assets\canvas.html" goto :source_failed

set "BACKUP_DIR=%BACKUP_ROOT%\app-%RANDOM%-%RANDOM%"
if exist "%APP_DIR%\app.py" (
    if not exist "%BACKUP_ROOT%" (
        mkdir "%BACKUP_ROOT%"
        if errorlevel 1 goto :source_failed
    )
    echo Saving existing app files in "%BACKUP_DIR%"...
    robocopy "%APP_DIR%" "%BACKUP_DIR%" /E /XD "%APP_DIR%\.venv" "%APP_DIR%\models" "%APP_DIR%\outputs" /NFL /NDL /NJH /NJS >nul
    if errorlevel 8 goto :source_failed
)
if not exist "%APP_DIR%" (
    mkdir "%APP_DIR%"
    if errorlevel 1 goto :source_failed
)
robocopy "%SOURCE_DIR%" "%APP_DIR%" /E /XD "%SOURCE_DIR%\models" "%SOURCE_DIR%\outputs" /NFL /NDL /NJH /NJS >nul
if errorlevel 8 goto :source_failed
echo App files are current. Models, outputs, and the private environment were left in place.

py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.11"

if not defined PYTHON_CMD (
    py -3.10 -c "import sys; assert sys.version_info[:2] == (3, 10)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3.10"
)

if not defined PYTHON_CMD (
    python -c "import sys; assert sys.version_info >= (3, 10) and sys.version_info < (3, 12)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo ERROR: Python 3.10 or 3.11 was not found.
    echo Install 64-bit Python from https://www.python.org/downloads/windows/
    echo Enable "Add python.exe to PATH", then run this installer again.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [2/6] Creating the private Python environment...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create the Python environment.
        pause
        exit /b 1
    )
) else (
    echo [2/6] Existing private Python environment found.
)

if "%UPDATE_ONLY%"=="0" (
    echo [3/6] Updating pip tools...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 goto :install_failed
    echo [4/6] Installing NVIDIA PyTorch 2.10 with CUDA 13.0...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
    if errorlevel 1 goto :install_failed
    echo [5/6] Installing standalone app dependencies...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade -r "%APP_DIR%\requirements.txt"
) else (
    echo [3/6] Keeping the existing pip installation.
    echo [4/6] Keeping the existing PyTorch installation.
    echo [5/6] Installing any new app dependencies...
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r "%APP_DIR%\requirements.txt"
)
if errorlevel 1 goto :install_failed

echo [6/6] Checking the base model and both LoRAs...
echo Missing models download automatically; existing valid LoRAs are reused.
"%VENV_DIR%\Scripts\python.exe" "%APP_DIR%\download_models.py"
if errorlevel 1 goto :install_failed

"%VENV_DIR%\Scripts\python.exe" "%APP_DIR%\app.py" --self-check
if errorlevel 1 goto :install_failed

echo.
echo Floyd Headliner is ready.
if exist "%~dp02-START-Floyd-Headliner.bat" (
    echo Start the app with 2-START-Floyd-Headliner.bat
) else (
    echo Start the app with "%APP_DIR%\run.bat"
)
pause
exit /b 0

:source_failed
echo ERROR: Could not fetch, verify, back up, or copy the Codeberg app source.
echo Existing installed files and downloaded model files have not been deleted.
pause
exit /b 1

:install_failed
echo ERROR: Installation or validation failed. Review the message above.
echo Run this installer again to resume downloads. Existing files have not been deleted.
pause
exit /b 1
