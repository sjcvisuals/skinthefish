@echo off
setlocal
cd /d "%~dp0"
title Skinthefish

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>&1
  if %errorlevel%==0 (
    set "PY=python"
  ) else (
    echo Python was not found.
    echo.
    echo 1. Install Python 3.12 from https://www.python.org/downloads/windows/
    echo 2. On the first page of the installer, tick "Add python.exe to PATH".
    echo 3. Double-click Skinthefish.bat again.
    start https://www.python.org/downloads/windows/
    pause
    exit /b 1
  )
)

echo Installing the Python packages this app needs...
%PY% -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo.
  echo pip could not install the packages. Leave this window open and read the message above.
  pause
  exit /b 1
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo.
  echo FFmpeg was not found. Trying to install it with winget...
  winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
  echo.
  echo Close this window, open it again, and double-click Skinthefish.bat.
  echo Windows only sees a new FFmpeg after the window is closed.
  pause
  exit /b 1
)

echo.
echo Opening Skinthefish in your browser.
echo Leave this window open while you use the app.
echo.
%PY% "%~dp0studio.py"
if errorlevel 1 pause
