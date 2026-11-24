@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Skinthefish

call "%~dp0select-python.bat"
if errorlevel 1 exit /b 1

echo Installing the Python packages this app needs...
!PY! -m pip install -r "%~dp0requirements.txt"
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
echo If export says it will take hours, double-click Skinthefish-GPU.bat once.
echo That installs Python 3.13 if needed, then puts the unwrap on the NVIDIA card.
echo.
!PY! "%~dp0studio.py"
if errorlevel 1 pause
