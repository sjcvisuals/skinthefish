@echo off
setlocal
cd /d "%~dp0"
title Skinthefish GPU

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)

echo Installing the CUDA build of PyTorch. This is a large download.
echo The NVIDIA card will then do the fisheye resample. ProRes still encodes on the CPU.
echo.
%PY% -m pip install torch --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
  echo.
  echo PyTorch did not install. The app still runs on the CPU.
  pause
  exit /b 1
)

echo.
echo GPU support is installed. Opening Skinthefish.
echo.
%PY% "%~dp0studio.py"
if errorlevel 1 pause
