@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Skinthefish GPU

call "%~dp0select-python.bat" gpu
if errorlevel 1 exit /b 1

echo.
echo Using !PY!
echo Installing the app packages, then the CUDA build of PyTorch.
echo The PyTorch download is large.
echo The NVIDIA card will then do the fisheye resample. ProRes still encodes on the CPU.
echo.

!PY! -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo.
  echo pip could not install the app packages. Leave this window open and read the message above.
  pause
  exit /b 1
)

set "INDEX="
for /f "delims=" %%I in ('!PY! "%~dp0cuda_index.py"') do set "INDEX=%%I"
if not defined INDEX set "INDEX=cu128"

echo Installing PyTorch !INDEX! ...
!PY! -m pip install torch --index-url https://download.pytorch.org/whl/!INDEX!
if errorlevel 1 (
  if /I not "!INDEX!"=="cu124" (
    echo.
    echo That wheel did not install. Trying the CUDA 12.4 build.
    !PY! -m pip install torch --index-url https://download.pytorch.org/whl/cu124
  )
)
if errorlevel 1 (
  echo.
  echo PyTorch did not install. The app still runs on the CPU, with one encode per core.
  pause
  exit /b 1
)

!PY! -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo.
  echo PyTorch is installed, but it cannot see the NVIDIA card.
  echo Update the Studio driver from https://www.nvidia.com/Download/index.aspx
  echo The app will still export on the CPU, using one encode per core.
  echo.
  pause
)

echo.
echo Opening Skinthefish.
echo Leave this window open while you use the app.
echo.
!PY! "%~dp0studio.py"
if errorlevel 1 pause
