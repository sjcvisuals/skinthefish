@echo off
rem Sets PY for the launcher that called this file.
rem PyTorch publishes Windows CUDA wheels for Python 3.11 through 3.14.
rem Python 3.15, which "py -3" picks when it is installed, has none.
rem Pass "gpu" to install Python 3.13 when no compatible runtime is present.

set "PY="
call :try 3.13
if defined PY exit /b 0
call :try 3.12
if defined PY exit /b 0
call :try 3.14
if defined PY exit /b 0
call :try 3.11
if defined PY exit /b 0

if /I "%~1"=="gpu" goto install313

py -3 -c "import sys" >nul 2>&1 && set "PY=py -3" && exit /b 0
python -c "import sys" >nul 2>&1 && set "PY=python" && exit /b 0
echo Python was not found.
echo Install Python 3.13 from https://www.python.org/downloads/windows/
echo Then double-click Skinthefish.bat again.
pause
exit /b 1

:install313
echo.
echo This PC's Python is 3.15. PyTorch has no Windows CUDA build for 3.15.
echo Installing Python 3.13 next to it. Python 3.15 stays installed.
echo.
py install 3.13
call :try 3.13
if defined PY exit /b 0

where winget >nul 2>&1
if not errorlevel 1 (
  echo.
  echo Trying the Python 3.13 installer from winget...
  winget install -e --id Python.Python.3.13 --scope user --accept-package-agreements --accept-source-agreements
  call :try 3.13
  if defined PY exit /b 0
)

echo.
echo Python 3.13 did not install.
echo 1. Open https://www.python.org/downloads/windows/
echo 2. Install Python 3.13, not 3.15.
echo 3. Double-click Skinthefish-GPU.bat again.
start https://www.python.org/downloads/windows/
pause
exit /b 1

:try
py -%1 -c "import sys" >nul 2>&1 && set "PY=py -%1" && exit /b 0
exit /b 0
