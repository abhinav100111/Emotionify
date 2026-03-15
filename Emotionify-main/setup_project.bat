@echo off
setlocal

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher ^(`py`^) was not found. Install Python 3.12 and retry.
    exit /b 1
)

if exist .venv (
    echo Removing existing virtual environment...
    rmdir /s /q .venv
)

echo Creating virtual environment with Python 3.12...
py -3.12 -m venv .venv
if errorlevel 1 (
    echo Failed to create the virtual environment. Ensure Python 3.12 is installed.
    exit /b 1
)

echo Upgrading pip...
call .\.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo Installing project dependencies...
call .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Setup complete.
